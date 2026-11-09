#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I1（2026-10-31）工程自检：零初始化等价 / 关闭开关等价 / 梯度通路 / 显存步时 / ckpt 键。

不是训练、不碰训练数据、不出评测数字。只证明"新算子按预登记的数学在跑、且初始化时
数值上精确回到基线网络"，以及量出真实的显存/步时增量（审核要求的工程闸）。

跑法（服务器，项目根）::

    CUDA_VISIBLE_DEVICES=5 /DataDisk/miniconda/envs/xreflection/bin/python3 \\
        scripts/selfcheck_photo1031.py --device cuda

    # 追加：把 100 it 冒烟产出的 ckpt 严格载回同构网（strict=True）
    ... scripts/selfcheck_photo1031.py --device cuda --ckpt /tmp/idea1030/smoke_i1_s101/ckpt_final.pth

检查项（与审核更正逐条对应）
------------------------------
S1  参数量/键名：适配器 18,467（删掉空间末层 bias 后的值）；out_conv 无 bias；
    四列共享同一实例（state_dict 只有一套 photo.*）。
S2  零初始化前向等价：photo 网 vs 基线网，**四个监督列**的后 3 通道（T̂）逐元素相同；
    前 3 通道（R̂）也逐元素相同（同一 storage 视图）。FP32 与 autocast 两种口径。
S3  关闭开关等价：`photo_enabled=False` -> 与基线逐元素相同（且适配器未被调用）。
S4  旧参数梯度等价（step 0）：photo(on, 零初始化) 与基线对**全部共有参数**梯度相同；
    新末层（fc2 / out_conv）梯度有限且非零；上游新层（fc1/in_conv/dw）首步梯度为 0
    —— 预期现象（零初始化），不是死支路。
S4b 关闭开关的梯度等价：photo(off) 与基线梯度相同。
S5  一步更新之后：上游新层梯度变为非零（支路真的被打开），且输出确实偏离基线
    （非退化；同时说明"输出恒等于基线"不是永久属性）。
S6  反向不写坏旁路特征：hook 记录进适配器时 c0/c3 的 `_version`，反向后仍未变
    （没有原地写入；可逆反向重建不受影响）。
S7  ckpt 键完整性：rdnet.pth -> 只缺 mona+photo 新增键、unexpected=0；
    同构网 strict=True 重载自己的 state_dict；`--ckpt` 时对冒烟 ckpt 做同一断言。
S8  显存/步时（bs4、三个 crop、完整 Adam 状态、末层已更新后测）：photo 与基线两臂的
    peak allocated / reserved 与稳态步时；报告增量与"是否 <=10 GB"。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, HERE)
sys.path.insert(0, RDNET)
os.chdir(RDNET)

from models.arch.RDnet_ import FullNet_NLP                                    # noqa: E402
from models.arch.rdnet_photo1031 import (                                    # noqa: E402
    FullNet_NLP_Photo, GlobalSpatialPhotometricAdapter, PHOTO_N_PARAMS)
from rdnet_peft import is_mona_name, patch_focalnet                          # noqa: E402

CKPT = os.path.join(PROJ, "pretrained/rdnet.pth")
FAIL = []
NET_ARGS = dict(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                loss_col=4, num_classes=1000, drop_path=0, save_memory=True,
                inter_supv=True, head_init_scale=None, kernel_size=3)


def check(name, ok, detail=""):
    print(f"[selfcheck] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def info(msg):
    print(f"[selfcheck] INFO  {msg}", flush=True)


# ---------------------------------------------------------------- 构造
def build_base(grad_ckpt=False):
    net = FullNet_NLP(**{**NET_ARGS, "channels": list(NET_ARGS["channels"])})
    patch_focalnet(net, "both", grad_ckpt=grad_ckpt, verbose=False)
    return net


def build_photo(grad_ckpt=False, enabled=True):
    net = FullNet_NLP_Photo(**{**NET_ARGS, "channels": list(NET_ARGS["channels"])},
                            photo_enabled=enabled)
    patch_focalnet(net, "both", grad_ckpt=grad_ckpt, verbose=False)
    return net


def load_shared(net_b, net_p, dev):
    """基线载 rdnet.pth；新网**整体复制**基线权重（含 mona 随机初值），只留 photo.* 未覆盖。"""
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck.get("icnn", ck)
    mb, ub = net_b.load_state_dict(sd, strict=False)
    miss_b = [k for k in mb if not is_mona_name(k)]
    check("S7a 基线载 rdnet.pth（非 mona missing=0）", not miss_b and not ub,
          f"missing={len(mb)}(非mona {len(miss_b)}) unexpected={len(ub)}")
    mp, up = net_p.load_state_dict(net_b.state_dict(), strict=False)
    miss_new = [k for k in mp if k.startswith("photo.")]
    miss_core = [k for k in mp if not k.startswith("photo.")]
    check("S7b 新网只缺 photo.* 新增键", not miss_core and not up and len(miss_new) > 0,
          f"非新增 missing={len(miss_core)} unexpected={len(up)} photo missing={len(miss_new)}")
    sdb, sdp = net_b.state_dict(), net_p.state_dict()
    diff = max(float((sdb[k].float() - sdp[k].float()).abs().max()) for k in sdb)
    check("S7c 共有权重逐一对一相同", diff == 0.0, f"max|Δw|={diff:.1e}")
    net_b.to(dev).eval()
    net_p.to(dev).eval()
    return net_b, net_p


def loss_fn(outs, tgt):
    """训练器的损失形状：所有监督列按 w_c=c+1 加权；只依赖 T̂（后 3 通道）。"""
    tot = None
    for i, o in enumerate(outs):
        v = (i + 1) * (o[:, 3:] - tgt).abs().mean()
        tot = v if tot is None else tot + v
    n = len(outs)
    return tot / (n * (n + 1) / 2.0)


# ---------------------------------------------------------------- S1
def s1_params(dev):
    m = GlobalSpatialPhotometricAdapter()
    n = sum(p.numel() for p in m.parameters())
    check("S1 适配器参数量 == 预登记 18,467", n == PHOTO_N_PARAMS,
          f"实际 {n}（审核更正：18,470−3）")
    parts = {k: sum(p.numel() for p in sub.parameters()) for k, sub in m.named_children()}
    info(f"S1 分项：{parts}  （合计 {n}）")
    check("S1 空间末层无 bias（常数场会被 mean_xy 吃掉）", m.out_conv.bias is None)
    check("S1 fc2 零初始化", float(m.fc2.weight.detach().abs().max()) == 0.0
          and float(m.fc2.bias.detach().abs().max()) == 0.0)
    check("S1 out_conv 零初始化", float(m.out_conv.weight.detach().abs().max()) == 0.0)
    check("S1 隐藏层非零初始化（fc1/in_conv/dw）",
          all(float(x.weight.detach().abs().max()) > 0 for x in (m.fc1, m.in_conv, m.dw)))

    net = build_photo(grad_ckpt=False)
    keys = sorted(k for k in net.state_dict() if k.startswith("photo."))
    n_ph = sum(p.numel() for p in net.photo.parameters())
    check("S1 网络里只有一套 photo.*（四列共享）", len(keys) == len(list(net.photo.state_dict())),
          f"photo keys={len(keys)} 参数={n_ph}")
    info(f"S1 photo keys: {keys}")
    check("S1 网络内适配器参数 == 18,467", n_ph == PHOTO_N_PARAMS)
    del net
    torch.cuda.empty_cache() if dev.type == "cuda" else None


# ---------------------------------------------------------------- S2/S3/S4
def s2s3_forward(dev, sizes=(96, 160)):
    net_b = build_base(grad_ckpt=False)
    net_p = build_photo(grad_ckpt=False)
    load_shared(net_b, net_p, dev)
    for size in sizes:
        torch.manual_seed(7)
        x = torch.rand(2, 3, size, size, device=dev)
        code = torch.rand(2, 6, device=dev)
        with torch.no_grad():
            _, ob = net_b(x, code, prompt=True)
            _, op = net_p(x, code, prompt=True)
        check(f"S2 列数=4 @{size}", len(ob) == len(op) == 4, f"{len(ob)} vs {len(op)}")
        for i in range(len(ob)):
            e = float((ob[i] - op[i]).abs().max())
            check(f"S2 col{i} 六通道逐元素 == 基线（FP32, 零初始化）", e == 0.0, f"max|Δ|={e:.1e}")
        # 前 3 通道本来就是同一 storage 的切片；再单独确认 R̂ 与 T̂ 都等价
        eR = max(float((ob[i][:, :3] - op[i][:, :3]).abs().max()) for i in range(len(ob)))
        eT = max(float((ob[i][:, 3:] - op[i][:, 3:] - 0.0).abs().max()) for i in range(len(ob)))
        check(f"S2 R̂/T̂ 分别等价 @{size}", eR == 0.0 and eT == 0.0, f"R={eR:.1e} T={eT:.1e}")

        # autocast 口径（训练时的真实精度）
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
            _, ob2 = net_b(x, code, prompt=True)
            _, op2 = net_p(x, code, prompt=True)
        e16 = max(float((ob2[i].float() - op2[i].float()).abs().max()) for i in range(len(ob2)))
        check(f"S3 autocast 口径逐元素 == 基线 @{size}", e16 == 0.0, f"max|Δ|={e16:.1e}")

    # S4 关开关
    net_o = build_photo(grad_ckpt=False, enabled=False)
    net_o.load_state_dict(net_b.state_dict(), strict=False)
    net_o.to(dev).eval()
    torch.manual_seed(11)
    x = torch.rand(2, 3, 96, 96, device=dev)
    code = torch.rand(2, 6, device=dev)
    with torch.no_grad():
        _, ob = net_b(x, code, prompt=True)
        _, oo = net_o(x, code, prompt=True)
    e = max(float((ob[i] - oo[i]).abs().max()) for i in range(len(ob)))
    check("S4 关闭开关 == 基线（逐元素）", e == 0.0, f"max|Δ|={e:.1e}")
    check("S4 关闭开关时适配器未被调用", net_o.photo_diag is None)
    del net_o
    return net_b, net_p


# ---------------------------------------------------------------- S4/S4b/S5
def freeze_backbone(net):
    """照训练脚本口径冻结 baseball.*（含 Mona 之外的全部 FocalNet），再显式解冻 Mona。"""
    from rdnet_peft import set_mona_trainable
    for n, p in net.named_parameters():
        if n.startswith("baseball."):
            p.requires_grad_(False)
    set_mona_trainable(net, True)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _grads(net, x, tgt, code, seed=13, keep=True):
    """一次前向+反向；keep=True 时返回 {参数名: grad.detach().clone()}（只含 requires_grad 的）。"""
    net.train()
    net.zero_grad(set_to_none=True)
    torch.manual_seed(seed)              # 复原 dropout 随机流，两网严格同相位
    _, outs = net(x, code, prompt=True)
    loss = loss_fn(outs, tgt)
    loss.backward()
    g = None
    if keep:
        g = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}
    return loss, outs, g


def _cmp(ga, gb_):
    """返回 (最大相对差, 该处绝对差, 参数名, 比较个数)。相对 = |Δg|max / |g|max。"""
    worst_rel, worst_abs, wk, n = 0.0, 0.0, "", 0
    for k, g in ga.items():
        h = gb_.get(k, None)
        if h is None:
            continue
        a = float((g - h).abs().max())
        scale = max(float(g.abs().max()), 1e-12)
        n += 1
        if a / scale > worst_rel:
            worst_rel, worst_abs, wk = a / scale, a, k
    return worst_rel, worst_abs, wk, n


def s4s5_grads(dev, net_b, net_p):
    """显存纪律：每步最多同时保留 2 份梯度字典（≈0.26 GB/份），用完立刻 del。"""
    size = 96
    n_tr = freeze_backbone(net_b)
    check("S4 冻结口径与训练一致（baseball.* 冻、Mona 解冻）", n_tr > 0,
          f"可训 {n_tr/1e6:.2f}M 参数")
    freeze_backbone(net_p)
    torch.manual_seed(13)
    x = torch.rand(2, 3, size, size, device=dev)
    tgt = torch.rand(2, 3, size, size, device=dev)
    code = torch.rand(2, 6, device=dev)

    net_p.photo_enabled = True
    loss_b, outs_b, gb = _grads(net_b, x, tgt, code)
    loss_p, _, gp = _grads(net_p, x, tgt, code)
    check("S4 零初始化时两网 loss 相同", float((loss_b - loss_p).abs()) == 0.0,
          f"Δ={float((loss_b - loss_p).abs()):.2e}")
    del loss_p

    # 先量参照噪声：**同一张基线网重跑一遍**的梯度差（GPU 非确定性 / cuDNN 算法差异）
    _, _, gb2 = _grads(net_b, x, tgt, code)
    noise_rel, noise_abs, nk, _ = _cmp(gb, gb2)
    del gb2
    cross_rel, cross_abs, ck_, ncmp = _cmp(gb, gp)
    del gp
    torch.cuda.empty_cache()
    tol = max(3.0 * noise_rel, 1e-6)
    check("S4 旧参数梯度等价（step 0，同相位、噪声内）", cross_rel <= tol,
          f"{ncmp} 参数 max_rel={cross_rel:.2e}（同网重跑噪声 {noise_rel:.2e}, "
          f"tol {tol:.2e}）abs={cross_abs:.2e} @{ck_}")
    info(f"S4 噪声参照：同网重跑 max_rel={noise_rel:.2e} abs={noise_abs:.2e} @{nk}")

    # 关掉 TF32 再测一次（把 cuDNN TF32 舍入从等价性判决里剔掉）
    old_tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _, _, gb3 = _grads(net_b, x, tgt, code)
        _, _, gp3 = _grads(net_p, x, tgt, code)
        c2, c2a, c2k, _ = _cmp(gb3, gp3)
        del gp3
        _, _, gb4 = _grads(net_b, x, tgt, code)
        n2, n2a, n2k, _ = _cmp(gb3, gb4)
        del gb3, gb4
        tol2 = max(3.0 * n2, 1e-7)
        check("S4 旧参数梯度等价（关 TF32，更严）", c2 <= tol2,
              f"max_rel={c2:.2e}（同网噪声 {n2:.2e}, tol {tol2:.2e}）abs={c2a:.2e} @{c2k}")
        info(f"S4 关TF32 噪声参照：max_rel={n2:.2e} abs={n2a:.2e} @{n2k}")
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.cuda.empty_cache()

    # ---- S4b 关闭开关的梯度等价（同网重跑噪声内）
    net_p.photo_enabled = False
    _, _, go = _grads(net_p, x, tgt, code)
    net_p.photo_enabled = True
    cross_o, abs_o, ck_o, _ = _cmp(gb, go)
    del go, gb
    torch.cuda.empty_cache()
    check("S4b 关闭开关：全部参数梯度等价（噪声内）", cross_o <= tol,
          f"max_rel={cross_o:.2e}（tol {tol:.2e}）abs={abs_o:.2e} @{ck_o}")

    # ---- 首步梯度通路：末层有梯度 / 上游为零（梯度就地读取，不另存副本）
    _grads(net_p, x, tgt, code, keep=False)
    g_out = net_p.photo.out_conv.weight.grad
    g_fc2 = net_p.photo.fc2.weight.grad
    g_fc2b = net_p.photo.fc2.bias.grad
    check("S4 新末层 out_conv 首步梯度有限且非零",
          g_out is not None and torch.isfinite(g_out).all() and float(g_out.abs().max()) > 0,
          f"max|g|={None if g_out is None else float(g_out.abs().max()):.3e}")
    check("S4 新末层 fc2 首步梯度有限且非零",
          g_fc2 is not None and torch.isfinite(g_fc2).all() and float(g_fc2.abs().max()) > 0
          and g_fc2b is not None and float(g_fc2b.abs().max()) > 0,
          f"fc2.max|g|={None if g_fc2 is None else float(g_fc2.abs().max()):.3e}")
    up = {k: getattr(net_p.photo, k).weight.grad for k in ("fc1", "in_conv", "dw")}
    zero_up = all(g is None or float(g.abs().max()) == 0.0 for g in up.values())
    check("S4 上游新层首步梯度为 0（零初始化预期，不是死支路）", zero_up,
          " ".join(f"{k}={None if g is None else float(g.abs().max()):.2e}" for k, g in up.items()))

    # ---- S5 一步更新后：上游新层打开 + 输出偏离基线
    opt = torch.optim.Adam([p for p in net_p.parameters() if p.requires_grad], lr=1e-3)
    opt.step()                      # 上一步 backward 的梯度已就位（on 臂）
    opt.zero_grad(set_to_none=True)
    net_p.train()
    torch.manual_seed(13)
    _, outs_p2 = net_p(x, code, prompt=True)
    loss_fn(outs_p2, tgt).backward()
    up2 = {k: getattr(net_p.photo, k).weight.grad for k in ("fc1", "in_conv", "dw")}
    check("S5 一步更新后上游新层梯度非零（支路已打开）",
          all(g is not None and float(g.abs().max()) > 0 for g in up2.values()),
          " ".join(f"{k}={None if g is None else float(g.abs().max()):.2e}" for k, g in up2.items()))
    net_p.zero_grad(set_to_none=True)
    net_p.eval()
    with torch.no_grad():
        _, outs_p3 = net_p(x, code, prompt=True)
    dev_out = max(float((outs_b[i][:, 3:] - outs_p3[i][:, 3:]).abs().max()) for i in range(len(outs_b)))
    check("S5 更新后输出确实偏离基线（非退化）", dev_out > 1e-6, f"max|ΔT̂|={dev_out:.2e}")
    d = net_p.photo_diag
    if d:
        st = [v.tolist() for v in d["stats"]]
        info(f"S5 诊断：cols={d['cols']} |b|={[round(r[0], 5) for r in st]} "
             f"stdS={[round(r[1], 5) for r in st]}")
    del opt, outs_p2, outs_p3


# ---------------------------------------------------------------- S6
def s6_no_inplace(dev, net_b, net_p):
    seen = {}

    def hook(mod, args):
        x_in, f0, f3, y_t = args
        seen["f0"] = f0
        seen["f3"] = f3
        seen["v"] = (f0._version, f3._version)

    h = net_p.photo.register_forward_pre_hook(hook)
    net_p.train()
    torch.manual_seed(17)
    x = torch.rand(2, 3, 96, 96, device=dev)
    code = torch.rand(2, 6, device=dev)
    _, outs = net_p(x, code, prompt=True)
    loss_fn(outs, torch.rand(2, 3, 96, 96, device=dev)).backward()
    h.remove()
    v0 = seen.get("v")
    v1 = (seen["f0"]._version, seen["f3"]._version)
    check("S6 反向未原地改写旁路特征 c0/c3（_version 不变）", v0 == v1, f"前={v0} 后={v1}")


# ---------------------------------------------------------------- S7 ckpt
def s7_ckpt(dev, path=None):
    net = build_photo(grad_ckpt=False)
    sd = net.state_dict()
    keys = [k for k in sd if k.startswith("photo.")]
    tmpdir = os.environ.get("SELFCHECK_TMP", "/tmp/idea1030")
    os.makedirs(tmpdir, exist_ok=True)
    tmp = os.path.join(tmpdir, "selfcheck_photo1031_sd.pth")
    torch.save({"icnn": sd, "peft": {"mode": "both"}, "arch": "photo1031"}, tmp)
    net2 = build_photo(grad_ckpt=False)
    ck = torch.load(tmp, map_location="cpu")
    try:
        net2.load_state_dict(ck["icnn"], strict=True)
        ok, msg = True, "strict=True 通过"
    except Exception as e:                       # noqa: BLE001
        ok, msg = False, str(e)[:160]
    check("S7d 同构网 strict=True 重载自身 state_dict", ok, msg)
    info(f"S7 photo keys 数={len(keys)}；ckpt 内 arch={ck.get('arch')}")
    del net, net2

    if path:
        fresh = build_photo(grad_ckpt=False)
        ck2 = torch.load(path, map_location="cpu")
        sd2 = ck2.get("icnn", ck2)
        miss, unex = fresh.load_state_dict(sd2, strict=False)
        miss_core = [k for k in miss if not (is_mona_name(k) or k.startswith("photo."))]
        miss_ph = [k for k in miss if k.startswith("photo.")]
        miss_mo = [k for k in miss if is_mona_name(k)]
        check(f"S7e 冒烟 ckpt 只缺新增键（{os.path.basename(path)}）",
              not miss_core and not unex and not miss_ph and not miss_mo,
              f"missing 非新增={len(miss_core)} photo={len(miss_ph)} mona={len(miss_mo)} "
              f"unexpected={len(unex)}")
        peft = ck2.get("peft") or {}
        info(f"S7e ckpt iter={ck2.get('iter')} peft={peft} "
             f"arch={ck2.get('arch')} photo={(ck2.get('photo') or {}).get('n_params')}")
        try:
            fresh.load_state_dict(sd2, strict=True)
            check("S7f 冒烟 ckpt strict=True 重载", True)
        except Exception as e:                   # noqa: BLE001
            check("S7f 冒烟 ckpt strict=True 重载", False, str(e)[:200])
        del fresh
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- S8 显存/步时
def bench(net, dev, crop, bs=4, steps=3, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(bs, 3, crop, crop, device=dev)
    tgt = torch.rand(bs, 3, crop, crop, device=dev)
    code = torch.rand(bs, 6, device=dev)
    net.train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    ts = []
    for s in range(steps + 1):
        if s == 1:                                   # warmup 之后 Adam 状态已分配、末层已更新
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        with torch.cuda.amp.autocast():
            _, outs = net(x, code, prompt=True)
            loss = loss_fn(outs, tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if s >= 1:
            ts.append(time.time() - t0)
    out = dict(peak_alloc=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
               peak_res=torch.cuda.max_memory_reserved(dev) / 2 ** 30,
               step=sum(ts) / len(ts), n_step=len(ts))
    del opt, scaler, x, tgt, code, outs, loss
    return out


def s8_mem(dev, crops=(224, 256, 320), bs=4):
    if dev.type != "cuda":
        info("S8 需要 CUDA，跳过（--device cpu 时不做显存闸）")
        return None
    res = {}
    for arm in ("base", "photo"):
        torch.cuda.empty_cache()
        net = build_base(grad_ckpt=True) if arm == "base" else build_photo(grad_ckpt=True)
        net.to(dev)
        res[arm] = {}
        for c in crops:
            r = bench(net, dev, c, bs=bs)
            res[arm][c] = r
            print(f"[selfcheck] bench {arm:5s} crop={c} bs={bs} "
                  f"peak_alloc={r['peak_alloc']:.2f}G peak_res={r['peak_res']:.2f}G "
                  f"step={r['step']*1000:.0f}ms", flush=True)
            torch.cuda.empty_cache()
        del net
        torch.cuda.empty_cache()
    for c in crops:
        da = res["photo"][c]["peak_alloc"] - res["base"][c]["peak_alloc"]
        dt = res["photo"][c]["step"] - res["base"][c]["step"]
        tot = res["photo"][c]["peak_alloc"]
        check(f"S8 crop={c} 显存增量 ≤2 GB", da <= 2.0,
              f"Δalloc={da*1024:+.0f} MiB（photo {res['photo'][c]['peak_alloc']:.2f}G）")
        check(f"S8 crop={c} 总占用 ≤10 GB（含 10 GiB 卡）", tot <= 10.0, f"{tot:.2f} GiB")
        info(f"S8 crop={c} 步时 {res['base'][c]['step']*1000:.0f} → "
             f"{res['photo'][c]['step']*1000:.0f} ms ({dt*1000:+.0f} ms, "
             f"{100*dt/max(res['base'][c]['step'],1e-9):+.1f}%)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="", help="额外做 strict 重载的 ckpt（冒烟产物）")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--sizes", nargs="+", type=int, default=[96, 160])
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"[selfcheck] device={dev} torch={torch.__version__} "
          f"cuda={torch.version.cuda}", flush=True)

    s1_params(dev)
    net_b, net_p = s2s3_forward(dev, sizes=tuple(args.sizes))
    s4s5_grads(dev, net_b, net_p)
    s6_no_inplace(dev, net_b, net_p)
    del net_b, net_p
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    s7_ckpt(dev, args.ckpt or None)
    if not args.skip_bench:
        s8_mem(dev)
    print(f"\n[selfcheck] {'ALL PASS' if not FAIL else 'FAILED: %s' % FAIL}", flush=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
