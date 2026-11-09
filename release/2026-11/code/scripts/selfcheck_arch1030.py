#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARCH_H1（2026-10-30）自检：Haar 完美重构 / 初值等价 / 块均值不变 / 梯度通路 / 参数量。

不是训练、不碰数据、不出评测数字；只证明"新算子按预登记的数学在跑"，且
`g≡1` 时数值上精确回到基线网络。

跑法（服务器，项目根）:
    CUDA_VISIBLE_DEVICES=4 /DataDisk/miniconda/envs/xreflection/bin/python3 \\
        scripts/selfcheck_arch1030.py --device cuda --size 96 128

检查项
------
S1 Haar 正交/完美重构（FP32, 多尺寸，含非 8 倍数尺寸走 E/C 补边裁剪）
S2 调制器：参数量 2969；末层零初始化 ⇒ d_mod == d；g∈(0,2)；门控生效后确实改动 d
S3 块均值不变：任意门值下，d_mod 与原 d 的**对齐 8×8 块均值**逐块相等（FP32 atol）
S4 全网等价：同一份权重下，FullNet_NLP_Subband 的**最后一列**输出 == FullNet_NLP 输出
   （FP32 与 autocast-fp16 两种口径）；中间列逐元素相等
S5 梯度通路：初值时 out_conv 有梯度、隐藏层 in_conv/feat_proj 梯度为零（数学审结论）
S6 参数量：subband 段 2,969，总可训规模与基线一致（冻结口径由训练脚本负责）
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, HERE)
sys.path.insert(0, RDNET)
os.chdir(RDNET)

from models.arch.RDnet_ import FullNet_NLP                       # noqa: E402
from models.arch.rdnet_subband1030 import (                      # noqa: E402
    FullNet_NLP_Subband, ResidualSubbandModulator, haar2, ihaar2)
from rdnet_peft import patch_focalnet                            # noqa: E402

CKPT = os.path.join(PROJ, "pretrained/rdnet.pth")
FAIL = []


def check(name, ok, detail=""):
    print(f"[selfcheck] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- S1 Haar
def s1_haar():
    torch.manual_seed(0)
    for (h, w) in [(8, 8), (16, 24), (40, 40), (13, 7)]:
        hp, wp = h + (-h) % 8, w + (-w) % 8
        x = torch.randn(2, 3, hp, wp)
        ll, lh, hl, hh = haar2(x)
        y = ihaar2(ll, lh, hl, hh)
        err = float((y - x).abs().max())
        check(f"S1 完美重构 {hp}x{wp}", err < 1e-5, f"max|y-x|={err:.2e}")
        # Parseval：||x||² = Σ||系数||²（补边域内成立）
        e1 = float((x ** 2).sum())
        e2 = float(sum((c ** 2).sum() for c in (ll, lh, hl, hh)))
        check(f"S1 Parseval {hp}x{wp}", abs(e1 - e2) / max(e1, 1e-12) < 1e-6,
              f"rel={abs(e1 - e2) / e1:.2e}")
    # 三级深度 + 非 8 倍数走 E/C（与网络里同一套补边/裁剪代码路径）
    x = torch.rand(1, 3, 13, 7)
    hp, wp = 13 + (-13) % 8, 7 + (-7) % 8
    xp = F.pad(x, (0, wp - 7, 0, hp - 13), mode="replicate")
    cur = xp
    pyr = []
    for _ in range(3):
        c = haar2(cur)
        pyr.append(c)
        cur = c[0]
    rec = cur
    for lev in (2, 1, 0):
        rec = ihaar2(rec, pyr[lev][1], pyr[lev][2], pyr[lev][3])
    err = float((rec[..., :13, :7] - x).abs().max())
    check("S1 三级 + E/C 补边裁剪", err < 1e-5, f"max|重构图-原图|={err:.2e}")


# ------------------------------------------------------- S2/S3 调制器（CPU）
def s2s3_modulator():
    torch.manual_seed(1)
    m = ResidualSubbandModulator(feat_dim=64)
    n = sum(p.numel() for p in m.parameters())
    check("S2 参数量 == 2969", n == 2969, f"实际 {n}")
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    print("[selfcheck] subband state_dict 形状:", shapes, flush=True)

    x = torch.rand(2, 3, 64, 64)
    d = 0.2 * torch.randn(2, 3, 64, 64)
    feat = torch.randn(2, 64, 32, 32)
    d0 = m(x, d, feat)
    check("S2 零初始化 ⇒ d_mod==d", float((d0 - d).abs().max().detach()) < 1e-6,
          f"max|d_mod-d|={float((d0 - d).abs().max().detach()):.2e}")
    d1 = m(x, d, feat, gate_identity=True)
    check("S2 gate_identity ⇒ d_mod==d", float((d1 - d).abs().max().detach()) < 1e-6)

    # 打乱末层（模拟训练后）→ 门控必须真的起作用
    with torch.no_grad():
        nn.init.normal_(m.out_conv.weight, std=0.5)
        nn.init.normal_(m.out_conv.bias, std=0.5)
    d2 = m(x, d, feat)
    dev = float((d2 - d).abs().max().detach())
    check("S2 门控生效后确实改动 d", dev > 1e-4, f"max|d_mod-d|={dev:.2e}")
    aux = [float(v) for v in m.last_aux]
    check("S2 |g-1| 诊断可用", len(aux) == 3 and all(a > 0 for a in aux),
          f"每级 mean|g-1|={['%.4f' % a for a in aux]}")

    # g 值域：数学上 g = 1+tanh(z) ∈ (0,2)；FP32 下 |z|>~10 饱和到端点（下确界不可达）
    z = torch.linspace(-50, 50, 1001)
    g = 1.0 + torch.tanh(z)
    check("S2 g∈[0,2]（FP32 饱和到端点）", float(g.min()) >= 0.0 and float(g.max()) <= 2.0
          and float(1.0 + torch.tanh(torch.tensor(0.0))) == 1.0,
          f"g∈[{float(g.min()):.3f},{float(g.max()):.3f}]，z=0 时 g=1")

    # S3 对齐 8×8 块均值不变（LL 不调制）
    def block_mean(t):
        n_, c_, h_, w_ = t.shape
        return t.reshape(n_, c_, h_ // 8, 8, w_ // 8, 8).mean(dim=(3, 5))

    err = float((block_mean(d2) - block_mean(d)).abs().max())
    check("S3 8×8 块均值不变（任意门值）", err < 1e-5, f"max|Δ块均值|={err:.2e}")
    # 非 8 倍数尺寸：E/C 后仍成立（只在补边域内成块，故按补边尺寸比较）
    x2 = torch.rand(1, 3, 40, 56)
    d2_ = 0.1 * torch.randn(1, 3, 40, 56)
    f2 = torch.randn(1, 64, 20, 28)
    out = m(x2, d2_, f2)
    check("S3 补边尺寸下块均值不变(40x56)", float((block_mean(out) - block_mean(d2_)).abs().max()) < 1e-5)
    # dtype 透传（内部 FP32 计算 -> 输出 cast 回 d_pred.dtype；fp16 由 S4 autocast 覆盖）
    out_d = m(x.double(), d.double(), feat.double())
    check("S2 dtype 透传（double→double）", out_d.dtype == torch.float64 and torch.isfinite(out_d).all(),
          str(out_d.dtype))


# --------------------------------------------------------- S4/S5 全网（GPU）
def s4s5_network(dev, size):
    net_b = FullNet_NLP(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                        loss_col=4, num_classes=1000, drop_path=0, save_memory=True,
                        inter_supv=True, head_init_scale=None, kernel_size=3)
    patch_focalnet(net_b, "both", grad_ckpt=False, verbose=False)
    net_s = FullNet_NLP_Subband(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                                loss_col=4, num_classes=1000, drop_path=0, save_memory=True,
                                inter_supv=True, head_init_scale=None, kernel_size=3)
    patch_focalnet(net_s, "both", grad_ckpt=False, verbose=False)
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck.get("icnn", ck)
    mb, ub = net_b.load_state_dict(sd, strict=False)
    n_mona_b = len([k for k in mb if "mona" in k])
    check("S4 基线载入 rdnet.pth", len(mb) == n_mona_b and not ub,
          f"missing={len(mb)}(全为 mona? {len(mb) == n_mona_b}) unexpected={len(ub)}")
    # 新网络**整体复制**基线权重（含 mona 的随机初值）——否则两个实例的 mona 初值不同，
    # 输出必然不同，等价性检查就失去意义。
    ms, us = net_s.load_state_dict(net_b.state_dict(), strict=False)
    is_new = lambda k: ("mona" in k) or k.startswith("subband.")   # noqa: E731
    ms_new = [k for k in ms if k.startswith("subband.")]
    ms_core = [k for k in ms if not k.startswith("subband.")]
    check("S4 新网络仅新增 subband key 缺失",
          not ms_core and not us and len(ms_new) == 8,
          f"非新增 missing={len(ms_core)} unexpected={len(us)} subband missing={len(ms_new)}")

    # 权重逐一对齐（新网络的非新增参数 == 基线）
    sdb, sds = net_b.state_dict(), net_s.state_dict()
    diff = max(float((sdb[k].float() - sds[k].float()).abs().max()) for k in sdb)
    check("S4 同一份权重", diff == 0.0, f"max|Δw|={diff:.1e}")

    net_b.to(dev).eval()
    net_s.to(dev).eval()
    torch.manual_seed(7)
    x = torch.rand(1, 3, *size, device=dev)
    code = torch.rand(1, 6, device=dev)                 # alpha 走 prompt，随机即可
    with torch.no_grad():
        _, ob = net_b(x, code, prompt=True)
        _, os_ = net_s(x, code, prompt=True)
    check("S4 列数一致", len(ob) == len(os_) == 4, f"{len(ob)} vs {len(os_)}")
    for i in range(len(ob) - 1):
        e = float((ob[i] - os_[i]).abs().max())
        check(f"S4 中间列 {i} 逐元素相等", e == 0.0, f"max|Δ|={e:.1e}")
    e = float((ob[-1] - os_[-1]).abs().max())
    scale = float(ob[-1].abs().max())
    check("S4 最后一列 == 基线（FP32, 初值 g≡1）", e < 2e-5 * max(1.0, scale),
          f"max|Δ|={e:.2e} (输出幅值 {scale:.3f})")

    # 诊断开关：gate_identity=True 也必须回到基线（与上面同一 FP32 相位内比较）
    net_s.subband_gate_identity = True
    with torch.no_grad():
        _, os3 = net_s(x, code, prompt=True)
    e3 = float((ob[-1] - os3[-1]).abs().max())
    check("S4 gate_identity 推理开关回到基线（同相位 FP32）",
          e3 < 2e-5 * max(1.0, scale), f"max|Δ|={e3:.2e}")
    net_s.subband_gate_identity = False
    with torch.no_grad():
        _, os4 = net_s(x, code, prompt=True)
    e4 = float((os_[-1] - os4[-1]).abs().max())
    check("S4 关掉 gate_identity 后逐元素回到本网输出", e4 == 0.0, f"max|Δ|={e4:.2e}")

    # autocast fp16 口径（训练时真实口径）；只与同一 autocast 相位内的基线比
    net_s.requires_grad_(False)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
        _, ob2 = net_b(x, code, prompt=True)
        _, os2 = net_s(x, code, prompt=True)
    e16 = float((ob2[-1].float() - os2[-1].float()).abs().max())
    check("S4 最后一列 == 基线（autocast 口径）", e16 < 5e-3,
          f"max|Δ|={e16:.2e} (fp16 ulp 量级)")
    # 记录（不计失败）：跨精度相位后，同一张网的 FP32 输出会出现 fp16/TF32 量级
    # （~2^-11）的漂移，这是本机 torch 2.12 + cuDNN 的算法选择行为，不是算子带状态
    # （隔离测：调制器 FP32 重跑差值恒为 0）。训练/推理各自内部相位一致，不受影响。
    with torch.no_grad():
        _, os5 = net_s(x, code, prompt=True)
    info = float((os_[-1] - os5[-1]).abs().max())
    print(f"[selfcheck] INFO  S4 跨 autocast 相位后本网 FP32 漂移 = {info:.2e}"
          f"（fp16/TF32 量级，见注释；训练全程 autocast 故不受影响）", flush=True)

    # S5 梯度通路（初值）
    net_s.train()
    for p in net_s.parameters():
        p.requires_grad_(False)
    for p in net_s.subband.parameters():
        p.requires_grad_(True)
    g_param = None
    x2 = x.detach().clone().requires_grad_(False)
    with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
        _, og = net_s(x2, code, prompt=True)
        loss = (og[-1][:, 3:] - 0.5).abs().mean()
    loss.backward()
    gw = net_s.subband.out_conv.weight.grad
    g1 = net_s.subband.in_conv.weight.grad
    g2 = net_s.subband.feat_proj.weight.grad
    check("S5 末层 out_conv 首步有梯度", gw is not None and torch.isfinite(gw).all()
          and float(gw.abs().max()) > 0, f"max|grad|={float(gw.abs().max()):.2e}" if gw is not None else "None")
    check("S5 隐藏层首步梯度为零（零初始化）",
          (g1 is None or float(g1.abs().max()) == 0.0) and (g2 is None or float(g2.abs().max()) == 0.0),
          f"in_conv={None if g1 is None else float(g1.abs().max()):.1e} "
          f"feat_proj={None if g2 is None else float(g2.abs().max()):.1e}")
    check("S5 基线网络未被改动的参数键数一致",
          len([k for k in net_s.state_dict() if not is_new(k)])
          == len([k for k in net_b.state_dict() if not is_new(k)]),
          f"{len([k for k in net_s.state_dict() if not is_new(k)])} vs "
          f"{len([k for k in net_b.state_dict() if not is_new(k)])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", nargs=2, type=int, default=[96, 128])
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"[selfcheck] device={dev} torch={torch.__version__}", flush=True)
    s1_haar()
    s2s3_modulator()
    if not args.skip_full:
        s4s5_network(dev, tuple(args.size))
    print(f"\n[selfcheck] {'ALL PASS' if not FAIL else 'FAILED: %s' % FAIL}", flush=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
