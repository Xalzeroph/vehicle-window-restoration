#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 有界残差倍率门：CPU 可学性探针（只读既有缓存；**不做任何 GPU 推理**）。

为什么有这个探针
----------------
I1（8x8 光度适配器 −0.06）与 N2（字典近端 −0.067 / Ω 内 −0.576）的共同教训：
"oracle 有空间"推不出"单帧可预测"。A1 的探针因此不测 oracle 上界，而测
**固定隐藏层 + 只拟合末层 17 个参数** 的条件预测能否在**未见场景**上产生正增益，
并与"训练折拟合的常数门"对照——后者能拿到同样的整体缩放收益，若动态门赢不过它，
就不该把收益归因于"空间路由"。

数学（与 scripts/rdnet_a1gate_1101.py 完全一致，只把末层当线性模型）
------------------------------------------------------------------
    T0, I, D = T0, I, I-T0
    h = SG(DW(Conv(u)))，u=[I,T0,D,Avg3(D)]
    h' = h * SCA(GAP(h))                      # 冻结的随机隐藏层（SCA 恒等初始化）
    a(x) = v^T [h'(x); 1]，clip 到 [-1,1]     # 只解 v（17 维）
    T^ = T0 - a * D
拟合目标（未 clip 的平方误差，审查已指出这不是真实损失下的最优解）：
    min_v Σ_x W(x) (a(x) - r(x))^2 + λ‖v‖²
    W(x)=Σ_c D_c(x)^2， r(x)=Σ_c D_c(x)(T_c(x)-T0_c(x)) / W(x)
闭式：A=Σ W [h;1][h;1]^T，b=Σ W r [h;1]，v=(A+λ·tr(A)/17·I)^{-1} b。

数据与分组（全部为非测试场景；禁用 real_pair_1019）
-------------------------------------------------
* I/T：data/train_sets/real_pairs_1020（may 130 / jun 44 = 174 对 / 7 场景）
* T0 ：runs/n2_1031/pcache/s{101,202,303}/<case>.png（既有缓存，由 R-mayjun_s* 生成）
* 每场景最多 2 个 64² 原尺度裁块，裁块位置由 sha256(case) 决定（不看 GT/不看误差）
* 预锁定划分：场景按 sha256(scene) 排序，后 3 个为验证、前 4 个为训练（写死在代码里）
* 另做 leave-one-scene-out（7 折）描述性统计（各折训练集重叠，不能当独立证据）

缓存资格（会写进 verdict，不可绕过）
----------------------------------
若产出 T0 的 ckpt 训练时见过这些场景，则本探针只能检验"固定已训练表示之上的新增头跨场景
预测"，**不足以放行三种子训练**。`cache_scope` 由 ckpt 的 manifest 自动判定，`release`
会被强制置为 blocked。用 `--cache-scope oof` 只在你能证明 T0 来自未见场景的冻结模型时使用。

预算：墙钟 300 s 硬上限（含读图/特征/拟合/评分），`OMP_NUM_THREADS<=4`，超时即记
status=timeout 且 release=blocked，不拿部分折放行。

用法
----
  PY=/DataDisk/miniconda/envs/xreflection/bin/python3
  $PY scripts/a1_gate_probe_cpu.py unit  --out runs/a1_1101/probe/unit.json
  $PY scripts/a1_gate_probe_cpu.py power --out runs/a1_1101/probe/power.json
  $PY scripts/a1_gate_probe_cpu.py probe --pcache-root runs/n2_1031/pcache \
       --out runs/a1_1101/probe/probe.json --time-budget 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rdnet_a1gate_1101 import DEFAULT_CFG, make_gate, param_report  # noqa: E402

POOL = os.path.join(PROJ, "data/train_sets/real_pairs_1020")
PCACHE = os.path.join(PROJ, "runs/n2_1031/pcache")

# ---------------- 预登记阈值（先写死，跑之前就存在） ----------------
GATE = dict(
    main_gain_db=0.26,      # 留出场景均值 ΔPSNR
    beat_constant_db=0.10,  # 必须超过同折常数门
    n_pos_frac=2 / 3,       # 至少 2/3 留出场景为正
    conservative_ge=0.0,    # 保守门（只允许 g in [0,1]）不得为负
    sat_frac_warn=0.05,     # 饱和率诊断阈值（不是放行门）
    lam_rel=1e-3,           # 岭回归相对正则（固定，不扫）
    crop=64,
    crops_per_scene=2,
    max_scenes=24,
    tau_quiet=0.05,         # L_quiet 同一尺度，仅用于事后分区诊断
    budget_s=300.0,
    threads=4,
)


def log(*a):
    print(*a, flush=True)


def h256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_rgb_u8(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # HWC uint8


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0.0, 255.0)).astype(np.uint8)


def bw_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """与 scripts/eval_1019_multi.bw_psnr 同口径：三通道 PSNR 的均值（HWC uint8）。"""
    v = []
    for c in range(a.shape[-1]):
        mse = float(np.mean((a[..., c].astype(np.float64) - b[..., c].astype(np.float64)) ** 2))
        v.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


# ======================================================================================
# unit
# ======================================================================================
def cmd_unit(args) -> int:
    t0 = time.time()
    cfg = dict(DEFAULT_CFG)
    rep = param_report(cfg)
    g = make_gate(cfg)

    # 1) 参数账
    n = sum(p.numel() for p in g.parameters())
    names = {k: sum(x.numel() for x in g._modules[k].parameters()) for k in g._modules}

    # 2) 零初始化精确性：a==0 -> T^==T0（float 逐位 + uint8 逐像素）
    torch.manual_seed(0)
    I = torch.rand(2, 3, 96, 128)
    T0 = torch.rand(2, 3, 96, 128) * 1.2 - 0.1         # 允许越界，模拟真实输出
    T = g(I, T0)
    Td = T.detach()
    dmax = float((Td - T0).abs().max())
    q_hat = np.rint(Td.mul(255).clamp(0, 255).numpy()).astype(np.uint8)
    q_0 = np.rint(T0.mul(255).clamp(0, 255).numpy()).astype(np.uint8)
    d_u8 = int((q_hat != q_0).sum())

    # 3) 数学一致性：把末层偏置设成 ±2，检查 a=±1 的两端
    with torch.no_grad():
        g.conv_out.weight.zero_()
        g.conv_out.bias.fill_(2.0)
    T_hi = g(I, T0).detach()
    err_hi = float((T_hi - (2 * T0 - I)).abs().max())      # a=+1 -> 2T0-I
    with torch.no_grad():
        g.conv_out.bias.fill_(-2.0)
    T_lo = g(I, T0).detach()
    err_lo = float((T_lo - I).abs().max())                 # a=-1 -> I
    # 恢复零初始化
    with torch.no_grad():
        g.conv_out.bias.zero_()

    # 4) 梯度：it1 只有末层有梯度；隐藏层在末层更新后才拿到梯度
    I2 = I.clone().requires_grad_(False)
    T0b = T0.clone()
    g.zero_grad()
    loss = ((g(I2, T0b) - 0.4) ** 2).mean()
    loss.backward()
    grad0 = {k: (None if p.grad is None else float(p.grad.norm())) for k, p in g.named_parameters()}
    with torch.no_grad():
        g.conv_out.weight.add_(torch.randn_like(g.conv_out.weight) * 0.05)
        g.conv_out.bias.add_(0.1)
    g.zero_grad()
    ((g(I2, T0b) - 0.4) ** 2).mean().backward()
    grad1 = {k: (None if p.grad is None else float(p.grad.norm())) for k, p in g.named_parameters()}

    # 5) hard clip 饱和区梯度为 0（审查要求必须报）
    with torch.no_grad():
        g.conv_out.bias.fill_(2.0)                          # a 恒被夹到 1
        g.conv_out.weight.zero_()
    g.zero_grad()
    ((g(I2, T0b) - 0.4) ** 2).mean().backward()
    grad_sat = float(g.conv_out.bias.grad.norm()) if g.conv_out.bias.grad is not None else None
    with torch.no_grad():
        g.conv_out.bias.zero_()

    # 6) RNG 隔离：构造/前向不得改变全局 RNG（否则同 seed A/B 的数据顺序会漂）
    st_cpu = torch.get_rng_state().clone()
    st_cuda = [torch.cuda.get_rng_state(i).clone() for i in range(torch.cuda.device_count())] \
        if torch.cuda.is_available() else None
    g3 = make_gate(cfg)
    _ = g3(I, T0)
    cpu_same = bool(torch.equal(st_cpu, torch.get_rng_state()))
    cuda_same = None
    if st_cuda is not None:
        cuda_same = all(bool(torch.equal(a, torch.cuda.get_rng_state(i)))
                        for i, a in enumerate(st_cuda))
    g4 = make_gate(cfg)
    same_w = all(bool(torch.equal(g3._modules[k].weight, g4._modules[k].weight))
                 for k in ("conv_in", "dw", "sca", "conv_out"))

    # 7) 回归自检：常数门应当精确恢复 a* = ΣWr/ΣW（防止列索引/符号写错）
    Xs = np.zeros((6, 17), np.float64)
    Xs[:, -1] = 1.0
    Ws = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rs = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    As = Xs.T @ (Ws[:, None] * Xs)
    bs = Xs.T @ (Ws * rs)
    vs = solve_ridge(As, bs, 1e-12)
    a_hat = float(vs[-1])
    a_ref = float((Ws * rs).sum() / Ws.sum())
    reg_err = abs(a_hat - a_ref)

    # 8) 预算
    flops = dict(mac_per_px=rep["mac_per_px"], gmac_320=rep["gmac_at_320"],
                 gmac_bs4_320=4 * rep["gmac_at_320"],
                 note="conv_in+DW+out；SCA 在 1x1 GAP 特征上，忽略")

    out = dict(
        status="ok",
        cfg=cfg, param_report=rep, n_param=n, per_module=names,
        exact_zero_init=dict(max_abs_diff=dmax, n_uint8_diff=d_u8,
                             asserted=bool(dmax == 0.0 and d_u8 == 0)),
        endpoint_math=dict(a_plus1_err=err_hi, a_minus1_err=err_lo,
                           asserted=bool(err_hi < 1e-5 and err_lo < 1e-5)),
        grads=dict(step1=grad0, after_out_update=grad1,
                   out_has_grad=bool(grad0.get("conv_out.weight")),
                   hidden_zero_at_step1=bool((grad0.get("conv_in.weight") or 0.0) == 0.0),
                   hidden_nonzero_after=bool((grad1.get("conv_in.weight") or 0.0) > 0.0)),
        clip_saturation=dict(sat_grad_norm=grad_sat,
                             note="hard clamp 饱和区 dL/dbias 严格为 0；训练必须报 sat_frac"),
        rng=dict(cpu_state_unchanged=cpu_same, cuda_state_unchanged=cuda_same,
                 same_seed_same_weights=same_w),
        regression_const=dict(a_hat=a_hat, a_ref=a_ref, err=reg_err,
                              asserted=bool(reg_err < 1e-9)),
        flops=flops,
        wall_s=round(time.time() - t0, 2),
    )
    ok = (out["exact_zero_init"]["asserted"] and out["endpoint_math"]["asserted"]
          and out["grads"]["out_has_grad"] and out["grads"]["hidden_zero_at_step1"]
          and out["grads"]["hidden_nonzero_after"] and cpu_same and (cuda_same in (None, True))
          and same_w and n == 1025 and out["regression_const"]["asserted"])
    out["unit_pass"] = bool(ok)
    _dump(out, args.out)
    log(f"[unit] pass={ok} param={n} exact={dmax:.3g}/{d_u8} "
        f"grad(it1 out={grad0.get('conv_out.weight'):.3g} hidden={grad0.get('conv_in.weight')}) "
        f"sat_grad={grad_sat} rng_cpu={cpu_same} wall={out['wall_s']}s")
    return 0 if ok else 2


# ======================================================================================
# power（纯算术，不读数据）
# ======================================================================================
def cmd_power(args) -> int:
    tcrit = 4.302652729911275          # t(0.975, df=2)
    m = 0.26
    sd_max = m * math.sqrt(3) / tcrit
    rows = []
    for sd in (0.05, 0.10, 0.15, 0.26, 0.44):
        half = tcrit * sd / math.sqrt(3)
        rows.append(dict(sd_paired=sd, ci_half_width=round(half, 4),
                         lower_if_mean_0p26=round(m - half, 4),
                         upper_if_mean_0p26=round(m + half, 4)))
    out = dict(
        status="ok",
        formula="95% CI = mean ± 4.303·sΔ/√3 (3 paired seeds)",
        target_effect_db=m,
        max_sd_for_ci_lower_gt0=round(sd_max, 4),
        note=("单臂 sd≈0.44 不能当配对 sd 用；若 sΔ 与 0.44 同量级，3 种子 CI 半宽≈1.09 dB，"
              "0.26 dB 的效应根本不可分辨。'3/3 为正'在零假设下的单侧 p=0.125，不是 5% 证据。"),
        table=rows,
        all_positive_p_one_sided=0.125,
        panel_note=("real_pair_1019 已被反复用于诊断与候选选择，不能再称完全未触碰的确认性测试集；"
                    "本轮冻结复测必须披露适应性使用。"),
    )
    _dump(out, args.out)
    log(f"[power] sΔ must be < {sd_max:.4f} dB for CI lower>0 at mean 0.26")
    return 0


# ======================================================================================
# probe
# ======================================================================================
def load_meta(pool: str, domains: str):
    with open(os.path.join(pool, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    keep = {d.strip().lower() for d in str(domains).split(",") if d.strip()}
    out = {}
    for k, m in meta.items():
        if keep and str(m.get("domain", "")).lower() not in keep:
            continue
        out[k] = m
    return out


def pick_crops(meta: dict, crop: int, per_scene: int):
    """按 sha256 选图与裁块位置（不看 GT、不看误差）。返回 [(scene, case, y0, x0)]。"""
    by_scene = {}
    for case, m in meta.items():
        by_scene.setdefault(str(m.get("scene")), []).append(case)
    picks = []
    for scene in sorted(by_scene):
        cases = sorted(by_scene[scene], key=lambda c: h256(c))
        for case in cases[:per_scene]:
            h = int(h256(case + "|crop")[:12], 16)
            picks.append([scene, case, h, None])       # 位置待读到 H,W 再定
    return picks


def fill_xy(picks, pool: str, crop: int):
    for p in picks:
        scene, case, h, _ = p
        im = read_rgb_u8(os.path.join(pool, "I", case + ".png"))
        H, W = im.shape[:2]
        y0 = h % max(1, H - crop + 1)
        x0 = int(h256(case + "|x")[:12], 16) % max(1, W - crop + 1)
        p[3] = (y0, x0, H, W)
    return picks

def image_design(gate, pool: str, pcache: str, seed: int, scene: str, case: str,
                 y0: int, x0: int, crop: int):
    I_u8 = read_rgb_u8(os.path.join(pool, "I", case + ".png"))
    T_u8 = read_rgb_u8(os.path.join(pool, "T", case + ".png"))
    T0_u8 = read_rgb_u8(os.path.join(pcache, f"s{seed}", case + ".png"))
    if not (I_u8.shape == T_u8.shape == T0_u8.shape):
        raise RuntimeError(f"shape mismatch {case}: {I_u8.shape} {T_u8.shape} {T0_u8.shape}")
    I = torch.from_numpy(I_u8.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    T0 = torch.from_numpy(T0_u8.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    with torch.no_grad():
        h, D = gate.features(I, T0)
    h = h[0].numpy()                                   # (16,H,W)
    D = D[0].numpy()                                   # (3,H,W)
    sl = (slice(None), slice(y0, y0 + crop), slice(x0, x0 + crop))
    hc = h[sl].reshape(h.shape[0], -1).T               # (N,16)
    Dc = D[sl].reshape(3, -1)                          # (3,N)
    Ic = I_u8[y0:y0 + crop, x0:x0 + crop].transpose(2, 0, 1).astype(np.float32).reshape(3, -1) / 255.0
    Tc = T_u8[y0:y0 + crop, x0:x0 + crop].transpose(2, 0, 1).astype(np.float32).reshape(3, -1) / 255.0
    T0c = T0_u8[y0:y0 + crop, x0:x0 + crop].transpose(2, 0, 1).astype(np.float32).reshape(3, -1) / 255.0
    W = (Dc ** 2).sum(0)                               # (N,)
    num = (Dc * (Tc - T0c)).sum(0)
    # a* = argmin_a Σ_c (T0_c - a·D_c - T_c)^2 = -Σ_c D_c (T_c - T0_c) / Σ_c D_c^2
    r = np.where(W > 1e-12, -num / np.maximum(W, 1e-12), 0.0)
    X = np.concatenate([hc, np.ones((hc.shape[0], 1), np.float32)], axis=1)   # (N,17)
    return dict(scene=scene, case=case, seed=seed, X=X.astype(np.float32),
                r=r.astype(np.float32), W=W.astype(np.float32),
                D=Dc.astype(np.float32), I=Ic, T=Tc, T0=T0c, crop=crop)


def scene_stats(recs):
    A = np.zeros((17, 17), np.float64)
    b = np.zeros(17, np.float64)
    n = 0
    for rec in recs:
        X, W, r = rec["X"].astype(np.float64), rec["W"].astype(np.float64), rec["r"].astype(np.float64)
        A += X.T @ (W[:, None] * X)
        b += X.T @ (W * r)
        n += X.shape[0]
    return A, b, n


def solve_ridge(A, b, lam_rel):
    tr = float(np.trace(A)) / A.shape[0]
    return np.linalg.solve(A + lam_rel * max(tr, 1e-12) * np.eye(A.shape[0]), b)


def eval_rec(rec, v):
    a = (rec["X"].astype(np.float64) @ v.astype(np.float64))
    a = np.clip(a, -1.0, 1.0)
    c = rec["crop"]
    D = rec["D"].astype(np.float64)
    Th = rec["T0"].astype(np.float64) - a[None, :] * D
    Th_u8 = to_u8(Th.reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    T_u8 = to_u8(rec["T"].reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    T0_u8 = to_u8(rec["T0"].reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    I_u8 = to_u8(rec["I"].reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    psnr = bw_psnr(Th_u8, T_u8)
    base = bw_psnr(T0_u8, T_u8)
    # 诊断分区（用 GT 的 |I-T|，oracle 划分，只事后看，不参与选择/放行）
    need = np.abs(I_u8.astype(np.float64) - T_u8.astype(np.float64)).mean(2) / 255.0
    quiet = need <= GATE["tau_quiet"] * 0.5
    loud = need >= GATE["tau_quiet"]
    mse_h = ((Th_u8.astype(np.float64) - T_u8.astype(np.float64)) ** 2).mean(2)
    mse_b = ((T0_u8.astype(np.float64) - T_u8.astype(np.float64)) ** 2).mean(2)
    q = dict(
        quiet_frac=float(quiet.mean()), loud_frac=float(loud.mean()),
        mse_ratio_quiet=float(mse_h[quiet].mean() / max(mse_b[quiet].mean(), 1e-9)) if quiet.any() else None,
        mse_ratio_loud=float(mse_h[loud].mean() / max(mse_b[loud].mean(), 1e-9)) if loud.any() else None,
    )
    return dict(case=rec["case"], scene=rec["scene"], seed=rec["seed"],
                psnr=psnr, psnr_base=base, delta=psnr - base,
                a_mean=float(a.mean()), a_abs=float(np.abs(a).mean()),
                a_sat=float((np.abs(a) >= 0.999).mean()), a_amp=float((a > 0).mean()),
                d_abs=float(np.abs(D).mean()), **q)


def eval_cons(rec, v):
    a = np.clip(rec["X"].astype(np.float64) @ v.astype(np.float64), -1.0, 0.0)  # g in [0,1]
    c = rec["crop"]
    Th = rec["T0"].astype(np.float64) - a[None, :] * rec["D"].astype(np.float64)
    Th_u8 = to_u8(Th.reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    T_u8 = to_u8(rec["T"].reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    return bw_psnr(Th_u8, T_u8)


def eval_const(rec, a_scalar):
    c = rec["crop"]
    Th = rec["T0"].astype(np.float64) - a_scalar * rec["D"].astype(np.float64)
    Th_u8 = to_u8(Th.reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    T_u8 = to_u8(rec["T"].reshape(3, c, c).transpose(1, 2, 0) * 255.0)
    return bw_psnr(Th_u8, T_u8)


def scene_mean(rows, key="delta"):
    by = {}
    for r in rows:
        by.setdefault(r["scene"], []).append(r[key])
    return {k: float(np.mean(v)) for k, v in by.items()}


def boot_deltas(vals, n_boot=10000, seed=0):
    """条件于固定拟合模型的逐图 bootstrap；<8 张图直接返回 None（不伪造 CI）。"""
    vals = np.asarray(vals, np.float64)
    if vals.size < 8:
        return None
    rng = np.random.default_rng(seed)
    means = [float(vals[rng.integers(0, vals.size, vals.size)].mean()) for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def cmd_probe(args) -> int:
    t0 = time.time()
    torch.set_num_threads(int(args.threads))
    cv2.setNumThreads(int(args.threads))
    budget = float(args.time_budget)

    pcache = args.pcache_root if os.path.isabs(args.pcache_root) else os.path.join(PROJ, args.pcache_root)
    pool = args.pool if os.path.isabs(args.pool) else os.path.join(PROJ, args.pool)
    with open(os.path.join(pcache, "meta.json"), encoding="utf-8") as fh:
        pc_meta = json.load(fh)
    meta = load_meta(pool, args.domains)
    scenes = sorted({str(m.get("scene")) for m in meta.values()})
    if len(scenes) < 6:
        _dump(dict(status="insufficient_scenes", n_scene=len(scenes), release="blocked"), args.out)
        log(f"[probe] 场景只有 {len(scenes)} 个 < 6，证据不足，release=blocked")
        return 3
    if len(scenes) > GATE["max_scenes"]:
        scenes = scenes[: GATE["max_scenes"]]

    # ---- 预锁定场景划分：按 sha256(scene) 排序，后 3 为验证 ----
    order = sorted(scenes, key=lambda s: h256(s))
    n_val = 3 if len(order) >= 7 else max(1, len(order) // 2)
    val_scenes = order[-n_val:]
    train_scenes = [s for s in order if s not in val_scenes]

    picks = pick_crops({k: v for k, v in meta.items() if str(v.get("scene")) in order},
                       GATE["crop"], GATE["crops_per_scene"])
    picks = fill_xy(picks, pool, GATE["crop"])

    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    recs, skipped = [], []
    for seed in seeds:
        g = make_gate(dict(DEFAULT_CFG))
        for scene, case, h, (y0, x0, H, W) in picks:
            if H < GATE["crop"] or W < GATE["crop"]:
                skipped.append((seed, case, "too_small"))
                continue
            if not os.path.exists(os.path.join(pcache, f"s{seed}", case + ".png")):
                skipped.append((seed, case, "no_pcache"))
                continue
            recs.append(image_design(g, pool, pcache, seed, scene, case, y0, x0, GATE["crop"]))
            if time.time() - t0 > budget:
                _dump(dict(status="timeout", elapsed=time.time() - t0, release="blocked"), args.out)
                log(f"[probe] TIMEOUT {time.time()-t0:.0f}s > {budget}s -> release=blocked")
                return 3
        log(f"[probe] seed {seed}: {len(recs)} recs, {time.time()-t0:.1f}s")

    # ---- 主协议：预锁定 4 训练 / 3 验证（每个 seed 单独拟合） ----
    per_seed = {}
    for seed in seeds:
        rs = [r for r in recs if r["seed"] == seed]
        tr = [r for r in rs if r["scene"] in train_scenes]
        va = [r for r in rs if r["scene"] in val_scenes]
        A, b, n = scene_stats(tr)
        v = solve_ridge(A, b, GATE["lam_rel"])
        a_const = float(b[-1] / max(A[-1, -1], 1e-12))   # X 最后一列恒为 1
        rows = []
        for r in va:
            e = eval_rec(r, v)
            e["delta_const"] = eval_const(r, a_const) - e["psnr_base"]
            e["delta_cons"] = eval_cons(r, v) - e["psnr_base"]
            rows.append(e)
        per_seed[seed] = dict(
            train_scenes=train_scenes, val_scenes=val_scenes, n_train_px=n,
            a_const=a_const, rows=rows,
            v=[float(x) for x in v],
            lam_sensitivity={str(l): float(np.mean([eval_rec(r, solve_ridge(A, b, l))["delta"]
                                                   for r in va]))
                             for l in (1e-6, 1e-4, 1e-3, 1e-2)},
        )

    # ---- 汇总：先按场景平均再跨 seed 平均（与面板"逐图/逐场景"口径分开） ----
    def agg(key):
        per = []
        for _seed, d in per_seed.items():
            sm_ = scene_mean(d["rows"], key)
            per.append(float(np.mean(list(sm_.values()))))
        return per

    main_per_seed = agg("delta")
    const_per_seed = agg("delta_const")
    cons_per_seed = agg("delta_cons")

    sm = scene_mean([r for d in per_seed.values() for r in d["rows"]], "delta")
    sm_const = scene_mean([r for d in per_seed.values() for r in d["rows"]], "delta_const")
    sm_cons = scene_mean([r for d in per_seed.values() for r in d["rows"]], "delta_cons")
    n_pos = int(sum(1 for x in sm.values() if x > 0))
    n_val = len(val_scenes)
    per_img = {}
    for d in per_seed.values():
        for r in d["rows"]:
            per_img.setdefault((r["scene"], r["case"]), []).append(r["delta"])
    img_delta = [float(np.mean(v)) for v in per_img.values()]
    ci = boot_deltas(img_delta)
    sat = float(np.mean([r["a_sat"] for d in per_seed.values() for r in d["rows"]]))
    amp = float(np.mean([r["a_amp"] for d in per_seed.values() for r in d["rows"]]))
    d_abs = float(np.mean([r["d_abs"] for d in per_seed.values() for r in d["rows"]]))
    quiet_ratio = [r["mse_ratio_quiet"] for d in per_seed.values() for r in d["rows"]
                   if r["mse_ratio_quiet"] is not None]
    loud_ratio = [r["mse_ratio_loud"] for d in per_seed.values() for r in d["rows"]
                  if r["mse_ratio_loud"] is not None]

    # ---- LOO 描述性（7 折；训练集重叠，不能当独立证据） ----
    loo = {}
    for scene in order:
        rs = [r for r in recs if r["seed"] == seeds[0]]
        tr = [r for r in rs if r["scene"] != scene]
        va = [r for r in rs if r["scene"] == scene]
        A, b, _ = scene_stats(tr)
        v = solve_ridge(A, b, GATE["lam_rel"])
        loo[scene] = float(np.mean([eval_rec(r, v)["delta"] for r in va]))

    # ---- 缓存资格 ----
    cache = judge_cache(pcache, pc_meta, pool, set(order), args)
    # ---- 判决 ----
    time_ok = (time.time() - t0) <= budget
    g_main = bool(np.mean(main_per_seed) >= GATE["main_gain_db"])
    g_ci = bool(ci is not None and ci[0] > 0)
    g_pos = bool(n_pos >= math.ceil(GATE["n_pos_frac"] * n_val))
    g_const = bool(np.mean(main_per_seed) - np.mean(const_per_seed) >= GATE["beat_constant_db"])
    g_cons = bool(np.mean(cons_per_seed) >= GATE["conservative_ge"])
    numeric = dict(main_gain_db=g_main, ci_lower_gt0=g_ci, n_pos_scenes=g_pos,
                   beat_constant_db=g_const, conservative_ge=g_cons)
    numeric_pass = all(numeric.values())
    release = "eligible_for_3seed_AB" if (numeric_pass and cache["cache_qualifies_for_release"]
                                          and time_ok) else "blocked"
    reasons = []
    if not cache["cache_qualifies_for_release"]:
        reasons.append(f"cache_scope={cache['cache_scope']}：T0 产自见过这些场景的基线，"
                       "本探针只能检验固定表示之上的新增头跨组预测，不足以放行")
    if ci is None:
        reasons.append(f"n_val_scenes={n_val} / n_val_images={len(img_delta)}，"
                       "条件 bootstrap 样本不足，统计放行不成立")
    if not numeric_pass:
        reasons.append("数值门未全过：" + ",".join(k for k, v in numeric.items() if not v))
    if not time_ok:
        reasons.append("超 300 s 预算")

    out = dict(
        status="ok",
        protocol=dict(
            pool=os.path.relpath(pool, PROJ), pcache=os.path.relpath(pcache, PROJ),
            domains=args.domains, seeds=seeds, scenes=order,
            train_scenes=train_scenes, val_scenes=val_scenes,
            crop=GATE["crop"], crops_per_scene=GATE["crops_per_scene"],
            n_rec_used=len(recs), skipped=skipped,
            split_rule="scenes sorted by sha256(scene); last 3 = val (pre-locked)",
            note="每场景 2 个 64² 原尺度裁块，位置由 sha256(case) 决定；未读 real_pair_1019",
        ),
        gate_cfg=GATE,
        per_seed={str(k): dict(train_scenes=v["train_scenes"], val_scenes=v["val_scenes"],
                              a_const=v["a_const"], n_train_px=v["n_train_px"],
                              delta_scene_mean=round(float(np.mean(list(scene_mean(v["rows"]).values()))), 4),
                              delta_const=round(float(np.mean(list(scene_mean(v["rows"], "delta_const").values()))), 4),
                              delta_cons=round(float(np.mean(list(scene_mean(v["rows"], "delta_cons").values()))), 4),
                              lam_sensitivity={k: round(x, 4) for k, x in v["lam_sensitivity"].items()},
                              rows=v["rows"])
                 for k, v in per_seed.items()},
        summary=dict(
            delta_main=round(float(np.mean(main_per_seed)), 4),
            delta_main_per_seed=[round(x, 4) for x in main_per_seed],
            delta_const=round(float(np.mean(const_per_seed)), 4),
            delta_cons=round(float(np.mean(cons_per_seed)), 4),
            scene_delta={k: round(v, 4) for k, v in sm.items()},
            scene_delta_const={k: round(v, 4) for k, v in sm_const.items()},
            scene_delta_cons={k: round(v, 4) for k, v in sm_cons.items()},
            n_pos_scenes=n_pos, n_val_scenes=n_val,
            ci_images=ci, ci_note="条件于固定拟合模型的逐图 bootstrap，留出图<8 时为 None",
            a_sat_frac=round(sat, 4), a_amp_frac=round(amp, 4), d_abs_mean=round(d_abs, 4),
            quiet_mse_ratio=round(float(np.mean(quiet_ratio)), 4) if quiet_ratio else None,
            loud_mse_ratio=round(float(np.mean(loud_ratio)), 4) if loud_ratio else None,
            quiet_note="quiet/loud 用 GT |I-T| 划分（oracle），只作机制诊断，不参与选择",
        ),
        loo_descriptive={k: round(v, 4) for k, v in loo.items()},
        cache=cache,
        gates=numeric,
        release=release, release_reasons=reasons,
        wall_s=round(time.time() - t0, 1),
    )
    _dump(out, args.out)
    log(f"[probe] main={out['summary']['delta_main']:+.3f} const={out['summary']['delta_const']:+.3f} "
        f"cons={out['summary']['delta_cons']:+.3f} n_pos={n_pos}/{n_val} sat={sat:.4f} "
        f"cache={cache['cache_scope']} release={release} wall={out['wall_s']}s")
    return 0 if release.startswith("eligible") else 3


def judge_cache(pcache, pc_meta, pool, scenes, args):
    """判定 T0 缓存是否来自"没见过这些场景"的冻结模型。"""
    if args.cache_scope == "oof":
        return dict(cache_scope="oof_declared", cache_qualifies_for_release=True,
                    source="--cache-scope oof（用户声明，脚本无法验证）")
    tmpl = str(pc_meta.get("ckpt_tmpl", ""))
    seeds_meta = pc_meta.get("seeds") or [101]
    man = ""
    if tmpl:
        cand = tmpl.replace("{seed}", str(seeds_meta[0])).replace("ckpt_final.pth", "manifest.json")
        cand = cand if os.path.isabs(cand) else os.path.join(PROJ, cand)
        if os.path.exists(cand):
            man = cand
    if not man:
        return dict(cache_scope="unknown", cache_qualifies_for_release=False,
                    source=f"未找到 {tmpl!r} 的 manifest.json，无法判定训练场景")
    with open(man, encoding="utf-8") as fh:
        m = json.load(fh)
    a = m.get("args", {})
    om_dir = a.get("omega_dir", "")
    dom = a.get("omega_domains", "")
    try:
        train_meta = load_meta(om_dir if os.path.isabs(om_dir) else os.path.join(PROJ, om_dir), dom)
        train_scenes = {str(v.get("scene")) for v in train_meta.values()}
    except Exception as e:  # noqa: BLE001
        return dict(cache_scope="unknown", cache_qualifies_for_release=False,
                    source=f"读训练池失败: {e}")
    overlap = sorted(set(scenes) & train_scenes)
    in_sample = (overlap == sorted(scenes))
    return dict(
        cache_scope="in_sample_T0" if in_sample else "partial_or_oof",
        cache_qualifies_for_release=bool(not in_sample),
        source=os.path.relpath(man, PROJ),
        ckpt_tmpl=tmpl,
        train_domains=dom, train_scenes=sorted(train_scenes),
        probe_scenes=sorted(scenes), overlap=overlap,
        note=("基线训练池包含全部探针场景：只能检验固定已训练表示之上的新增头跨场景预测，"
              "不能证明完整系统泛化，不得据此开三种子" if in_sample else
              "探针场景未被该 ckpt 训练过；结论只针对该冻结源模型的残差分布"),
    )


def _dump(obj, path):
    path = path if os.path.isabs(path) else os.path.join(PROJ, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=float)
    log(f"[probe] -> {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unit")
    u.add_argument("--out", default="runs/a1_1101/probe/unit.json")
    u.set_defaults(func=cmd_unit)
    p = sub.add_parser("power")
    p.add_argument("--out", default="runs/a1_1101/probe/power.json")
    p.set_defaults(func=cmd_power)
    q = sub.add_parser("probe")
    q.add_argument("--pool", default=POOL)
    q.add_argument("--pcache-root", default=PCACHE)
    q.add_argument("--domains", default="may,jun")
    q.add_argument("--seeds", default="101,202,303")
    q.add_argument("--cache-scope", choices=["auto", "oof"], default="auto")
    q.add_argument("--time-budget", type=float, default=GATE["budget_s"])
    q.add_argument("--threads", type=int, default=GATE["threads"])
    q.add_argument("--out", default="runs/a1_1101/probe/probe.json")
    q.set_defaults(func=cmd_probe)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
