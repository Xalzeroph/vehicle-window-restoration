#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N2 可学性探针（CPU 版，2026-10-31，审查意见 modify 后重做）

两道闸
------
第一道（本脚本 gate1）：**训练域构造 → 测试域直接执行**。
  只用原训练池的 (I, T) 造 DT/DR，锁死采样权重、DC 处理、τ、ISTA 次数、h、η、padding、聚合；
  面板只把 I 与缓存 P（三个 F0 种子）喂给算子，GT 与人工 Ω **只在输出生成之后**用于评分。
  同时跑同口径对照：MODE_TONLY（只有透射字典、分母 2、无 i 项）与共享 RGB 仿射。
补充（本脚本 transfer）：训练域拟合 → 面板直测。
  α 诊断：LOMO 场景划分，α*=clip(〈d,e〉/‖d‖²,0,1) 只在拟合折求，留出场景直接预测 P+αd。
  共享仿射对照：只在训练域拟合（禁测试逐图拟合），再到面板直测。
第二道（训练桥接，GPU，200 步）：**不在本脚本**。冒烟脚本
  scripts/finetune_rdnet_s2_n2.py 的 bridge.json 提供 η/字典梯度与逐项损失梯度证据；
  缺 bridge.json 或步数不足 200 ⇒ 第二道记"未判"，不得声称可学性已验证。

铁律（写在代码里，别靠记忆）
----------------------------
  * 面板只提供 I 与 P 给算子；GT/Ω 在算子输出之后才读。
  * 同一场景的全部帧/种子同折；不做随机像素划分当"跨场景可学"证据。
  * 三种子结果**先按场景平均**，再做 15 场景簇 bootstrap；不把 45 个结果当独立样本。
  * ΔPSNR 逐图算（per-channel PSNR 再平均，与权威表 bw_psnr 同口径），不用池化 MSE。
  * 抽样只用于探针，不改面板与官方出表口径；最终判决必须是全图官方评测。
  * 超时/缺输入 = 未判，不记为通过，也不记为失败。
  * 判据阈值在文件顶 CRIT 里预登记；本脚本不扫超参、不按结果改判据。

用法
----
  # 0) 造字典（CPU，几十秒）
  python3 scripts/n2_probe_cpu.py build-dict \
      --pool data/train_sets/real_pairs_1020 --domains may,jun \
      --out runs/n2_1031/n2_dict_mayjun.json --per-case 2000
  # 1) 单元测试（η=0 精确性 / i=p 退化 / τ 全零分支 / 稠密 vs 卷积 / 收敛性 / 存盘一致）
  python3 scripts/n2_probe_cpu.py unit --dict runs/n2_1031/n2_dict_mayjun.json \
      --out runs/n2_1031/unit.json
  # 2) 第一道闸：面板直测（三种子）
  python3 scripts/n2_probe_cpu.py gate1 --dict runs/n2_1031/n2_dict_mayjun.json \
      --eta 1.0 --seeds 101,202,303 --out runs/n2_1031/gate1.json
  # 3) 训练域→面板（需要 pcache；没有就记未判）
  python3 scripts/n2_probe_cpu.py transfer --dict runs/n2_1031/n2_dict_mayjun.json \
      --pcache-root runs/n2_1031/pcache --out runs/n2_1031/transfer.json
  # 4) 汇总判决
  python3 scripts/n2_probe_cpu.py verdict --dir runs/n2_1031
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import torch                                                     # noqa: E402

from n2_dict_core import (                                       # noqa: E402
    K_INNER, MODE_DUAL, MODE_TONLY, N_ATOM, N_OUTER, N_PARAM_TOTAL, PATCH, RHO_TAU,
    H_TRACE_BOUND, build_dictionary, forward_dense_np, ista_step, load_config,
    n2_from_config, patch_bank, save_config, tau_from_bank,
)

# --------------------------- 预登记判据（不得事后改） ---------------------------
CRIT = {
    "main_gain_db": 0.50,          # 三种子平均 ΔPSNR（场景平均后再取均值）
    "ci_lower_gt": 0.0,            # 15 场景簇 95% CI 下界
    "min_pos_scenes": 11,          # ≥11/15 场景为正
    "omega_in_degrade_max": 0.05,  # Ω 内平均退化上限（dB，负=退化）
    "omega_out_degrade_max": 0.05,  # Ω 外平均退化上限
    "tonly_extra_db": 0.26,        # N2 − 同口径 T_only 对照（机制：DR 有额外价值）
    "affine_extra_db": 0.30,       # N2 − 训练域共享 RGB 仿射对照（N1 协议）
    "beta_move_min": 0.05,         # |β| 至少缩小 0.05
    "beta_overshoot_max": 0.05,    # 但不得跨零过冲到 +0.05 以上
    "split_max": 0.05,             # 两组不重叠抽样的平均增益差（dB）
    "ista_resid_max": 0.25,        # 内迭代相对变化；超过=配置未收敛，判未判而非失败
    "base_psnr_tol": 0.15,         # 缓存 P 复算 PSNR 与权威表之差（自检）
    "expect_beta_base": -0.289,    # 账本 may+jun 的 β（自检用）
    "beta_def_tol": 0.05,
}
PANEL_TABLE = {101: 23.07, 202: 22.87, 303: 22.20}   # eval/table_final_1019_poolcmp_1028.md

PANEL_DIR = os.path.join(PROJ, "data/test_sets/real_pair_1019")
PRED_ROOT = os.path.join(PROJ, "runs/pred")
PRED_TAG = "real_pair_1019_mayjun15_s{seed}"
OMEGA_DIR = os.path.join(PROJ, "data/annot_1019/omega_masks_1030")
POOL_DEFAULT = os.path.join(PROJ, "data/train_sets/real_pairs_1020")


# ------------------------------- 基础 IO / 指标 -------------------------------
def log(*a):
    print(*a, flush=True)


def imread_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def psnr_u8(a, b):
    """逐通道 PSNR 再平均（与 scripts/eval_1019_multi.bw_psnr / 权威表同口径）。"""
    the = 99.0
    v = []
    for c in range(a.shape[-1]):
        aa = a[..., c].astype(np.float64)
        bb = b[..., c].astype(np.float64)
        mse = float(np.mean((aa - bb) ** 2))
        v.append(the if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


def region_psnr(a, b, mask):
    if mask is None or not mask.any():
        return None
    v = []
    for c in range(a.shape[-1]):
        aa = a[..., c].astype(np.float64)[mask]
        bb = b[..., c].astype(np.float64)[mask]
        mse = float(np.mean((aa - bb) ** 2))
        v.append(99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


def beta_of(I, GT, That):
    """β 残留反射系数：−1 = 完全没去（T̂=I），0 = 完全去净（T̂=GT）。

    β = mean|I−T̂| / mean|I−GT| − 1（全图、通道池化）。定义由账本值 −0.289 反查校验。
    """
    num = float(np.abs(I.astype(np.float64) - That.astype(np.float64)).mean())
    den = float(np.abs(I.astype(np.float64) - GT.astype(np.float64)).mean())
    return num / max(den, 1e-8) - 1.0


def grad_mean(x):
    """平均梯度幅值（luma 前向差分，越高越锐）。"""
    g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2.0)


def to_u8(x_float01):
    """与 OpenCV imwrite(CV_32F) 同口径的四舍五入量化。"""
    return np.rint(np.clip(x_float01 * 255.0, 0.0, 255.0)).astype(np.uint8)


# ------------------------------- 面板 / 训练池 -------------------------------
def panel_pairs(panel_dir=PANEL_DIR):
    out = []
    for sc in sorted(os.listdir(panel_dir)):
        d = os.path.join(panel_dir, sc)
        if not os.path.isdir(d):
            continue
        gt = os.path.join(d, "GT.png")
        if not os.path.exists(gt):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                out.append(dict(scene=sc, stem=os.path.splitext(f)[0],
                                i=os.path.join(d, f), gt=gt))
    return out


def omega_mask(scene, shape=None, omega_dir=OMEGA_DIR):
    p = os.path.join(omega_dir, scene + ".png")
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = m > 127
    if shape is not None and m.shape != tuple(shape[:2]):
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def pool_cases(pool_dir, domains):
    with open(os.path.join(pool_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    keep = {d.strip().lower() for d in str(domains).split(",") if d.strip()}
    cases = []
    for k in sorted(meta):
        m = meta[k]
        if keep and str(m.get("domain", "")).lower() not in keep:
            continue
        cases.append(dict(case=k, scene=str(m.get("scene", "")), domain=m.get("domain", ""),
                          i=os.path.join(pool_dir, "I", k + ".png"),
                          t=os.path.join(pool_dir, "T", k + ".png")))
    return cases


# ------------------------------- 算子执行 -------------------------------
def run_op_float(n2, I_u8, P_u8, eta=1.0, mode=MODE_DUAL, want_diag=False):
    """跑一次部署算子，返回 [H,W,3] float32 [0,1] 与 diag。"""
    i = torch.from_numpy((I_u8.astype(np.float32) / 255.0).transpose(2, 0, 1))[None]
    p = torch.from_numpy((P_u8.astype(np.float32) / 255.0).transpose(2, 0, 1))[None]
    et = torch.tensor([float(eta)] * n2.n_outer, dtype=torch.float32)
    with torch.no_grad():
        if want_diag:
            out, diag = n2(i, p, eta=et, mode=mode, return_diag=True)
        else:
            out, diag = n2(i, p, eta=et, mode=mode), None
    return out[0].clamp(0, 1).numpy().transpose(1, 2, 0).astype(np.float32), diag


def run_op(n2, I_u8, P_u8, eta=1.0, mode=MODE_DUAL, want_diag=False):
    t, diag = run_op_float(n2, I_u8, P_u8, eta=eta, mode=mode, want_diag=want_diag)
    return to_u8(t), diag


def sample_delta_psnr(P, That, GT, idx, mse_base_c):
    """抽样 ΔPSNR 估计（审查给的公式）。idx 为扁平像素索引，逐通道。"""
    est = []
    for c in range(3):
        p = P[..., c].astype(np.float64).reshape(-1)[idx]
        q = That[..., c].astype(np.float64).reshape(-1)[idx]
        g = GT[..., c].astype(np.float64).reshape(-1)[idx]
        dmse = float(np.mean((q - g) ** 2) - np.mean((p - g) ** 2))
        est.append(-10.0 * math.log10(max(1.0 + dmse / max(mse_base_c[c], 1e-12), 1e-9)))
    return float(np.mean(est))


def boot_ci(vals, n_boot=10000, seed=20261031, alpha=0.05):
    v = np.asarray(vals, np.float64)
    if v.size < 2:
        return [float(v.mean()), float(v.mean())] if v.size else [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


# ================================ build-dict ================================
def act_build_dict(args):
    t0 = time.time()
    cases = pool_cases(args.pool, args.domains)
    if not cases:
        raise SystemExit(f"[build-dict] {args.pool} domains={args.domains} 为空")
    log(f"[build-dict] {len(cases)} 例 / {len({c['scene'] for c in cases})} 场景 "
        f"pool={os.path.relpath(args.pool, PROJ)} domains={args.domains}")
    tb, sb = [], []
    for c in cases:
        I = imread_rgb(c["i"]).astype(np.float32) / 255.0
        T = imread_rgb(c["t"]).astype(np.float32) / 255.0
        H, W = T.shape[:2]
        step = max(1, int(round(math.sqrt(H * W / max(1, args.per_case)))))
        pt = patch_bank(T.transpose(2, 0, 1), step)
        # I−T 有符号残差：同一网格位置（对齐）
        ps = patch_bank((I - T).transpose(2, 0, 1), step)
        tb.append(pt)
        sb.append(ps)
        log(f"  {c['case'][:44]:44s} {H}x{W} step={step} patch={len(pt)}")
    TB = np.concatenate(tb, 0)
    SB = np.concatenate(sb, 0)
    dt, it = build_dictionary(TB)
    dr, ir = build_dictionary(SB)
    tau, ti = tau_from_bank(TB)
    h, hi = ista_step(dt, dr)
    meta = dict(
        built=time.strftime("%Y-%m-%d %H:%M:%S"),
        pool=os.path.relpath(args.pool, PROJ), domains=args.domains,
        n_case=len(cases), n_scene=len({c["scene"] for c in cases}),
        per_case=args.per_case, n_patch=int(len(TB)),
        dt_info=it, dr_info=ir, tau_info=ti, step_info=hi,
        dt_dc=[float(x) for x in dt[0]], dr_dc=[float(x) for x in dr[0]],
        note=("DT 来自训练池 T patch；DR 来自训练池 (I−T) 有符号残差 patch；"
              "均值/DC 原子显式保留；单位列范数；eigh 降序 + 最大绝对分量取正。"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_config(args.out, dt, dr, tau, meta)
    # 交叉核对：两本字典的重叠度（层混淆的直接量化）
    G = dt @ dr.T
    sv = np.linalg.svd(G, compute_uv=False)
    # DC 原子被拿掉后的残余重叠（审查："两套字典都含相似的低频/DC 原子尤其值得警惕"）
    Gx = dt[1:] @ dr[1:].T
    svx = np.linalg.svd(Gx, compute_uv=False)
    out = dict(action="build-dict", dict_path=args.out, tau=tau, h=h, k_inner=K_INNER,
               n_outer=N_OUTER, n_param=N_PARAM_TOTAL,
               eta_init=0.0, rho_tau=RHO_TAU,
               median_patch_norm=ti["median_patch_norm"],
               lam_max=hi["lam_max"], h_trace_bound=H_TRACE_BOUND,
               cross_gram_sv_max=float(sv.max()),
               cross_gram_cos_max=float(np.abs(G).max()),
               cross_gram_cos_max_excl_dc=float(np.abs(Gx).max()),
               cross_gram_sv_max_excl_dc=float(svx.max()),
               dc_cos=float(abs(float(dt[0] @ dr[0]))),
               meta=meta, sec=round(time.time() - t0, 1))
    _dump(out, args.out_json)
    return out


def _dump(obj, path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    log(f"[json] -> {path}")


# ================================== unit ==================================
def act_unit(args):
    t0 = time.time()
    n2 = n2_from_config(args.dict)
    tau = float(n2.tau)
    h = n2.h
    rng = np.random.default_rng(20261031)
    res = {"action": "unit", "dict_path": args.dict, "h": h, "tau": tau,
           "k_inner": n2.k_inner, "n_outer": n2.n_outer, "tests": {}}

    # ---- T1: η=0 数值 & 存盘一致（含梯度签名） ----
    I = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    GTr = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    Pr = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    i_t = torch.from_numpy(I.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    p_t = torch.from_numpy(Pr.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    gt_t = torch.from_numpy(GTr.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    n2.zero_grad(set_to_none=True)
    torch.set_grad_enabled(True)
    out = n2(i_t, p_t)                     # 用模块自带的 η（初值 0），才是部署路径
    loss = torch.nn.functional.mse_loss(out, gt_t)
    loss.backward()
    torch.set_grad_enabled(False)
    d_eta = n2.eta.grad.detach().cpu().numpy().tolist() if n2.eta.grad is not None else None
    d_dict = float(n2.dt.grad.abs().sum() + n2.dr.grad.abs().sum()) \
        if n2.dt.grad is not None else None
    maxdiff = float((out - p_t).detach().abs().max())
    u8_same = bool(np.array_equal(to_u8(out[0].detach().permute(1, 2, 0).numpy()), Pr))
    res["tests"]["T1_eta0_exact"] = dict(
        max_abs_diff_float=maxdiff, uint8_identical=u8_same,
        grad_eta=d_eta, grad_dict_abs_sum=d_dict,
        eta_still_zero=[float(x) for x in n2.eta.detach()],
        note=("η=0 ⇒ 逐像素等于 P（float 与存盘 uint8 都一致）；字典梯度为 0 是预期"
              "（启动屏障），η 梯度应非零 —— 这决定第二道桥接能不能起来。"))

    # ---- T2: i=p 的常量 / 低对比纹理退化（量化幅度） ----
    deg = {}
    for lvl in (0.05, 0.10, 0.20, 0.40, 0.60, 0.90):
        pimg = np.full((64, 64, 3), int(round(lvl * 255)), np.uint8)
        for eta in (1.0, 0.25):
            o, dg = run_op_float(n2, pimg, pimg, eta=eta, want_diag=True)
            deg[f"const_{lvl:.2f}_eta{eta}"] = dict(
                delta_mean=float(o.mean() - pimg.mean() / 255.0),
                delta_rel=float(o.mean() / (pimg.mean() / 255.0) - 1.0),
                frac_both_zero=dg["frac_both_zero"],
                ista_resid=dg["ista_a_rel_change"])
    # 低对比纹理：P = 0.3 + 0.01*sin（三通道共用同一条纹）
    _, xx = np.mgrid[0:64, 0:64]
    base = 0.30 + 0.01 * np.sin(xx / 3.0)[..., None] * np.ones((1, 1, 3))
    ptex = to_u8(base)
    for eta in (1.0, 0.25):
        o, dg = run_op_float(n2, ptex, ptex, eta=eta, want_diag=True)
        deg[f"lowcontrast_eta{eta}"] = dict(
            delta_mean=float(o.mean() - ptex.mean() / 255.0),
            delta_rel=float(o.mean() / (ptex.mean() / 255.0) - 1.0),
            frac_both_zero=dg["frac_both_zero"])
    res["tests"]["T2_i_eq_p_degeneracy"] = dict(
        cases=deg,
        analytic_note=("全零编码分支的解析值：t*=(p+i)/3，i=p ⇒ 2p/3（单步 −33% 亮度）；"
                       "常量且编码未死时 t*=p−(τ/3)u 的压暗量级由 τ/3 给出。"
                       "这里是**结构性代价**，只量化、不宣布解决。"),
        tau=float(tau), tau_over_3=float(tau / 3.0))

    # ---- T3: τ 极大 ⇒ 编码全零分支 ----
    n2_big = n2_from_config(args.dict)
    n2_big.tau.fill_(1e6)
    pimg = np.full((48, 48, 3), 128, np.uint8)
    iimg = pimg.copy()
    o, dg = run_op_float(n2_big, iimg, pimg, eta=1.0, want_diag=True)
    expect = (pimg.astype(np.float64) / 255.0 + iimg.astype(np.float64) / 255.0) / 3.0
    md = float(np.abs(o - expect).max())
    res["tests"]["T3_tau_kills_codes"] = dict(
        frac_both_zero=dg["frac_both_zero"],
        delta_mean=float(o.mean() - pimg.mean() / 255.0),
        delta_rel=float(o.mean() / (pimg.mean() / 255.0) - 1.0),
        max_abs_diff_vs_2p_over_3=md,
        matches_2p_over_3=bool(md <= 1e-6),
        note=("τ→∞ ⇒ a=b=0 ⇒ t*=(p+i)/3；i=p 时 =2p/3（−33% 亮度）。"
              "审查给出的反例，逐像素核对（这里用 float 输出，避免量化噪声）。"))

    # ---- T4: 稠密参考 vs 卷积实现 ----
    ii = rng.random((3, 40, 48)).astype(np.float32) * 0.8 + 0.1
    pp = rng.random((3, 40, 48)).astype(np.float32) * 0.8 + 0.1
    dt, dr, tau_c, _ = load_config(args.dict)
    with torch.no_grad():
        et = torch.tensor([1.0, 1.0])
        o_conv = n2(torch.from_numpy(ii)[None], torch.from_numpy(pp)[None], eta=et)[0].numpy()
    o_dense, _, _, dg_dense = forward_dense_np(ii, pp, dt, dr, tau_c, [1.0, 1.0],
                                               k_inner=n2.k_inner, n_outer=n2.n_outer, h=h)
    # 边框环误差单独量化（分析用 reflect、合成用 convT 的非共轭近似）
    npx = o_conv.shape[1] * o_conv.shape[2]
    inner = o_conv[:, 2:-2, 2:-2] - o_dense[:, 2:-2, 2:-2]
    ring = np.ones(o_conv.shape[1:], bool)
    ring[2:-2, 2:-2] = False
    res["tests"]["T4_dense_vs_conv"] = dict(
        max_abs_diff=float(np.abs(o_conv - o_dense).max()),
        max_abs_diff_interior=float(np.abs(inner).max()),
        max_abs_diff_ring=float(np.abs((o_conv - o_dense)[:, ring]).max()) if ring.any() else 0.0,
        ring_pixel_frac=float(ring.sum() / npx),
        ista_resid_dense=dg_dense["ista_a_rel_change"],
        note="两条独立实现（sliding_window_view+25 偏移累加 vs conv2d+convT）逐像素核对。")

    # ---- T5: 收敛性（h=谱界 K 步 vs h=1/16 200 步的不动点一致性） ----
    o_trace, _, _, dg_tr = forward_dense_np(ii, pp, dt, dr, tau_c, [1.0, 1.0],
                                            k_inner=200, n_outer=2, h=H_TRACE_BOUND)
    o_trace4, _, _, _ = forward_dense_np(ii, pp, dt, dr, tau_c, [1.0, 1.0],
                                         k_inner=n2.k_inner, n_outer=2, h=h)
    res["tests"]["T5_convergence"] = dict(
        h_used=h, h_trace_bound=H_TRACE_BOUND,
        max_abs_diff_h_vs_trace200=float(np.abs(o_trace4 - o_trace).max()),
        ista_resid_used=dg_dense["ista_a_rel_change"],
        ista_resid_trace200=dg_tr["ista_a_rel_change"],
        pass_resid=bool(dg_dense["ista_a_rel_change"] <= CRIT["ista_resid_max"]),
        note=("h 只改收敛速度、不改 ISTA 不动点；此项核对"
              "『h=谱界 K 步』与『h=1/16 200 步』的输出是否同量级。"))

    # ---- T6: 单位列范数 / 参数量 / η 投影 ----
    res["tests"]["T6_contract"] = dict(
        n_param_dict=int(n2.dt.numel() + n2.dr.numel()),
        n_param_total=N_PARAM_TOTAL,
        dt_row_norm=[float(x) for x in n2.dt.norm(dim=1)],
        dr_row_norm=[float(x) for x in n2.dr.norm(dim=1)],
        eta_init=[float(x) for x in n2.eta.detach()],
        eta_projection="optimizer 每步后 clamp 到 [0,1]（训练脚本内实现）",
        patch_dims=[PATCH, PATCH, 3], n_atom=N_ATOM)
    res["tests"]["T6_contract"]["rows_unit"] = bool(
        np.allclose(n2.dt.detach().norm(dim=1).numpy(), 1.0, atol=1e-5) and
        np.allclose(n2.dr.detach().norm(dim=1).numpy(), 1.0, atol=1e-5))

    ok = (res["tests"]["T1_eta0_exact"]["uint8_identical"]
          and res["tests"]["T1_eta0_exact"]["max_abs_diff_float"] == 0.0
          and res["tests"]["T4_dense_vs_conv"]["max_abs_diff"] < 2e-4
          and res["tests"]["T6_contract"]["rows_unit"])
    res["unit_ok"] = bool(ok)
    res["sec"] = round(time.time() - t0, 1)
    _dump(res, args.out_json or args.out)
    return res


# ================================= gate1 =================================
def _center_idx(H, W, n, seed=20261031):
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, H, n)
    xs = rng.integers(0, W, n)
    return ys * W + xs


def act_gate1(args):
    t0 = time.time()
    n2 = n2_from_config(args.dict)
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    tonly_seeds = [int(x) for x in str(args.tonly_seeds).split(",") if x.strip()]
    pairs = panel_pairs(args.panel)
    if args.limit_scenes:
        keep = sorted({p["scene"] for p in pairs})[: args.limit_scenes]
        pairs = [p for p in pairs if p["scene"] in keep]
    log(f"[gate1] dict={os.path.basename(args.dict)} eta={args.eta} K={n2.k_inner} "
        f"outer={n2.n_outer} h={n2.h:.5f} tau={float(n2.tau):.5f} 种子={seeds} 面板={len(pairs)} 对")
    out = {"action": "gate1", "dict_path": args.dict, "eta_primary": float(args.eta),
           "k_inner": n2.k_inner, "n_outer": n2.n_outer, "h": n2.h,
           "tau": float(n2.tau), "seeds": seeds, "n_scene": len({p["scene"] for p in pairs}),
           "mode_tonly_note": "T_only 目标 ½‖t−p‖²+½‖t−DTa‖² ⇒ t*=(p+DTa)/2（分母 2，无 i 项）",
           "timing": {}, "per_seed": {}, "notes": []}

    budget = float(args.time_budget)
    rec = []
    t_first = None
    for si, seed in enumerate(seeds):
        tag = PRED_TAG.format(seed=seed)
        pred_dir = os.path.join(args.pred_root, tag)
        if not os.path.isdir(pred_dir):
            raise SystemExit(f"[gate1] 缺缓存预测：{pred_dir}")
        out["per_seed"][str(seed)] = {"tag": tag, "scene": {}, "psnr_mean": None}
        for k, pr in enumerate(pairs):
            sc = pr["scene"]
            I = imread_rgb(pr["i"])
            GT = imread_rgb(pr["gt"])
            pp = os.path.join(pred_dir, f"{sc}__{pr['stem']}", "t_hat.png")
            if not os.path.exists(pp):
                raise SystemExit(f"[gate1] 缺预测图：{pp}")
            P = imread_rgb(pp)
            if P.shape != GT.shape:
                raise SystemExit(f"[gate1] 尺寸不一致 {sc}: P{P.shape} GT{GT.shape}")
            H, W = GT.shape[:2]
            om = omega_mask(sc, GT.shape, args.omega_dir)
            # ---- 算子（只用 I 与 P） ----
            Tn2, dg = run_op(n2, I, P, eta=args.eta, mode=MODE_DUAL, want_diag=True)
            o = {}
            o["psnr_base"] = psnr_u8(P, GT)
            o["psnr_n2"] = psnr_u8(Tn2, GT)
            o["d_n2"] = o["psnr_n2"] - o["psnr_base"]
            o["beta_base"] = beta_of(I, GT, P)
            o["beta_n2"] = beta_of(I, GT, Tn2)
            o["grad_base"] = grad_mean(P.astype(np.float32) / 255.0)
            o["grad_n2"] = grad_mean(Tn2.astype(np.float32) / 255.0)
            o["grad_gt"] = grad_mean(GT.astype(np.float32) / 255.0)
            o["ista_resid"] = dg["ista_a_rel_change"]
            o["frac_both_zero"] = dg["frac_both_zero"]
            o["delta_abs_mean"] = dg["delta_abs_mean"]
            o["delta_signed_mean"] = dg["delta_signed_mean"]
            if om is not None:
                o["omega_frac"] = float(om.mean())
                b_in, n_in = region_psnr(P, GT, om), region_psnr(Tn2, GT, om)
                b_out, n_out = region_psnr(P, GT, ~om), region_psnr(Tn2, GT, ~om)
                o["omega_in"] = None if b_in is None else n_in - b_in
                o["omega_out"] = None if b_out is None else n_out - b_out
            # ---- 抽样核查：中心点 2048（两组不重叠） ----
            idx = _center_idx(H, W, 2048)
            mse_base_c = [float(np.mean((P[..., c].astype(np.float64) -
                                         GT[..., c].astype(np.float64)) ** 2)) for c in range(3)]
            a_hat = sample_delta_psnr(P, Tn2, GT, idx[0::2], mse_base_c)
            b_hat = sample_delta_psnr(P, Tn2, GT, idx[1::2], mse_base_c)
            f_hat = sample_delta_psnr(P, Tn2, GT, idx, mse_base_c)
            o["sample_dA"], o["sample_dB"], o["sample_dFull"] = a_hat, b_hat, f_hat
            o["sample_split"] = abs(a_hat - b_hat)
            o["sample_vs_full"] = abs(f_hat - o["d_n2"])
            # ---- T_only 对照（同 η、同 K、同聚合，分母 2） ----
            if tonly_seeds and seed in tonly_seeds:
                Tto, _ = run_op(n2, I, P, eta=args.eta, mode=MODE_TONLY)
                o["psnr_tonly"] = psnr_u8(Tto, GT)
                o["d_tonly"] = o["psnr_tonly"] - o["psnr_base"]
                del Tto
            del Tn2
            out["per_seed"][str(seed)]["scene"][sc] = o
            rec.append(o)
            if t_first is None and si == 0 and k == 0:
                t_first = time.time() - t0
                log(f"  首对耗时 {t_first:.1f}s（含字典/库加载）；后续按此估计总预算")
            if (k + 1) % 5 == 0:
                log(f"  seed {seed} {k + 1}/{len(pairs)}  {time.time() - t0:.0f}s")
            gc.collect()
            if time.time() - t0 > budget:
                out["notes"].append(f"time_budget {budget}s 触发：seed {seed} 第 {k + 1} 对后中止，"
                                    f"未跑部分记未判")
                log(f"[gate1] 触发时间预算 {budget}s，中止（未判）")
                break
        out["per_seed"][str(seed)]["psnr_mean"] = float(np.mean(
            [v["psnr_n2"] for v in out["per_seed"][str(seed)]["scene"].values()]))
        if out["notes"] and "time_budget" in out["notes"][-1]:
            break
    out["timing"]["total_sec"] = round(time.time() - t0, 1)
    out["timing"]["first_pair_sec"] = None if t_first is None else round(t_first, 1)
    _aggregate_gate1(out, pairs)
    _dump(out, args.out)
    return out


def _aggregate_gate1(out, pairs):
    scenes = sorted({p["scene"] for p in pairs})
    seeds = [str(s) for s in out["seeds"]]
    per_scene = {}
    for sc in scenes:
        rows = [out["per_seed"][s]["scene"][sc] for s in seeds
                if sc in out["per_seed"].get(s, {}).get("scene", {})]
        if not rows:
            continue
        def m(key):
            vs = [r[key] for r in rows if r.get(key) is not None]
            return None if not vs else float(np.mean(vs))
        per_scene[sc] = dict(
            n_seed=len(rows),
            d_n2=m("d_n2"), d_tonly=m("d_tonly"),
            d_omega_in=m("omega_in"), d_omega_out=m("omega_out"),
            beta_base=m("beta_base"), beta_n2=m("beta_n2"),
            psnr_base=m("psnr_base"), psnr_n2=m("psnr_n2"),
            d_by_seed=[r["d_n2"] for r in rows],
            sample_split=m("sample_split"), sample_vs_full=m("sample_vs_full"),
            ista_resid=m("ista_resid"), frac_both_zero=m("frac_both_zero"),
            grad_ratio_n2_gt=(m("grad_n2") / m("grad_gt")) if m("grad_gt") else None,
        )
    out["per_scene"] = per_scene
    d = [v["d_n2"] for v in per_scene.values() if v["d_n2"] is not None]
    n = len(d)
    summ = dict(n_scene=n, d_psnr_mean=float(np.mean(d)) if d else None,
                d_psnr_median=float(np.median(d)) if d else None,
                n_pos=int(sum(1 for x in d if x > 0)),
                d_by_seed={s: float(np.mean([out["per_seed"][s]["scene"][sc]["d_n2"]
                                             for sc in scenes if sc in out["per_seed"][s]["scene"]]))
                           for s in seeds if out["per_seed"].get(s, {}).get("scene")})
    if d:
        summ["d_psnr_ci95"] = boot_ci(d)
        summ["d_psnr_sd_scene"] = float(np.std(d, ddof=1)) if n > 1 else None
        summ["d_psnr_min_scene"] = float(np.min(d))
        summ["d_psnr_max_scene"] = float(np.max(d))
    for key, kk in (("d_omega_in", "d_omega_in"), ("d_omega_out", "d_omega_out"),
                    ("sample_split", "sample_split"), ("sample_vs_full", "sample_vs_full"),
                    ("ista_resid", "ista_resid"), ("frac_both_zero", "frac_both_zero"),
                    ("grad_ratio_n2_gt", "grad_ratio_n2_gt")):
        vs = [v[kk] for v in per_scene.values() if v.get(kk) is not None]
        summ[key] = float(np.mean(vs)) if vs else None
    if all(v.get("d_tonly") is not None for v in per_scene.values()):
        dt_ = [out["per_scene"][sc]["d_tonly"] for sc in per_scene]
        summ["d_tonly_mean"] = float(np.mean(dt_))
        summ["extra_vs_tonly"] = float(np.mean(d) - np.mean(dt_))
    b0 = [v["beta_base"] for v in per_scene.values() if v["beta_base"] is not None]
    b1 = [v["beta_n2"] for v in per_scene.values() if v["beta_n2"] is not None]
    if b0 and b1:
        summ["beta_base"] = float(np.mean(b0))
        summ["beta_n2"] = float(np.mean(b1))
        summ["beta_move"] = float(np.mean([abs(v["beta_base"]) - abs(v["beta_n2"])
                                           for v in per_scene.values()]))
    # β 定义自检：与账本 may+jun 的 β=−0.289 比（定义锁定不住 ⇒ 机制判据未判）
    if "beta_base" in summ:
        diff = abs(summ["beta_base"] - CRIT["expect_beta_base"])
        summ["beta_def_selfcheck"] = dict(
            beta_base=summ["beta_base"], expect=CRIT["expect_beta_base"], diff=diff,
            ok=bool(diff <= CRIT["beta_def_tol"]),
            note="β=mean|I−T̂|/mean|I−GT|−1（全图通道池化）；对不上账本值则 β 判据记未判")
    # 完整性：缺种子/缺场景 ⇒ 汇总判据一律未判（未判≠通过）
    summ["complete"] = bool(len(per_scene) == out.get("n_scene", 0)
                            and all(per_scene[sc]["n_seed"] == len(seeds) for sc in per_scene))
    # 缓存 P 复算 PSNR 自检（与权威表比）
    chk = {}
    for s in seeds:
        rows = out["per_seed"].get(s, {}).get("scene", {})
        if not rows:
            continue
        got = float(np.mean([v["psnr_base"] for v in rows.values()]))
        want = PANEL_TABLE.get(int(s))
        chk[s] = dict(recomputed=got, table=want,
                      diff=None if want is None else got - want,
                      ok=bool(want is not None and abs(got - want) <= CRIT["base_psnr_tol"]))
    out["base_psnr_selfcheck"] = chk
    # 判据
    complete = bool(summ.get("complete"))
    c = {}
    def st(v, thr):
        return _st(v, thr) if complete else "undetermined"
    def sti(v, thr):
        return _st_inv(v, thr) if complete else "undetermined"
    c["main_gain_db"] = dict(
        value=summ.get("d_psnr_mean"), need=CRIT["main_gain_db"],
        state=st(summ.get("d_psnr_mean"), CRIT["main_gain_db"]))
    c["ci_lower_gt0"] = dict(
        value=None if "d_psnr_ci95" not in summ else summ["d_psnr_ci95"][0],
        need=">0",
        state=st(None if "d_psnr_ci95" not in summ else summ["d_psnr_ci95"][0], 1e-9))
    c["n_pos_scenes"] = dict(value=summ.get("n_pos"), need=f">={CRIT['min_pos_scenes']}",
                             state=st(summ.get("n_pos"), CRIT["min_pos_scenes"] - 1e-9))
    c["omega_in_no_degrade"] = dict(
        value=summ.get("d_omega_in"), need=f">={-CRIT['omega_in_degrade_max']}",
        state=st(summ.get("d_omega_in"), -CRIT["omega_in_degrade_max"] - 1e-9))
    c["omega_out_no_degrade"] = dict(
        value=summ.get("d_omega_out"), need=f">={-CRIT['omega_out_degrade_max']}",
        state=st(summ.get("d_omega_out"), -CRIT["omega_out_degrade_max"] - 1e-9))
    c["extra_vs_tonly"] = dict(
        value=summ.get("extra_vs_tonly"), need=f">={CRIT['tonly_extra_db']}",
        state=st(summ.get("extra_vs_tonly"), CRIT["tonly_extra_db"] - 1e-9))
    mv = summ.get("beta_move")
    bo = summ.get("beta_n2")
    bdef = (summ.get("beta_def_selfcheck") or {}).get("ok", False)
    c["beta_toward_zero"] = dict(
        value=mv, need=(f">={CRIT['beta_move_min']} 且 beta_n2<={CRIT['beta_overshoot_max']}"
                        f"（β 定义自检 {'通过' if bdef else '未通过⇒未判'}）"),
        state=("undetermined" if (mv is None or bo is None or not bdef) else
               ("pass" if (mv >= CRIT["beta_move_min"] and bo <= CRIT["beta_overshoot_max"])
                else "fail")))
    c["sample_split"] = dict(value=summ.get("sample_split"),
                             need=f"<={CRIT['split_max']}",
                             state=sti(summ.get("sample_split"), CRIT["split_max"]))
    c["ista_converged"] = dict(value=summ.get("ista_resid"),
                               need=f"<={CRIT['ista_resid_max']}",
                               state=sti(summ.get("ista_resid"), CRIT["ista_resid_max"]))
    c["cache_selfcheck"] = dict(
        value={k: v.get("diff") for k, v in chk.items()},
        need=f"|diff|<={CRIT['base_psnr_tol']}",
        state=("undetermined" if not chk else
               ("pass" if all(v["ok"] for v in chk.values()) else "fail")))
    # 抽样实现一致性是诊断项（不是判据），单列
    c["sample_vs_full_diag"] = dict(value=summ.get("sample_vs_full"), note="诊断：抽样估计与全图差")
    out["summary"] = summ
    out["criteria"] = c
    states = [v["state"] for v in c.values() if isinstance(v, dict) and "state" in v]
    out["gate1_verdict"] = ("fail" if "fail" in states else
                            ("undetermined" if "undetermined" in states else "pass"))
    log(f"[gate1] ΔPSNR={summ.get('d_psnr_mean')} CI={summ.get('d_psnr_ci95')} "
        f"n_pos={summ.get('n_pos')}/{summ.get('n_scene')} tonic_extra={summ.get('extra_vs_tonly')} "
        f"verdict={out['gate1_verdict']}")


def _st(value, threshold):
    if value is None:
        return "undetermined"
    return "pass" if value >= threshold else "fail"


def _st_inv(value, threshold):
    if value is None:
        return "undetermined"
    return "pass" if value <= threshold else "fail"


# ================================ transfer ================================
def act_transfer(args):
    """训练域拟合 → 面板直测：α 幅度诊断 + 共享 RGB 仿射对照。"""
    t0 = time.time()
    n2 = n2_from_config(args.dict)
    res = {"action": "transfer", "dict_path": args.dict, "eta": float(args.eta),
           "crop": int(args.crop), "pcache_root": args.pcache_root}
    if not os.path.isdir(args.pcache_root):
        res["state"] = "undetermined"
        res["reason"] = (f"缺训练域 P 缓存 {args.pcache_root}：α 诊断与共享仿射对照都不能"
                         f"在训练域拟合，按审查记未判（不得用测试 GT 顶替）。"
                         f"生成命令见 scripts/n2_build_pcache.py。")
        log(f"[transfer] 未判：{res['reason']}")
        _dump(res, args.out)
        return res
    all_cases = pool_cases(args.pool, args.domains)
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    # 算子只在**确定性等距子集**上跑（覆盖各场景；不看 GT、不看结果，纯预算用）
    cases = all_cases
    if args.max_cases and len(cases) > args.max_cases:
        sel = np.linspace(0, len(cases) - 1, args.max_cases).round().astype(int)
        cases = [cases[i] for i in sorted(set(int(x) for x in sel))]
        res["subsample"] = dict(n_all=len(all_cases), n_used=len(cases),
                                rule="linspace 等距（确定性，预登记预算规则）")
    res["n_case"], res["n_scene"] = len(cases), len({c["scene"] for c in cases})
    # ---- 逐例：固定中心裁剪上的 d = H−P, e = T−P ----
    per_case = {}
    for seed in seeds:
        d = os.path.join(args.pcache_root, f"s{seed}")
        if not os.path.isdir(d):
            res["state"] = "undetermined"
            res["reason"] = f"缺 {d}"
            _dump(res, args.out)
            return res
        for c in cases:
            pp = os.path.join(d, c["case"] + ".png")
            if not os.path.exists(pp):
                res.setdefault("missing", []).append(pp)
                continue
            I = imread_rgb(c["i"])
            T = imread_rgb(c["t"])
            P = imread_rgb(pp)
            H, W = T.shape[:2]
            k = min(args.crop, H, W)
            y0, x0 = (H - k) // 2, (W - k) // 2
            I, T, P = I[y0:y0 + k, x0:x0 + k], T[y0:y0 + k, x0:x0 + k], P[y0:y0 + k, x0:x0 + k]
            Hn, _ = run_op(n2, I, P, eta=args.eta)
            dlt = (Hn.astype(np.float64) - P.astype(np.float64)) / 255.0
            err = (T.astype(np.float64) - P.astype(np.float64)) / 255.0
            per_case.setdefault(c["case"], dict(
                scene=c["scene"], de=0.0, dd=0.0, ee=0.0, n=0.0,
                mse_base=[0.0] * 3, dd_c=[0.0] * 3, de_c=[0.0] * 3, npx=0.0))
            r = per_case[c["case"]]
            r["de"] += float((dlt * err).sum())
            r["dd"] += float((dlt * dlt).sum())
            r["ee"] += float((err * err).sum())
            r["n"] += float(dlt.size)
            for ch in range(3):
                r["mse_base"][ch] += float(((P[..., ch].astype(np.float64) -
                                             T[..., ch].astype(np.float64)) ** 2).sum()) / 255.0 ** 2
                r["dd_c"][ch] += float((dlt[..., ch] ** 2).sum())
                r["de_c"][ch] += float((dlt[..., ch] * err[..., ch]).sum())
            r["npx"] += float(dlt[..., 0].size)
            del Hn
    if res.get("missing"):
        res["state"] = "undetermined"
        res["reason"] = f"缺 {len(res['missing'])} 张训练域预测"
        _dump(res, args.out)
        return res
    # ---- α：LOMO 场景划分 ----
    scenes = sorted({v["scene"] for v in per_case.values()})
    lomo = []
    for sc in scenes:
        fit = [v for k, v in per_case.items() if v["scene"] != sc]
        num = sum(v["de"] for v in fit)
        den = sum(v["dd"] for v in fit)
        alpha = 0.0 if den <= 1e-12 else float(np.clip(num / den, 0.0, 1.0))
        held = [v for k, v in per_case.items() if v["scene"] == sc]
        dms = []
        for v in held:
            ps = []
            for ch in range(3):
                dmse = (alpha ** 2 * v["dd_c"][ch] - 2 * alpha * v["de_c"][ch]) / max(v["npx"], 1)
                mse_base = v["mse_base"][ch] / max(v["npx"], 1)
                ps.append(-10.0 * math.log10(max(1.0 + dmse / max(mse_base, 1e-12), 1e-9)))
            dms.append(float(np.mean(ps)))
        lomo.append(dict(scene=sc, alpha=alpha, n_case=len(held),
                         d_psnr_fold=float(np.mean(dms))))
    dl = [x["d_psnr_fold"] for x in lomo]
    res["alpha_lomo"] = dict(
        per_scene=lomo,
        alpha_mean=float(np.mean([x["alpha"] for x in lomo])),
        alpha_median=float(np.median([x["alpha"] for x in lomo])),
        d_psnr_mean=float(np.mean(dl)), n_pos=int(sum(1 for x in dl if x > 0)),
        de_positive_folds=int(sum(1 for x in lomo if x["alpha"] > 0)),
        note="α 只在拟合折求（clip 到 [0,1]，分母≤0 时 α=0）；留出场景直接预测 P+αd。")
    # ---- 共享 RGB 仿射：只在训练域拟合 ----
    aff = {}
    for seed in seeds:
        d = os.path.join(args.pcache_root, f"s{seed}")
        accP = np.zeros(3); accT = np.zeros(3); accPP = np.zeros(3); accPT = np.zeros(3)
        npx = 0
        for c in all_cases:
            P = imread_rgb(os.path.join(d, c["case"] + ".png")).astype(np.float64) / 255.0
            T = imread_rgb(c["t"]).astype(np.float64) / 255.0
            for ch in range(3):
                accP[ch] += P[..., ch].sum(); accT[ch] += T[..., ch].sum()
                accPP[ch] += (P[..., ch] ** 2).sum(); accPT[ch] += (P[..., ch] * T[..., ch]).sum()
            npx += P[..., 0].size
        a = np.zeros(3); b = np.zeros(3)
        for ch in range(3):
            mp, mt = accP[ch] / npx, accT[ch] / npx
            var = accPP[ch] / npx - mp * mp
            cov = accPT[ch] / npx - mp * mt
            a[ch] = 1.0 if var <= 1e-12 else cov / var
            b[ch] = mt - a[ch] * mp
        aff[str(seed)] = dict(a=[float(x) for x in a], b=[float(x) for x in b])
    res["affine_fit"] = aff
    # 面板直测（应用拟合好的仿射，不做任何逐图拟合）
    pairs = panel_pairs(args.panel)
    per_scene_aff = {}
    for seed in seeds:
        pred_dir = os.path.join(args.pred_root, PRED_TAG.format(seed=seed))
        a = np.asarray(aff[str(seed)]["a"]); b = np.asarray(aff[str(seed)]["b"])
        for pr in pairs:
            sc = pr["scene"]
            GT = imread_rgb(pr["gt"])
            P = imread_rgb(os.path.join(pred_dir, f"{sc}__{pr['stem']}", "t_hat.png"))
            base = psnr_u8(P, GT)
            q = np.clip(P.astype(np.float64) / 255.0 * a.reshape(1, 1, 3) + b.reshape(1, 1, 3), 0, 1)
            per_scene_aff.setdefault(sc, []).append(psnr_u8(to_u8(q), GT) - base)
    res["affine_panel"] = dict(
        d_psnr_mean=float(np.mean([np.mean(v) for v in per_scene_aff.values()])),
        per_scene={k: float(np.mean(v)) for k, v in per_scene_aff.items()},
        note="6 参数共享仿射，训练域最小二乘拟合（per-channel），面板只应用不拟合。")
    res["state"] = "judged"
    res["sec"] = round(time.time() - t0, 1)
    _dump(res, args.out)
    return res


# ================================= verdict =================================
def act_verdict(args):
    def rd(name):
        if not name:
            return None
        p = name if os.path.isabs(name) else os.path.join(args.dir, name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    unit = rd("unit.json")
    gate1 = rd("gate1.json")
    transfer = rd("transfer.json")
    bridge = rd(args.bridge_json) if args.bridge_json else rd("bridge.json")
    out = {"action": "verdict", "dir": args.dir,
           "inputs": {"unit": unit is not None, "gate1": gate1 is not None,
                      "transfer": transfer is not None, "bridge": bridge is not None},
           "criteria_def": CRIT}
    rows = {}
    if unit is not None:
        t5 = unit.get("tests", {}).get("T5_convergence", {})
        rows["unit_ok"] = dict(state="pass" if unit.get("unit_ok") else "fail",
                               detail="η=0 精确/稠密一致/单位范数")
        rows["unit_ista_converged"] = dict(
            state="pass" if t5.get("pass_resid") else "undetermined",
            detail=f"resid={t5.get('ista_resid_used')}")
    else:
        rows["unit_ok"] = dict(state="undetermined", detail="缺 unit.json")
    if gate1 is not None:
        c = gate1.get("criteria", {})
        for k, v in c.items():
            if isinstance(v, dict) and "state" in v:
                rows[k] = dict(seen=args.dir, **v)
        if transfer is not None and transfer.get("state") == "judged":
            d_n2 = gate1.get("summary", {}).get("d_psnr_mean")
            d_aff = transfer.get("affine_panel", {}).get("d_psnr_mean")
            extra = None if (d_n2 is None or d_aff is None) else d_n2 - d_aff
            rows["extra_vs_shared_affine"] = dict(
                value=extra, need=f">={CRIT['affine_extra_db']}",
                state=_st(extra, CRIT["affine_extra_db"] - 1e-9))
            al = transfer.get("alpha_lomo", {})
            rows["alpha_direction"] = dict(
                value=al.get("d_psnr_mean"), n_pos_fold=al.get("n_pos"),
                state=("pass" if (al.get("d_psnr_mean") or -1) > 0 and
                       (al.get("de_positive_folds") or 0) >= (transfer.get("n_scene", 0) // 2 + 1)
                       else "fail"),
                note="方向诊断，不单独放行；负值=同一修正方向在留出场景不可迁移")
        else:
            rows["extra_vs_shared_affine"] = dict(state="undetermined",
                                                  detail="缺训练域 P 缓存/transfer.json")
            rows["alpha_direction"] = dict(state="undetermined", detail="缺训练域 P 缓存")
        rows["gate1"] = dict(state=gate1.get("gate1_verdict"))
    else:
        rows["gate1"] = dict(state="undetermined", detail="缺 gate1.json")
    rows["gate2_training_bridge"] = dict(
        state=("pass" if (bridge and bridge.get("bridge_pass")) else "undetermined"),
        detail=("缺 bridge.json：200 步训练桥接未跑，按审查第二道记未判"
                if not bridge else f"bridge_pass={bridge.get('bridge_pass')}, "
                                   f"iters={bridge.get('iters')}"))
    states = [v.get("state") for v in rows.values()]
    out["rows"] = rows
    if "fail" in states:
        out["release"] = "blocked: 有判据 fail（不得开跑正式训练）"
    elif "undetermined" in states:
        out["release"] = "blocked: 有未判项（补输入后再判；未判≠通过）"
    else:
        out["release"] = "allowed: 第一道 + 第二道全过，可进三种子配对正式实验"
    log(json.dumps({"release": out["release"],
                    "rows": {k: v.get("state") for k, v in rows.items()}},
                   ensure_ascii=False, indent=1))
    _dump(out, args.out)
    return out


# ================================== CLI ==================================
def main():
    ap = argparse.ArgumentParser(description="N2 CPU 可学性探针")
    sub = ap.add_subparsers(dest="action", required=True)

    b = sub.add_parser("build-dict", help="训练域字典 + τ + h（CPU）")
    b.add_argument("--pool", default=POOL_DEFAULT)
    b.add_argument("--domains", default="may,jun")
    b.add_argument("--per-case", type=int, default=2000)
    b.add_argument("--out", default=os.path.join(PROJ, "runs/n2_1031/n2_dict_mayjun.json"))
    b.add_argument("--out-json", default=os.path.join(PROJ, "runs/n2_1031/build_dict.json"))

    u = sub.add_parser("unit", help="单元测试（CPU）")
    u.add_argument("--dict", required=True)
    u.add_argument("--out", default=os.path.join(PROJ, "runs/n2_1031/unit.json"))
    u.add_argument("--out-json", default=None)

    g = sub.add_parser("gate1", help="面板直测（CPU，三种子）")
    g.add_argument("--dict", required=True)
    g.add_argument("--eta", type=float, default=1.0, help="预登记主值；不得按结果挑")
    g.add_argument("--seeds", default="101,202,303")
    g.add_argument("--panel", default=PANEL_DIR)
    g.add_argument("--pred-root", default=PRED_ROOT)
    g.add_argument("--omega-dir", default=OMEGA_DIR)
    g.add_argument("--tonly-seeds", default="", help="跑 T_only 对照的种子（空=不跑）")
    g.add_argument("--limit-scenes", type=int, default=0)
    g.add_argument("--time-budget", type=float, default=290.0)
    g.add_argument("--threads", type=int, default=16)
    g.add_argument("--out", default=os.path.join(PROJ, "runs/n2_1031/gate1.json"))

    t = sub.add_parser("transfer", help="训练域拟合 → 面板直测（需 pcache）")
    t.add_argument("--dict", required=True)
    t.add_argument("--eta", type=float, default=1.0)
    t.add_argument("--pool", default=POOL_DEFAULT)
    t.add_argument("--domains", default="may,jun")
    t.add_argument("--seeds", default="101,202,303")
    t.add_argument("--crop", type=int, default=512, help="每例固定中心裁剪（原分辨率，不缩放）")
    t.add_argument("--max-cases", type=int, default=48,
                   help="算子只跑确定性等距子集（预算；仿射仍用全池拟合）")
    t.add_argument("--pcache-root", default=os.path.join(PROJ, "runs/n2_1031/pcache"))
    t.add_argument("--panel", default=PANEL_DIR)
    t.add_argument("--pred-root", default=PRED_ROOT)
    t.add_argument("--threads", type=int, default=16)
    t.add_argument("--out", default=os.path.join(PROJ, "runs/n2_1031/transfer.json"))

    v = sub.add_parser("verdict", help="汇总判据")
    v.add_argument("--dir", default=os.path.join(PROJ, "runs/n2_1031"))
    v.add_argument("--bridge-json", default="")
    v.add_argument("--out", default=None)

    args = ap.parse_args()
    torch.set_num_threads(int(getattr(args, "threads", 16)))
    if args.action in ("gate1", "transfer"):
        torch.set_grad_enabled(False)
    if args.action == "build-dict":
        act_build_dict(args)
    elif args.action == "unit":
        act_unit(args)
    elif args.action == "gate1":
        act_gate1(args)
    elif args.action == "transfer":
        act_transfer(args)
    elif args.action == "verdict":
        args.out = args.out or os.path.join(args.dir, "verdict.json")
        act_verdict(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
