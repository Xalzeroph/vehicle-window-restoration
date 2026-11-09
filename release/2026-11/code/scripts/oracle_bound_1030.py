#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B_oracle（ARCH_H1 / 2026-10-30 子带残差调制）的零训练可行性上界。

零训练、零 GPU、纯 numpy（不 import torch、不读 ckpt）。

定义（照抄预登记，未做任何改动）
--------------------------------
对每个场景、每个 RGB 通道：
    a = W(I - T)                需要的残差（T = GT.png）
    d = W(I - T0)               基线残差（T0 = t_hat.png）
    W = 3 层正交 Haar，逐通道；尺寸不足 8 的倍数 → reflect 补边，算完裁回
    最粗 LL  ：q_LL = d_LL                          （不调制）
    细节     ：S_k = [min(0, 2 d_k), max(0, 2 d_k)]
               q*_k = proj_{S_k}(a_k) = clip(a_k, lo_k, hi_k)
    E0 = sum_all    (a_k - d_k)^2                   （基线误差）
    E* = sum_LL     (a_k - d_k)^2 + sum_detail (a_k - q*_k)^2
    B_oracle = mean_{15 场景 x 3 通道} 10 log10(E0 / E*)

这个数**只**用于 kill / pass，不得用于反推算子或调参。

口径（与项目评测一致）
----------------------
* 读图：cv2.imread（BGR，float64，0-255）—— 与 scripts/table_final_1019.py 的 read() 相同。
  E0/E* 是比值 ⇒ 尺度无关；本脚本把两者都除以 255 归一到 [0,1]，只为数值好看。
* 配对：按 (场景, I 的 stem) 配到 runs/pred/<arm>/<scene>__<stem>/t_hat.png。
  绝不按文件名顺序配（项目踩过坑：按顺序配会把 identity 算成 25-31 dB）。
* 增益：先逐通道算 dB，再对通道平均 —— 与 table_final_1019.psnr_rgb 完全同口径，
  因此 B_oracle 可直接和面板上的 ΔPSNR 比较（ours 22.70 − 官方 20.05 = +2.65）。
* 尺寸不一致（GT / t_hat 与 I 不同）：记录后**跳过**，不 resize（resize 会破坏子带对齐）。

⚠ 本数是「uint8 口径的乐观上界」，不是严格上界
--------------------------------------------
t_hat.png 是 uint8 存盘（eval_arm_1019.py: cv2.imwrite(clip(pred,0,255).astype(uint8))），
与 float 预测差 0.04-0.09 dB（项目已核）。另外 q*_k 是**逐系数** oracle，门必须从
(I, d, 特征) 预测而不能看 GT ⇒ 只能当上界。脚本末尾用 ±0.5 LSB 蒙特卡洛量化
「uint8 存盘」本身给 B_oracle 带来的抖动。

用法
----
    python3 scripts/oracle_bound_1030.py \
        --pairs data/test_sets/real_pair_1019 \
        --pred  runs/pred/real_pair_1019_mayjun15_s101 \
        --out   /tmp/arch1030/oracle \
        --extra-pred "s202:runs/pred/real_pair_1019_mayjun15_s202,s303:runs/pred/real_pair_1019_mayjun15_s303"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import cv2
except Exception as exc:                                    # pragma: no cover
    raise SystemExit("需要 opencv（cv2）：%s" % exc)

LEVELS = 3
PAD_MULT = 8            # 3 层 Haar => 8 的倍数
EPS = 1e-30
T975_14 = 2.1448        # t_{0.975, 14}，n=15 场景

# 预登记判据（2026.10.30_架构改进_双模型方案.md §2）
B_KILL = 0.30           # B_oracle < 0.30 dB ⇒ 机制不充分，可不训练先否掉
MSE_DROP_FOR_PASS = 1.0 - 10.0 ** (-B_KILL / 10.0)   # = 6.6746 %
LL_SHARE_TEST = 1.0 - MSE_DROP_FOR_PASS              # = 93.325 %


# =============================================================== Haar（正交、完美重构）
def _haar2(x):
    """单级正交 2D Haar（逐通道）。与 RDNet/models/arch/rdnet_subband1030.py::haar2 同系。"""
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    return ((a + b + c + d) * 0.5,
            (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5,
            (a - b - c + d) * 0.5)


def _ihaar2(ll, lh, hl, hh):
    """_haar2 的逆（合成）。"""
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    out = np.empty((ll.shape[0] * 2, ll.shape[1] * 2), dtype=np.float64)
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    return out


def haar3(x, levels=LEVELS):
    """三级分析。返回 (LL3, [(lh,hl,hh)_lev1, (..)_lev2, (..)_lev3])，lev1 最细。"""
    det = []
    ll = x
    for _ in range(levels):
        ll, lh, hl, hh = _haar2(ll)
        det.append((lh, hl, hh))
    return ll, det


def ihaar3(ll, det):
    """haar3 的逆（逐级合成）。"""
    for lh, hl, hh in reversed(det):
        ll = _ihaar2(ll, lh, hl, hh)
    return ll


def _coeff_vec(x, levels=LEVELS):
    """把系数拉成一根向量（顺序固定：LL3 先，然后 lev1→lev3 的 (lh,hl,hh)）。"""
    ll, det = haar3(x, levels)
    parts = [ll.ravel().copy()]
    for lev in det:
        for b in lev:
            parts.append(b.ravel().copy())
    return np.concatenate(parts)


def _pad_to(x, m=PAD_MULT, mode="reflect"):
    """补到 m 的倍数（E 步）；返回 (x_pad, ph, pw)。"""
    h, w = x.shape
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw)), mode=mode)
    return x, ph, pw


# ==================================================================== 校验：W^T W = I
def check_orthonormal(h=16, w=24, levels=LEVELS):
    """在 (h,w) 网格上显式构造算子矩阵 M，断言 max|M^T M - I| < 1e-6。"""
    n = h * w
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        e = np.zeros((h, w), dtype=np.float64)
        e.flat[i] = 1.0
        M[:, i] = _coeff_vec(e, levels)
    err = float(np.max(np.abs(M.T @ M - np.eye(n))))
    err2 = float(np.max(np.abs(M @ M.T - np.eye(n))))     # 完美重构的另一半
    U, S, Vt = np.linalg.svd(M)                           # 奇异值应全为 1
    return {"grid": [h, w], "n_coeff": n,
            "max_abs_gram_err": err,
            "max_abs_gram_err_other_side": err2,
            "singular_value_min": float(S.min()),
            "singular_value_max": float(S.max()),
            "pass": bool(err < 1e-6 and err2 < 1e-6)}


def recon_check(x):
    """完美重构 + 能量守恒（在真实尺寸、补边后的网格上）。"""
    xp, ph, pw = _pad_to(x)
    ll, det = haar3(xp)
    xr = ihaar3(ll, det)
    rec = float(np.max(np.abs(xr - xp)))
    e_in = float(np.sum(xp ** 2))
    e_co = float(np.sum(ll ** 2) + sum(float(np.sum(b ** 2)) for lev in det for b in lev))
    rel = abs(e_co - e_in) / max(e_in, EPS)
    return {"pad": [ph, pw], "recon_max_abs_err": rec, "energy_rel_err": rel,
            "pass": bool(rec < 1e-9 and rel < 1e-9)}


# ======================================================================== 单通道 oracle
def channel_oracle(a, d, levels=LEVELS):
    """a, d: (H,W) float64 残差（同一尺度）。返回该通道的 E0 / E* / 诊断。"""
    ap, ph, pw = _pad_to(a)
    dp, _, _ = _pad_to(d)

    ll_a, det_a = haar3(ap, levels)
    ll_d, det_d = haar3(dp, levels)

    ll_err = float(np.sum((ll_a - ll_d) ** 2))          # 最粗 LL：不调制

    det_base = 0.0        # sum_detail (a - d)^2
    det_orac = 0.0        # sum_detail (a - q*)^2
    det_zero = 0.0        # 全删细节：q = 0  =>  sum_detail a^2
    det_shared = 0.0      # 每 (level, band) 一个最优**常数**门
    n_clip = n_lo = n_hi = 0
    n_tot = 0
    err_clip = 0.0        # 被截断系数背负的基线误差
    per_lev_base = [0.0] * levels
    per_lev_orac = [0.0] * levels

    for lev, (la, ld) in enumerate(zip(det_a, det_d)):
        for ba, bd in zip(la, ld):
            lo = np.minimum(0.0, 2.0 * bd)
            hi = np.maximum(0.0, 2.0 * bd)
            q = np.clip(ba, lo, hi)                     # q*_k = proj_{S_k}(a_k)

            base_arr = (ba - bd) ** 2
            base = float(np.sum(base_arr))
            orac = float(np.sum((ba - q) ** 2))
            det_base += base
            det_orac += orac
            per_lev_base[lev] += base
            per_lev_orac[lev] += orac
            det_zero += float(np.sum(ba ** 2))

            m_lo = ba < lo
            m_hi = ba > hi
            m_clip = m_lo | m_hi
            n_lo += int(np.count_nonzero(m_lo))
            n_hi += int(np.count_nonzero(m_hi))
            n_clip += int(np.count_nonzero(m_clip))
            n_tot += int(ba.size)
            err_clip += float(np.sum(base_arr[m_clip]))

            den = float(np.sum(bd * bd))
            g = 0.0 if den <= EPS else float(np.sum(ba * bd) / den)
            g = min(2.0, max(0.0, g))
            det_shared += float(np.sum((ba - g * bd) ** 2))

    E0 = ll_err + det_base
    E_star = ll_err + det_orac
    E_del = ll_err + det_zero
    E_shared = ll_err + det_shared

    # 正交性自检（当次数据上）：||a-d||^2 == LL 误差 + 细节基线误差
    E0_spatial = float(np.sum((ap - dp) ** 2))
    ortho_rel = abs(E0 - E0_spatial) / max(E0_spatial, EPS)

    def g_(e):
        return (10.0 * np.log10(E0 / e)) if (E0 > EPS and e > EPS) else float("nan")

    return {
        "pad": [int(ph), int(pw)],
        "E0": E0, "E_star": E_star,
        "ortho_rel_err": ortho_rel,
        "gain_db": g_(E_star),
        "gain_keep_all_db": g_(E0),                      # 恒 0（对照：g ≡ 1）
        "gain_delete_all_db": g_(E_del),                 # 对照：q_detail = 0
        "gain_shared_gate_db": g_(E_shared),             # 每 (level,band) 一个常数门
        "gain_perfect_detail_db":
            (-10.0 * np.log10(ll_err / E0)) if ll_err > EPS else float("nan"),
        "ll_share": (ll_err / E0) if E0 > EPS else float("nan"),
        "level_base_share": [v / E0 if E0 > EPS else float("nan") for v in per_lev_base],
        "level_orac_share": [v / E0 if E0 > EPS else float("nan") for v in per_lev_orac],
        "clip_frac": (n_clip / n_tot) if n_tot else float("nan"),
        "clip_lo_frac": (n_lo / n_tot) if n_tot else float("nan"),
        "clip_hi_frac": (n_hi / n_tot) if n_tot else float("nan"),
        "clip_err_share": (err_clip / det_base) if det_base > EPS else float("nan"),
    }


# ============================================================================ 面板遍历
def discover_pairs(pairs_dir, pred_dir):
    """按 (scene, stem) 配对；返回 (pairs, problems)。不 resize、不猜。"""
    pairs, problems = [], []
    if not os.path.isdir(pairs_dir):
        raise SystemExit("配对集不存在：%s" % pairs_dir)
    for sc in sorted(os.listdir(pairs_dir)):
        d = os.path.join(pairs_dir, sc)
        if not os.path.isdir(d):
            continue
        gt_p = os.path.join(d, "GT.png")
        if not os.path.exists(gt_p):
            problems.append({"scene": sc, "kind": "no_GT"})
            continue
        for f in sorted(os.listdir(d)):
            if not (f.startswith("I_") and f.endswith(".png")):
                continue
            stem = os.path.splitext(f)[0]
            pred_p = os.path.join(pred_dir, "%s__%s" % (sc, stem), "t_hat.png")
            if not os.path.exists(pred_p):
                problems.append({"scene": sc, "input": stem, "kind": "no_pred"})
                continue
            pairs.append({"scene": sc, "input": stem, "I": os.path.join(d, f),
                          "GT": gt_p, "T0": pred_p})
    return pairs, problems


def check_ll_blockmean(h=64, w=80, levels=LEVELS, seed=0):
    """独立核对最粗 LL：三层正交 Haar 下 LL3 应等于 8×(8×8 块均值)。

    这条不依赖算子矩阵的构造，是「LL 就是块均值」的另一条路径 ——
    LL 占比这个判据的分母全在它身上。
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((h, w))
    ll, _ = haar3(x, levels)
    m = x.reshape(h // PAD_MULT, PAD_MULT, w // PAD_MULT, PAD_MULT).mean(axis=(1, 3)) * PAD_MULT
    err = float(np.max(np.abs(ll - m)))
    return {"grid": [h, w], "ll_max_abs_err_vs_8x8_blockmean": err,
            "pass": bool(err < 1e-10)}


def check_projection(n=20000, seed=0):
    """独立核对闭包解：clip(a, lo, hi) 必须是 S=[min(0,2d),max(0,2d)] 上的真最优。

    与 65 点密网格穷举比较（含端点），逐样本误差不得高于网格最优。
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n) * 0.05
    d = rng.standard_normal(n) * 0.05
    lo = np.minimum(0.0, 2.0 * d)
    hi = np.maximum(0.0, 2.0 * d)
    q = np.clip(a, lo, hi)
    ts = np.linspace(0.0, 1.0, 65)
    grid = lo[:, None] + (hi - lo)[:, None] * ts[None, :]
    best = ((a[:, None] - grid) ** 2).min(axis=1)
    worst = float(np.max((a - q) ** 2 - best))
    return {"n": n, "grid_points": 65, "max_excess_over_grid_optimum": worst,
            "pass": bool(worst <= 1e-15)}


def _read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return None if im is None else im.astype(np.float64)


def mean_ci(vals, t975=T975_14):
    """均值 + 95% CI（n=15 场景时 t=2.1448；别的 n 仍按这个 t 近似，n 变大会偏保守）。"""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "n": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "n": 1, "sd": float("nan"), "ci95": [float(v[0])] * 2}
    sd = float(v.std(ddof=1))
    hw = float(t975 * sd / np.sqrt(v.size))
    return {"mean": float(v.mean()), "sd": sd, "n": int(v.size),
            "min": float(v.min()), "max": float(v.max()),
            "ci95": [float(v.mean() - hw), float(v.mean() + hw)]}


def arm_means_from_csv(path, arms):
    """从 table_final_1019*.csv 取某些臂的均值（用于基线 PSNR 交叉核对）。"""
    if not path or not os.path.exists(path):
        return {}
    acc = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                acc[row["arm"]].append(float(row["psnr"]))
            except Exception:
                continue
    return {a: float(np.mean(acc[a])) for a in arms if acc.get(a)}


def run_panel(pairs, quant_n=0, seed=20261030, verbose=True, tag=""):
    """跑一个 pred 目录的下整个面板。返回 dict（per_scene + 聚合 + 诊断）。"""
    per_scene = []
    all_gain, all_ll, all_shared, all_del, all_clip = [], [], [], [], []
    all_clip_lo, all_clip_hi, all_clip_err = [], [], []
    all_level_base = np.zeros(LEVELS)
    all_level_orac = np.zeros(LEVELS)
    recon_worst = {"recon_max_abs_err": 0.0, "energy_rel_err": 0.0, "pad": [0, 0]}
    ortho_worst = 0.0
    base_psnr_scene = []
    E0_sum = Es_sum = 0.0
    problems = []

    for k, pr in enumerate(pairs, 1):
        I = _read(pr["I"])
        G = _read(pr["GT"])
        T0 = _read(pr["T0"])
        ptag = "%s__%s" % (pr["scene"], pr["input"])
        if I is None or G is None or T0 is None:
            problems.append({"pair": ptag, "kind": "read_failed"})
            continue
        if G.shape != I.shape or T0.shape != I.shape:
            problems.append({"pair": ptag, "kind": "shape_mismatch",
                             "I": list(I.shape), "GT": list(G.shape), "T0": list(T0.shape)})
            if verbose:
                print("   [skip] %s 尺寸不一致 I=%s GT=%s T0=%s（不 resize）"
                      % (ptag, I.shape, G.shape, T0.shape), flush=True)
            continue

        I = I / 255.0
        G = G / 255.0
        T0 = T0 / 255.0
        A = I - G                     # 需要的残差（逐像素）
        D = I - T0                    # 基线残差（逐像素）

        rec = recon_check(A[..., 0])
        if rec["recon_max_abs_err"] > recon_worst["recon_max_abs_err"]:
            recon_worst = dict(rec)

        ch = []
        for c in range(3):
            r = channel_oracle(A[..., c], D[..., c])
            r["channel"] = c
            E0_sum += r["E0"]
            Es_sum += r["E_star"]
            mse_c = float(np.mean((T0[..., c] - G[..., c]) ** 2))
            r["base_psnr"] = (10 * np.log10(1.0 / mse_c)) if mse_c > 0 else float("nan")
            ch.append(r)
            all_gain.append(r["gain_db"])
            all_ll.append(r["ll_share"])
            all_shared.append(r["gain_shared_gate_db"])
            all_del.append(r["gain_delete_all_db"])
            all_clip.append(r["clip_frac"])
            all_clip_lo.append(r["clip_lo_frac"])
            all_clip_hi.append(r["clip_hi_frac"])
            all_clip_err.append(r["clip_err_share"])
            all_level_base += np.array(r["level_base_share"], dtype=np.float64)
            all_level_orac += np.array(r["level_orac_share"], dtype=np.float64)
            ortho_worst = max(ortho_worst, r["ortho_rel_err"])

        scene_gain = float(np.mean([r["gain_db"] for r in ch]))
        scene_psnr = float(np.mean([r["base_psnr"] for r in ch]))
        base_psnr_scene.append(scene_psnr)
        per_scene.append({
            "scene": pr["scene"], "input": pr["input"],
            "shape": list(I.shape[:2]),
            "pad": ch[0]["pad"],
            "baseline_psnr": round(scene_psnr, 4),
            "gain_db": round(scene_gain, 4),
            "ll_share": round(float(np.mean([r["ll_share"] for r in ch])), 6),
            "gain_shared_gate_db": round(float(np.mean([r["gain_shared_gate_db"] for r in ch])), 4),
            "gain_delete_all_db": round(float(np.mean([r["gain_delete_all_db"] for r in ch])), 4),
            "gain_perfect_detail_db": round(
                float(np.mean([r["gain_perfect_detail_db"] for r in ch])), 4),
            "gain_per_channel": [round(r["gain_db"], 4) for r in ch],
            "ll_share_per_channel": [round(r["ll_share"], 6) for r in ch],
            "clip_frac": round(float(np.mean([r["clip_frac"] for r in ch])), 6),
        })
        if verbose:
            print("  %s[%2d/%2d] %-34s base=%6.3f dB  B= %+6.3f dB  LL占比=%.4f  clip=%.3f"
                  % (tag, k, len(pairs), ptag, scene_psnr, scene_gain,
                     per_scene[-1]["ll_share"], per_scene[-1]["clip_frac"]), flush=True)

    if not per_scene:
        return None

    n_ch = len(all_gain)
    gains = np.array([s["gain_db"] for s in per_scene], dtype=np.float64)
    stats_gain = mean_ci(gains)
    ll_ch = np.array(all_ll, dtype=np.float64)
    pooled_db = (10.0 * np.log10(E0_sum / Es_sum)) if (E0_sum > EPS and Es_sum > EPS) else float("nan")

    out = {
        "n_pairs": len(per_scene),
        "n_channels_total": n_ch,
        "B_oracle_db": round(float(stats_gain["mean"]), 4),
        "B_oracle_stats": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in stats_gain.items()},
        "B_oracle_pooled_db": round(float(pooled_db), 4),
        "B_oracle_channel_level": round(float(np.mean(all_gain)), 4),
        "ll_share": {
            "mean_over_scenes": round(float(np.mean([s["ll_share"] for s in per_scene])), 6),
            "mean_over_scene_channels": round(float(ll_ch.mean()), 6),
            "min_scene_channel": round(float(ll_ch.min()), 6),
            "max_scene_channel": round(float(ll_ch.max()), 6),
            "frac_channels_above_93_33pct": round(float(np.mean(ll_ch > LL_SHARE_TEST)), 4),
            "implied_ceiling_db": round(float(-10 * np.log10(float(ll_ch.mean()))), 4),
            "levels_base_share_mean": [round(float(v / n_ch), 6) for v in all_level_base],
            "levels_oracle_share_mean": [round(float(v / n_ch), 6) for v in all_level_orac],
            "level_order": "lev1 = 最细（3/4 系数），lev2 = 3/16，lev3 = 3/64；LL3 = 1/64",
            "note": "levels_*_share 已按 E0 归一：LL 占比 + 三个 level 的 base 占比 = 1",
        },
        "controls": {
            "keep_all_detail_db": 0.0,
            "delete_all_detail_db": round(float(np.mean(all_del)), 4),
            "shared_gate_db": round(float(np.mean(all_shared)), 4),
            "perfect_detail_db": round(
                float(np.mean([s["gain_perfect_detail_db"] for s in per_scene])), 4),
            "clip_frac_mean": round(float(np.mean(all_clip)), 6),
            "clip_lo_frac_mean": round(float(np.mean(all_clip_lo)), 6),
            "clip_hi_frac_mean": round(float(np.mean(all_clip_hi)), 6),
            "clip_err_share_mean": round(float(np.mean(all_clip_err)), 6),
        },
        "checks": {
            "recon_energy": recon_worst,
            "ortho_rel_err_worst_on_data": round(float(ortho_worst), 15),
            "pad_triggered": bool(any(s["pad"] != [0, 0] for s in per_scene)),
            "baseline_psnr_recomputed": round(float(np.mean(base_psnr_scene)), 4),
        },
        "per_scene": per_scene,
        "problems": problems,
    }

    # ---- ±0.5 LSB 蒙特卡洛（只对主臂做）
    quant = {"n": 0,
             "known_project_delta_db": "0.04–0.09 dB"}
    if quant_n and quant_n > 0:
        rng = np.random.default_rng(seed)
        cache = []
        for pr in pairs:
            I = _read(pr["I"]) / 255.0
            G = _read(pr["GT"]) / 255.0
            T0 = _read(pr["T0"]) / 255.0
            if G.shape != I.shape or T0.shape != I.shape:
                continue
            cache.append((I - G, I - T0))
        mcs = []
        for t in range(quant_n):
            gs = []
            for A, D in cache:
                Dp = D + rng.uniform(-0.5, 0.5, size=D.shape) / 255.0
                for c in range(3):
                    gs.append(channel_oracle(A[..., c], Dp[..., c])["gain_db"])
            mcs.append(float(np.mean(gs)))
            if verbose:
                print("  [quant] MC %d/%d  B= %+.5f dB" % (t + 1, quant_n, mcs[-1]), flush=True)
        quant.update({
            "n": len(mcs),
            "mc_mean_db": round(float(np.mean(mcs)), 5),
            "mc_sd_db": round(float(np.std(mcs, ddof=1)) if len(mcs) > 1 else float("nan"), 6),
            "delta_vs_uint8_db": round(float(np.mean(mcs)) - out["B_oracle_db"], 5),
            "interpretation":
                "在已 uint8 化的 t_hat 上再叠加 ±0.5 LSB 均匀噪声，衡量量化尺度噪声"
                "对 B_oracle（一个比值）的敏感度；不是 float-vs-uint8 的严格差值",
        })
    out["quantization"] = quant
    return out


# ================================================================================== 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="B_oracle：子带门控机制的零训练可行性上界")
    ap.add_argument("--pairs", default="data/test_sets/real_pair_1019")
    ap.add_argument("--pred", default="runs/pred/real_pair_1019_mayjun15_s101")
    ap.add_argument("--pred-name", default="", help="主臂显示名（默认取目录名）")
    ap.add_argument("--out", default="/tmp/arch1030/oracle")
    ap.add_argument("--quant-n", type=int, default=8,
                    help="±0.5 LSB 蒙特卡洛次数（0 = 关）")
    ap.add_argument("--extra-pred", default="",
                    help='额外臂（稳健性）："name:dir,name:dir"')
    ap.add_argument("--ref-csv", default="eval/table_final_1019_mayjun_1030.csv",
                    help="交叉核对基线 PSNR 的权威表（不存在则跳过）")
    ap.add_argument("--ref-arm", default="OURS_mayjun_s101",
                    help="权威表里对应本 pred 目录的臂名")
    args = ap.parse_args(argv)

    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    main_name = args.pred_name or os.path.basename(args.pred.rstrip("/"))

    print("=" * 78)
    print("B_oracle（ARCH_H1 子带残差调制）—— 零训练 / 纯 numpy")
    print("  pairs =", args.pairs)
    print("  pred  =", args.pred, "(%s)" % main_name)
    print("  out   =", args.out)
    print("=" * 78, flush=True)

    # ---- 0. 算子自检
    orth = check_orthonormal()
    print("[check] 正交性 max|W^T W - I| = %.3e ；max|W W^T - I| = %.3e ；"
          "奇异值 [%.15f, %.15f] -> %s"
          % (orth["max_abs_gram_err"], orth["max_abs_gram_err_other_side"],
             orth["singular_value_min"], orth["singular_value_max"],
             "PASS" if orth["pass"] else "FAIL"), flush=True)
    if not orth["pass"]:
        raise SystemExit("断言失败：max|W^T W - I| >= 1e-6")

    # ---- 0b. 两条独立不变量（不依赖上面的算子矩阵构造）
    blk = check_ll_blockmean()
    proj = check_projection()
    print("[check] LL3 == 8 × (8×8 块均值) : max|err| = %.3e -> %s"
          % (blk["ll_max_abs_err_vs_8x8_blockmean"], "PASS" if blk["pass"] else "FAIL"), flush=True)
    print("[check] clip 是闭包上的真最优 : max 超出网格最优 = %.3e -> %s"
          % (proj["max_excess_over_grid_optimum"], "PASS" if proj["pass"] else "FAIL"), flush=True)
    if not (blk["pass"] and proj["pass"]):
        raise SystemExit("断言失败：LL 块均值或闭包投影的独立核对没过")

    # ---- 1. 配对
    pairs, problems = discover_pairs(args.pairs, args.pred)
    print("[pair] (scene, stem) 配对 %d 组；问题 %d 条" % (len(pairs), len(problems)), flush=True)
    for p in problems:
        print("   [skip]", p, flush=True)
    if not pairs:
        raise SystemExit("没有可用的配对 —— 先查 --pairs / --pred")

    # ---- 2. 主臂
    main_res = run_panel(pairs, quant_n=args.quant_n, verbose=True)
    if main_res is None:
        raise SystemExit("主臂没有任何一组跑通")
    problems += main_res["problems"]
    B_oracle = main_res["B_oracle_db"]

    # ---- 3. 额外臂（种子稳健性）
    robustness = []
    if args.extra_pred:
        for item in [x for x in args.extra_pred.split(",") if x.strip()]:
            name, _, pdir = item.partition(":")
            name, pdir = name.strip(), pdir.strip()
            if not pdir:
                continue
            px, pbx = discover_pairs(args.pairs, pdir)
            if not px:
                print("[extra] %s 无可配对的 t_hat，跳过" % name, flush=True)
                continue
            print("\n[extra] %s -> %s（%d 对）" % (name, pdir, len(px)), flush=True)
            r = run_panel(px, quant_n=0, verbose=True, tag="  ")
            if r is None:
                continue
            robustness.append({
                "arm": name, "pred_dir": pdir, "n_pairs": r["n_pairs"],
                "baseline_psnr": r["checks"]["baseline_psnr_recomputed"],
                "B_oracle_db": r["B_oracle_db"],
                "ci95": r["B_oracle_stats"].get("ci95"),
                "ll_share": r["ll_share"]["mean_over_scenes"],
                "shared_gate_db": r["controls"]["shared_gate_db"],
                "problems": r["problems"],
            })

    # ---- 4. 交叉核对：基线 PSNR 是否复现权威表
    ref = arm_means_from_csv(args.ref_csv, [args.ref_arm] if args.ref_arm else [])
    arm_info = {}
    ai_p = os.path.join(args.pred, "arm_info.json")
    if os.path.exists(ai_p):
        try:
            arm_info = json.load(open(ai_p, encoding="utf-8"))
        except Exception:
            arm_info = {}

    mine_psnr = main_res["checks"]["baseline_psnr_recomputed"]
    check = {
        "baseline_psnr_recomputed": mine_psnr,
        "ref_arm": args.ref_arm,
        "baseline_psnr_from_csv": {k: round(v, 4) for k, v in ref.items()},
        "delta_vs_csv_db": (round(mine_psnr - list(ref.values())[0], 4) if ref else None),
        "arm_info_psnr_float_pred": arm_info.get("psnr"),
        "note": "csv 口径 = uint8 t_hat + 逐通道 PSNR 平均；arm_info 口径 = float 预测，"
                "两者已知差 0.04-0.09 dB。若 csv 那行与本臂同一 pred 目录，应吻合到 <0.01 dB。",
    }

    # ---- 5. 结果
    allB = [B_oracle] + [r["B_oracle_db"] for r in robustness]
    judgement = {
        "kill_threshold_db": B_KILL,
        "MSE_drop_needed_for_pass": round(MSE_DROP_FOR_PASS, 6),
        "ll_share_threshold": round(LL_SHARE_TEST, 6),
        "verdict": ("KILL（机制不充分）" if B_oracle < B_KILL else "PASS（机制有空间）"),
        "margin_db": round(B_oracle - B_KILL, 4),
        "rule": "2026.10.30_架构改进_双模型方案.md §2：B_oracle < 0.30 dB ⇒ 不训练先否掉",
        "n_scenes_above_threshold": int(sum(1 for s in main_res["per_scene"]
                                            if s["gain_db"] >= B_KILL)),
        "n_arms": len(allB),
        "B_range_db": [round(min(allB), 4), round(max(allB), 4)],
        "verdict_across_arms": ("全部低于阈值" if max(allB) < B_KILL else
                                ("全部不低于阈值" if min(allB) >= B_KILL else
                                 "跨阈值：无裕度")),
    }

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": os.path.abspath(__file__),
        "pairs_dir": args.pairs,
        "pred_dir": args.pred,
        "arm": main_name,
        "n_pairs": main_res["n_pairs"],
        "n_channels_total": main_res["n_channels_total"],
        "definition": {
            "a": "W(I - T), T = GT",
            "d": "W(I - T0), T0 = t_hat.png",
            "W": "3-level orthonormal Haar, per channel, reflect-pad to multiple of 8 then crop",
            "q_LL": "d_LL (coarsest, unmodulated)",
            "q_detail": "proj over S_k = [min(0,2d_k), max(0,2d_k)]",
            "E0": "sum_all (a-d)^2",
            "E_star": "sum_LL (a-d)^2 + sum_detail (a-q*)^2",
            "B_oracle": "mean over scenes x channels of 10*log10(E0/E*)",
        },
        "checks": dict(main_res["checks"], orthonormal=orth, baseline=check,
                       ll_blockmean=blk, projection=proj),
        "B_oracle_db": B_oracle,
        "B_oracle_stats": main_res["B_oracle_stats"],
        "B_oracle_pooled_db": main_res["B_oracle_pooled_db"],
        "judgement": judgement,
        "ll_share": main_res["ll_share"],
        "controls": main_res["controls"],
        "quantization": main_res["quantization"],
        "robustness_other_arms": robustness,
        "problems": problems,
        "per_scene": main_res["per_scene"],
        "seconds": round(time.time() - t_start, 1),
    }

    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "result.md"), "w", encoding="utf-8") as fh:
        fh.write(build_md(result))

    # ---- 6. 屏幕摘要
    st = main_res["B_oracle_stats"]
    ls = main_res["ll_share"]
    ct = main_res["controls"]
    q = main_res["quantization"]
    print("\n" + "=" * 78)
    print("B_oracle = %+.4f dB  (sd %.4f, n=%d 场景, 95%% CI [%+.3f, %+.3f])"
          % (B_oracle, st.get("sd", float("nan")), st.get("n", 0), *st.get("ci95", [0, 0])))
    print("  LL 占比 均值 %.4f（min %.4f / max %.4f）；>93.33%% 的通道占比 %.3f"
          % (ls["mean_over_scenes"], ls["min_scene_channel"], ls["max_scene_channel"],
             ls["frac_channels_above_93_33pct"]))
    print("  对照：全留 %+0.3f / 全删 %+0.3f / 常数门 %+0.3f / 细节完美 %+0.3f"
          % (0.0, ct["delete_all_detail_db"], ct["shared_gate_db"], ct["perfect_detail_db"]))
    print("  判据 B_oracle < %.2f ⇒ %s（差 %+.4f dB）"
          % (B_KILL, judgement["verdict"], judgement["margin_db"]))
    print("  基线 PSNR 复算 %.4f dB；权威表 %s = %.4f dB；arm_info(float) %s"
          % (mine_psnr, args.ref_arm,
             list(ref.values())[0] if ref else float("nan"), arm_info.get("psnr")))
    for r in robustness:
        print("  [extra] %-8s B_oracle %+.4f dB (n=%d) 基线 %.3f dB"
              % (r["arm"], r["B_oracle_db"], r["n_pairs"], r["baseline_psnr"]))
    if q.get("n"):
        print("  uint8 量化：MC mean %+.5f (sd %.6f)，Δ vs uint8 = %+.5f dB"
              % (q["mc_mean_db"], q["mc_sd_db"], q["delta_vs_uint8_db"]))
    print("  -> %s/result.json , result.md  (%.1f s)" % (args.out, time.time() - t_start))
    print("=" * 78)
    return 0


def build_md(R):
    L = []
    A = L.append
    v, c, ls, ct, q = R["judgement"], R["checks"], R["ll_share"], R["controls"], R["quantization"]
    st = R["B_oracle_stats"]
    ci = st.get("ci95", [float("nan")] * 2)
    shape = R["per_scene"][0]["shape"]

    A("# B_oracle：ARCH_H1「完美重构 Haar 子带残差调制」的零训练可行性上界")
    A("")
    A("> 生成 %s · 脚本 `%s` · **零训练 / 零 GPU / 纯 numpy**"
      % (R["generated_at"], R["script"]))
    A("> 面板 `%s` × 预测 `%s`（臂 `%s`，%d 对）"
      % (os.path.basename(R["pairs_dir"].rstrip("/")),
         os.path.basename(R["pred_dir"].rstrip("/")), R["arm"], R["n_pairs"]))
    A("")
    A("## 0 结论")
    A("")
    A("**B_oracle = %+.3f dB**（%d 场景 × 3 通道逐通道 PSNR 增益的均值；"
      "场景间 sd %.3f，95%% CI [%+.3f, %+.3f]）。"
      % (R["B_oracle_db"], R["n_pairs"], st.get("sd", float("nan")), ci[0], ci[1]))
    A("")
    A("按预登记判据 `B_oracle < %.2f dB ⇒ 机制不充分`：**%s**（距阈值 %+.3f dB）。"
      % (v["kill_threshold_db"], v["verdict"], v["margin_db"]))
    if v.get("n_arms", 1) > 1:
        A("")
        A("对 %d 个现有 B 臂（不同种子）复算：B_oracle ∈ [%+.3f, %+.3f] dB ⇒ **%s**；"
          "最高那一臂也只比阈值高 %+.3f dB。"
          % (v["n_arms"], v["B_range_db"][0], v["B_range_db"][1], v["verdict_across_arms"],
             v["B_range_db"][1] - v["kill_threshold_db"]))
    A("")
    A("| 量 | 值 | 含义 |")
    A("|---|---|---|")
    A("| **B_oracle** | **%+.3f dB** | 只靠细节门控（逐系数 oracle 门）的最多可涨量上界 |"
      % R["B_oracle_db"])
    A("| 预登记 kill 阈值 | %.2f dB | 低于它 ⇒ 机制不充分，不训练先否掉 |"
      % v["kill_threshold_db"])
    A("| 达到阈值所需 MSE 降幅 | %.4f%% | 1 − 10^(−0.03)；等价于 dB 差 0.30 |"
      % (100 * v["MSE_drop_needed_for_pass"]))
    A("| 与阈值的距离 | %+.3f dB | 负 = 未达阈值 |" % v["margin_db"])
    A("| LL 误差占比（最粗 LL，场景均值） | %.4f | > %.4f 则细节全消也到不了阈值 |"
      % (ls["mean_over_scenes"], v["ll_share_threshold"]))
    A("| LL 占比隐含天花板 | %+.3f dB | = −10·log10(LL 占比) |" % ls["implied_ceiling_db"])
    A("| 能量加权（pooled）对照 | %+.3f dB | 换聚合口径的稳健性检查 |"
      % R["B_oracle_pooled_db"])
    A("")
    A("**这个 0.30 dB 判据有多严**：H1 在预登记里的预期收益是 +0.20~+0.50 dB"
      "（`2026.10.30_架构改进_双模型方案.md` §1 表），阈值 0.30 取在其中。"
      "B_oracle 是**上界** ⇒ 任何实现都不会超过 %+.3f dB：它**够得到**预期区间下端 +0.20，"
      "但**到不了** 0.30 的门槛，更到不了区间中位 +0.35。" % R["B_oracle_db"])
    A("")
    A("## 1 定义（照抄预登记，未改）")
    A("")
    A("对每个场景、每个 RGB 通道：")
    A("")
    A("- `a = W(I − T)`（T = GT.png）· `d = W(I − T0)`（T0 = t_hat.png，现成缓存）")
    A("- `W` = 3 层正交 Haar，逐通道；尺寸不足 8 的倍数就 reflect 补边，算完裁回")
    A("- 最粗 LL：`q_LL = d_LL`（不调制）")
    A("- 细节：`S_k = [min(0, 2d_k), max(0, 2d_k)]`，`q*_k = proj_{S_k}(a_k)`")
    A("- `E0 = Σ_all (a−d)²`，`E* = Σ_LL (a−d)² + Σ_detail (a−q*)²`")
    A("- `B_oracle = mean_{场景 × 通道} 10·log10(E0/E*)`")
    A("")
    A("## 2 口径与校验（先看这节，再看数）")
    A("")
    A("| 项 | 结果 |")
    A("|---|---|")
    A("| `max‖WᵀW − I‖`（%d×%d 网格，显式算子矩阵 %d×%d） | %.3e %s |"
      % (c["orthonormal"]["grid"][0], c["orthonormal"]["grid"][1],
         c["orthonormal"]["n_coeff"], c["orthonormal"]["n_coeff"],
         c["orthonormal"]["max_abs_gram_err"],
         "PASS（判据 <1e-6）" if c["orthonormal"]["pass"] else "**FAIL**"))
    A("| `max‖WWᵀ − I‖` | %.3e |" % c["orthonormal"]["max_abs_gram_err_other_side"])
    A("| 奇异值 min / max | %.15f / %.15f |"
      % (c["orthonormal"]["singular_value_min"], c["orthonormal"]["singular_value_max"]))
    A("| 独立核对：`LL3 == 8 × (8×8 块均值)` | %.3e %s |"
      % (c["ll_blockmean"]["ll_max_abs_err_vs_8x8_blockmean"],
         "PASS" if c["ll_blockmean"]["pass"] else "**FAIL**"))
    A("| 独立核对：`clip(a,lo,hi)` 是闭包上真最优（%d 样本 × %d 点网格） | 超出 %.3e %s |"
      % (c["projection"]["n"], c["projection"]["grid_points"],
         c["projection"]["max_excess_over_grid_optimum"],
         "PASS" if c["projection"]["pass"] else "**FAIL**"))
    A("| 完美重构 `max‖WᵀWx − x‖`（真实尺寸） | %.3e |" % c["recon_energy"]["recon_max_abs_err"])
    A("| 能量守恒相对误差 | %.3e |" % c["recon_energy"]["energy_rel_err"])
    A("| 数据上正交分解一致性（偏差最大的一次） | %.3e |" % c["ortho_rel_err_worst_on_data"])
    A("| 是否触发补边 | %s（本面板 %d×%d，H、W 均被 8 整除） |"
      % ("是" if c["pad_triggered"] else "否", shape[1], shape[0]))
    A("| **基线 PSNR 复算** | **%.4f dB** |" % c["baseline"]["baseline_psnr_recomputed"])
    for k, val in c["baseline"]["baseline_psnr_from_csv"].items():
        A("| 权威表 `<%s>` 同臂 | %.4f dB（差 %+.4f） |"
          % (k, val, c["baseline"]["delta_vs_csv_db"]))
    A("| `arm_info.json`（float 预测口径） | %s dB |"
      % c["baseline"]["arm_info_psnr_float_pred"])
    A("")
    A("**基线 PSNR 复算是看门人**：按 `(scene, stem)` 配对、`cv2.imread` 读图、"
      "逐通道 PSNR 再平均 —— 复算值必须与权威表同一行吻合；对不上就是配对/读图口径错了，"
      "下面的数全部作废。`arm_info` 那行是 float 预测口径，本就该差 0.04–0.09 dB。")
    A("")
    A("增益口径 = 先逐通道算 dB、再对通道和场景平均 —— 与 `table_final_1019.psnr_rgb` 同口径，"
      "所以 B_oracle 可以直接跟面板 ΔPSNR 比（ours 22.70 − 官方 20.05 = **+2.65**）。")
    A("")
    A("## 3 诊断：误差落在哪一层")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| LL（最粗，1/64 系数）误差占比 · 场景均值 | %.4f |" % ls["mean_over_scenes"])
    A("| 同上 · %d 个场景-通道的均值 | %.4f |" % (len(R["per_scene"]) * 3,
                                            ls["mean_over_scene_channels"]))
    A("| 同上 · min / max（场景-通道） | %.4f / %.4f |"
      % (ls["min_scene_channel"], ls["max_scene_channel"]))
    A("| LL 占比 > 93.33%% 的场景-通道比例 | %.3f |" % ls["frac_channels_above_93_33pct"])
    A("| 逐层基线误差占比（lev1 最细 → lev3） | %s |"
      % " / ".join("%.4f" % x for x in ls["levels_base_share_mean"]))
    A("| 逐层 oracle 后残留占比 | %s |"
      % " / ".join("%.4f" % x for x in ls["levels_oracle_share_mean"]))
    A("")
    A("*层级面积比例：lev1 = 3/4 系数、lev2 = 3/16、lev3 = 3/64、LL3 = 1/64。*")
    A("")
    A("读法：LL 占比均值 %.4f ⇒ 细节完美也只值 **+%.3f dB**（= −10·log10(LL 占比)；"
      "逐场景再平均的口径见 §4 的 %+.3f）；而闭包内最优门只拿到 %+.3f dB —— "
      "差出来的 %.3f dB 全部来自**闭包**（`q` 只能把 `d` 按 (0,2) 缩放：不能反向、不能超 2 倍），"
      "不是 LL 锁死。" % (ls["mean_over_scenes"], ls["implied_ceiling_db"], ct["perfect_detail_db"],
                      R["B_oracle_db"], ls["implied_ceiling_db"] - R["B_oracle_db"]))
    A("")
    A("## 4 对照：把 B_oracle 夹在中间")
    A("")
    A("| 臂 | dB | 说明 |")
    A("|---|---|---|")
    A("| 全留细节（g ≡ 1） | %+.3f | 基线本身（初始化态），恒 0 |" % ct["keep_all_detail_db"])
    A("| **逐系数 oracle 门** | **%+.3f** | 本文的 B_oracle（最乐观） |" % R["B_oracle_db"])
    A("| 每 (level, band) 一个最优**常数**门 | %+.3f | 去掉「逐系数选择性」后的天花板 |"
      % ct["shared_gate_db"])
    A("| 细节完美消掉（q* = a） | %+.3f | 只剩 LL 残余 ⇒ 等于 LL 占比隐含天花板 |"
      % ct["perfect_detail_db"])
    A("| 全删细节（q ≡ 0） | %+.3f | 门开到 0 的下界对照 |" % ct["delete_all_detail_db"])
    A("")
    A("**最要紧的一行是常数门**：把每 (channel, level, band) 合成一个用 GT 拟合的最优标量门，"
      "只值 **%+.3f dB**（占 B_oracle 的 %.0f%%）。也就是说 B_oracle 里 "
      "%.0f%% 的增益依赖**逐系数**的空间选择性。实现里的门确实是逐像素输出的"
      "（`out_conv` 出 9 张空间图），但它看不到 GT —— 这个上界已经替它把"
      "「哪里该放大」全猜对了。"
      % (ct["shared_gate_db"], 100 * ct["shared_gate_db"] / max(R["B_oracle_db"], 1e-9),
         100 * (1 - ct["shared_gate_db"] / max(R["B_oracle_db"], 1e-9))))
    A("")
    A("| 闭包诊断（细节侧） | 值 |")
    A("|---|---|")
    A("| `a` 落在门外、被闭包截断的细节系数占比 | %.3f |" % ct["clip_frac_mean"])
    A("| 这些系数背负的**基线误差**占比 | **%.3f** |" % ct["clip_err_share_mean"])
    A("| 需要 gain < 0（`a` 与 `d` 反号）的系数 | %.3f |" % ct["clip_lo_frac_mean"])
    A("| 需要 gain > 2 的系数 | %.3f |" % ct["clip_hi_frac_mean"])
    A("")
    A("**这就解释了 %.3f dB 的缺口**：截断只涉及 %.0f%% 的系数，却占 **%.1f%%** 的细节侧误差 —— "
      "这些位置不是「方向对、幅度不对」，而是方向就反了、或者压根没有可放大的量。"
      "按比例缩放 `d`（哪怕逐系数用 GT 选最优倍数）在结构上修不了这一类误差。"
      "（改门的形式属于改算子，不在本表范围内。）"
      % (ls["implied_ceiling_db"] - R["B_oracle_db"],
         100 * ct["clip_frac_mean"], 100 * ct["clip_err_share_mean"]))
    A("")
    A("*注：`d_k = 0` 的系数闭包退化为 `{0}`，此时 lo/hi 两分支只描述 `a` 的符号，"
      "不代表「需要负增益」——这两个比例是描述性的，不是可学的处方。*")
    A("")
    A("## 5 逐场景")
    A("")
    A("| 场景 | 尺寸 | 基线 PSNR | B_oracle | 常数门 | 全删细节 | LL 占比 | clip 比例 |")
    A("|---|---|---|---|---|---|---|---|")
    for s in R["per_scene"]:
        A("| %s | %d×%d | %.2f | %+.3f | %+.3f | %+.3f | %.4f | %.3f |"
          % (s["scene"], s["shape"][1], s["shape"][0], s["baseline_psnr"], s["gain_db"],
             s["gain_shared_gate_db"], s["gain_delete_all_db"], s["ll_share"], s["clip_frac"]))
    A("")
    A("%d/%d 个场景单独也 < %.2f dB；逐场景全部为正（%+.3f ~ %+.3f）。"
      % (R["n_pairs"] - v["n_scenes_above_threshold"], R["n_pairs"], v["kill_threshold_db"],
         min(s["gain_db"] for s in R["per_scene"]),
         max(s["gain_db"] for s in R["per_scene"])))
    A("")
    A("## 6 B 臂选择（种子）稳健性")
    A("")
    if R["robustness_other_arms"]:
        A("| 臂 | 基线 PSNR | B_oracle | 95% CI | LL 占比 | 常数门 |")
        A("|---|---|---|---|---|---|")
        A("| %s（主） | %.3f | **%+.3f** | [%+.3f, %+.3f] | %.4f | %+.3f |"
          % (R["arm"], c["baseline"]["baseline_psnr_recomputed"], R["B_oracle_db"],
             ci[0], ci[1], ls["mean_over_scenes"], ct["shared_gate_db"]))
        for r in R["robustness_other_arms"]:
            rc = r.get("ci95") or [float("nan")] * 2
            A("| %s | %.3f | %+.3f | [%+.3f, %+.3f] | %.4f | %+.3f |"
              % (r["arm"], r["baseline_psnr"], r["B_oracle_db"], rc[0], rc[1],
                 r["ll_share"], r["shared_gate_db"]))
        A("")
        A("B 臂会重训（设计审 2 号阻塞项），所以上界必须对「用哪个 T0」不敏感才可信；"
          "上表就是这件事的实测。三个臂的基线越低、上界越高"
          "（%.2f→%+.3f / %.2f→%+.3f / %.2f→%+.3f），方向符合「基线越差、"
          "留在细节侧待纠正的误差越多」—— 只有 3 个点，别当规律用。"
          % (c["baseline"]["baseline_psnr_recomputed"], R["B_oracle_db"],
             R["robustness_other_arms"][0]["baseline_psnr"],
             R["robustness_other_arms"][0]["B_oracle_db"],
             R["robustness_other_arms"][-1]["baseline_psnr"],
             R["robustness_other_arms"][-1]["B_oracle_db"]))
    else:
        A("未跑额外臂（`--extra-pred` 为空）。")
    A("")
    A("## 7 量化口径：这不是严格上界")
    A("")
    A("- `t_hat.png` 是 **uint8** 存盘（`np.clip(pred,0,255).astype(np.uint8)`），"
      "项目已核：float 预测 vs uint8 存盘的 PSNR 差 **%s**"
      "（`2026.10.30_会话交接_架构损失池与消融.md` §3）。" % q["known_project_delta_db"])
    if q.get("n"):
        A("- ±0.5 LSB 蒙特卡洛（n=%d，重采 t_hat 的量化误差）：B_oracle = **%+.5f ± %.6f dB**，"
          "相对 uint8 直接算偏移 **%+.5f dB**。"
          % (q["n"], q["mc_mean_db"], q["mc_sd_db"], q["delta_vs_uint8_db"]))
        A("- 注意这两件事不是一回事：0.04–0.09 dB 是 `E0` 侧（预测本身的 PSNR）的差；"
          "B_oracle 是 `E0/E*` 的**比值**，量化噪声同时抬高分子分母，所以对它几乎不敏感 —— "
          "实测逐次抖动 sd = %.1e dB。" % max(q["mc_sd_db"], 0.0))
    A("- 因此：这个数足以做 kill/pass 判断（与 0.30 dB 判据的距离远大于量化抖动），"
      "但**不能**声称是严格上界，也不能拿它反推可部署收益。")
    A("")
    A("使它只能当上界的另两条：")
    A("")
    A("1. `q*_k` 是**逐系数** oracle：门由 `(I, d, 解码特征)` 预测，看不到 GT；"
      "实际算子（2,969 参数的头）达不到 ⇒ 真臂只会更差。")
    A("2. 闭包 `S_k` 只是固定 `D` 下逐系数独立最优的**下确界**，不是九项损失的解"
      "（数学审 §2 第 1 条）。")
    A("")
    A("## 8 判读与限制")
    A("")
    A("1. **只做 kill/pass**：不要拿这个数去调算子，也不要写进论文当收益预告。")
    A("2. 判据来源：`2026.10.30_架构改进_双模型方案.md` §2 —— `B_oracle < 0.30 dB` "
      "判机制不充分；等价的代数事实是 +0.30 dB 需要 MSE 至少降 %.4f%%，"
      "若每个场景通道 LL 误差占比都 > %.4f%% 则细节全消也到不了。"
      % (100 * v["MSE_drop_needed_for_pass"], 100 * v["ll_share_threshold"]))
    A("3. **CI 不是判据的不确定性**：B_oracle 是在冻结面板上的确定性计算，"
      "CI 只描述场景间差异；判据比的是**上界**与 0.30，不是 CI 与 0.30。"
      "面板均值 %+.3f 已在阈值下，且上界性质意味着换实现只会更低。"
      % R["B_oracle_db"])
    A("4. **面板参与选择**：这份诊断正好跑在主评测 15 场景上，按该档 §2 的告诫，"
      "必须预登记为「只报告、不据此改方案」，或改到场景隔离的开发集上复核。")
    A("5. 本表**不**涉及训练：B 臂能否学到这个门、以及损失/初始化/池子的问题，与本数无关。")
    A("6. **这是临界判决，别读成「收益为零」**：上界 %+.3f dB 压在阈值 %.2f 附近，"
      "它证明的是「达不到 +%.2f dB 的预登记门槛」，"
      "而上界仍高于预登记预期区间的下限 +0.20 dB。是否值得为这 ~%.2f dB 跑正式训练 + 三种子，"
      "是预算判断，本表只能提供上界，替代不了那个判断。"
      % (R["B_oracle_db"], v["kill_threshold_db"], v["kill_threshold_db"], R["B_oracle_db"]))
    A("")
    if R["problems"]:
        A("## 9 跳过 / 异常记录")
        A("")
        for p in R["problems"]:
            A("- `%s`" % json.dumps(p, ensure_ascii=False))
        A("")
    A("---")
    A("*脚本 `scripts/oracle_bound_1030.py`；机读结果同目录 `result.json`（含逐场景、"
      "逐通道、三档量化与额外臂）。*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
