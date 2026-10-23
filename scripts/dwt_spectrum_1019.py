#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DWT 频域分布：我们的真实车窗反射 vs 合成/RRW 反射（2026-10-20）。

要回答的问题
------------
我们的反射是"弱反射、无炫光、偏暗"。它在频率上是否**比常规反射更偏低频**？
若是 → 可在频域强约束 R̂ 为低频，把高频让给 T（方向 B 有依据）；
若否 → 别走 B，改走结构保护（方向 A）。

口径（**尺度无关**，避免被分辨率/亮度带偏）
-------------------------------------------
对每张图做 L 级 Haar DWT，得到各子带能量 E[band]。
真实数据：R = I − GT（f0 配对，15 对），GT 为透射真值。
参照数据：合成集的 r/（已知反射层）与 gt/（已知透射层）。

关键归一化：**用"该子带能量 / 同尺度透射子带能量"**作为相对频响 rho[band]，
这样与绝对亮度无关，只反映"反射在该频带的相对占比"。

输出：eval/dwt_r_spectrum.json + 终端表 + fig_1019/dwt_r_spectrum.png
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
SYNTH = os.path.join(P, "data/test_sets/synth_rrw_v21_0913")
EVAL = os.path.join(P, "eval")
FIG = os.path.join(EVAL, "fig_1019")
LEVELS = 3
BANDS = ["LL1", "LH1", "HL1", "HH1",
         "LL2", "LH2", "HL2", "HH2",
         "LL3", "LH3", "HL3", "HH3"]


def dwt_level(img_gray, levels=LEVELS):
    """返回 [(LL, LH, HL, HH), ...]，每项为该级 4 个子带（float32）。"""
    out = []
    cur = img_gray.astype(np.float32)
    for _ in range(levels):
        # Haar: 沿行再沿列做 [1,1]/sqrt2 与 [1,-1]/sqrt2
        h, w = cur.shape
        if h % 2 or w % 2:
            cur = cur[: h // 2 * 2, : w // 2 * 2]
        a = cur[:, 0::2]; b = cur[:, 1::2]
        lo_r = (a + b) / np.sqrt(2.0)
        hi_r = (a - b) / np.sqrt(2.0)
        c = lo_r[0::2, :]; d = lo_r[1::2, :]
        LL = (c + d) / np.sqrt(2.0)
        LH = (c - d) / np.sqrt(2.0)
        c2 = hi_r[0::2, :]; d2 = hi_r[1::2, :]
        HL = (c2 + d2) / np.sqrt(2.0)
        HH = (c2 - d2) / np.sqrt(2.0)
        out.append((LL, LH, HL, HH))
        cur = LL
    return out


def band_energy(img_gray):
    """返回 {band_name: mean(|X|^2)}。"""
    subs = dwt_level(img_gray)
    e = {}
    for lvl, (LL, LH, HL, HH) in enumerate(subs, 1):
        for nm, s in (("LL", LL), ("LH", LH), ("HL", HL), ("HH", HH)):
            e[f"{nm}{lvl}"] = float(np.mean(s.astype(np.float64) ** 2))
    return e


def gray_of(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im.astype(np.float32) if im is not None else None


def ratio_spectrum(R_gray, T_gray):
    """rho[band] = E_R[band] / E_T[band]（尺度无关）。"""
    er = band_energy(R_gray)
    et = band_energy(T_gray)
    return {b: (er[b] / et[b] if et[b] > 1e-9 else float("nan")) for b in BANDS}, er, et


def main() -> int:
    os.makedirs(FIG, exist_ok=True)

    # ---------- 1) 真实 1019 ----------
    real_rho, real_rel = [], []
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    n = 0
    for sc in scenes:
        I = gray_of(os.path.join(PAIR, sc, "I_000000.png"))
        G = gray_of(os.path.join(PAIR, sc, "GT.png"))
        if I is None or G is None:
            continue
        R = I - G
        rho, er, et = ratio_spectrum(R, G)
        tot = sum(er.values())
        real_rho.append(rho)
        real_rel.append({b: (er[b] / tot if tot > 0 else 0.0) for b in BANDS})
        n += 1
    print(f"真实数据: {n} 对")

    def agg(list_of_dicts):
        return {b: float(np.nanmedian([d[b] for d in list_of_dicts])) for b in BANDS}

    real_rho_m = agg(real_rho)
    real_rel_m = agg(real_rel)

    # ---------- 2) 合成参照（RRW 合成集，含已知 r/ 与 gt/）----------
    r_files = sorted(glob.glob(os.path.join(SYNTH, "r", "*")))
    g_files = sorted(glob.glob(os.path.join(SYNTH, "gt", "*")))
    syn_rho, syn_rel = [], []
    for rf, gf in zip(r_files, g_files):
        Rg = gray_of(rf); Tg = gray_of(gf)
        if Rg is None or Tg is None:
            continue
        h = min(Rg.shape[0], Tg.shape[0]); w = min(Rg.shape[1], Tg.shape[1])
        Rg, Tg = Rg[:h, :w], Tg[:h, :w]
        rho, er, _ = ratio_spectrum(Rg, Tg)
        tot = sum(er.values())
        syn_rho.append(rho)
        syn_rel.append({b: (er[b] / tot if tot > 0 else 0.0) for b in BANDS})
    syn_rho_m = agg(syn_rho) if syn_rho else None
    syn_rel_m = agg(syn_rel) if syn_rel else None
    print(f"合成参照: {len(syn_rho)} 对")

    # ---------- 3) 打印与判定 ----------
    def hf_share(d):
        hf = sum(d[b] for b in BANDS if not b.startswith("LL"))
        tot = sum(d[b] for b in BANDS)
        return hf / tot if tot > 0 else float("nan")

    print("\n=== 相对频响 rho = E_R[band] / E_T[band]（中位，>1 表示反射在该带强于透射）===")
    hdr = f"{'band':<6}" + "".join(f"{x:>10}" for x in ["真实1019"] +
                                   (["合成RRW"] if syn_rho_m else []))
    print(hdr); print("-" * len(hdr))
    for b in BANDS:
        line = f"{b:<6}{real_rho_m[b]:>10.3f}"
        if syn_rho_m:
            line += f"{syn_rho_m[b]:>10.3f}"
        print(line)

    print("\n=== 反射自身的频带能量占比（归一化到 1）===")
    hdr2 = f"{'band':<6}" + "".join(f"{x:>10}" for x in ["真实1019"] +
                                    (["合成RRW"] if syn_rel_m else []))
    print(hdr2); print("-" * len(hdr2))
    for b in BANDS:
        line = f"{b:<6}{real_rel_m[b]:>10.3f}"
        if syn_rel_m:
            line += f"{syn_rel_m[b]:>10.3f}"
        print(line)

    hs_r = hf_share(real_rel_m)
    print(f"\n真实反射的高频占比(HF/all) = {hs_r:.3f}")
    if syn_rel_m:
        hs_s = hf_share(syn_rel_m)
        print(f"合成反射的高频占比        = {hs_s:.3f}   (差 {hs_r-hs_s:+.3f})")
    hr_r = sum(real_rho_m[b] for b in BANDS if not b.startswith("LL")) / \
           sum(real_rho_m[b] for b in BANDS)
    print(f"真实 rho 的高频占比        = {hr_r:.3f}")

    out = {"n_real": n, "n_synth": len(syn_rho),
           "real_rho": real_rho_m, "real_energy_share": real_rel_m,
           "synth_rho": syn_rho_m, "synth_energy_share": syn_rel_m,
           "real_hf_share": hs_r}
    sp = os.path.join(EVAL, "dwt_r_spectrum.json")
    json.dump(out, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {sp}")

    # ---------- 4) 出图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        x = np.arange(len(BANDS))
        ax[0].bar(x - 0.2, [real_rho_m[b] for b in BANDS], 0.4, label="real 1019 (R=I-GT)")
        if syn_rho_m:
            ax[0].bar(x + 0.2, [syn_rho_m[b] for b in BANDS], 0.4, label="synth RRW (R known)")
        ax[0].axhline(1.0, color="k", ls="--", lw=.8)
        ax[0].set_xticks(x); ax[0].set_xticklabels(BANDS, rotation=45, ha="right", fontsize=8)
        ax[0].set_ylabel("E_R[band] / E_T[band]")
        ax[0].set_title("Relative reflection spectrum (rho)")
        ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)

        ax[1].bar(x - 0.2, [real_rel_m[b] for b in BANDS], 0.4, label="real 1019")
        if syn_rel_m:
            ax[1].bar(x + 0.2, [syn_rel_m[b] for b in BANDS], 0.4, label="synth RRW")
        ax[1].set_xticks(x); ax[1].set_xticklabels(BANDS, rotation=45, ha="right", fontsize=8)
        ax[1].set_ylabel("share of reflection energy")
        ax[1].set_title("Reflection energy distribution")
        ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)
        plt.tight_layout()
        p = os.path.join(FIG, "dwt_r_spectrum.png")
        plt.savefig(p, dpi=140)
        print(f"图 -> {p}")
    except Exception as e:
        print(f"[跳过出图] {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
