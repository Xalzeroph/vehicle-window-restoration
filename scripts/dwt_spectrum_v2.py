#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DWT 频域分析 —— 加入"抑制 JPEG 伪影"的对照版（2026-10-20）。

上一版的教训
------------
`R = I − GT` 里 I 与 GT **都经过 JPEG 压缩**（quality 92），差分会把压缩块噪声抬到高频，
使"反射的高频占比"被高估。上一版右图（能量占比）里两边都挤在 LL3，正是压缩伪影的特征。

本版补三个对照，用来把"真反射的高频"与"压缩噪声的高频"分开：
  A. real        : R = I − GT
  B. real_denoise: R = blur(I) − blur(GT)，blur 为 3x3 高斯（压掉 JPEG 8x8 块噪声）
  C. synth       : RRW 合成集已知 r/（PNG，无 JPEG 块）
判据：若 B 的高频占比显著低于 A，说明 A 的高频主要是压缩噪声，不是反射。
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
LL_BANDS = [f"LL{i}" for i in (1, 2, 3)]
HF_BANDS = [f"{o}{i}" for i in (1, 2, 3) for o in ("LH", "HL", "HH")]
ALL_BANDS = LL_BANDS + HF_BANDS


def dwt_subs(gray, levels=LEVELS):
    out, cur = [], gray.astype(np.float32)
    for _ in range(levels):
        h, w = cur.shape
        cur = cur[: h // 2 * 2, : w // 2 * 2]
        a, b = cur[:, 0::2], cur[:, 1::2]
        lo_r, hi_r = (a + b) / np.sqrt(2.0), (a - b) / np.sqrt(2.0)
        c, d = lo_r[0::2, :], lo_r[1::2, :]
        LL, LH = (c + d) / np.sqrt(2.0), (c - d) / np.sqrt(2.0)
        c2, d2 = hi_r[0::2, :], hi_r[1::2, :]
        HL, HH = (c2 + d2) / np.sqrt(2.0), (c2 - d2) / np.sqrt(2.0)
        out.append({"LL": LL, "LH": LH, "HL": HL, "HH": HH})
        cur = LL
    return out


def band_energy(gray):
    e = {}
    for lvl, sub in enumerate(dwt_subs(gray), 1):
        for nm, s in sub.items():
            e[f"{nm}{lvl}"] = float(np.mean(s.astype(np.float64) ** 2))
    return e


def hf_share_of(energy: dict) -> float:
    tot = sum(energy.values())
    if tot <= 0:
        return float("nan")
    return sum(energy[b] for b in HF_BANDS) / tot


def gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im.astype(np.float32) if im is not None else None


def blur(g):
    return cv2.GaussianBlur(g, (3, 3), 0.8)


def ratio_rho(R, T):
    er, et = band_energy(R), band_energy(T)
    return {b: (er[b] / et[b] if et[b] > 1e-9 else float("nan")) for b in ALL_BANDS}


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))

    res = {"real": [], "real_denoise": []}
    for sc in scenes:
        I = gray(os.path.join(PAIR, sc, "I_000000.png"))
        G = gray(os.path.join(PAIR, sc, "GT.png"))
        if I is None or G is None:
            continue
        # A: 原始差分
        eA = band_energy(I - G)
        res["real"].append(eA)
        # B: 先去噪再差分
        eB = band_energy(blur(I) - blur(G))
        res["real_denoise"].append(eB)

    # C: 合成参照
    syn = []
    for rf, gf in zip(sorted(glob.glob(os.path.join(SYNTH, "r", "*"))),
                      sorted(glob.glob(os.path.join(SYNTH, "gt", "*")))):
        Rg, Tg = gray(rf), gray(gf)
        if Rg is None or Tg is None:
            continue
        h, w = min(Rg.shape[0], Tg.shape[0]), min(Rg.shape[1], Tg.shape[1])
        syn.append(band_energy(Rg[:h, :w]))

    def share(lst):
        return {b: float(np.nanmedian([e[b] / max(sum(e.values()), 1e-9) for e in lst]))
                for b in ALL_BANDS}

    sA, sB, sC = share(res["real"]), share(res["real_denoise"]), share(syn)

    def hf(d):
        return float(sum(d[b] for b in HF_BANDS))

    print(f"真实 {len(res['real'])} 对，合成 {len(syn)} 对\n")
    print("=== 反射能量在各频带的占比（中位，归一化）===")
    print(f"{'band':<6}{'real(原始)':>12}{'real(去噪后)':>14}{'synth':>9}")
    print("-" * 44)
    for b in ALL_BANDS:
        print(f"{b:<6}{sA[b]:>12.4f}{sB[b]:>14.4f}{sC[b]:>9.4f}")

    print("\n=== 高频占比合计（LH+HL+HH 全部尺度）===")
    print(f"  real(原始)   : {hf(sA):.4f}")
    print(f"  real(去噪后) : {hf(sB):.4f}   "
          f"(相对原始 {hf(sB)-hf(sA):+.4f})")
    print(f"  synth        : {hf(sC):.4f}")

    # 结论判据
    drop = hf(sA) - hf(sB)
    print("\n=== 判定 ===")
    if drop > 0.5 * hf(sA):
        print("  → 原始差分的高频**主要是 JPEG 压缩噪声**（去噪后大幅下降）")
        print("    真实反射应以『去噪后』的分布为准")
    else:
        print("  → 高频并非压缩伪影主导，真实反射自身含可观高频")
    print(f"  真实反射（去噪后）高频占比 {hf(sB):.4f} vs 合成 {hf(sC):.4f} "
          f"→ 真实{'更' if hf(sB)>hf(sC) else '更不'}偏高/低频")

    out = {"n_real": len(res["real"]), "n_synth": len(syn),
           "share_real_raw": sA, "share_real_denoise": sB, "share_synth": sC,
           "hf_raw": hf(sA), "hf_denoise": hf(sB), "hf_synth": hf(sC)}
    json.dump(out, open(os.path.join(EVAL, "dwt_r_spectrum_v2.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {EVAL}/dwt_r_spectrum_v2.json")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4))
        x = np.arange(len(ALL_BANDS))
        w = 0.27
        ax[0].bar(x - w, [sA[b] for b in ALL_BANDS], w, label="real (raw diff)")
        ax[0].bar(x, [sB[b] for b in ALL_BANDS], w, label="real (denoised diff)")
        ax[0].bar(x + w, [sC[b] for b in ALL_BANDS], w, label="synth RRW (known R)")
        ax[0].set_xticks(x); ax[0].set_xticklabels(ALL_BANDS, rotation=45, ha="right", fontsize=8)
        ax[0].set_ylabel("share of reflection energy")
        ax[0].set_title("Reflection energy per DWT band")
        ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)

        names = ["real\n(raw)", "real\n(denoised)", "synth\nRRW"]
        vals = [hf(sA), hf(sB), hf(sC)]
        bars = ax[1].bar(names, vals, color=["#4C72B0", "#55A868", "#DD8452"])
        for bb, v in zip(bars, vals):
            ax[1].text(bb.get_x() + bb.get_width() / 2, v + 0.002, f"{v:.3f}",
                       ha="center", fontsize=9)
        ax[1].set_ylabel("HF share (LH+HL+HH)")
        ax[1].set_title("How much of the reflection lives in high frequencies")
        ax[1].grid(axis="y", alpha=.3)
        plt.tight_layout()
        p = os.path.join(FIG, "dwt_r_spectrum_v2.png")
        plt.savefig(p, dpi=140)
        print(f"图 -> {p}")
    except Exception as e:
        print(f"[跳过出图] {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
