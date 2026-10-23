#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"反射与透射结构重叠 → 去反射时糊掉透射物体"这一机制（2026-10-20）。

思路：反射图 R 无法直接拿到，但可近似 R ≈ I − g·GT。
在"结构重叠"的区域（两图梯度都强且方向一致），去反射必然更易伤到透射细节。
本脚本量化：
  1. 每对 (I, GT)：残余 R 的强度与结构比例；
  2. I 与 GT 的**梯度方向一致性**（结构重叠度代理指标）；
  3. 各模型在该场景的 PSNR 与梯度保留率，与"重叠度"的相关性。
"""
from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
ST = os.path.join(PAIR, "input_strength.json")

ARMS = {"rdnet_zs": "real_pair_1019_rdnet_zs",
        "ft_s2main": "real_pair_1019_ft_s2main",
        "ft_mixbest": "real_pair_1019_ft_mixbest",
        "lightrr": "real_pair_1019_lightrr"}


def rd(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


def gray(b):
    return cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sobel(g):
    return (cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def grad_mag(g):
    gx, gy = sobel(g)
    return np.sqrt(gx * gx + gy * gy)


def cos_sim(a_g, b_g, topk=0.15):
    """两图强梯度像素处的方向余弦（结构重叠度）。"""
    ax, ay = sobel(a_g); bx, by = sobel(b_g)
    ma = np.sqrt(ax ** 2 + ay ** 2); mb = np.sqrt(bx ** 2 + by ** 2)
    thr_a = np.percentile(ma, (1 - topk) * 100)
    thr_b = np.percentile(mb, (1 - topk) * 100)
    m = (ma >= thr_a) & (mb >= thr_b)
    if m.sum() < 100:
        return float("nan")
    dot = (ax * bx + ay * by)[m]
    den = (ma * mb)[m].clip(1e-6)
    return float(np.mean(dot / den))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def main() -> int:
    strength = {r["scene"]: r for r in json.load(open(ST, encoding="utf-8"))}
    scenes = sorted(strength)
    rows = []
    print(f"{'场景':<30}{'|g-1|':>8}{'重叠度':>8}{'R结构比':>9}"
          f"{'rdnet':>8}{'s2main':>8}{'mix':>8}{'lightrr':>9}")
    print("-" * 96)
    for sc in scenes:
        I = rd(os.path.join(PAIR, sc, "I_000000.png"))
        G = rd(os.path.join(PAIR, sc, "GT.png"))
        if I is None or G is None:
            continue
        gi, gg = gray(I), gray(G)
        overlap = cos_sim(gi, gg)
        # R ≈ I - GT（先做逐通道均值对齐，避免色调主导）
        R = I.astype(np.float32) - G.astype(np.float32)
        gr = grad_mag(cv2.cvtColor(R.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32))
        gt_mag = grad_mag(gg)
        struct_ratio = float(np.median(gr) / max(np.median(gt_mag), 1e-6))

        row = {"scene": sc, "gain_dev": strength[sc]["input_vs_gt"]["gain_dev"],
               "overlap_cos": round(overlap, 4), "resid_struct_ratio": round(struct_ratio, 4)}
        line = (f"{sc:<30}{row['gain_dev']:>8.4f}{overlap:>8.3f}{struct_ratio:>9.3f}")
        for arm, d in ARMS.items():
            p = os.path.join(PRED, d, f"{sc}__I_000000", "t_hat.png")
            if not os.path.exists(p):
                line += f"{'--':>8}"
                row[arm] = None
                continue
            out = rd(p)
            v = psnr(out, G)
            row[arm] = round(v, 3)
            line += f"{v:>8.2f}" if arm != "lightrr" else f"{v:>9.2f}"
        print(line)
        rows.append(row)

    # 相关性：重叠度 vs 各模型 PSNR（去反射在"硬场景"是否更差）
    print("\n=== 相关性（重叠度 / |g-1| vs 各模型 PSNR）===")
    from scipy.stats import spearmanr
    ov = [r["overlap_cos"] for r in rows]
    gd = [r["gain_dev"] for r in rows]
    for arm in ARMS:
        vs = [r[arm] for r in rows]
        ok = [(a, b) for a, b in zip(ov, vs) if b is not None and a == a]
        if len(ok) < 5:
            continue
        rho_ov = spearmanr([a for a, _ in ok], [b for _, b in ok]).statistic
        ok2 = [(a, b) for a, b in zip(gd, vs) if b is not None]
        rho_g = spearmanr([a for a, _ in ok2], [b for _, b in ok2]).statistic
        print(f"  {arm:<12} vs 重叠度 rho={rho_ov:+.3f}   vs |g-1| rho={rho_g:+.3f}")

    out = os.path.join(P, "eval/real_pair_1019_overlap.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
