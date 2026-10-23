#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归因诊断：为什么 identity 反而最高、所有模型都低（2026-10-20）。

要回答三个问题
1. INPUT 与 GT 到底差多少？（逐通道增益 g、亮度差）
   若 |g-1| 很小 → 说明 I 与 GT 已高度接近，identity 自然赢。
2. 各模型相对 INPUT 改了什么？（亮度 luma 比、梯度能量比）
   若模型普遍"变暗 + 变糊" → 与交接页 §3 结论一致（模型做的是色调+平滑）。
3. 这些差异是否量级上足以解释 PSNR 落差？

只读预测图，零 GPU。
"""
from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(PROJ, "data/test_sets/real_pair_1019")
PRED = os.path.join(PROJ, "runs/pred")
EVAL = os.path.join(PROJ, "eval")

ARMS = [("identity", None), ("rdnet_zs", "real_pair_1019_rdnet_zs"),
        ("ft_s2main", "real_pair_1019_ft_s2main"),
        ("ft_mixbest", "real_pair_1019_ft_mixbest"),
        ("lightrr", "real_pair_1019_lightrr")]


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


def luma(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def grad(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 +
                         cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2).mean())


def robust_gain(a, b, keep=0.7, iters=3):
    """g 使 a ≈ g*b（逐通道稳健 LS）。"""
    A = a.astype(np.float64); B = b.astype(np.float64)
    g = np.ones(3)
    for c in range(3):
        x = A[..., c].ravel(); y = B[..., c].ravel()
        for _ in range(iters):
            den = float((y * y).sum())
            gc = float((x * y).sum() / den) if den > 1e-9 else 1.0
            r = np.abs(x - gc * y)
            k = r <= np.percentile(r, keep * 100)
            x, y = x[k], y[k]
            if len(x) < 100:
                break
        g[c] = gc
    return g


def main() -> int:
    # 逐对清单
    pairs = []
    for sc in sorted(os.listdir(PAIR)):
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                pairs.append((sc, os.path.splitext(f)[0]))
    print(f"配对 {len(pairs)} 组\n")

    # ---- 1) INPUT vs GT ----
    devs, lum_ratio = [], []
    for sc, stem in pairs:
        I = read(os.path.join(PAIR, sc, f"{stem}.png"))
        G = read(os.path.join(PAIR, sc, "GT.png"))
        if I is None or G is None:
            continue
        g = robust_gain(I, G)
        devs.append(np.abs(g - 1).mean())
        lum_ratio.append(luma(I) / max(luma(G), 1e-6))
    print("=== 1) INPUT 与 GT 的关系 ===")
    print(f"  逐通道 |g-1| 中位 = {np.median(devs):.4f}  "
          f"P90 = {np.percentile(devs,90):.4f}")
    print(f"  luma(INPUT)/luma(GT) 中位 = {np.median(lum_ratio):.3f}")

    # ---- 2) 各臂相对 INPUT 的改动 ----
    print("\n=== 2) 各臂相对 INPUT 的改动 ===")
    print(f"{'arm':<12}{'luma比':>9}{'梯度比':>9}{'|g-1|中位':>11}")
    print("-" * 44)
    out = {}
    for arm, preddir in ARMS:
        lr, gr, gd = [], [], []
        for sc, stem in pairs:
            I = read(os.path.join(PAIR, sc, f"{stem}.png"))
            if I is None:
                continue
            if arm == "identity":
                P = I
            else:
                p = os.path.join(PRED, preddir, f"{sc}__{stem}", "t_hat.png")
                if not os.path.exists(p):
                    continue
                P = read(p)
                if P is None:
                    continue
            lr.append(luma(P) / max(luma(I), 1e-6))
            gr.append(grad(P) / max(grad(I), 1e-9))
            gd.append(np.abs(robust_gain(P, I) - 1).mean())
        if not lr:
            print(f"{arm:<12}{'(无预测)':>9}")
            continue
        out[arm] = {"luma_ratio": round(float(np.median(lr)), 4),
                    "grad_ratio": round(float(np.median(gr)), 4),
                    "gain_dev": round(float(np.median(gd)), 4),
                    "n": len(lr)}
        print(f"{arm:<12}{np.median(lr):>9.3f}{np.median(gr):>9.3f}"
              f"{np.median(gd):>11.4f}")

    # ---- 3) 逐场景 PSNR（看是否普遍一致）----
    print("\n=== 3) 各臂逐场景 PSNR 中位/极差 ===")
    print(f"{'arm':<12}{'中位':>8}{'最小':>8}{'最大':>8}{'胜 identity 的场景数':>20}")
    print("-" * 58)
    base = {}
    for arm, _ in ARMS:
        p = os.path.join(EVAL, f"real_pair_1019_{arm}.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        by = {}
        for r in rows:
            by.setdefault(r["scene"], []).append(float(r["psnr"]))
        med = {k: float(np.mean(v)) for k, v in by.items()}
        if arm == "identity":
            base = med
        wins = sum(1 for k in med if k in base and med[k] > base[k]) if base else 0
        print(f"{arm:<12}{np.median(list(med.values())):>8.2f}"
              f"{min(med.values()):>8.2f}{max(med.values()):>8.2f}{wins:>14}/{len(med):<5}")

    sp = os.path.join(EVAL, "real_pair_1019_diagnosis.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump({"input_vs_gt": {"gain_dev_median": round(float(np.median(devs)), 4),
                                   "luma_ratio_median": round(float(np.median(lum_ratio)), 4)},
                   "arms": out}, fh, ensure_ascii=False, indent=1)
    print(f"\n写入 {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
