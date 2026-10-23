#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭环验证：**暗区的反射是不是真的弱？**（2026-10-20）

你提出的洞机制是："物体正面受光在车窗形成弱反射，同时物体遮蔽环境光 →
该区域偏暗 + 反射贡献弱"。

这给出一个**可直接证伪**的推论：
    若洞假设成立，则 **T 越暗的区域，r 应当越小**（正相关：亮度↑ → r↑）。

本脚本按 T 的亮度分位分箱，统计每箱的 r 中位数，并给出
   - 亮度-反射的 Spearman 相关
   - 暗区（T 最低 20%）与亮区（T 最高 20%）的 r 之比
同时输出"暗区"的可视化，供你目检这些暗区里有没有你说的那种"漫反射黑区"。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from scipy.stats import spearmanr

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/hole_test"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
MAXW = 640
SCENES = ["VID_20261019_165126", "VID_20261019_165530",
          "VID_20261019_170406", "VID_20261019_164257", "VID_20261019_165849"]
NBIN = 5


def load(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def robust_affine(src, ref, keep=0.7, iters=3):
    A = np.ones(3); B = np.zeros(3)
    for c in range(3):
        x = src[..., c].ravel().astype(np.float64)
        y = ref[..., c].ravel().astype(np.float64)
        for _ in range(iters):
            V = np.vstack([y, np.ones_like(y)]).T
            coef, *_ = np.linalg.lstsq(V, x, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            rr = np.abs(x - (a * y + b))
            k = rr <= np.percentile(rr, keep * 100)
            x, y = x[k], y[k]
            if len(x) < 500:
                break
        A[c], B[c] = a, b
    return A, B


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    results = {}
    print(f"{'场景':<30}{'rho(T,r)':>10}{'暗区r':>9}{'亮区r':>9}{'比值':>8}{'结论':>22}")
    print("-" * 92)
    for sc in SCENES:
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        I = load(os.path.join(d, "I_000000.png"))
        G = load(os.path.join(d, "GT.png"))
        if I is None or G is None:
            continue
        h, w = I.shape[:2]
        sw = MAXW; sh = max(2, int(round(h * sw / w)))
        I = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G, (sw, sh), interpolation=cv2.INTER_AREA)

        A, B = robust_affine(I, G)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
        T = G
        r = np.abs((I_al - T) / np.maximum(T, 8.0)).mean(axis=2)
        Tg = cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 分箱统计
        qs = np.percentile(Tg, np.linspace(0, 100, NBIN + 1))
        bins = []
        for i in range(NBIN):
            m = (Tg >= qs[i]) & (Tg < qs[i + 1] if i < NBIN - 1 else Tg <= qs[i + 1])
            if m.sum() < 500:
                continue
            bins.append({"T_lo": round(float(qs[i]), 1), "T_hi": round(float(qs[i + 1]), 1),
                         "r_med": round(float(np.median(r[m])), 4),
                         "n_px": int(m.sum())})
        rho = spearmanr(Tg.ravel()[::13], r.ravel()[::13]).statistic

        dark = Tg <= np.percentile(Tg, 20)
        bright = Tg >= np.percentile(Tg, 80)
        r_dark = float(np.median(r[dark])); r_bright = float(np.median(r[bright]))
        ratio = r_dark / max(r_bright, 1e-9)

        verdict = ("暗区反射更弱 → 支持洞假设"
                   if ratio < 0.9 else
                   ("暗区反射更强 → 反对洞假设" if ratio > 1.1 else "无明显关系"))
        print(f"{sc:<30}{rho:>10.3f}{r_dark:>9.4f}{r_bright:>9.4f}{ratio:>8.3f}{verdict:>22}")
        results[sc] = {"rho_T_r": round(float(rho), 4), "bins": bins,
                       "r_dark": round(r_dark, 4), "r_bright": round(r_bright, 4),
                       "ratio_dark_over_bright": round(ratio, 4), "verdict": verdict}

        # 可视化：暗区叠加
        base = T.astype(np.uint8)
        over = base.copy()
        over[dark] = (0.35 * over[dark] + 0.65 * np.array([0, 128, 255])).astype(np.uint8)
        ww = base.shape[1]
        bar = np.zeros((20, ww, 3), np.uint8)
        cv2.putText(bar, f"darkest 20% (T)   r_dark={r_dark:.3f} r_bright={r_bright:.3f}",
                    (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_dark.jpg"),
                    np.vstack([bar, over]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(results, open(os.path.join(EVAL, "hole_test_luminance_r.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n输出 {OUT}\n      {EVAL}/hole_test_luminance_r.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
