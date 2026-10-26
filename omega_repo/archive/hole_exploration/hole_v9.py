#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 v9 —— 反射内容内部的暗斑（2026-10-20 定稿口径）。

从 v7 图看清楚了
---------------
"不被 T 解释的边缘"（2.0–6.0%）**落在反射的树冠上** —— 那就是反射内容。
所以洞不是"整片没有反射的区域"，而是：

    **反射内容里的一块暗斑**（如反射树冠的背光部分）：
      · 局部反射弱（暗物体反射光少）
      · 有清晰的枝条/轮廓边缘
      · 该边缘在 T 里**没有对应结构**（不是真实物体）

判据（四条件）
------------
1. **在反射区内部**：`|R| >= p60`（排除"本来就没反射"的大片背景）
2. **局部偏暗**：`|R| <= 0.60 * blur(|R|, σ=15)`（相对周围反射偏暗）
3. **有边缘**：`∇I >= p75`
4. **边缘不是 T 的**：`∇T <= p60`

输出：_review_1019/hole_v9/<scene>.jpg   四联
      eval/hole_v9.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v9")
EVAL = os.path.join(P, "eval")
W = 700
DIP_RATIO = 0.60
DIP_SIGMA = 15
E_I_PCT = 75
E_T_PCT = 60
R_PRESENT_PCT = 60
MIN_AREA = 30


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


def grad(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def turbo(x, p=(1, 99.5)):
    lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
    return cv2.applyColorMap((np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)


def title(img, t, h=20):
    w = img.shape[1]
    bar = np.zeros((h, w, 3), np.uint8)
    cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    res = {}
    print(f"{'场景':<30}{'洞区%':>8}{'洞数':>6}{'最大洞':>8}{'洞内|R|':>9}"
          f"{'周围|R|':>9}{'对比度':>8}")
    print("-" * 76)

    for sc in scenes:
        d = os.path.join(PAIR, sc)
        I0 = load(os.path.join(d, "I_000000.png"))
        G0 = load(os.path.join(d, "GT.png"))
        if I0 is None or G0 is None:
            continue
        h, w = I0.shape[:2]
        sh = max(2, int(round(h * W / w)))
        I = cv2.resize(I0, (W, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G0, (W, sh), interpolation=cv2.INTER_AREA)
        A, B = robust_affine(I, G)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)

        R_a = np.abs((I_al - G).mean(axis=2)).astype(np.float32)
        T = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gI, gT = grad(Ig), grad(T)

        in_refl = R_a >= np.percentile(R_a, R_PRESENT_PCT)
        R_bg = cv2.GaussianBlur(R_a, (0, 0), DIP_SIGMA)
        dip = R_a <= DIP_RATIO * R_bg
        has_edge = gI >= np.percentile(gI, E_I_PCT)
        not_T = gT <= np.percentile(gT, E_T_PCT)

        hole = in_refl & dip & has_edge & not_T
        m = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        out = np.zeros_like(m); areas = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a >= MIN_AREA:
                out |= (lab == i).astype(np.uint8); areas.append(a)
        areas.sort(reverse=True)
        holeF = out.astype(bool)

        ring = cv2.dilate(holeF.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~holeF
        r_in = float(np.median(R_a[holeF])) if holeF.any() else float("nan")
        r_out = float(np.median(R_a[ring])) if ring.any() else float("nan")
        contrast = (r_in / r_out) if (r_out == r_out and r_out > 1e-6) else float("nan")

        res[sc] = {"hole_pct": round(float(holeF.mean()) * 100, 2),
                   "n_holes": len(areas),
                   "largest": areas[0] if areas else 0,
                   "R_inside": None if r_in != r_in else round(r_in, 2),
                   "R_outside": None if r_out != r_out else round(r_out, 2),
                   "contrast": None if contrast != contrast else round(contrast, 3)}
        print(f"{sc:<30}{res[sc]['hole_pct']:>7.2f}%{len(areas):>6}"
              f"{res[sc]['largest']:>8}"
              f"{(f'{r_in:.1f}' if r_in == r_in else '-'):>9}"
              f"{(f'{r_out:.1f}' if r_out == r_out else '-'):>9}"
              f"{(f'{contrast:.2f}' if contrast == contrast else '-'):>8}")

        over = I.astype(np.uint8).copy()
        over[holeF] = (0.25 * over[holeF] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
        cells = [
            title(over, f"HOLE = dark patch INSIDE reflection: {res[sc]['hole_pct']:.1f}%"),
            title(turbo(R_a), f"|R|  (in={r_in:.1f} vs out={r_out:.1f})"
                  if r_in == r_in and r_out == r_out else "|R|"),
            title(turbo(R_bg - R_a), "local dip: R_bg - |R|  (bright=darker than surround)"),
            title(turbo(gI), "grad I (edges exist)"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v9.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 76)
    if res:
        hp = [v["hole_pct"] for v in res.values()]
        print(f"洞区 中位 {np.median(hp):.2f}%  范围 [{min(hp):.2f}%, {max(hp):.2f}%]")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
