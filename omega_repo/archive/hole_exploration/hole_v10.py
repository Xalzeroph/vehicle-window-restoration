#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 v10 —— 区域级判据 + 剥掉轮廓带（2026-10-20）。

v9 的两个偏差（用户指出）
-----------------------
1. **太严格**：v9 要求**每个像素** `∇I >= p75`，而边缘只占区域的一圈 →
   只有靠近轮廓的像素能通过，选出来是"真洞的子集"。
2. **混进边界点**：正因为钉在边缘上，选出的很多是**洞的轮廓**而非洞的内部。

v10 改为**区域级**判据
-------------------
1. 候选体：`|R|` 比大尺度背景暗（`|R| <= α·blur(|R|)`），且在反射区内；
2. 对每个连通域**整体**检验它的轮廓带：
   · 轮廓带上 `∇I` 中位 >= 全图 p60  → **这个块有可见边界**
   · 轮廓带上 `∇T` 中位 <= 全图 p60  → **这个边界不是透射层的结构**
3. 通过检验的块，**腐蚀若干像素剥掉轮廓带**，只保留内部。

这样：判据放宽（整块通过即可，不再逐像素）→ 覆盖更全；
      再剥掉边界带 → 只留洞的内部。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v10")
EVAL = os.path.join(P, "eval")
W = 700
DIP = 0.70          # 候选体：比背景暗 30% 即可（v9 是 0.60，偏严）
DIP_SIGMA = 15
IN_REFL_PCT = 40    # 在反射区内（v9 用 p60，放宽到 p40）
BAND_I_PCT = 60     # 轮廓带 ∇I 中位 >= p60
BAND_T_PCT = 60     # 轮廓带 ∇T 中位 <= p60
MIN_AREA = 60
ERODE_PX = 3        # 剥掉轮廓带的宽度


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
    print(f"{'场景':<30}{'候选%':>8}{'通过%':>8}{'剥边后%':>9}{'洞数':>6}{'最大洞':>8}"
          f"{'带宽∇I分位':>11}{'带宽∇T分位':>11}")
    print("-" * 92)

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
        R_bg = cv2.GaussianBlur(R_a, (0, 0), DIP_SIGMA)

        cand = ((R_a >= np.percentile(R_a, IN_REFL_PCT))
                & (R_a <= DIP * R_bg))
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)

        pI60, pT60 = np.percentile(gI, BAND_I_PCT), np.percentile(gT, BAND_T_PCT)
        kept = np.zeros_like(cand)
        areas = []
        band_I_q, band_T_q = [], []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a < MIN_AREA:
                continue
            comp = (lab == i).astype(np.uint8)
            band = (cv2.dilate(comp, np.ones((7, 7), np.uint8)).astype(bool)
                    & ~cv2.erode(comp, np.ones((7, 7), np.uint8)).astype(bool))
            if not band.any():
                continue
            mi = float(np.median(gI[band])); mt = float(np.median(gT[band]))
            # 区域级检验：整块轮廓既有可见边缘，又不属于透射层
            if mi >= pI60 and mt <= pT60:
                kept |= comp
                areas.append(a)
                band_I_q.append(float((gI < mi).mean() * 100))
                band_T_q.append(float((gT < mt).mean() * 100))
        areas.sort(reverse=True)

        # 剥掉轮廓带，只留内部
        if ERODE_PX > 0:
            k = np.ones((ERODE_PX * 2 + 1, ERODE_PX * 2 + 1), np.uint8)
            inner = cv2.erode(kept, k)
            # 若侵蚀后什么都不剩，退回原块（小洞无法剥边）
            if inner.sum() == 0:
                inner = kept
        else:
            inner = kept
        innerF = inner.astype(bool)

        res[sc] = {"cand_pct": round(float(cand.mean()) * 100, 2),
                   "pass_pct": round(float(kept.mean()) * 100, 2),
                   "hole_pct": round(float(innerF.mean()) * 100, 2),
                   "n_holes": len(areas),
                   "largest": areas[0] if areas else 0,
                   "band_gI_pct": (round(float(np.median(band_I_q)), 1)
                                   if band_I_q else None),
                   "band_gT_pct": (round(float(np.median(band_T_q)), 1)
                                   if band_T_q else None)}
        print(f"{sc:<30}{res[sc]['cand_pct']:>7.2f}%{res[sc]['pass_pct']:>7.2f}%"
              f"{res[sc]['hole_pct']:>8.2f}%{len(areas):>6}{res[sc]['largest']:>8}"
              f"{(f'{np.median(band_I_q):.0f}' if band_I_q else '-'):>11}"
              f"{(f'{np.median(band_T_q):.0f}' if band_T_q else '-'):>11}")

        over = I.astype(np.uint8).copy()
        over[innerF] = (0.25 * over[innerF] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
        cont, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cont, -1, (0, 255, 0), 1)
        cells = [
            title(over, f"HOLE interior (red) + its contour (green)] : "
                        f"{res[sc]['hole_pct']:.1f}%"),
            title(turbo(R_a), f"|R| p50={np.median(R_a):.1f}"),
            title(turbo(R_bg - R_a), "local dip R_bg - |R|"),
            title(turbo(gI), "grad I"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v10.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 92)
    if res:
        hp = [v["hole_pct"] for v in res.values()]
        pp = [v["pass_pct"] for v in res.values()]
        print(f"通过率（区域级） 中位 {np.median(pp):.2f}%   "
              f"剥边后洞区 中位 {np.median(hp):.2f}%  "
              f"范围 [{min(hp):.2f}%, {max(hp):.2f}%]")
        print(f"对比 v9（逐像素判据）洞区中位 1.17% -> 现在覆盖面变化见上表")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
