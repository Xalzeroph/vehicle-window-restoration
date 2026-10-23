#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""洞掩膜松紧档位对比（2026-10-20）。

用户反馈 v9 "太严格、只是子集" 且 "混进了洞边缘的点"。
这里给三档松紧，其余条件不变（在反射区内 / 有边缘 / 边缘不属于 T）：

  紧  : |R| <= 0.60*blur,  ∇I>=p75  → v9 原设
  中  : |R| <= 0.75*blur,  ∇I>=p65
  松  : |R| <= 0.88*blur,  ∇I>=p55

并对每档给出"剥边后"（腐蚀 2px）的面积，用于排除洞边缘点。
输出：_review_1019/hole_levels/<scene>.jpg（三档并排）
      eval/hole_levels.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_levels")
EVAL = os.path.join(P, "eval")
W = 640
SCENES = ["VID_20261019_165126", "VID_20261019_165530", "VID_20261019_164257"]
LEVELS = [("tight v9", 0.60, 75), ("mid", 0.75, 65), ("loose", 0.88, 55)]
R_PRESENT_PCT, E_T_PCT, MIN_AREA, ERODE_PX = 60, 60, 30, 2


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


def clean(m, min_area=MIN_AREA):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) >= min_area:
            out |= (lab == i)
    return out


def title(img, t, h=20):
    w = img.shape[1]
    bar = np.zeros((h, w, 3), np.uint8)
    cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    allres = {}
    for sc in SCENES:
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
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
        R_bg = cv2.GaussianBlur(R_a, (0, 0), 15)
        base_in = R_a >= np.percentile(R_a, R_PRESENT_PCT)
        base_notT = gT <= np.percentile(gT, E_T_PCT)

        rec = {"scene": sc, "levels": {}}
        cells = []
        for name, dip, gpct in LEVELS:
            m = clean(base_in & (R_a <= dip * R_bg)
                      & (gI >= np.percentile(gI, gpct)) & base_notT)
            frac = float(m.mean()) * 100
            if ERODE_PX > 0 and m.any():
                k = np.ones((ERODE_PX * 2 + 1, ERODE_PX * 2 + 1), np.uint8)
                me = cv2.erode(m.astype(np.uint8), k).astype(bool)
                if me.sum() > 0:
                    m = me
            frac_e = float(m.mean()) * 100
            rec["levels"][name] = {"dip": dip, "edge_pct": gpct,
                                   "hole_pct": round(frac, 2),
                                   "hole_pct_eroded": round(frac_e, 2)}
            over = I.astype(np.uint8).copy()
            over[m] = (0.25 * over[m] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
            cells.append(title(over, f"{name}: |R|<={dip}*bg, edge>p{gpct} "
                                     f"-> {frac:.1f}% (eroded {frac_e:.1f}%)"))
        allres[sc] = rec
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}   strictness levels", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{sc}: " + "  ".join(
            f"{k}={v['hole_pct']}%(->{v['hole_pct_eroded']}%)"
            for k, v in rec["levels"].items()))

    json.dump(allres, open(os.path.join(EVAL, "hole_levels.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
