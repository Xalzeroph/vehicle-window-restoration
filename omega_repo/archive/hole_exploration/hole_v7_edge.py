#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7 的矛盾诊断 + "不被 T 解释的边缘"可视化（2026-10-20）。

发现的矛盾
----------
在 `I = a·T·(1+r)`、`∇T≈0` 处有 `∇I ≈ a·T·∇r`。
所以"**∇I 强但 ∇T 弱**"等价于"**∇r 强**" —— 即**反射在跨这条边发生了明显变化**。
但"反射弱"要求 `|R| = |r·T|` 小。两者在数学上互相拉扯：

    反射要产生看得见的边缘 → 反射不能太弱
    反射要"确实弱"       → 就不能产生强边缘

本脚本做的事
-----------
1. 把"不被 T 解释的边缘"单独画出来（这就是"有边缘但不能由透射解释"的部分）；
2. 量一下：这些边缘**两侧的反射比 r 差多少**（是否真的存在"一边弱"）；
3. 若 r 在边缘两侧差异显著 → 说明这条边是**反射层的边界**，
   洞就是该边界围起来的、r 更低的那一侧。

输出：_review_1019/hole_v7_edge/<scene>.jpg
      eval/hole_v7_edge.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v7_edge")
EVAL = os.path.join(P, "eval")
W = 700
LAMBDA = 12.0
EDGE_I_PCT = 85
EDGE_T_PCT = 55
SCENES = ["VID_20261019_165530", "VID_20261019_170406", "VID_20261019_165126",
          "VID_20261019_164901"]


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
    res = {}
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

        R_s = (I_al - G).mean(axis=2).astype(np.float32)
        T = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        r_w = (R_s * T / (T * T + LAMBDA * LAMBDA)).astype(np.float32)

        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gI, gT = grad(Ig), grad(T)
        unexplained = (gI >= np.percentile(gI, EDGE_I_PCT)) & \
                      (gT <= np.percentile(gT, EDGE_T_PCT))
        # 取边缘两侧（膨胀环与内部）比较 r
        ker = np.ones((13, 13), np.uint8)
        ring = cv2.dilate(unexplained.astype(np.uint8), ker).astype(bool) & ~unexplained
        r_edge = float(np.median(r_w[unexplained])) if unexplained.any() else float("nan")
        r_ring = float(np.median(r_w[ring])) if ring.any() else float("nan")
        # 边缘处 |R| 相对全图的分位
        R_abs = np.abs(R_s)
        R_pct_at_edge = (float((R_abs < np.median(R_abs[unexplained])).mean() * 100)
                         if unexplained.any() else float("nan"))

        res[sc] = {"unexplained_edge_pct": round(float(unexplained.mean()) * 100, 2),
                   "r_at_edge": None if r_edge != r_edge else round(r_edge, 4),
                   "r_in_ring": None if r_ring != r_ring else round(r_ring, 4),
                   "r_contrast": (None if (r_edge != r_edge or r_ring != r_ring)
                                  else round(abs(r_edge - r_ring), 4)),
                   "Rabs_pct_at_edge": (None if R_pct_at_edge != R_pct_at_edge
                                        else round(R_pct_at_edge, 1))}
        print(f"{sc}: 不被T解释的边缘 {res[sc]['unexplained_edge_pct']:.2f}%  "
              f"r(边)={res[sc]['r_at_edge']}  r(环)={res[sc]['r_in_ring']}  "
              f"|R|在边处的分位={res[sc]['Rabs_pct_at_edge']}%")

        over = I.astype(np.uint8).copy()
        m3 = cv2.dilate(unexplained.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        over[m3] = (0.25 * over[m3] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
        cells = [
            title(over, f"edge in I but NOT in T: {res[sc]['unexplained_edge_pct']:.1f}% (red)"),
            title(turbo(gI), "grad I"),
            title(turbo(gT), "grad T  (where this is LOW, the edge is not from T)"),
            title(turbo(np.abs(r_w)), f"|r| = |R/T|  at edge={res[sc]['r_at_edge']}"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v7_edge.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
