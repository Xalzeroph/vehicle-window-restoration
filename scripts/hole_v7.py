#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 v7 —— 按"**边缘在 I 有、在 T 没有、且反射弱**"（2026-10-20 最终口径）。

用户定义（原话）
-------------
"首先不能在 T 中出现这个结构；其次在 R 中有没有出现我不知道；
 我只知道它反射确实弱，但是又边缘。"

判据
----
    ∇I 强   （合成图里看得见边缘）
  ∧ ∇T 弱   （该边缘**不能由透射层解释** —— 排除建筑物立柱、车架等真实物体轮廓）
  ∧ |R| 弱  （反射确实弱）

为什么这组判据在物理上自洽
------------------------
`I = a·T·(1+r)` ⇒ `∇I ≈ a[(1+r)∇T + T∇r]`。
若 `∇T ≈ 0` 而 `∇I` 很强，则边缘必来自 `T·∇r` 项 —— 即**反射层内部的一条边界**。
这正是"反射很弱、但有清晰边缘"的情形：反射层上一块弱反射区的轮廓。

输出
----
_review_1019/hole_v7/<scene>.jpg   四联：I(洞标红) | ∇I | ∇T | |R|
_review_1019/hole_v7/summary.jpg   代表场景合图
eval/hole_v7.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v7")
EVAL = os.path.join(P, "eval")
W = 700
R_WEAK_PCT = 40     # |R| 低于该分位 → 反射弱
EDGE_I_PCT = 85     # ∇I 高于该分位 → 看得见边缘
EDGE_T_PCT = 55     # ∇T 低于该分位 → 该边缘不能由透射解释
MIN_AREA = 20


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
    os.makedirs(EVAL, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    res = {}
    print(f"{'场景':<30}{'边缘不被T解释%':>15}{'洞区%':>8}{'连通域':>7}{'最大块':>8}"
          f"{'洞区∇I/∇T':>11}")
    print("-" * 82)

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

        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        Tg = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gI, gT = grad(Ig), grad(Tg)

        eI = gI >= np.percentile(gI, EDGE_I_PCT)       # I 里有边缘
        eT = gT <= np.percentile(gT, EDGE_T_PCT)       # T 里没有这个结构
        unexplained = eI & eT
        weakR = R_a <= np.percentile(R_a, R_WEAK_PCT)  # 反射弱

        hole = unexplained & weakR
        m = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        out = np.zeros_like(m); areas = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a >= MIN_AREA:
                out |= (lab == i).astype(np.uint8); areas.append(a)
        areas.sort(reverse=True)
        holeF = out.astype(bool)

        ratio_grad = (float(np.median(gI[holeF])) /
                      max(float(np.median(gT[holeF])), 1e-6)) if holeF.any() else float("nan")

        res[sc] = {"unexplained_edge_pct": round(float(unexplained.mean()) * 100, 2),
                   "hole_pct": round(float(holeF.mean()) * 100, 2),
                   "n_comp": len(areas),
                   "largest": areas[0] if areas else 0,
                   "gradI_over_gradT": (None if ratio_grad != ratio_grad
                                        else round(ratio_grad, 2))}
        print(f"{sc:<30}{res[sc]['unexplained_edge_pct']:>14.2f}%"
              f"{res[sc]['hole_pct']:>7.2f}%{len(areas):>7}{res[sc]['largest']:>8}"
              f"{(f'{ratio_grad:.1f}x' if ratio_grad == ratio_grad else '-'):>11}")

        # 四联图
        over = I.astype(np.uint8).copy()
        over[holeF] = (0.30 * over[holeF] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        cells = [
            title(over, f"HOLE: edge in I, NOT in T, weak R : "
                        f"{res[sc]['hole_pct']:.1f}%"),
            title(turbo(gI), "grad I (edges visible in composite)"),
            title(turbo(gT), "grad T (edges explained by transmission)"),
            title(turbo(R_a), f"|R|  p50={np.median(R_a):.1f}"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v7.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 82)
    if res:
        hp = [v["hole_pct"] for v in res.values()]
        up = [v["unexplained_edge_pct"] for v in res.values()]
        print(f"洞区 中位 {np.median(hp):.2f}%  范围 [{min(hp):.2f}%, {max(hp):.2f}%]")
        print(f"'边缘不被 T 解释' 的像素 中位 {np.median(up):.2f}%")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
