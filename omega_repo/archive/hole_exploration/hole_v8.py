#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 v8 最终版 —— `r` 的陡降边界 + 其弱反射侧（2026-10-20）。

从 v7 的实测得到的完整图景
-----------------------
"不被 T 解释的边缘"处 `r = 0.31~0.47`（强），周围环带 `r = 0.04~0.20`（弱）。
→ 该边缘是**反射比 r 的一条陡降边界**（从强反射陡降到弱反射）。
→ 你说的"洞"就是这条陡降边界**旁边那块弱反射区**。

定义（三步，全部可测）
------------------
1. **陡降边界**：`∇I` 强 ∧ `∇T` 弱（边缘不能由透射解释 ⇒ 必来自 T·∇r）
2. **弱反射区**：`r` 低于全图 45 分位
3. **洞 = 与陡降边界相连的弱反射连通域**
   （只保留接触边界的那些弱反射块 —— 排除大片"本来就没反射"的背景）

输出
----
_review_1019/hole_v8/<scene>.jpg   I(洞标红) | r | 陡降边界 | |R|
eval/hole_v8.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v8")
EVAL = os.path.join(P, "eval")
W = 700
LAMBDA = 12.0
EDGE_I_PCT = 85
EDGE_T_PCT = 55
R_WEAK_PCT = 45
MIN_AREA = 40
MAX_DIST = 30      # 洞只取陡降边界 30px 以内的弱反射（限制"洞"的空间范围）


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
    print(f"{'场景':<30}{'陡降边%':>9}{'洞区%':>8}{'洞数':>6}{'最大洞':>8}"
          f"{'洞内r':>8}{'洞外邻r':>8}")
    print("-" * 78)

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

        R_s = (I_al - G).mean(axis=2).astype(np.float32)
        T = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = (R_s * T / (T * T + LAMBDA * LAMBDA)).astype(np.float32)

        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gI, gT = grad(Ig), grad(T)
        steep = (gI >= np.percentile(gI, EDGE_I_PCT)) & \
                (gT <= np.percentile(gT, EDGE_T_PCT))

        weak = r <= np.percentile(r, R_WEAK_PCT)
        # ★ 关键约束：只取**陡降边界附近**的弱反射（否则整片背景 45% 都会被算进来）
        # 用距离变换：到最近陡降边的像素距离 <= MAX_DIST
        dist = cv2.distanceTransform((~steep).astype(np.uint8), cv2.DIST_L2, 5)
        near_edge = dist <= MAX_DIST
        weak = weak & near_edge
        # 弱反射连通域
        n, lab, st, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
        steep_d = cv2.dilate(steep.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        hole = np.zeros_like(weak)
        areas = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            comp = (lab == i)
            if a < MIN_AREA:
                continue
            if (comp & steep_d).sum() < 5:      # 必须接触陡降边界
                continue
            hole |= comp
            areas.append(a)
        areas.sort(reverse=True)

        ring = cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~hole
        r_in = float(np.median(r[hole])) if hole.any() else float("nan")
        r_out = float(np.median(r[ring])) if ring.any() else float("nan")

        res[sc] = {"steep_edge_pct": round(float(steep.mean()) * 100, 2),
                   "hole_pct": round(float(hole.mean()) * 100, 2),
                   "n_holes": len(areas),
                   "largest": areas[0] if areas else 0,
                   "r_inside": None if r_in != r_in else round(r_in, 4),
                   "r_outside": None if r_out != r_out else round(r_out, 4)}
        print(f"{sc:<30}{res[sc]['steep_edge_pct']:>8.2f}%{res[sc]['hole_pct']:>7.2f}%"
              f"{len(areas):>6}{res[sc]['largest']:>8}"
              f"{(f'{r_in:.3f}' if r_in == r_in else '-'):>8}"
              f"{(f'{r_out:.3f}' if r_out == r_out else '-'):>8}")

        over = I.astype(np.uint8).copy()
        over[hole] = (0.30 * over[hole] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        cont, _ = cv2.findContours(hole.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cont, -1, (0, 255, 0), 1)
        sd = cv2.dilate(steep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        iso = I.astype(np.uint8).copy()
        iso[sd] = (0, 255, 255)
        cells = [
            title(over, f"HOLE (weak-reflection side of a steep r drop): "
                        f"{res[sc]['hole_pct']:.1f}%"),
            title(iso, f"steep drop edges (in I, not in T): "
                       f"{res[sc]['steep_edge_pct']:.1f}%"),
            title(turbo(r), f"r = R/T   inside={res[sc]['r_inside']}"),
            title(turbo(np.abs(R_s)), f"|R|  p50={np.median(np.abs(R_s)):.1f}"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v8.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 78)
    if res:
        hp = [v["hole_pct"] for v in res.values()]
        print(f"洞区 中位 {np.median(hp):.2f}%  范围 [{min(hp):.2f}%, {max(hp):.2f}%]")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
