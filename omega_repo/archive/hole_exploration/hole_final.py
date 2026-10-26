#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 = **残差无反射信号 + 存在边缘**（2026-10-20，用户最终口径）。

定义
----
    洞 = { |R| 小（数据上没有反射） } ∧ { 边缘强（但有结构轮廓） }

其中 `R = I/a − T`（已做逐通道仿射，去掉黑布造成的整体曝光偏置）。
另外并行用 `r = R/T`（维纳正则）复算一次，因为 `R = r·T` 会受 T 暗区影响，
两个口径一起报，避免单一口径误判。

输出（**无网格、无花哨标注**，一眼能看懂）
--------------------------------------
_review_1019/hole_final/<scene>.jpg   三联：I(洞区标红) | |R| | 边缘
_review_1019/hole_final/summary.jpg   挑 4 个代表场景拼一张
eval/hole_final.json                  每场景统计
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/hole_final"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
W = 760
LAMBDA = 12.0
R_LOW_PCT = 30      # |R| 低于该分位 → "没有反射"
EDGE_HI_PCT = 80    # 边缘高于该分位 → "有边缘"
MIN_AREA = 25       # 注意：原设 150 + 5x5 开运算会把结果全磨掉（实测 0.00%）。
                    # 原始交集有 2.2~4.0%，且呈**分散的沿边细条**，不是连片 blob。


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


def turbo(x, p=(1, 99.5)):
    lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
    return cv2.applyColorMap((np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)


def cleanup(m, min_area=MIN_AREA):
    """轻量清理：只去孤立噪点，不做强开运算（强开会把沿边细条全磨掉）。"""
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    areas = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out |= (lab == i).astype(np.uint8)
            areas.append(a)
    areas.sort(reverse=True)
    return out.astype(bool), areas


def title(img, t, h=22):
    w = img.shape[1]
    bar = np.zeros((h, w, 3), np.uint8)
    cv2.putText(bar, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    res = {}
    print(f"{'场景':<30}{'洞%(|R|)':>10}{'洞%(r)':>9}{'连通域':>7}{'最大块':>8}"
          f"{'洞区边缘增益':>13}")
    print("-" * 82)

    for sc in scenes:
        d = os.path.join(PAIR, sc)
        I = load(os.path.join(d, "I_000000.png"))
        G = load(os.path.join(d, "GT.png"))
        if I is None or G is None:
            continue
        h, w = I.shape[:2]
        sh = max(2, int(round(h * W / w)))
        I = cv2.resize(I, (W, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G, (W, sh), interpolation=cv2.INTER_AREA)

        A, B = robust_affine(I, G)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)

        R_s = (I_al - G).mean(axis=2).astype(np.float64)
        R_a = np.abs(R_s)
        T = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        r_w = np.abs(R_s * T / (T * T + LAMBDA * LAMBDA))

        # 边缘：在合成图 I 上算（洞的"边缘"是图像里看得见的结构）
        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(Ig, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(Ig, cv2.CV_32F, 0, 1)
        edge = np.sqrt(gx * gx + gy * gy)

        weak_R = R_a <= np.percentile(R_a, R_LOW_PCT)
        weak_r = r_w <= np.percentile(r_w, R_LOW_PCT)
        strong_edge = edge >= np.percentile(edge, EDGE_HI_PCT)

        hole_R, areas_R = cleanup(weak_R & strong_edge)
        hole_r, areas_r = cleanup(weak_r & strong_edge)

        if hole_R.any():
            eg = float(np.median(edge[hole_R])) / max(float(np.median(edge)), 1e-6)
        else:
            eg = float("nan")

        res[sc] = {"hole_pct_R": round(float(hole_R.mean()) * 100, 2),
                   "hole_pct_r": round(float(hole_r.mean()) * 100, 2),
                   "n_comp": len(areas_R),
                   "largest": areas_R[0] if areas_R else 0,
                   "edge_gain_in_hole": (None if eg != eg else round(eg, 2))}
        print(f"{sc:<30}{res[sc]['hole_pct_R']:>9.2f}%{res[sc]['hole_pct_r']:>8.2f}%"
              f"{len(areas_R):>7}{res[sc]['largest']:>8}"
              f"{(f'{eg:.2f}' if eg == eg else '-'):>13}")

        # ---- 三联图 ----
        over = I.astype(np.uint8).copy()
        over[hole_R] = (0.30 * over[hole_R] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        cells = [
            title(over, f"HOLE = no reflection + has edges : "
                        f"{res[sc]['hole_pct_R']:.1f}%  (red)"),
            title(turbo(R_a), f"|R| reflection strength  p50={np.median(R_a):.1f}"),
            title(cv2.applyColorMap(np.clip(edge / max(np.percentile(edge, 99), 1e-6),
                                            0, 1).astype(np.uint8) if False else
                                    (np.clip(edge / max(np.percentile(edge, 99), 1e-6),
                                             0, 1) * 255).astype(np.uint8),
                                    cv2.COLORMAP_BONE),
                  "edges in the image"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_final.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 82)
    if res:
        pr = [v["hole_pct_R"] for v in res.values()]
        print(f"洞占比(|R|口径) 中位 {np.median(pr):.2f}%  "
              f"范围 [{min(pr):.2f}%, {max(pr):.2f}%]")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
