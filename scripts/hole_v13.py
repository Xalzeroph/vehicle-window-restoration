#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 v13 —— 在 `r = R/T` 上做两步阈值（2026-10-20）。

为什么必须用 r 而不是 |R|
--------------------
`R = r·T`。暗的透射区（树干、阴影）天然 `R` 小，即使反射正常。
所以用 `|R|` 找"弱反射"，必然把**暗的透射物体**误当成洞 —— v12 红区盖住树干/袋子就是这个错。

`r = R/T` 消掉 T 的影响，只留"反射-透射比"，才是反射层自身的量。

两步阈值
-------
1. **有反射**：`r >= t1`（t1 取 r 分布的 p55，把"没反射"的背景 r≈0 排除）
2. **暗反射**：在"有反射"区内对 `r` 做 **Otsu**，低的一类 = 洞
3. 连通域过滤 + 腐蚀剥边

输出：_review_1019/hole_v13/<scene>.jpg
      eval/hole_v13.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v13")
EVAL = os.path.join(P, "eval")
W = 700
LAMBDA = 12.0
T_FLOOR = 20.0
HAS_REFL_PCT = 55    # 已废弃（分位数会被"无反射"背景拉低）
T_HAS_REFL = 0.15    # **绝对阈值**：r >= 0.15 视为"有反射"（袋区 r≈0 被排除）
SMOOTH = 5
MIN_AREA = 200
ERODE_PX = 2


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


def title(img, t, h=20):
    w = img.shape[1]
    bar = np.zeros((h, w, 3), np.uint8)
    cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    res = {}
    print(f"{'场景':<30}{'有反射%':>9}{'t1':>7}{'Otsu':>7}{'洞%':>7}{'剥边后%':>9}"
          f"{'洞数':>6}{'最大洞':>8}")
    print("-" * 86)

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
        r = (R_s / np.maximum(T, T_FLOOR)).astype(np.float32)
        rs = cv2.blur(r, (SMOOTH, SMOOTH))

        # r 是**三级**结构，不是两级：
        #   无反射   r ≈ 0（袋区、背景）      -> 必须排除
        #   暗反射   r ≈ 0.15~0.30            -> **洞**
        #   亮反射   r ≈ 0.40~0.60（树冠亮部）
        # 所以先用**绝对阈值** T_HAS_REFL 划出"有反射"，再用 Otsu 二分。
        t1 = T_HAS_REFL
        has_refl = rs >= t1
        vals = rs[has_refl]
        if vals.size < 500:
            print(f"{sc:<30} [跳过] 有反射像素过少")
            continue
        lo, hi = float(vals.min()), float(vals.max())
        v8 = (np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
        thr_n, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_r = lo + (thr_n / 255.0) * (hi - lo)      # 换算回 r 域
        hole = has_refl & (rs <= otsu_r)

        m = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m, dtype=bool); areas = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a >= MIN_AREA:
                keep |= (lab == i); areas.append(a)
        areas.sort(reverse=True)

        pre = float(keep.mean()) * 100
        inner = keep
        if ERODE_PX > 0 and keep.any():
            k = np.ones((ERODE_PX * 2 + 1, ERODE_PX * 2 + 1), np.uint8)
            e = cv2.erode(keep.astype(np.uint8), k).astype(bool)
            if e.sum() > 0:
                inner = e
        post = float(inner.mean()) * 100

        ring = cv2.dilate(inner.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~inner
        ri = float(np.median(rs[inner])) if inner.any() else float("nan")
        ro = float(np.median(rs[ring])) if ring.any() else float("nan")

        res[sc] = {"has_refl_pct": round(float(has_refl.mean()) * 100, 2),
                   "t1": round(t1, 4), "otsu_r": round(otsu_r, 4),
                   "hole_pct": round(pre, 2), "hole_pct_eroded": round(post, 2),
                   "n_holes": len(areas), "largest": areas[0] if areas else 0,
                   "r_in": None if ri != ri else round(ri, 4),
                   "r_out": None if ro != ro else round(ro, 4)}
        print(f"{sc:<30}{res[sc]['has_refl_pct']:>8.2f}%{t1:>7.3f}{otsu_r:>7.3f}"
              f"{pre:>6.2f}%{post:>8.2f}%{len(areas):>6}{res[sc]['largest']:>8}")

        over = I.astype(np.uint8).copy()
        over[inner] = (0.25 * over[inner] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
        cont, _ = cv2.findContours(keep.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cont, -1, (0, 255, 0), 1)
        cells = [
            title(over, f"HOLE = dark reflection in r space: {post:.1f}% interior"),
            title(turbo(rs), f"r = R/T smoothed  (t1={t1:.3f}, otsu={otsu_r:.3f})"),
            title(cv2.cvtColor((has_refl * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                  f"reflection present ({res[sc]['has_refl_pct']:.0f}%)"),
            title(turbo(T), "T (transmission) — no longer contaminates the criterion"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v13.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 86)
    if res:
        he = [v["hole_pct_eroded"] for v in res.values()]
        hp = [v["hole_pct"] for v in res.values()]
        print(f"洞（剥边前）中位 {np.median(hp):.2f}%   剥边后 中位 {np.median(he):.2f}%  "
              f"范围 {min(he):.2f}% – {max(he):.2f}%")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
