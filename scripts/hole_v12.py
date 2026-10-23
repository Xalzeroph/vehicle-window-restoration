#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 v12 —— 反射区内 `|R|` 的 Otsu 二分（2026-10-20）。

为什么放弃"相对背景凹陷"
----------------------
v9~v11 都用 `|R| <= α·blur(|R|)`，本质是找**局部凹陷**。
它只能抓"明亮反射之间的小暗点"，抓不到**整片暗的树影** ——
所以三档松紧都不够覆盖（用户："松也不够"）。

v12 改为**绝对分界**
------------------
`|R| = r·T`，在被反射覆盖的区域里（如树冠），`|R|` 呈**双峰**：
  · 亮的反射（天空、亮背景）→ 高
  · 暗的遮挡物（树影、人影）→ 低
对这个区域内的 `|R|` 做 **Otsu 自动阈值**，低的一类就是**暗反射 = 洞**。

步骤
----
1. `|R|` 轻微平滑（去纹理噪声）
2. **反射覆盖区**：`|R|_smooth >= 全图 p45`（排除"本来就没反射"的背景）
3. 在覆盖区内对 `|R|` 做 **Otsu**，低类 → 洞候选
4. 连通域过滤 + **腐蚀 2px 剥掉轮廓带**（避免 v9 的"边缘点"偏差）

输出：_review_1019/hole_v12/<scene>.jpg
      eval/hole_v12.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v12")
EVAL = os.path.join(P, "eval")
W = 700
COVER_PCT = 45      # 反射覆盖区：|R| 高于该分位
SMOOTH = 5          # |R| 平滑核
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
    print(f"{'场景':<30}{'覆盖区%':>9}{'Otsu阈值':>9}{'洞%':>7}{'剥边后%':>9}"
          f"{'洞数':>6}{'最大洞':>8}")
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
        Rs = cv2.blur(R_a, (SMOOTH, SMOOTH))

        cover = Rs >= np.percentile(Rs, COVER_PCT)
        # Otsu：在覆盖区内按 |R| 二分
        vals = Rs[cover]
        if vals.size < 500:
            continue
        v8 = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
        thr, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr)
        hole = cover & (Rs <= thr)

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
        ri = float(np.median(R_a[inner])) if inner.any() else float("nan")
        ro = float(np.median(R_a[ring])) if ring.any() else float("nan")

        res[sc] = {"cover_pct": round(float(cover.mean()) * 100, 2),
                   "otsu_thr": round(thr, 2),
                   "hole_pct": round(pre, 2),
                   "hole_pct_eroded": round(post, 2),
                   "n_holes": len(areas),
                   "largest": areas[0] if areas else 0,
                   "R_in": None if ri != ri else round(ri, 2),
                   "R_out": None if ro != ro else round(ro, 2)}
        print(f"{sc:<30}{res[sc]['cover_pct']:>8.2f}%{thr:>9.1f}{pre:>6.2f}%"
              f"{post:>8.2f}%{len(areas):>6}{res[sc]['largest']:>8}")

        over = I.astype(np.uint8).copy()
        over[inner] = (0.25 * over[inner] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
        cont, _ = cv2.findContours(keep.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cont, -1, (0, 255, 0), 1)
        cells = [
            title(over, f"HOLE (dark reflection, Otsu): {post:.1f}% interior"),
            title(turbo(R_a), f"|R|  Otsu thr={thr:.1f}"),
            title(cv2.cvtColor((cover * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                  f"reflection coverage ({res[sc]['cover_pct']:.0f}%)"),
            title(turbo(Rs), "|R| smoothed (Otsu applied here)"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v12.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 82)
    if res:
        hp = [v["hole_pct"] for v in res.values()]
        he = [v["hole_pct_eroded"] for v in res.values()]
        print(f"洞（剥边前）中位 {np.median(hp):.2f}%   剥边后中位 {np.median(he):.2f}%")
        print(f"范围 {min(he):.2f}% – {max(he):.2f}%")
        print("对比：v9 剥边后 0.12%；松档 1.31%")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
