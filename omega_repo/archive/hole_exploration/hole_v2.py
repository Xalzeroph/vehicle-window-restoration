#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""洞区判定 v2：修正 v1 的判据冲突（2026-10-20）。

v1 的失败
--------
v1 判据 = 弱反射 ∧ **强梯度** ∧ 偏暗。
但 |∇T| 高只出现在**边缘**，而"弱反射"区域内往往是平坦的 →
两者交集必然退化成一圈边缘像素，绝不是连通的"洞"。

物理上"反射洞"到底是什么
------------------------
按用户的描述：物体正面受光在车窗形成弱反射，同时物体遮蔽了环境光 →
**该区域整体偏暗、反射贡献弱**。它更像一片"**暗淡且低反射的连片区域**"，
而不是一条边。

v2 三个变体（都去掉"强梯度"，改成可选）
------------------------------------
  V1 纯弱反射+暗        : r 低 ∧ T 暗
  V2 弱反射+暗+**低**梯度: r 低 ∧ T 暗 ∧ ∇T 低   （连通、平滑的暗区）
  V3 弱反射+暗（相对）  : r 低 ∧ T 低于全图中位 ∧ r 的**局部**均值低（抑制孤立像素）

用形态学开运算去掉孤立像素，让结果"连成片"，并统计连通域大小，
以便判断"是不是真的有一片洞"。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/holesB2"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
MAXW = 640
SCHEDULE = {"scaffold": ["VID_20261019_165126", "VID_20261019_165530",
                         "VID_20261019_170406", "VID_20261019_164257"]}


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
            r = np.abs(x - (a * y + b))
            k = r <= np.percentile(r, keep * 100)
            x, y = x[k], y[k]
            if len(x) < 500:
                break
        A[c], B[c] = a, b
    return A, B


def analyze(sc):
    I = load(os.path.join(PAIR, sc, "I_000000.png"))
    G = load(os.path.join(PAIR, sc, "GT.png"))
    if I is None or G is None:
        return None
    h, w = I.shape[:2]
    sw = MAXW; sh = max(2, int(round(h * sw / w)))
    I = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
    G = cv2.resize(G, (sw, sh), interpolation=cv2.INTER_AREA)

    A, B = robust_affine(I, G)
    I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
    T = G
    r = np.abs((I_al - T) / np.maximum(T, 8.0)).mean(axis=2)
    Tg = cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(Tg, cv2.CV_32F, 1, 0); gy = cv2.Sobel(Tg, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)
    # r 的局部均值（抑制孤立像素，让洞连片）
    r_loc = cv2.blur(r.astype(np.float32), (15, 15))

    r_lo = np.percentile(r_loc, 25)
    T_dark = np.percentile(Tg, 40)
    g_lo = np.percentile(grad, 40)

    weak = r_loc <= r_lo
    dark = Tg <= T_dark
    smooth = grad <= g_lo
    kern = np.ones((7, 7), np.uint8)

    Vs = {
        "V1_r+T": weak & dark,
        "V2_r+T+smooth": weak & dark & smooth,
    }
    res = {"scene": sc, "A": [round(float(x), 4) for x in A],
           "r_loc_median": round(float(np.median(r_loc)), 4),
           "variants": {}}
    vis_cells = []

    def norm(x, p=(1, 99)):
        lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
        return np.clip((x - lo) / (hi - lo), 0, 1)

    base = T.astype(np.uint8)
    for name, m in Vs.items():
        m2 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kern).astype(bool)
        n, lab, stats_c, _ = cv2.connectedComponentsWithStats(m2.astype(np.uint8), 8)
        areas = sorted([int(a) for a in stats_c[1:, cv2.CC_STAT_AREA]], reverse=True)
        big = sum(a for a in areas if a >= 200)
        res["variants"][name] = {
            "frac": round(float(m2.mean()), 4),
            "n_components": int(n - 1),
            "largest_area": areas[0] if areas else 0,
            "big_area_frac": round(big / m2.size, 4),
        }
        over = base.copy()
        over[m2] = (0.30 * over[m2] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        ww = over.shape[1]
        bar = np.zeros((20, ww, 3), np.uint8)
        cv2.putText(bar, f"{name}: {m2.mean()*100:.1f}%  "
                         f"largest={areas[0] if areas else 0}px",
                    (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        vis_cells.append(np.vstack([bar, over]))

    # r 图与梯度图也放进去
    ww = base.shape[1]
    bar = np.zeros((20, ww, 3), np.uint8)
    cv2.putText(bar, f"r (local mean) med={res['r_loc_median']:.3f}", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    vis_cells.insert(0, np.vstack([bar, cv2.cvtColor((norm(r_loc) * 255).astype(np.uint8),
                                                   cv2.COLOR_GRAY2BGR)]))
    hh = min(c.shape[0] for c in vis_cells)
    sheet = np.hstack([c[:hh] for c in vis_cells])
    title = np.zeros((30, sheet.shape[1], 3), np.uint8)
    cv2.putText(title, f"{sc}  hole v2  (weak reflection + dark, no strong-gradient requirement)",
                (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA)
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, f"{sc}_hole_v2.jpg"),
                np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return res


def main() -> int:
    allres = {}
    for sc in SCHEDULE["scaffold"]:
        if not os.path.isdir(os.path.join(PAIR, sc)):
            print(f"[跳过] {sc}")
            continue
        r = analyze(sc)
        if r is None:
            continue
        allres[sc] = r
        print(f"\n=== {sc} ===")
        print(f"  r_loc 中位 {r['r_loc_median']:.4f}")
        for name, v in r["variants"].items():
            print(f"  {name:<16} 占比 {v['frac']*100:5.2f}%  "
                  f"连通域 {v['n_components']:>4}  最大 {v['largest_area']:>6}px  "
                  f"大片占比 {v['big_area_frac']*100:.2f}%")
    os.makedirs(EVAL, exist_ok=True)
    json.dump(allres, open(os.path.join(EVAL, "hole_b2_stats.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
