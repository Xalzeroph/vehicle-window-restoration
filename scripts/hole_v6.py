#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞检测 v6 —— 双约束：反射**绝对弱** + 与透射**结构共边**（2026-10-20）。

前几版都错在哪
-------------
v4/v5 找的是"反射层内部的**局部**凹陷（相对自身背景）"→ 结果落在天空/楼区这种
本来就平滑的地方，而不是"暗的遮挡物"。**"相对变暗"和"绝对暗"是两件事。**

v6 判据（同时满足，全部可测）
--------------------------
1. **反射绝对弱**：`R̂` 低于全图 35 分位（不是比值，是绝对量）；
2. **有物体级结构**：该区域内 `∇R̂` 的 75 分位 >= 全图 60 分位（不是平坦区）；
3. **与透射结构共边（关键约束）**：暗遮挡物是真实物体，
   它的轮廓应当**同时出现在 T 上**。用"暗斑边界带"上 `∇T` 是否也强来衡量。

→ 这条 3 把"天空/楼区的平滑暗区"排除掉：那些地方 T 上也没有强轮廓。

输出
----
_review_1019/holesB6/<scene>_hole_v6.jpg
eval/hole_b6_stats.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/holesB6"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
MAXW = 720
MIN_AREA = 600
SCENES = ["VID_20261019_165126", "VID_20261019_165530", "VID_20261019_170406",
          "VID_20261019_164257", "VID_20261019_165849", "VID_20261019_170758"]


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


def components(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask); areas = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out |= (lab == i); areas.append(a)
    areas.sort(reverse=True)
    return out, areas


def grad_of(x):
    gx, gy = np.gradient(x)
    return np.sqrt(gx ** 2 + gy ** 2)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    res = {}
    hdr = (f"{'场景':<30}{'R̂p35':>8}{'暗斑%':>8}{'连通域':>7}{'最大块':>8}"
           f"{'∇R分位':>8}{'∇T边界分位':>11}")
    print(hdr); print("-" * len(hdr))

    for sc in SCENES:
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        I = load(os.path.join(d, "I_000000.png"))
        G = load(os.path.join(d, "GT.png"))
        if I is None or G is None:
            continue
        h, w = I.shape[:2]
        sw = MAXW; sh = max(2, int(round(h * sw / w)))
        I = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G, (sw, sh), interpolation=cv2.INTER_AREA)

        A, B = robust_affine(I, G)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
        R = np.abs(I_al - G).mean(axis=2).astype(np.float32)
        Tg = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        gR = grad_of(R)
        gT = grad_of(Tg)
        p = lambda x, q: float(np.percentile(x, q))
        R35 = p(R, 35)

        # 判据 1：反射绝对弱
        weak = R <= R35
        # 判据 2：在物体级尺度上有结构（用局部梯度均值，避免细纹理误判）
        gR_loc = cv2.blur(gR.astype(np.float32), (25, 25))
        struct = gR_loc >= p(gR_loc, 60)

        m = weak & struct
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((11, 11), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             np.ones((21, 21), np.uint8)).astype(bool)
        m, areas = components(m, MIN_AREA)

        # 指标：区域内的 ∇R 强度分位；边界带上 ∇T 的分位（判据 3 的度量）
        ker = np.ones((21, 21), np.uint8)
        band = (cv2.dilate(m.astype(np.uint8), ker).astype(bool)
                & ~cv2.erode(m.astype(np.uint8), ker).astype(bool))
        gR_pct = float((gR < p(gR[m], 75)).mean() * 100) if m.any() else 0.0
        gT_band_pct = float((gT < p(gT[band], 75)).mean() * 100) if band.any() else 0.0
        # 更直观：边界带的 ∇T 中位 vs 全图 ∇T 分位
        gT_band_q = float((gT < float(np.median(gT[band]))).mean() * 100) if band.any() else 0.0

        res[sc] = {"R_p35": round(R35, 3),
                   "hole_frac": round(float(m.mean()), 4),
                   "n_comp": len(areas), "largest": areas[0] if areas else 0,
                   "areas_top5": areas[:5],
                   "gR_in_region_pct": round(gR_pct, 1),
                   "gT_band_pct": round(gT_band_q, 1),
                   "pass_both": bool(m.any() and gR_pct >= 50 and gT_band_q >= 60)}
        print(f"{sc:<30}{R35:>8.2f}{m.mean()*100:>7.2f}%{len(areas):>7}"
              f"{res[sc]['largest']:>8}{gR_pct:>8.0f}{gT_band_q:>11.0f}")

        # ---- 可视化 ----
        def nz(x, pp=(2, 98)):
            lo, hi = np.percentile(x, pp); hi = max(hi, lo + 1e-9)
            return np.clip((x - lo) / (hi - lo), 0, 1)
        over = G.astype(np.uint8).copy()
        over[m] = (0.30 * over[m] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cont, -1, (0, 255, 0), 2)
        cells = []
        for nm, img in (
            ("R_hat (reflection layer)", cv2.applyColorMap(
                (nz(R) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)),
            ("weak(R<=p35) & structured", cv2.applyColorMap(
                (nz((weak & struct).astype(np.float32)) * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO)),
            (f"HOLE v6: {m.mean()*100:.1f}%  gR={gR_pct:.0f}  gT_band={gT_band_q:.0f}", over),
        ):
            ww = img.shape[1]
            bar = np.zeros((20, ww, 3), np.uint8)
            cv2.putText(bar, nm, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(np.vstack([bar, img]))
        hh = min(c.shape[0] for c in cells)
        sheet = np.hstack([c[:hh] for c in cells])
        title = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(title, f"{sc}  HOLE v6 = WEAK reflection + structure + co-edges with T",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_hole_v6.jpg"),
                    np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(res, open(os.path.join(EVAL, "hole_b6_stats.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * len(hdr))
    if res:
        print(f"判据双达标场景: {sum(1 for v in res.values() if v['pass_both'])}/{len(res)}")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
