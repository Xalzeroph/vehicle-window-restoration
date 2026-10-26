#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞检测 v4 —— 按"**反射层自身的暗区**"定义（2026-10-20 对齐定义后重写）。

定义（用户口径）
---------------
"反射洞" = **反射层里出现的一片暗区**，其物理来源是：
  · 场景中的遮挡物（树影、人影）自身是深色的、且遮挡了背景环境光（漫反射遮蔽）；
  · 该物体正面受光后照到玻璃上，发生**较弱的反射**；
  · 于是反射层在该处**变暗**，但**仍保留该物体的清晰边缘轮廓**。

→ 不是"反射缺失"，而是"反射层上的一块暗区 + 边界清楚"。

判据（按此定义重写）
------------------
  R̂  = |I_a − T| 的逐通道均值  （反射层估计；I_a 已做逐通道仿射去曝光偏置）
  1. **在反射层内偏暗**：R̂ 显著低于其大尺度局部背景 R̂_bg（即局部凹陷）
  2. **边界清晰**：该凹陷的边界处 ∇R̂ 强（不是模糊过渡）
  3. **连成片**：开运算后仍成块

实现要点
--------
- 用 R̂ / R̂_bg 的**比值**判"局部暗"（尺度无关，避免被整体反射强度差异干扰）；
- 用对数比 `log((R̂_bg + c)/(R̂ + c))` 更稳，c 取全图中位；
- 边界清晰度用"暗斑边界带上的 ∇R̂ 与全图 ∇R̂ 的比值"衡量。

输出
----
_review_1019/holesB4/<scene>_hole_v4.jpg   （R̂ 图 / 暗斑掩膜叠加 / 边界清晰度）
eval/hole_b4_stats.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/holesB4"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
MAXW = 720
SCENES = ["VID_20261019_165126", "VID_20261019_165530", "VID_20261019_170406",
          "VID_20261019_164257", "VID_20261019_165849", "VID_20261019_170758"]

# 阈值（可解释）
DIP_RATIO = 0.55      # R̂ / R̂_bg <= 0.55  → 局部反射明显偏暗
EDGE_GAIN = 1.0       # 边界处 ∇R̂ 相对全图中位至少 1.0 倍
MIN_AREA = 300        # 最小连通块（px @ 720 宽）


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


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    res = {}
    hdr = (f"{'场景':<30}{'R̂中位':>9}{'暗斑占比':>9}{'连通域':>7}"
           f"{'最大块':>8}{'边界增益':>9}")
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
        # 反射层估计 R̂
        R = np.abs(I_al - G).mean(axis=2).astype(np.float32)

        # 大尺度背景（反射层的趋势）
        R_bg = cv2.GaussianBlur(R, (0, 0), 31)
        c0 = float(np.median(R))
        # 对数比：>0 表示该处反射比背景暗
        dip = np.log((R_bg + c0) / (R + c0)).astype(np.float32)

        # 反射层梯度（用于边界清晰度）
        gx = cv2.Sobel(R, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(R, cv2.CV_32F, 0, 1)
        gR = np.sqrt(gx * gx + gy * gy)

        # 判据 1：局部偏暗（比值阈值换算到对数域）
        dark = (R <= DIP_RATIO * R_bg) & (R <= np.percentile(R, 60))
        # 判据 3：连片
        m = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((7, 7), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)

        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        keep = np.zeros_like(m)
        areas = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a >= MIN_AREA:
                keep |= (lab == i)
                areas.append(a)
        m = keep
        areas.sort(reverse=True)

        # 判据 2：边界清晰度 —— 暗斑边界带上的 ∇R̂ vs 全图中位
        ker = np.ones((9, 9), np.uint8)
        m_dil = cv2.dilate(m.astype(np.uint8), ker).astype(bool)
        m_ero = cv2.erode(m.astype(np.uint8), ker).astype(bool)
        band = m_dil & ~m_ero
        g_med = float(np.median(gR))
        edge_gain = (float(np.median(gR[band])) / max(g_med, 1e-6)) if band.any() else float("nan")

        res[sc] = {"R_median": round(float(np.median(R)), 3),
                   "hole_frac": round(float(m.mean()), 4),
                   "n_comp": len(areas), "largest": areas[0] if areas else 0,
                   "areas_top5": areas[:5],
                   "edge_gain": None if edge_gain != edge_gain else round(edge_gain, 3),
                   "pass_edge": bool(edge_gain == edge_gain and edge_gain >= EDGE_GAIN)}
        print(f"{sc:<30}{res[sc]['R_median']:>9.2f}{m.mean()*100:>8.2f}%"
              f"{len(areas):>7}{res[sc]['largest']:>8}"
              f"{(f'{edge_gain:.2f}' if edge_gain == edge_gain else '-'):>9}")

        # ---- 可视化 ----
        def nz(x, p=(2, 98)):
            lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
            return np.clip((x - lo) / (hi - lo), 0, 1)
        base = G.astype(np.uint8)
        over = base.copy()
        over[m] = (0.30 * over[m] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        # 边界描边
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cont, -1, (0, 255, 0), 2)

        cells = []
        for nm, img in (
            ("R_hat = |I_a - T|  (reflection layer)",
             cv2.applyColorMap((nz(R) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)),
            ("log((R_bg+c)/(R+c))  (large=locally dark)",
             cv2.applyColorMap((nz(dip) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)),
            (f"HOLE in reflection: {m.mean()*100:.1f}%  edge_gain={edge_gain:.2f}"
             if edge_gain == edge_gain else f"HOLE {m.mean()*100:.1f}%", over),
        ):
            ww = img.shape[1]
            bar = np.zeros((20, ww, 3), np.uint8)
            cv2.putText(bar, nm, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(np.vstack([bar, img]))
        hh = min(c.shape[0] for c in cells)
        sheet = np.hstack([c[:hh] for c in cells])
        title = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(title, f"{sc}  HOLE = dark patch INSIDE the reflection layer",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_hole_v4.jpg"),
                    np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(res, open(os.path.join(EVAL, "hole_b4_stats.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * len(hdr))
    if res:
        fr = [v["hole_frac"] for v in res.values()]
        eg = [v["edge_gain"] for v in res.values() if v["edge_gain"]]
        print(f"暗斑占比 中位 {np.median(fr)*100:.2f}%   边界增益 中位 "
              f"{(np.median(eg) if eg else float('nan')):.2f}   达标场景 "
              f"{sum(1 for v in res.values() if v['pass_edge'])}/{len(res)}")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
