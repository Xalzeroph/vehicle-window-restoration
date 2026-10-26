#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""洞区 v3：改用**绝对阈值**而非分位数，并验证"白袋子是不是洞"（2026-10-20）。

v2 为何 165126 判 0%
-------------------
v2 的"暗"用的是 `T ≤ 40分位`，而画面里**树干的深色纹理占了大量暗像素**，
把袋子的名额抢走了 → 袋子没进"暗"集合 → 交集为空。

物理上"洞"的判据应该是**绝对量**，不是相对名次：
  · 反射弱：|R| 绝对残差小（比如 < 某阈值，单位灰度级）
  · 透射暗：T 的局部亮度显著低于其邻域（局部对比度定义，避免被大面积暗区抢名额）
  · 连成片：形态学开运算后仍有足够大的连通域

同时做**针对性验证**：在白色袋子/箱子这类"大面积平坦暗区"上直接算 |R|，
看它是否真的接近 0（若是，说明"洞"确实存在，只是判据选错了）。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/holesB3"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
MAXW = 640
SCENES = ["VID_20261019_165126", "VID_20261019_165530",
          "VID_20261019_170406", "VID_20261019_164257", "VID_20261019_165849"]


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
    hdr = (f"{'场景':<30}{'|R|中位':>9}{'|R|p20':>8}{'|R|p80':>8}"
           f"{'洞(绝对)':>10}{'洞连通域':>9}{'最大块':>8}")
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
        T = G
        R_abs = np.abs(I_al - T).mean(axis=2)          # 绝对残差（灰度级）
        Tg = cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 局部对比度：T 比其大尺度邻域暗多少（正值=局部更暗）
        big = cv2.GaussianBlur(Tg, (0, 0), 25)
        local_dark = big - Tg          # >0 表示比周围暗
        # 平滑度：局部梯度低（连片平坦）
        gx = cv2.Sobel(Tg, cv2.CV_32F, 1, 0); gy = cv2.Sobel(Tg, cv2.CV_32F, 0, 1)
        grad = np.sqrt(gx * gx + gy * gy)
        flat = cv2.blur(grad, (25, 25))

        # 绝对阈值（可解释、可迁移）
        R_THR = 8.0        # 灰度级：残余不到 8 级 → 认为反射可忽略
        DARK_THR = 5.0     # 比周围暗 5 级以上
        FLAT_THR = float(np.percentile(flat, 50))

        m = (R_abs <= R_THR) & (local_dark >= DARK_THR) & (flat <= FLAT_THR)
        m2 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((9, 9), np.uint8)).astype(bool)
        n, lab, stats_c, _ = cv2.connectedComponentsWithStats(m2.astype(np.uint8), 8)
        areas = sorted([int(a) for a in stats_c[1:, cv2.CC_STAT_AREA]], reverse=True)

        res[sc] = {"R_median": round(float(np.median(R_abs)), 3),
                   "R_p20": round(float(np.percentile(R_abs, 20)), 3),
                   "R_p80": round(float(np.percentile(R_abs, 80)), 3),
                   "hole_frac": round(float(m2.mean()), 4),
                   "n_comp": int(n - 1),
                   "largest": areas[0] if areas else 0,
                   "areas_top5": areas[:5]}
        print(f"{sc:<30}{res[sc]['R_median']:>9.2f}{res[sc]['R_p20']:>8.2f}"
              f"{res[sc]['R_p80']:>8.2f}{m2.mean()*100:>9.2f}%{n-1:>9}{res[sc]['largest']:>8}")

        # 可视化：T + 洞叠加 + |R| 热图
        def nz(x, p=(2, 98)):
            lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
            return np.clip((x - lo) / (hi - lo), 0, 1)
        base = T.astype(np.uint8)
        over = base.copy()
        over[m2] = (0.3 * over[m2] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
        cells = []
        for nm, img in (("|R| abs residual (blue=small)", cv2.applyColorMap(
                            (nz(R_abs) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)),
                        (f"HOLE: |R|<={R_THR}, local-dark>={DARK_THR}, flat "
                         f"-> {m2.mean()*100:.1f}%", over)):
            ww = img.shape[1]
            bar = np.zeros((20, ww, 3), np.uint8)
            cv2.putText(bar, nm, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(np.vstack([bar, img]))
        hh = min(c.shape[0] for c in cells)
        sheet = np.hstack([c[:hh] for c in cells])
        title = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(title, f"{sc}  hole v3 = ABSOLUTE thresholds",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_hole_v3.jpg"),
                    np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(res, open(os.path.join(EVAL, "hole_b3_stats.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n=== 判读 ===")
    fr = [v["hole_frac"] for v in res.values()]
    big = [v["largest"] for v in res.values()]
    print(f"洞占比 中位 {np.median(fr)*100:.2f}%  最大连通块 中位 {np.median(big):.0f}px")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
