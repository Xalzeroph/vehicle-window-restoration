#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""残差图 + 网格坐标 + r = R/T（真正的反射层）（2026-10-20）。

两个目的
--------
1. **可指认**：给每行图打 8×6 网格与行列标签，便于用「R3-B2」这种坐标精确指认区域。
2. **算对量**：`R = I/a − T = r·T` 里含 T 的结构；真正的反射层是
      r = (I/a − T) / T
   本脚本用**维纳正则** `r ≈ R·T / (T² + λ²)` 避免暗区除零爆炸，
   并同时给出直接除法（floor 版）作对照。

输出
----
_review_1019/residual_grid/<scene>_grid.jpg   （网格标注版，6 行）
_review_1019/residual_grid/<scene>_r_plain.png（纯 r 灰度，可放大）
"""
from __future__ import annotations

import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/residual_grid"
SCENES = ["VID_20261019_165126", "VID_20261019_165530", "VID_20261019_170406"]
W = 700          # 每格渲染宽
GR, GC = 6, 8    # 网格 6 行 × 8 列
LAMBDA = 12.0    # 维纳正则强度（灰度级）


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


def jet_signed(x):
    m = float(np.percentile(np.abs(x), 99.5)) or 1.0
    u = ((np.clip(x / m, -1, 1) + 1) * 0.5 * 255).astype(np.uint8)
    # JET: 0=蓝(负) 128=绿(≈0) 255=红(正)
    return cv2.applyColorMap(u, cv2.COLORMAP_JET)


def grid(img, row_tag):
    """打网格 + 行列标签。"""
    out = img.copy()
    H, Wd = out.shape[:2]
    bh, bw = H // GR, Wd // GC
    for j in range(1, GR):
        cv2.line(out, (0, j * bh), (Wd, j * bh), (255, 255, 255), 1)
    for i in range(1, GC):
        cv2.line(out, (i * bw, 0), (i * bw, H), (255, 255, 255), 1)
    for j in range(GR):
        for i in range(GC):
            lbl = f"{chr(65 + i)}{j + 1}"
            cv2.putText(out, lbl, (i * bw + 3, j * bh + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, lbl, (i * bw + 3, j * bh + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    # 左侧行标签
    bar = np.zeros((H, 46, 3), np.uint8)
    for k, ch in enumerate(row_tag):
        cv2.putText(bar, ch, (4, 26 + k * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([bar, out])


def label(img, t, h=20):
    ww = img.shape[1]
    bar = np.zeros((h, ww, 3), np.uint8)
    cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for sc in SCENES:
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
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

        R_s = (I_al - G).mean(axis=2).astype(np.float64)     # 有符号残差
        R_a = np.abs(R_s)
        T = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 维纳正则版反射比
        r_w = R_s * T / (T * T + LAMBDA * LAMBDA)
        # 直接除法（floor 版），仅作对照
        r_d = R_s / np.maximum(T, 20.0)

        rows = [
            ("R1 I|GT", label(np.hstack([I.astype(np.uint8), G.astype(np.uint8)]),
                              "I (f0)          |          GT")),
            ("R2 |R|", label(grid(turbo(R_a), ["R", "2"]),
                             f"|R| = |I/a - T|   p50={np.median(R_a):.2f}")),
            ("R3 Rsigned", label(grid(jet_signed(R_s), ["R", "3"]),
                                 "R signed (JET: blue=darker, green=0, red=brighter)")),
            ("R4 T", label(grid(cv2.applyColorMap(
                cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_BONE),
                ["R", "4"]), "T (transmission, grayscale)")),
            ("R5 r=R/T", label(grid(turbo(r_w), ["R", "5"]),
                               f"r = R/(T) Wiener  blue=LOW reflection ratio (candidate hole)")),
            ("R6 r direct", label(grid(turbo(r_d), ["R", "6"]),
                                  "r = R/max(T,20) direct division (noisy in dark)")),
        ]
        # R1 是两图并排，宽度不与其它行一致 -> 单独缩放到同宽
        target_w = rows[1][1].shape[1]
        r1 = cv2.resize(rows[0][1], (target_w, rows[0][1].shape[0]))
        rows[0] = (rows[0][0], r1)
        sheet = np.vstack([r for _, r in rows])
        ttl = np.zeros((32, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}   grid = A-H (cols) x 1-6 (rows)   "
                         f"affine a={[round(float(x),3) for x in A]}",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_grid.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

        lo, hi = np.percentile(r_w, (1, 99.5))
        gg = np.clip((r_w - lo) / max(hi - lo, 1e-9), 0, 1)
        cv2.imwrite(os.path.join(OUT, f"{sc}_r_plain.png"), (gg * 255).astype(np.uint8))
        print(f"  {sc}: r_w p10={np.percentile(r_w,10):+.3f} "
              f"p50={np.percentile(r_w,50):+.3f} p90={np.percentile(r_w,90):+.3f}")
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
