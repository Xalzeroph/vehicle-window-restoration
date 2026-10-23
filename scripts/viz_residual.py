#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""残差图可视化（2026-10-20）：把反射层估计 R̂ 直接摊开给你看。

四种残差同时给出，避免任何单一显示的误读：
  A. |R̂| = mean|(I−b)/a − T|        绝对残差（灰度级），turbo 色图
  B. R̂signed = mean((I−b)/a − T)    带符号（蓝=反射使该处变暗，红=变亮）
  C. |I − T| 原始差分（不做仿射对齐）  —— 用来对照"对齐到底改了什么"
  D. 原始 I 与 T 并排，便于定位

输出：_review_1019/residual/<scene>_residual.jpg（每张含 A/B/C/D 四联）
另出 <scene>_residual_plain.png（只有 A，灰度，可直接放大看细节）
"""
from __future__ import annotations

import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/residual"
SCENES = ["VID_20261019_165126", "VID_20261019_165530", "VID_20261019_170406",
          "VID_20261019_164257", "VID_20261019_165849", "VID_20261019_170758",
          "VID_20261019_164730", "VID_20261019_170033", "VID_20261019_171408"]
MAXW = 760


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
    lo, hi = np.percentile(x, p)
    hi = max(hi, lo + 1e-9)
    v = np.clip((x - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def signed_cmap(x):
    """对称发散色图：0 为绿，负蓝，正红。"""
    m = float(np.percentile(np.abs(x), 99.5)) or 1.0
    v = np.clip(x / m, -1, 1)
    u = ((v + 1) * 0.5 * 255).astype(np.uint8)
    return cv2.applyColorMap(u, cv2.COLORMAP_JET)


def label(img, t, h=20):
    """在图上加一条标题栏。img 宽度决定栏宽。"""
    ww = img.shape[1]
    bar = np.zeros((h, ww, 3), np.uint8)
    cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
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
        sw = MAXW; sh = max(2, int(round(h * sw / w)))
        I = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G, (sw, sh), interpolation=cv2.INTER_AREA)

        A, B = robust_affine(I, G)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)

        R_abs = np.abs(I_al - G).mean(axis=2).astype(np.float32)
        R_sign = (I_al - G).mean(axis=2).astype(np.float32)
        raw_abs = np.abs(I - G).mean(axis=2).astype(np.float32)

        # 所有行统一宽度 W = 2*sw（第 1 行 = I 与 GT 各占 sw 宽并排）
        half = sw  # 注意：必须用完整 sw，否则 row1 宽度是 sw、其余行是 2*sw，vstack 会报错
        c_I = cv2.resize(I.astype(np.uint8), (half, sh))
        c_G = cv2.resize(G.astype(np.uint8), (half, sh))
        row1 = label(np.hstack([c_I, c_G]), "INPUT (f0)      |      GT (no reflection)")

        def wide(img, t):
            """把单图缩放到 (2*sw, sh) 并加标签，保证所有行等宽等高。"""
            out = cv2.resize(img, (sw * 2, sh), interpolation=cv2.INTER_AREA)
            return label(out, t)

        cells = [
            row1,
            wide(turbo(R_abs),
                 f"|R_hat| aligned   p50={np.median(R_abs):.2f}  "
                 f"p90={np.percentile(R_abs,90):.1f}"),
            wide(signed_cmap(R_sign),
                 "R_hat signed   blue = reflection makes it DARKER, red = brighter"),
            wide(turbo(raw_abs),
                 f"|I - T| RAW (no affine)  p50={np.median(raw_abs):.2f}"),
        ]
        sheet = np.vstack(cells)
        ttl = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}   RESIDUAL / REFLECTION-LAYER ESTIMATE",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_residual.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 单独的纯 |R̂| 灰度大图（便于放大看结构）
        lo, hi = np.percentile(R_abs, (1, 99.5))
        g = np.clip((R_abs - lo) / max(hi - lo, 1e-9), 0, 1)
        cv2.imwrite(os.path.join(OUT, f"{sc}_Rhat_gray.png"),
                    (g * 255).astype(np.uint8))
        print(f"  {sc}: |R| p50={np.median(R_abs):.2f} p90={np.percentile(R_abs,90):.1f} "
              f"max={R_abs.max():.1f}  gain={[round(float(x),3) for x in A]}")
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
