#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正上一步的除法伪影：改用**绝对残差**重验暗区假设（2026-10-20）。

上一版的错误
------------
用了 `r = |R| / T`，而 `T` 在暗区很小 → 除法把暗区的 r 自动放大，
于是必然得到"暗区 r 更大"的结论。**这是定义伪影，不是物理发现。**

本版改用三个**无除法**的量，互相印证：
  1. `R_abs = mean|(I/a − b) − T|`       绝对残差（灰度级）
  2. `R_lum = mean((I/a − b) − T)`       带符号残差（看是"变亮"还是"变暗"）
  3. `R / (T + c)`，c 取全图中位（加了常数项，避免除小数的奇点）

并同时给出：暗区/亮区的 T、I 的亮度，判断"暗区到底是真暗，还是被反射压暗"。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from scipy.stats import spearmanr

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/hole_test2"
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
    hdr = (f"{'场景':<30}{'rho(T,|R|)':>12}{'暗区|R|':>9}{'亮区|R|':>9}"
           f"{'比值':>7}{'暗区T':>8}{'亮区T':>8}")
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
        R_signed = (I_al - T).mean(axis=2)          # 带符号
        R_abs = np.abs(I_al - T).mean(axis=2)       # 绝对
        Tg = cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        c0 = float(np.median(Tg))
        R_rel = R_abs / (Tg + c0)                   # 加常数，避免奇点

        dark = Tg <= np.percentile(Tg, 20)
        bright = Tg >= np.percentile(Tg, 80)
        a_d, a_b = float(np.median(R_abs[dark])), float(np.median(R_abs[bright]))
        s_d, s_b = float(np.median(R_signed[dark])), float(np.median(R_signed[bright]))
        rl_d, rl_b = float(np.median(R_rel[dark])), float(np.median(R_rel[bright]))
        rho_abs = spearmanr(Tg.ravel()[::13], R_abs.ravel()[::13]).statistic
        rho_rel = spearmanr(Tg.ravel()[::13], R_rel.ravel()[::13]).statistic
        ratio = a_d / max(a_b, 1e-9)

        print(f"{sc:<30}{rho_abs:>12.3f}{a_d:>9.2f}{a_b:>9.2f}{ratio:>7.2f}"
              f"{np.median(Tg[dark]):>8.1f}{np.median(Tg[bright]):>8.1f}")
        res[sc] = {"rho_T_Rabs": round(float(rho_abs), 4),
                   "rho_T_Rrel": round(float(rho_rel), 4),
                   "Rabs_dark": round(a_d, 3), "Rabs_bright": round(a_b, 3),
                   "ratio_abs": round(ratio, 3),
                   "Rsigned_dark": round(s_d, 3), "Rsigned_bright": round(s_b, 3),
                   "Rrel_dark": round(rl_d, 4), "Rrel_bright": round(rl_b, 4),
                   "T_dark_median": round(float(np.median(Tg[dark])), 1),
                   "T_bright_median": round(float(np.median(Tg[bright])), 1)}

        # 可视化：三个量的热图 + 暗区叠加
        def nz(x, p=(2, 98)):
            lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
            return np.clip((x - lo) / (hi - lo), 0, 1)
        cells = []
        for nm, x in (("|R| abs residual", R_abs),
                      ("R signed (+ = I brighter)", R_signed),
                      ("R/(T+c) rel residual", R_rel)):
            img = (nz(x) * 255).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
            ww = img.shape[1]
            bar = np.zeros((20, ww, 3), np.uint8)
            cv2.putText(bar, nm, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(np.vstack([bar, img]))
        base = T.astype(np.uint8)
        over = base.copy()
        over[dark] = (0.4 * over[dark] + 0.6 * np.array([0, 128, 255])).astype(np.uint8)
        ww = base.shape[1]
        bar = np.zeros((20, ww, 3), np.uint8)
        cv2.putText(bar, "darkest 20% of T (orange)", (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(np.vstack([bar, over]))
        hh = min(c.shape[0] for c in cells)
        sheet = np.hstack([c[:hh] for c in cells])
        title = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(title, f"{sc}  dark-vs-bright reflection (NO division artifact)",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_dark2.jpg"),
                    np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(res, open(os.path.join(EVAL, "hole_test2.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n=== 判读 ===")
    rs = [v["ratio_abs"] for v in res.values()]
    print(f"暗/亮 的绝对残差比：中位 {np.median(rs):.2f}（>1 表示暗区反射更强）")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
