#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位诊断：R̂（反射层估计）在空间上到底怎么分布（2026-10-20）。

不断调阈值之前，先看清事实：
  1. R̂ 的直方图
  2. 把图像分成 8×8 网格，看每格的 R̂ 中位与 gT 中位 —— 找出
     "暗反射 + 有结构"的格子到底存不存在
  3. 关键检验：用户说的"树影"（左上大片反射区里的暗斑）
     与"袋子"（下方）在 R̂ / gT 上的差异

输出：eval/rh_locator.json + _review_1019/rh_locator/<scene>_grid.jpg
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/rh_locator"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
MAXW = 640
G = 8   # 8x8 网格
SCENES = ["VID_20261019_165126", "VID_20261019_165530", "VID_20261019_170406"]


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
    allres = {}
    for sc in SCENES:
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        I = load(os.path.join(d, "I_000000.png"))
        Gt = load(os.path.join(d, "GT.png"))
        if I is None or Gt is None:
            continue
        h, w = I.shape[:2]
        sw = MAXW; sh = max(2, int(round(h * sw / w)))
        I = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
        Gt = cv2.resize(Gt, (sw, sh), interpolation=cv2.INTER_AREA)
        A, B = robust_affine(I, Gt)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
        R = np.abs(I_al - Gt).mean(axis=2).astype(np.float32)
        Tg = cv2.cvtColor(Gt.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx, gy = np.gradient(Tg)
        gT = np.sqrt(gx ** 2 + gy ** 2)

        H, W = R.shape
        bh, bw = H // G, W // G
        table = []
        print(f"\n=== {sc}   R̂ 全局: p10={np.percentile(R,10):.1f} "
              f"p50={np.percentile(R,50):.1f} p90={np.percentile(R,90):.1f} ===")
        print("  R̂ 网格（每格中位；行=上到下，列=左到右）")
        for j in range(G):
            row = []
            for i in range(G):
                blk = R[j*bh:(j+1)*bh, i*bw:(i+1)*bw]
                row.append(f"{np.median(blk):6.1f}")
            print("   " + " ".join(row))
        print("  gT 网格（每格中位）")
        for j in range(G):
            row = []
            for i in range(G):
                blk = gT[j*bh:(j+1)*bh, i*bw:(i+1)*bw]
                row.append(f"{np.median(blk):6.1f}")
            print("   " + " ".join(row))

        # 关键：按 R̂ 分位分箱，看每箱的 gT 中位 → "弱反射处有没有结构"？
        qs = np.percentile(R, [0, 20, 40, 60, 80, 100])
        bins = []
        for k in range(5):
            m = (R >= qs[k]) & (R <= qs[k+1])
            if m.sum() < 200:
                continue
            bins.append({"R_lo": round(float(qs[k]), 2), "R_hi": round(float(qs[k+1]), 2),
                         "gT_med": round(float(np.median(gT[m])), 2),
                         "n": int(m.sum())})
        print("  R̂ 分箱 → gT 中位（看弱反射处是否有结构）")
        for b in bins:
            print(f"    R̂∈[{b['R_lo']:6.2f},{b['R_hi']:6.2f}]  gT中位={b['gT_med']:6.2f}  "
                  f"n={b['n']}")
        allres[sc] = {"R_p": [round(float(x), 2) for x in np.percentile(R, [10, 50, 90])],
                      "bins": bins}

        # 可视化：R̂ 热图 + 网格线 + gT 热图
        def nz(x, pp=(2, 98)):
            lo, hi = np.percentile(x, pp); hi = max(hi, lo + 1e-9)
            return np.clip((x - lo) / (hi - lo), 0, 1)
        vr = cv2.applyColorMap((nz(R) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vg = cv2.applyColorMap((nz(gT) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        for img in (vr, vg):
            for k in range(1, G):
                cv2.line(img, (0, k*bh), (W, k*bh), (255, 255, 255), 1)
                cv2.line(img, (k*bw, 0), (k*bw, H), (255, 255, 255), 1)
        base = Gt.astype(np.uint8)
        sheet = np.hstack([base, vr, vg])
        ttl = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}   GT | R_hat | grad(T)   (8x8 grid for reading the tables)",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_grid.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(allres, open(os.path.join(EVAL, "rh_locator.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
