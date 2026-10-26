#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9 洞掩膜的分区域评测（2026-10-20 定稿）。

洞定义见 hole_v9.py：反射区内的局部暗斑 + 有边缘 + 边缘不属于 T。
以 identity 为基线破除"洞由 |R| 定义"带来的循环性。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
MASKDIR = os.path.join(P, "_review_1019/hole_masks_v9")
EVAL = os.path.join(P, "eval")
W = 700
DIP_RATIO, DIP_SIGMA = 0.60, 15
E_I_PCT, E_T_PCT, R_PRESENT_PCT, MIN_AREA = 75, 60, 60, 30

ARMS = {"rdnet_zs": "real_pair_1019_rdnet_zs",
        "ft_s2main": "real_pair_1019_ft_s2main",
        "ft_mixbest": "real_pair_1019_ft_mixbest",
        "lightrr": "real_pair_1019_lightrr"}

GLOBAL = {"identity": [28.30, 0.0]}  # 占位，实际由下面重算


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


def grad(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def psnr_mask(a, b, mask):
    if mask.sum() < 50:
        return float("nan")
    vals = []
    for c in range(3):
        av = a[..., c][mask].astype(np.float64)
        bv = b[..., c][mask].astype(np.float64)
        mse = float(np.mean((av - bv) ** 2))
        vals.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(vals))


def main() -> int:
    os.makedirs(MASKDIR, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    rows = []
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
        T = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gI, gT = grad(Ig), grad(T)
        R_bg = cv2.GaussianBlur(R_a, (0, 0), DIP_SIGMA)

        hole = ((R_a >= np.percentile(R_a, R_PRESENT_PCT))
                & (R_a <= DIP_RATIO * R_bg)
                & (gI >= np.percentile(gI, E_I_PCT))
                & (gT <= np.percentile(gT, E_T_PCT)))
        m = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        out = np.zeros_like(m, dtype=bool)
        for i in range(1, n):
            if int(st[i, cv2.CC_STAT_AREA]) >= MIN_AREA:
                out |= (lab == i)
        cv2.imwrite(os.path.join(MASKDIR, f"{sc}.png"), (out.astype(np.uint8) * 255))

        rec = {"scene": sc, "hole_frac": round(float(out.mean()), 4)}
        rec["identity"] = {"h": round(psnr_mask(I, G, out), 3),
                           "n": round(psnr_mask(I, G, ~out), 3)}
        for arm, sub in ARMS.items():
            pth = os.path.join(PRED, sub, f"{sc}__I_000000", "t_hat.png")
            if not os.path.exists(pth):
                continue
            Pr = cv2.imread(pth, cv2.IMREAD_COLOR).astype(np.float64)
            if Pr.shape[:2] != (sh, W):
                Pr = cv2.resize(Pr, (W, sh), interpolation=cv2.INTER_AREA)
            rec[arm] = {"h": round(psnr_mask(Pr, G, out), 3),
                        "n": round(psnr_mask(Pr, G, ~out), 3)}
        rows.append(rec)

    summ = {}
    for arm in ["identity"] + list(ARMS):
        hh = [r[arm]["h"] for r in rows if arm in r]
        nn = [r[arm]["n"] for r in rows if arm in r]
        if hh:
            summ[arm] = {"hole": round(float(np.nanmean(hh)), 3),
                         "nonhole": round(float(np.nanmean(nn)), 3),
                         "gap": round(float(np.nanmean(hh) - np.nanmean(nn)), 3)}
    bh, bn = summ["identity"]["hole"], summ["identity"]["nonhole"]

    print(f"{'臂':<12}{'洞区':>8}{'非洞区':>9}{'gap':>8}{'Δ洞区':>9}{'Δ非洞区':>10}{'Δgap':>8}")
    print("-" * 66)
    for arm, s in summ.items():
        print(f"{arm:<12}{s['hole']:>8.2f}{s['nonhole']:>9.2f}{s['gap']:>8.2f}"
              f"{s['hole']-bh:>9.2f}{s['nonhole']-bn:>10.2f}{s['gap']-summ['identity']['gap']:>8.2f}")

    json.dump({"per_scene": rows, "summary": summ},
              open(os.path.join(EVAL, "hole_region_v9.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n洞区平均占比 {np.mean([r['hole_frac'] for r in rows])*100:.2f}%")
    print(f"掩膜 -> {MASKDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
