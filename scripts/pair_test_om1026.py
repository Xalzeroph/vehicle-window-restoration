#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired scene-level comparison for the Omega-mask mechanism test. ASCII-only."""
from __future__ import annotations

import os
import sys

import numpy as np
import cv2

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")

ARMS = {
    "real":   os.path.join(PRED, "real_pair_1019_om1026_real_s101"),
    "random": os.path.join(PRED, "real_pair_1019_om1026_random_s101"),
    "C0":     os.path.join(PRED, "real_pair_1019_abl_C0_s101"),
    "identity": None,
}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def main():
    scenes = [s for s in sorted(os.listdir(PAIR)) if os.path.isdir(os.path.join(PAIR, s))]
    print("scenes:", len(scenes))
    res = {k: [] for k in ARMS}
    for sc in scenes:
        ip = os.path.join(PAIR, sc)
        gp = os.path.join(ip, "GT.png")
        ins = [f for f in sorted(os.listdir(ip)) if f.startswith("I_") and f.endswith(".png")]
        if not (os.path.exists(gp) and ins):
            continue
        G = cv2.imread(gp)
        I = cv2.imread(os.path.join(ip, ins[0]))
        for k, d in ARMS.items():
            if d is None:
                res[k].append(psnr(I, G)); continue
            hit = [x for x in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, x)) and x.startswith(sc)]
            if not hit or not os.path.exists(os.path.join(d, hit[0], "t_hat.png")):
                res[k].append(np.nan); continue
            Pr = cv2.imread(os.path.join(d, hit[0], "t_hat.png"))
            if Pr.shape[:2] != G.shape[:2]:
                Pr = cv2.resize(Pr, (G.shape[1], G.shape[0]))
            res[k].append(psnr(Pr, G))

    for k in ARMS:
        v = np.array(res[k], dtype=float)
        print("  %-9s mean PSNR = %.3f  (n=%d)" % (k, np.nanmean(v), int(np.isfinite(v).sum())))

    print()
    for a, b in (("real", "random"), ("real", "C0"), ("random", "C0")):
        x = np.array(res[a], float); y = np.array(res[b], float)
        ok = np.isfinite(x) & np.isfinite(y)
        d = x[ok] - y[ok]
        n = d.size
        m = d.mean(); sd = d.std(ddof=1)
        se = sd / max(np.sqrt(n), 1e-9)
        try:
            from scipy import stats
            t = m / se if se > 0 else float("nan")
            p = float(2 * stats.t.sf(abs(t), n - 1))
            tc = float(stats.t.ppf(0.975, n - 1))
        except Exception:
            t = m / se if se > 0 else float("nan")
            p = float("nan"); tc = 2.145
        lo, hi = m - tc * se, m + tc * se
        print("%-7s - %-7s : delta = %+0.3f dB  sd = %.3f  95%% CI [%+0.3f, %+0.3f]  t = %+.2f  p = %.4f  win %d/%d"
              % (a, b, m, sd, lo, hi, t, p, int((d > 0).sum()), n))
    print()
    d = np.array(res["real"], float) - np.array(res["random"], float)
    ok = np.isfinite(d)
    print("per-scene delta (real - random):", np.round(d[ok], 2).tolist())


if __name__ == "__main__":
    main()
