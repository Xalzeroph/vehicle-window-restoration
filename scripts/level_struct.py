#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Level vs structure split for existing arms (ASCII-only, no training).

For each arm's prediction on real_pair_1019_testsplit12:
  psnr_raw        : as reported
  psnr_gauge      : after a closed-form global gain g = mu(T)/mu(T_hat)  (level only)
  dGauge          : how many dB the LEVEL alone is worth
  hf_corr         : Pearson correlation of high-pass residual vs GT high-pass (structure)
  lvl_off         : mean-luma offset in gray levels (0-255)

Reading:
  if dGauge is large and positive for many arms -> the photometric level IS the dominant error
  -> a closed-form gauge (remove level from learning) is justified.
  if dGauge ~ 0 -> level is already fine -> the gauge would not help; the earlier -4.8 dB was a
  training artifact, not an architectural one.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import cv2

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
TE = os.path.join(P, "data/test_sets/real_pair_1019_testsplit12")


def luma(x):
    return 0.299 * x[..., 2] + 0.587 * x[..., 1] + 0.114 * x[..., 0]


def hp(x):
    b = cv2.GaussianBlur(x, (0, 0), 3.0)
    return x - b


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def main():
    scenes = [s for s in sorted(os.listdir(TE)) if os.path.isdir(os.path.join(TE, s))]
    arms = {}
    for d in sorted(glob.glob(os.path.join(P, "runs/pred/real_pair_1019_dom12_*"))):
        arms[os.path.basename(d).replace("real_pair_1019_dom12_", "")] = d
    rows = []
    for name, d in arms.items():
        raw, gau, hfc, lvl = [], [], [], []
        for sc in scenes:
            ip = os.path.join(TE, sc)
            gp = os.path.join(ip, "GT.png")
            ins = [f for f in sorted(os.listdir(ip)) if f.startswith("I_") and f.endswith(".png")]
            hit = [x for x in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, x)) and x.startswith(sc)]
            if not (ins and os.path.exists(gp) and hit):
                continue
            pp = os.path.join(d, hit[0], "t_hat.png")
            if not os.path.exists(pp):
                continue
            G = cv2.imread(gp).astype(np.float32)
            Pt = cv2.imread(pp).astype(np.float32)
            if Pt.shape[:2] != G.shape[:2]:
                Pt = cv2.resize(Pt, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_AREA)
            lg, lp = luma(G), luma(Pt)
            g = float(lg.mean() / max(lp.mean(), 1e-6))
            Pt2 = np.clip(Pt * g, 0, 255)
            raw.append(psnr(Pt, G)); gau.append(psnr(Pt2, G))
            lvl.append(float(lp.mean() - lg.mean()))
            a = hp(luma(np.clip(Pt, 0, 255))).ravel(); b = hp(lg).ravel()
            hfc.append(float(np.corrcoef(a, b)[0, 1]))
        if not raw:
            continue
        rows.append((name, float(np.mean(raw)), float(np.mean(gau)),
                     float(np.mean(gau) - np.mean(raw)) * -1.0, float(np.mean(hfc)), float(np.mean(lvl))))
    rows.sort(key=lambda r: -r[5])
    print("=== level vs structure (12 scenes, native res) ===")
    print("%-22s %8s %8s %9s %8s %9s" % ("arm", "PSNR", "PSNR_gauge", "gain_dB", "hf_corr", "level_off"))
    print("   (gain_dB = PSNR after fixing ONLY the global level, minus PSNR as-is)")
    for n, p, pg, gain, hc, lo in rows:
        print("%-22s %8.3f %8.3f %+9.3f %8.3f %+9.2f" % (n, p, pg, -gain * -1.0 if False else (pg - p), hc, lo))
    print()
    gains = np.array([r[2] - r[1] for r in rows])
    print("mean gain from level-only correction = %+.3f dB  (max %+.3f, min %+.3f)"
          % (gains.mean(), gains.max(), gains.min()))
    print("mean |level offset| = %.2f gray levels" % np.mean([abs(r[5]) for r in rows]))
    print()
    if gains.mean() > 0.5:
        print("READING: the level IS a large part of the error -> closed-form gauge is justified.")
    elif gains.mean() > 0.1:
        print("READING: level is a modest part of the error -> gauge helps a little, not decisive.")
    else:
        print("READING: level is already near-correct -> gauge will NOT help; earlier -4.8 dB was training.")


if __name__ == "__main__":
    main()
