#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inside-Omega vs outside-Omega signed error per arm (ASCII-only).

Question raised by the user: why not give BOTH brightness AND location?
If the arms are too bright OUTSIDE Omega but too dark INSIDE Omega, the two corrections are in
TENSION: a global brightness fix would make Omega worse. That would be a positive, measurable
reason to provide both conditions.

Omega is recomputed from (I, GT) with the repo's omega_of, same definition as the training pool.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import cv2

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
os.chdir(os.path.join(P, "scripts"))
from eval_g2_baseline import omega_of  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

TE = os.path.join(P, "data/test_sets/real_pair_1019_testsplit12")


def luma(x):
    return 0.299 * x[..., 2] + 0.587 * x[..., 1] + 0.114 * x[..., 0]


def main():
    scenes = [s for s in sorted(os.listdir(TE)) if os.path.isdir(os.path.join(TE, s))]
    # cache Omega per scene
    OMs = {}
    for sc in scenes:
        ip = os.path.join(TE, sc)
        gp = os.path.join(ip, "GT.png")
        ins = [f for f in sorted(os.listdir(ip)) if f.startswith("I_") and f.endswith(".png")]
        if not (os.path.exists(gp) and ins):
            continue
        G = cv2.imread(gp).astype(np.float64)
        I = cv2.imread(os.path.join(ip, ins[0])).astype(np.float64)
        if I.shape[:2] != G.shape[:2]:
            I = cv2.resize(I, (G.shape[1], G.shape[0]))
        rec = omega_of(I, G)
        OMs[sc] = (rec["Om"] if rec is not None else np.zeros(I.shape[:2], bool))
    fr = {s: float(m.mean()) for s, m in OMs.items()}
    print("Omega area fraction per scene: median %.3f  range [%.3f, %.3f]"
          % (float(np.median(list(fr.values()))), min(fr.values()), max(fr.values())))

    rows = []
    for d in sorted(glob.glob(os.path.join(P, "runs/pred/real_pair_1019_dom12_*"))):
        arm = os.path.basename(d).replace("real_pair_1019_dom12_", "")
        din, dout = [], []
        for sc in scenes:
            if sc not in OMs:
                continue
            ip = os.path.join(TE, sc)
            gp = os.path.join(ip, "GT.png")
            hit = [x for x in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, x)) and x.startswith(sc)]
            if not hit:
                continue
            pp = os.path.join(d, hit[0], "t_hat.png")
            if not os.path.exists(pp):
                continue
            G = cv2.imread(gp).astype(np.float32)
            Pt = cv2.imread(pp).astype(np.float32)
            if Pt.shape[:2] != G.shape[:2]:
                Pt = cv2.resize(Pt, (G.shape[1], G.shape[0]))
            m = OMs[sc]
            if m.sum() < 500 or (~m).sum() < 500:
                continue
            e = luma(Pt) - luma(G)          # signed: >0 = too bright
            din.append(float(e[m].mean()))
            dout.append(float(e[~m].mean()))
        if din:
            rows.append((arm, float(np.mean(din)), float(np.mean(dout)), len(din)))
    rows.sort(key=lambda r: r[2] - r[1])
    print()
    print("=== signed error: prediction - GT  (positive = too bright) ===")
    print("%-22s %12s %12s %10s" % ("arm", "inside Om", "outside Om", "gap"))
    for a, i, o, n in rows:
        print("%-22s %+12.2f %+12.2f %+10.2f" % (a, i, o, o - i))
    ai = np.array([r[1] for r in rows]); ao = np.array([r[2] for r in rows])
    print()
    print("mean inside = %+.2f gray | mean outside = %+.2f gray | mean gap = %+.2f"
          % (ai.mean(), ao.mean(), (ao - ai).mean()))
    print("arms with inside<0 and outside>0 (TENSION): %d / %d"
          % (int(((ai < 0) & (ao > 0)).sum()), len(rows)))
    print("arms with both >0 (uniformly too bright)   : %d / %d"
          % (int(((ai > 0) & (ao > 0)).sum()), len(rows)))
    if len(rows) > 3:
        rho = spearmanr(ai, ao).statistic
        print("Spearman(inside, outside) = %+.3f" % float(rho))


if __name__ == "__main__":
    main()
