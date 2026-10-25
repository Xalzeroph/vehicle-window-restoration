#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C) Is the spatial gain field a STABLE illumination pattern, or does it vary per frame?
ASCII-only.

Idea: fit the tone field (G x G bilinear + offset) that maps I -> its scene GT.
If the field is a property of the capture setup (illumination / vignetting), it should be
nearly IDENTICAL across frames of the same scene.
If it differs frame to frame, it is scene content / reflection variation, not illumination.

Report, per scene: mean pairwise correlation of the fitted fields, and their relative spread.
"""
from __future__ import annotations

import json
import os

import numpy as np
import cv2

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
G = 4
SUBSAMPLE = 80000


def bilinear_basis(h, w, G):
    ys, xs = np.mgrid[0:h, 0:w]
    u = xs.ravel() / max(w - 1, 1) * (G - 1)
    v = ys.ravel() / max(h - 1, 1) * (G - 1)
    i0 = np.clip(np.floor(u).astype(int), 0, G - 1); j0 = np.clip(np.floor(v).astype(int), 0, G - 1)
    i1 = np.clip(i0 + 1, 0, G - 1); j1 = np.clip(j0 + 1, 0, G - 1)
    fu = u - i0; fv = v - j0
    N = u.size
    A = np.zeros((N, G * G), np.float64)
    idx = lambda i, j: i * G + j  # noqa: E731
    r = np.arange(N)
    A[r, idx(i0, j0)] += (1 - fu) * (1 - fv)
    A[r, idx(i0, j1)] += (1 - fu) * fv
    A[r, idx(i1, j0)] += fu * (1 - fv)
    A[r, idx(i1, j1)] += fu * fv
    return A


def fit_field(I, T, rng):
    h, w = I.shape
    A = bilinear_basis(h, w, G)
    flat = I.ravel().astype(np.float64); tgt = T.ravel().astype(np.float64)
    sub = rng.choice(flat.size, size=min(SUBSAMPLE, flat.size), replace=False)
    D = np.concatenate([A[sub] * flat[sub][:, None], np.ones((sub.size, 1))], 1)
    coef, *_ = np.linalg.lstsq(D, tgt[sub], rcond=None)
    return coef  # G*G gains + 1 offset


def main():
    base = os.path.join(P, "data/train_sets/real_pairs_1020")
    meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
    by = {}
    for k, v in meta.items():
        by.setdefault(v["scene"], []).append(k)
    rng = np.random.RandomState(0)
    print("=== C) stability of the fitted tone field across frames of the same scene ===")
    print("    (high corr / low spread  =>  the field is a FIXED illumination pattern)")
    print("%-24s %5s %10s %12s %12s" % ("scene", "n", "meanCorr", "relSpread", "meanGain"))
    allc, alls = [], []
    for sc, ks in sorted(by.items()):
        ks = sorted(ks)[:12]
        coefs = []
        for k in ks:
            I = cv2.imread(os.path.join(base, "I", k + ".png"))
            T = cv2.imread(os.path.join(base, "T", k + ".png"))
            if I is None or T is None:
                continue
            Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
            Tg = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
            coefs.append(fit_field(Ig, Tg, rng))
        if len(coefs) < 3:
            continue
        C = np.stack(coefs)                        # (n, 17)
        g = C[:, :-1]
        cs = []
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                cs.append(float(np.corrcoef(g[a], g[b])[0, 1]))
        mc = float(np.mean(cs))
        relspread = float(g.std(0).mean() / max(abs(g.mean()), 1e-6))
        allc.append(mc); alls.append(relspread)
        print("%-24s %5d %10.3f %12.3f %12.3f" % (sc[:24], len(coefs), mc, relspread, float(g.mean())))
    print()
    print("overall mean corr = %.3f | mean relSpread = %.3f" % (np.mean(allc), np.mean(alls)))
    print()
    if np.mean(allc) > 0.9 and np.mean(alls) < 0.15:
        print("READING: the field is STABLE across frames -> it is a fixed illumination/vignetting pattern.")
    else:
        print("READING: the field VARIES frame to frame -> it is not a fixed illumination pattern;")
        print("         it reflects scene/reflection changes, so 'spatial illumination' is NOT established.")


if __name__ == "__main__":
    main()
