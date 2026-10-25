#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two decisive checks (ASCII-only, zero GPU):

A) Is the I->T photometric mismatch GLOBAL (uniform) or SPATIAL (blocky)?
   Fit a GxG bilinear gain field + global offset to map I->T, sweep G.
   G=1 (+offset) is exactly the "2-parameter affine" of 2026.10.08 C1.
   If PSNR saturates at G=1 -> global only. If it keeps rising -> spatial component exists.

B) Do the fine-tuned arms beat "identity + per-image optimal tone"?
   Their C1 showed official RDNet's net gain over identity+affine is ~0.
   Same question for OUR arms, on real_pair_1019_testsplit12.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import cv2

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
GS = [1, 2, 3, 4, 6, 8]
SUBSAMPLE = 150000


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def bilinear_basis(xy, G):
    """xy: (N,2) in [0,1]. Returns (N, G*G) bilinear weights."""
    u = xy[:, 0] * (G - 1)
    v = xy[:, 1] * (G - 1)
    i0 = np.clip(np.floor(u).astype(int), 0, G - 1)
    j0 = np.clip(np.floor(v).astype(int), 0, G - 1)
    i1 = np.clip(i0 + 1, 0, G - 1)
    j1 = np.clip(j0 + 1, 0, G - 1)
    fu = (u - i0)[:, None]
    fv = (v - j0)[:, None]
    N = xy.shape[0]
    A = np.zeros((N, G * G), np.float64)
    idx = lambda i, j: i * G + j  # noqa: E731
    rows = np.arange(N)
    A[rows, idx(i0, j0)] += (1 - fu[:, 0]) * (1 - fv[:, 0])
    A[rows, idx(i0, j1)] += (1 - fu[:, 0]) * fv[:, 0]
    A[rows, idx(i1, j0)] += fu[:, 0] * (1 - fv[:, 0])
    A[rows, idx(i1, j1)] += fu[:, 0] * fv[:, 0]
    return A


def fit_tonemap(I, T, G, rng):
    """Fit T ~ I * field_G(x,y) + b. Returns (fields, offset) and psnr."""
    h, w = I.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    xy = np.stack([(xs.ravel() / max(w - 1, 1)), (ys.ravel() / max(h - 1, 1))], 1)
    flat = I.ravel().astype(np.float64)
    tgt = T.ravel().astype(np.float64)
    n = flat.size
    sub = rng.choice(n, size=min(SUBSAMPLE, n), replace=False)
    A = bilinear_basis(xy, G)
    Asub = A[sub] * flat[sub][:, None]
    D = np.concatenate([Asub, np.ones((sub.size, 1))], 1)
    coef, *_ = np.linalg.lstsq(D, tgt[sub], rcond=None)
    pred = A * flat[:, None] @ coef[:-1] + coef[-1]
    pred = np.clip(pred, 0, 255)
    return psnr(pred, tgt), pred.reshape(I.shape), G * G + 1


def part_a():
    base = os.path.join(P, "data/train_sets/real_pairs_1020")
    cases = sorted(os.listdir(os.path.join(base, "I")))[:60]
    rng = np.random.RandomState(0)
    print("=== A) sweep field resolution: does the mismatch saturate at 'global'? ===")
    print("    (G=1 + offset  ==  the 2-parameter affine of 2026.10.08 C1)")
    print("%5s %8s %10s %10s" % ("G", "params", "meanPSNR", "delta vs G=1"))
    rows = {}
    for c in cases[:60]:
        cp = c[:-4] if c.endswith(".png") else c
        I = cv2.imread(os.path.join(base, "I", c))
        T = cv2.imread(os.path.join(base, "T", c))
        if I is None or T is None:
            continue
        if I.shape[:2] != T.shape[:2]:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))
        Il = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        Tl = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
        for G in GS:
            p, _, np_ = fit_tonemap(Il, Tl, G, rng)
            rows.setdefault((G, np_), []).append(p)
    ref = None
    for (G, npr), v in sorted(rows.items()):
        m = float(np.mean(v))
        if G == 1:
            ref = m
        print("%5d %8d %10.3f %10.3f" % (G, npr, m, m - (ref if ref else m)))
    print("    reading: if the last columns stop rising near G=1..2 -> mismatch is GLOBAL.")


def part_b():
    te = os.path.join(P, "data/test_sets/real_pair_1019_testsplit12")
    scenes = [s for s in sorted(os.listdir(te)) if os.path.isdir(os.path.join(te, s))]
    arms = {}
    for d in sorted(glob.glob(os.path.join(P, "runs/pred/real_pair_1019_dom12_*"))):
        arms[os.path.basename(d).replace("real_pair_1019_dom12_", "")] = d
    if not arms:
        print("(no arm predictions found)")
        return
    rng = np.random.RandomState(0)
    acc = {k: [] for k in ["identity", "identity+affine(G1)", "identity+field(G4)"]}
    for a in arms:
        acc[a] = []
    for sc in scenes:
        ip = os.path.join(te, sc)
        gp = os.path.join(ip, "GT.png")
        ins = [f for f in sorted(os.listdir(ip)) if f.startswith("I_") and f.endswith(".png")]
        if not (os.path.exists(gp) and ins):
            continue
        Gt = cv2.imread(gp).astype(np.float32)
        Ic = cv2.imread(os.path.join(ip, ins[0])).astype(np.float32)
        if Ic.shape[:2] != Gt.shape[:2]:
            Ic = cv2.resize(Ic, (Gt.shape[1], Gt.shape[0]))
        Ig = cv2.cvtColor(Ic.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        Tg = cv2.cvtColor(Gt.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        acc["identity"].append(psnr(Ic, Gt))
        p1, m1, _ = fit_tonemap(Ig, Tg, 1, rng)
        p4, m4, _ = fit_tonemap(Ig, Tg, 4, rng)
        acc["identity+affine(G1)"].append(p1)
        acc["identity+field(G4)"].append(p4)
        for a, d in arms.items():
            hit = [x for x in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, x)) and x.startswith(sc)]
            if not hit or not os.path.exists(os.path.join(d, hit[0], "t_hat.png")):
                acc[a].append(np.nan)
                continue
            Pr = cv2.imread(os.path.join(d, hit[0], "t_hat.png")).astype(np.float32)
            if Pr.shape[:2] != Gt.shape[:2]:
                Pr = cv2.resize(Pr, (Gt.shape[1], Gt.shape[0]))
            acc[a].append(psnr(Pr, Gt))
    print("\n=== B) our arms vs 'identity + per-image optimal tone' (12 scenes) ===")
    print("%-28s %8s %6s" % ("arm", "PSNR", "vs id+aff"))
    base = float(np.nanmean(acc["identity+affine(G1)"]))
    order = ["identity", "identity+affine(G1)", "identity+field(G4)"] + \
            [k for k in arms if k not in ("identity",)]
    for k in order:
        v = np.array(acc[k], dtype=np.float64)
        if np.all(np.isnan(v)):
            print("%-28s %8s %6s   (no pred)" % (k, "-", "-"))
            continue
        m = float(np.nanmean(v))
        print("%-28s %8.3f %+6.3f" % (k, m, m - base))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if "a" in which:
        part_a()
    if "b" in which:
        part_b()
