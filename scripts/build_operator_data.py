#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build labels + features for the operator screening (ASCII-only).

Data reality (verified 2026-10-25):
  real_pairs_1020: 300 frames -> only **13 unique T** (one GT still per scene).
  So per-pair *scene-level* labels have effective n = 13 scenes.
  Per-pair labels that vary within a scene are functions of I given T, i.e. partly trivial.

Outputs /tmp/opdata.npz with:
  cases, domain, scene
  y_<scalar>: rho, g, gain_dev, C_I, R_abs_mean, omega_frac, neg_frac, omega_depth
  y_<map64>:  omega, sign, depth   (64x64 float, NaN-free)
  feat_f1: simple image statistics (D1)
  feat_f2: frozen ConvNeXt 768 + 6-dim code (D2)
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import cv2

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
os.chdir(os.path.join(P, "scripts"))

from eval_g2_baseline import omega_of, G  # noqa: E402

SC = 64  # map resolution


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


def luma255(im):
    """Linear-luma at 0-255 scale, gamma-consistent with omega_of."""
    x = np.clip(im, 0, 255).astype(np.float64) / 255.0
    lin = x ** G
    return (0.299 * lin[..., 2] + 0.587 * lin[..., 1] + 0.114 * lin[..., 0])


def robust_gain(I, T):
    """Per-channel median ratio (same spirit as build_train_real_pairs.robust_gain)."""
    a = np.clip(I, 0, 255).astype(np.float64) / 255.0
    b = np.clip(T, 0, 255).astype(np.float64) / 255.0
    ok = b > (10 / 255.0)
    gs = []
    for c in range(3):
        r = a[..., c][ok[..., c]] / np.maximum(b[..., c][ok[..., c]], 1e-8)
        gs.append(float(np.median(r)) if r.size else 1.0)
    return np.array(gs, dtype=np.float64)


def down(a, sc=SC):
    return cv2.resize(np.asarray(a, np.float32), (sc, sc), interpolation=cv2.INTER_AREA)


def feat_f1(I):
    """Simple, deployment-cheap image statistics."""
    x = np.clip(I, 0, 255).astype(np.float32) / 255.0
    L = luma255(I)  # 0..~1
    g = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(g, cv2.CV_32F)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    blur = cv2.GaussianBlur(g, (0, 0), 3.0)
    hf = g - blur
    q = np.percentile(g, [1, 5, 25, 50, 75, 95, 99])
    return np.concatenate([
        [g.mean(), g.std(), L.mean(), L.std()],
        q,
        [x[..., 0].mean(), x[..., 1].mean(), x[..., 2].mean()],
        [x[..., 0].std(), x[..., 1].std(), x[..., 2].std()],
        [float(lap.var()), float(np.abs(gx).mean()), float(np.abs(gy).mean()),
         float(hf.std()), float(np.abs(hf).mean())],
        [float((g < 0.04).mean()), float((g > 0.96).mean())],
        [float(np.log(g.mean() + 1e-6)), float(np.log(g.std() + 1e-6))],
    ]).astype(np.float32)


def process_pair(im_path, gt_path, extra_meta=None):
    I = read(im_path)
    T = read(gt_path)
    if I is None or T is None:
        return None
    if I.shape[:2] != T.shape[:2]:
        T = cv2.resize(T, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
    out = {}
    # --- refit gain / signed R on disk convention: R = I - g*T ---
    gch = robust_gain(I, T)
    g = float(gch.mean())
    R = I.astype(np.float64) - T.astype(np.float64) * gch.reshape(1, 1, 3)
    Rs = R.mean(axis=2)  # signed, 0-255
    # --- photometric ratio rho (linear-luma) ---
    rho = float(luma255(T).mean() / max(luma255(I).mean(), 1e-9))
    # --- omega definition quantities ---
    rec = omega_of(I.astype(np.float64), T.astype(np.float64))
    if rec is None:
        Om = np.zeros(I.shape[:2], bool); S = np.zeros(I.shape[:2]); bg = np.zeros(I.shape[:2])
    else:
        Om, S, bg = rec["Om"], rec["S"], rec["bg"]
    # --- scalars ---
    out["rho"] = rho
    out["g"] = g
    out["gain_dev"] = float(np.abs(gch - 1).mean())
    out["neg_frac"] = float((Rs < 0).mean())
    out["omega_frac"] = float(Om.mean())
    out["omega_depth"] = float(np.median((bg - Rs)[Om])) if Om.sum() > 500 else 0.0
    out["R_abs_mean"] = float(np.abs(Rs).mean())
    out["C_I"] = float(np.percentile(np.abs(Rs), 90))
    # --- maps (64x64) ---
    out["map_omega"] = down(Om.astype(np.float32))
    out["map_sign"] = down((Rs < 0).astype(np.float32))
    depth = np.clip((bg - Rs) / 40.0, 0, 1)
    out["map_depth"] = down(depth.astype(np.float32))
    out["feat_f1"] = feat_f1(I)
    out["_I224"] = cv2.resize(I, (224, 224), interpolation=cv2.INTER_AREA)
    return out


def collect():
    recs = []
    # ---- training pool 1020 (300 frames / 13 scenes) ----
    # NOTE (2026-10-25): real_pairs_1019split (331 pairs / 3 scenes) is DELIBERATELY SKIPPED.
    # It has only 3 scenes -> not decidable in the screening gates, and omega_of is ~2 s/pair.
    base = os.path.join(P, "data/train_sets/real_pairs_1020")
    meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
    n = 0
    for case, v in sorted(meta.items()):
        ip = os.path.join(base, "I", case + ".png"); tp = os.path.join(base, "T", case + ".png")
        if not (os.path.exists(ip) and os.path.exists(tp)):
            continue
        r = process_pair(ip, tp)
        if r is None:
            continue
        r.update(case=case, domain=v.get("domain", "?"), scene=v.get("scene", "?"), src="1020")
        recs.append(r)
        n += 1
        if n % 25 == 0:
            print("[1020] %d/300" % n, flush=True)
    # ---- held-out test set: 12 clean scenes ----
    te = os.path.join(P, "data/test_sets/real_pair_1019_testsplit12")
    for sc in sorted(os.listdir(te)):
        d = os.path.join(te, sc)
        if not os.path.isdir(d):
            continue
        gp = os.path.join(d, "GT.png")
        ins = [f for f in sorted(os.listdir(d)) if f.startswith("I_") and f.endswith(".png")]
        if not (os.path.exists(gp) and ins):
            continue
        r = process_pair(os.path.join(d, ins[0]), gp)
        if r is None:
            continue
        r.update(case=sc, domain="1019test", scene=sc, src="1019test")
        recs.append(r)
        print("[1019test] %s done (total=%d)" % (sc[:24], len(recs)), flush=True)
    return recs


def main():
    recs = collect()
    print("records:", len(recs))
    scal = ["rho", "g", "gain_dev", "C_I", "R_abs_mean", "omega_frac", "neg_frac", "omega_depth"]
    np.savez("/tmp/opdata_partial.npz",
             cases=np.array([r["case"] for r in recs]),
             domain=np.array([r["domain"] for r in recs]),
             scene=np.array([r["scene"] for r in recs]),
             src=np.array([r["src"] for r in recs]),
             feat_f1=np.stack([r["feat_f1"] for r in recs]),
             **{f"y_{k}": np.array([r[k] for r in recs]) for k in scal},
             **{f"m_{k}": np.stack([r[f"map_{k}"] for r in recs]) for k in ("omega", "sign", "depth")},
             I224=np.stack([r["_I224"] for r in recs]))
    print("[saved] /tmp/opdata_partial.npz")
    import collections
    c = collections.Counter((r["domain"], r["scene"]) for r in recs)
    print("domains:", collections.Counter(r["domain"] for r in recs))
    print("scenes:", {d: len({s for dd, s in c if dd == d}) for d in {r["domain"] for r in recs}})


if __name__ == "__main__":
    main()
