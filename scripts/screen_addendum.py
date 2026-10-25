#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Addendum checks for the operator screening (ASCII-only).

Two things the main run did not separate:
  (a) within-domain CONSTANT baseline (needed to read the LOSO MAEs honestly)
  (b) for cross-domain winners (omega_frac / C_I / R_abs_mean): is the gain within-test-domain
      RANKING skill (an operator) or only domain-level identification (a prior)?
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
sys.path.insert(0, os.path.join(P, "RDNet"))

from sklearn.linear_model import Ridge  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SCALARS = ["rho", "g", "gain_dev", "C_I", "R_abs_mean", "omega_frac", "neg_frac", "omega_depth"]


def to_scene(vals, scenes):
    ks = sorted(set(scenes))
    return ks, np.array([np.mean([v for v, s in zip(vals, scenes) if s == k]) for k in ks])


def main():
    d = np.load("/tmp/opdata_partial.npz", allow_pickle=True)
    domain = d["domain"].astype(str); scene = d["scene"].astype(str); src = d["src"].astype(str)
    f1 = d["feat_f1"].astype(np.float64)

    from models.arch.classifier import PretrainedConvNext
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = PretrainedConvNext("convnext_small_in22k").to(dev)
    ck = torch.load(os.path.join(P, "pretrained/cls_model.pth"), map_location="cpu")
    m.load_state_dict(ck.get("icnn", ck.get("model", ck)), strict=False)
    m.eval()
    x = torch.from_numpy(d["I224"]).permute(0, 3, 1, 2).float().div(255.0).to(dev)
    gs, cs = [], []
    with torch.no_grad():
        for i in range(0, x.shape[0], 32):
            bb = F.interpolate(x[i:i + 32], size=(224, 224), mode="bilinear", align_corners=True)
            g = m.model(bb); gs.append(g.cpu().numpy()); cs.append(m.head(g).cpu().numpy())
    f2 = np.concatenate([np.concatenate(gs, 0), np.concatenate(cs, 0)], 1).astype(np.float64)

    def fitpred(Xtr, ytr, Xte, alpha=1.0):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        return Ridge(alpha=alpha).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)

    print("=========== (a) within-domain: predictor vs CONSTANT baseline ===========")
    print("%-6s %-12s %9s %9s %9s %9s" % ("domain", "target", "spread", "constMAE", "f2MAE", "verdict"))
    for dom in ("apr", "may"):
        msk = domain == dom
        sc = scene[msk]
        ks = sorted(set(sc))
        for t in SCALARS:
            y = d["y_" + t][msk].astype(np.float64)
            _, ytrue = to_scene(y, sc)
            # constant baseline: predict the mean of the OTHER scenes
            cmae = []
            for k in ks:
                tr = np.array([s != k for s in sc])
                cmae.append(abs(float(np.mean(y[tr])) - float(np.mean(y[~tr]))))
            cmae = float(np.mean(cmae))
            preds, truth = [], []
            for k in ks:
                tr = np.array([s != k for s in sc]); te = ~tr
                pr = fitpred(f2[msk][tr], y[tr], f2[msk][te])
                preds.append(float(pr.mean())); truth.append(float(y[te].mean()))
            fmae = float(np.mean(np.abs(np.array(preds) - np.array(truth))))
            verdict = "SKILL" if fmae < 0.9 * cmae else "no"
            print("%-6s %-12s %9.4f %9.4f %9.4f %9s" % (dom, t, float(np.std(ytrue)), cmae, fmae, verdict))

    print("\n=========== (b) cross-domain winners: within-test ranking skill? ===========")
    tr = src == "1020"
    te = domain == "1019test"
    print("%-12s %8s %8s %10s %10s %s" % ("target", "constMAE", "f2MAE", "spearman_in", "spread_te", "reading"))
    for t in ("omega_frac", "C_I", "R_abs_mean", "rho", "g"):
        y = d["y_" + t].astype(np.float64)
        pr = fitpred(f2[tr], y[tr], f2[te])
        ks, ytrue = to_scene(y[te], scene[te])
        _, ypred = to_scene(pr, scene[te])
        const = float(np.mean(y[tr]))
        cmae = float(np.mean(np.abs(const - ytrue)))
        fmae = float(np.mean(np.abs(ypred - ytrue)))
        sp = float(spearmanr(ypred, ytrue).statistic) if len(set(ypred)) > 1 else float("nan")
        spread = float(np.std(ytrue))
        if sp >= 0.5:
            rd = "OPERATOR (within-domain ranking skill)"
        elif fmae < 0.8 * cmae:
            rd = "prior only (domain-level, no ranking)"
        else:
            rd = "no skill"
        print("%-12s %8.4f %8.4f %10.3f %10.4f %s" % (t, cmae, fmae, sp, spread, rd))

    print("\n=========== (c) does the test domain really differ? ===========")
    for t in ("omega_frac", "C_I", "rho"):
        y = d["y_" + t].astype(np.float64)
        print("  %-12s train(1020) mean=%.4f +/- %.4f | test(1019) mean=%.4f +/- %.4f"
              % (t, float(y[tr].mean()), float(y[tr].std()), float(y[te].mean()), float(y[te].std())))


if __name__ == "__main__":
    main()
