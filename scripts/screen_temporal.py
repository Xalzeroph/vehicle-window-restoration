#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal-misalignment audit + re-screening on temporally-near pairs (ASCII-only).

Finding (2026-10-25): in real_pairs_1020 each scene has ONE GT still, captured at gt_frame,
which is *after* the I-frame range. Some pairs are up to ~1450 frames (~48 s) away from the GT.
So R = I - g*T mixes reflection with scene motion / lighting drift.

This script:
  1. prints the |frame - gt_frame| distribution per domain
  2. re-runs the decisive screening comparisons on NEAR pairs only
  3. reports whether predictability changes -> is the earlier negative a misalignment artifact?
"""
from __future__ import annotations

import json
import os
import re
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


def frame_index(case):
    m = re.search(r"_f(\d+)$", case)
    return int(m.group(1)) if m else None


def main():
    meta = json.load(open(os.path.join(P, "data/train_sets/real_pairs_1020/meta.json"), encoding="utf-8"))
    d = np.load("/tmp/opdata_partial.npz", allow_pickle=True)
    cases = d["cases"].astype(str); domain = d["domain"].astype(str)
    scene = d["scene"].astype(str); src = d["src"].astype(str)
    f1 = d["feat_f1"].astype(np.float64)

    # ---- delta frame ----
    delta = np.full(len(cases), -1.0)
    for i, c in enumerate(cases):
        if c in meta:
            fi = frame_index(c)
            gt = meta[c].get("gt_frame")
            if fi is not None and gt is not None:
                delta[i] = abs(fi - int(gt))
    print("=========== |frame - gt_frame| distribution (1020 only) ===========")
    m1020 = delta >= 0
    for dom in ("apr", "may", "jun"):
        v = delta[(domain == dom) & m1020]
        if v.size == 0:
            continue
        q = np.percentile(v, [10, 50, 90, 100])
        print("  %-4s n=%-3d  p10=%4.0f  med=%4.0f  p90=%4.0f  max=%4.0f | <=50: %d (%.0f%%)  >200: %d (%.0f%%)"
              % (dom, v.size, q[0], q[1], q[2], q[3],
                 int((v <= 50).sum()), 100 * (v <= 50).mean(),
                 int((v > 200).sum()), 100 * (v > 200).mean()))
    print("  : n=%d  med=%.0f  max=%.0f | <=50: %d (%.0f%%)"
          % (m1020.sum(), np.median(delta[m1020]), delta[m1020].max(),
             int((delta[m1020] <= 50).sum()), 100 * (delta[m1020] <= 50).mean()))

    # ---- features ----
    from models.arch.classifier import PretrainedConvNext
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PretrainedConvNext("convnext_small_in22k").to(dev)
    ck = torch.load(os.path.join(P, "pretrained/cls_model.pth"), map_location="cpu")
    net.load_state_dict(ck.get("icnn", ck.get("model", ck)), strict=False)
    net.eval()
    x = torch.from_numpy(d["I224"]).permute(0, 3, 1, 2).float().div(255.0).to(dev)
    gs, cs = [], []
    with torch.no_grad():
        for i in range(0, x.shape[0], 32):
            bb = F.interpolate(x[i:i + 32], size=(224, 224), mode="bilinear", align_corners=True)
            g = net.model(bb); gs.append(g.cpu().numpy()); cs.append(net.head(g).cpu().numpy())
    f2 = np.concatenate([np.concatenate(gs, 0), np.concatenate(cs, 0)], 1).astype(np.float64)

    def fitpred(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        return Ridge(alpha=1.0).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)

    def within(dom, near_thr):
        msk = (domain == dom) & (delta >= 0) & (delta <= near_thr)
        if msk.sum() < 20:
            return None
        sc = scene[msk]; ks = sorted(set(sc))
        out = {}
        for t in SCALARS:
            y = d["y_" + t][msk].astype(np.float64)
            cmae, fmae, pr_, tr_ = [], [], [], []
            for k in ks:
                trm = np.array([s != k for s in sc]); tem = ~trm
                if trm.sum() < 5 or tem.sum() < 1:
                    continue
                cmae.append(abs(float(y[trm].mean()) - float(y[tem].mean())))
                p = fitpred(f2[msk][trm], y[trm], f2[msk][tem])
                pr_.append(float(p.mean())); tr_.append(float(y[tem].mean()))
            if len(cmae) >= 3:
                out[t] = (float(np.mean(cmae)), float(np.mean(np.abs(np.array(pr_) - np.array(tr_)))),
                          float(spearmanr(pr_, tr_).statistic) if len(set(pr_)) > 1 else float("nan"),
                          len(cmae))
        return out

    print("\n=========== within-domain: near pairs only (|delta| <= 50) ===========")
    print("%-6s %-12s %9s %9s %9s %8s %s" % ("dom", "target", "constMAE", "f2MAE", "gain%", "rho_s", "verdict"))
    for dom in ("apr", "may"):
        r_all = within(dom, 10 ** 9)
        r_near = within(dom, 50)
        if not r_near:
            print("  %-6s (too few near pairs)" % dom)
            continue
        for t in SCALARS:
            ca, fa, sa, n = r_near[t]
            ca_all = r_all[t][0] if (r_all and t in r_all) else float("nan")
            gain = 100 * (ca - fa) / max(ca, 1e-9)
            vd = "SKILL" if fa < 0.9 * ca else "no"
            print("%-6s %-12s %9.4f %9.4f %+8.1f %8.2f %s   (all-pairs const=%.4f)"
                  % (dom, t, ca, fa, gain, sa, vd, ca_all))

    print("\n=========== cross-domain within-test Spearman, near pairs only ===========")
    tr = (src == "1020") & (delta >= 0) & (delta <= 50)
    te = domain == "1019test"
    print("train used: %d frames (%d scenes)" % (tr.sum(), len(set(scene[tr]))))
    for t in ("rho", "C_I", "omega_frac", "R_abs_mean"):
        y = d["y_" + t].astype(np.float64)
        if tr.sum() < 20:
            print("  %-12s (too few near train pairs)" % t); continue
        p = fitpred(f2[tr], y[tr], f2[te])
        ks = sorted(set(scene[te]))
        yt = np.array([float(np.mean([v for v, s in zip(y[te], scene[te]) if s == k])) for k in ks])
        pt = np.array([float(np.mean([v for v, s in zip(p, scene[te]) if s == k])) for k in ks])
        sp = float(spearmanr(pt, yt).statistic)
        cmae = float(np.mean(np.abs(float(np.mean(y[tr])) - yt)))
        fmae = float(np.mean(np.abs(pt - yt)))
        print("  %-12s constMAE=%.4f  f2MAE=%.4f  spearman_in_test=%+.3f" % (t, cmae, fmae, sp))


if __name__ == "__main__":
    main()
