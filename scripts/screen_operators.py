#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operator screening: can any pair-derived quantity be predicted from a single image?

Reads /tmp/opdata_partial.npz (built by build_operator_data.py), computes F2 (frozen
ConvNeXt) features, then screens:
  scalars : Ridge on {muI(trivial), f1, f2, f1+f2}, within-domain leave-scene-out + cross-domain
  maps    : tiny conv net on 64x64, LOSO within 1020 + cross-domain on 1019test
Controls  :
  G4  omega map must reproduce M1-level AUROC (~0.5-0.65)  -> harness sanity
  G5  shuffled labels must fail

ASCII-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
sys.path.insert(0, os.path.join(P, "RDNet"))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SCALARS = ["rho", "g", "gain_dev", "C_I", "R_abs_mean", "omega_frac", "neg_frac", "omega_depth"]
MAPS = ["omega", "sign", "depth"]


# ---------------------------------------------------------------- F2 features
def convnext_features(I224, device):
    import timm
    from models.arch.classifier import PretrainedConvNext
    m = PretrainedConvNext("convnext_small_in22k").to(device)
    ck = torch.load(os.path.join(P, "pretrained/cls_model.pth"), map_location="cpu")
    sd = ck.get("icnn", ck.get("model", ck))
    miss, unex = m.load_state_dict(sd, strict=False)
    print("[F2] cls_model.pth missing=%d unexpected=%d" % (len(miss), len(unex)))
    m.eval()
    feats, codes = [], []
    x = torch.from_numpy(I224).permute(0, 3, 1, 2).float().div(255.0).to(device)
    with torch.no_grad():
        for i in range(0, x.shape[0], 32):
            b = x[i:i + 32]
            bb = F.interpolate(b, size=(224, 224), mode="bilinear", align_corners=True)
            g = m.model(bb)          # 768-d global
            c = m.head(g)            # 6-d code
            feats.append(g.cpu().numpy())
            codes.append(c.cpu().numpy())
    return np.concatenate(feats, 0), np.concatenate(codes, 0)


# ---------------------------------------------------------------- helpers
def scene_stats(vals, scenes):
    """mean per scene, in sorted scene order."""
    ks = sorted(set(scenes))
    return ks, np.array([np.mean([v for v, s in zip(vals, scenes) if s == k]) for k in ks])


def ridge_fit_predict(Xtr, ytr, Xte, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    m = Ridge(alpha=alpha).fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd)


def eval_within_domain(y, scenes, Xlist, names, order="scene"):
    """LOSO by scene. Returns dict name -> (scene_mae, spearman, n_scenes)."""
    ks = sorted(set(scenes))
    if len(ks) < 3:
        return None, len(ks)
    out = {}
    for name, X in zip(names, Xlist):
        preds, truth = [], []
        for k in ks:
            tr = np.array([s != k for s in scenes])
            te = ~tr
            if tr.sum() < 5 or te.sum() < 1:
                continue
            pr = ridge_fit_predict(X[tr], y[tr], X[te])
            preds.append(float(pr.mean()))
            truth.append(float(y[te].mean()))
        if len(preds) >= 3:
            mae = float(np.mean(np.abs(np.array(preds) - np.array(truth))))
            rho = float(spearmanr(preds, truth).statistic) if len(set(preds)) > 1 else float("nan")
            out[name] = (mae, rho, len(preds))
    return out, len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/opdata_partial.npz")
    ap.add_argument("--out", default="/tmp/operator_screen.csv")
    ap.add_argument("--map-epochs", type=int, default=25)
    ap.add_argument("--skip-maps", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = np.load(args.npz, allow_pickle=True)
    domain = d["domain"].astype(str); scene = d["scene"].astype(str); src = d["src"].astype(str)
    print("records:", len(domain), "scenes:", len(set(scene)))
    for k in sorted(set(domain)):
        print("  %-10s frames=%-4d scenes=%d" % (k, int((domain == k).sum()), len(set(scene[domain == k]))))

    f1 = d["feat_f1"].astype(np.float64)
    muI = f1[:, [0]]  # luma mean only -> the "trivial physics" baseline
    g768, code6 = convnext_features(d["I224"], dev)
    f2 = np.concatenate([g768, code6], 1).astype(np.float64)
    f12 = np.concatenate([f1, f2], 1)

    feats = [("muI", muI), ("f1", f1), ("f2", f2), ("f1+f2", f12)]
    names = [n for n, _ in feats]

    rows = []
    print("\n================ SCALARS: within-domain leave-scene-out ================")
    for dom in ("apr", "may", "jun", "1019split"):
        m = domain == dom
        if m.sum() < 10:
            continue
        sc = scene[m]
        nsc = len(set(sc))
        print("\n--- %s (frames=%d, scenes=%d)%s ---" % (dom, int(m.sum()), nsc, "  [NOT DECIDABLE: <3 scenes]" if nsc < 3 else ""))
        for t in SCALARS:
            y = d["y_" + t][m].astype(np.float64)
            res, n = eval_within_domain(y, sc, [X[m] for _, X in feats], names)
            spread = float(np.std([np.mean([v for v, s in zip(y, sc) if s == k]) for k in set(sc)]))
            if res is None:
                print("  %-12s spread=%.4f  (too few scenes)" % (t, spread))
                continue
            line = "  %-12s spread=%.4f | " % (t, spread)
            for nme in names:
                mae, rho, k = res[nme]
                line += "%s: mae=%.4f rho=%+.2f | " % (nme, mae, rho)
                rows.append(dict(kind="scalar_within", domain=dom, target=t, features=nme,
                                 n_scenes=k, spread=spread, mae=mae, spearman=rho))
            print(line)

    print("\n================ SCALARS: cross-domain (train 1020 -> test 1019test) ================")
    tr = (src == "1020") & (domain != "1019test")
    te = domain == "1019test"
    print("train frames=%d scenes=%d | test frames=%d scenes=%d"
          % (tr.sum(), len(set(scene[tr])), te.sum(), len(set(scene[te]))))
    for t in SCALARS:
        y = d["y_" + t].astype(np.float64)
        line = "  %-12s | " % t
        for nme, X in feats:
            pr = ridge_fit_predict(X[tr], y[tr], X[te])
            ks, ytrue = scene_stats(y[te], scene[te])
            _, ypred = scene_stats(pr, scene[te])
            b_const = float(np.mean(y[tr]))
            mae_m = float(np.mean(np.abs(ypred - ytrue)))
            mae_c = float(np.mean(np.abs(b_const - ytrue)))
            line += "%s: mae=%.4f (const=%.4f) | " % (nme, mae_m, mae_c)
            rows.append(dict(kind="scalar_cross", domain="1019test", target=t, features=nme,
                             n_scenes=len(ks), spread=float(np.std(ytrue)), mae=mae_m,
                             spearman=float("nan")))
        print(line)

    # ---------------- G5 shuffled control ----------------
    print("\n================ G5 negative control (shuffled rho labels) ================")
    rng = np.random.RandomState(0)
    y = d["y_rho"].astype(np.float64).copy()
    perm = rng.permutation(len(y))
    ys = np.empty_like(y); ys[:] = y[perm]
    for dom in ("apr", "may"):
        m = domain == dom
        res, n = eval_within_domain(ys[m], scene[m], [X[m] for _, X in feats], names)
        if res:
            print("  %-4s shuffled: " % dom + " ".join("%s mae=%.4f" % (k, v[0]) for k, v in res.items()))

    # ---------------- maps ----------------
    if not args.skip_maps:
        print("\n================ MAPS (64x64 tiny conv, LOSO in 1020 + cross-domain) ================")
        class TinyNet(nn.Module):
            def __init__(self, cout):
                super().__init__()
                self.b = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                    nn.Conv2d(48, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                    nn.Conv2d(48, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(32, cout, 3, 1, 1),
                )
            def forward(self, x):
                return self.b(x)

        imgs = d["I224"].astype(np.float32)  # 224x224
        def to64(batch):
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255.0)
            return F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)

        for mk in MAPS:
            M = d["m_" + mk].astype(np.float32)
            binary = mk in ("omega", "sign")
            res_cross = {}
            m1020 = (src == "1020")
            for name, trmask, temask in (
                ("cross(1020->1019test)", m1020, (domain == "1019test")),
            ):
                net = TinyNet(1).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=2e-3)
                Xtr = to64(imgs[trmask]).to(dev)
                Ytr = torch.from_numpy(M[trmask]).unsqueeze(1).to(dev)
                net.train()
                for ep in range(args.map_epochs):
                    opt.zero_grad()
                    out = net(Xtr)
                    if binary:
                        loss = F.binary_cross_entropy_with_logits(out, Ytr)
                    else:
                        loss = F.mse_loss(torch.sigmoid(out), Ytr)
                    loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    po = net(to64(imgs[temask]).to(dev)).squeeze(1).cpu().numpy()
                yt = M[temask]
                if binary:
                    yb = (yt > 0.5).astype(np.int32).ravel()
                    if yb.min() == yb.max():
                        auc = float("nan")
                    else:
                        auc = float(roc_auc_score(yb, po.ravel()))
                    res_cross[name] = ("auc", auc, float(yt.mean()))
                else:
                    r = float(spearmanr(po.ravel(), yt.ravel()).statistic)
                    res_cross[name] = ("rho", r, float(yt.mean()))
                print("  %-8s %-22s %s=%.3f | pos-rate=%.3f" % (mk, name, res_cross[name][0], res_cross[name][1], res_cross[name][2]))
                rows.append(dict(kind="map", domain="1019test", target=mk, features="tinyconv",
                                 n_scenes=len(set(scene[temask])), spread=float(yt.mean()),
                                 mae=res_cross[name][1], spearman=float("nan")))

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n[write]", args.out, "rows=", len(rows))


if __name__ == "__main__":
    main()
