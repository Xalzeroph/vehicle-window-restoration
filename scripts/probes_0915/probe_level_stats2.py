#!/usr/bin/env python3
"""档位信号探针 2: 哪些统计量携带档位信号 (CPU 只读).

- v2.1 / indomain / train pool: 逐维 pooled AUC(1.0 vs 0.3) + 区域均值/均值差
- 跨域线性探针: 训练池拟合 -> v2.1/indomain, 特征组对比:
    gs(21) 原始 | gs_z 逐图 z-score | gs_mean 逐图除均值 | gap 双区域均值差 | 组合
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import uniform_filter
from PIL import Image

ROOT = "/DataDisk/zmy_code/TintedGlass_Enhance"
LEVELS = ["1", "0.6", "0.3"]
FILM_TAU, BAND_TAU = 0.95, 0.02


def load_img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0


def masks(a):
    band = (np.abs(np.diff(a, axis=0, prepend=a[:1])) + np.abs(np.diff(a, axis=1, prepend=a[:, :1]))) > BAND_TAU
    return (a < FILM_TAU) & ~band, (a >= FILM_TAU) & ~band


def stats21(x):
    v = x.reshape(3, -1)
    qs = np.quantile(v, [0.1, 0.5, 0.9], axis=1)
    lum = v.mean(0)
    lq = np.quantile(lum, [0.05, 0.95])
    return np.concatenate([v.mean(1), v.std(1), qs.ravel(),
                           [v[0].mean() - v[1].mean(), v[1].mean() - v[2].mean()],
                           [lum.std(), lq[1] - lq[0], (lum < 0.1).mean(), (lum > 0.9).mean()]])


NAMES = ["meanR", "meanG", "meanB", "stdR", "stdG", "stdB",
         "q10R", "q10G", "q10B", "q50R", "q50G", "q50B", "q90R", "q90G", "q90B",
         "R-G", "G-B", "lumstd", "p95-5", "<0.1", ">0.9"]


def one(sid, lvl, base, kind):
    if kind == "synth":
        I = load_img(os.path.join(base, "levels", f"alpha_{lvl}", sid + ".png"))
        A = np.load(os.path.join(base, "alpha", f"{sid}_a{lvl}.npy")).astype(np.float32)
    else:
        I = load_img(os.path.join(base, "I", sid + ".png"))
        A = np.load(os.path.join(base, "alpha", sid + ".npy")).astype(np.float32)
    fm, nm = masks(A)
    if fm.sum() < 50 or nm.sum() < 50:
        return None
    gs = stats21(I)
    mf = I[:, fm].mean(1); mn = I[:, nm].mean(1)
    gap = (mf - mn) / (np.abs(mf) + np.abs(mn) + 1e-6)      # 相对均值差 (3)
    return dict(sid=sid, lvl=lvl, gs=gs, gap=gap, cov=float(fm.mean()),
                mf=mf, mn=mn,
                hp_f=None)


def load_synth_pool(base):
    ids = sorted(os.path.splitext(os.path.basename(f))[0]
                 for f in glob.glob(os.path.join(base, "gt", "*.png")))
    out = []
    for sid in ids:
        for lv in LEVELS:
            r = one(sid, lv, base, "synth")
            if r: out.append(r)
    return out


def load_triplet_pool(base, ids, tags=("l100", "l060", "l030")):
    out = []
    for sid in ids:
        for tag, lv in zip(tags, LEVELS):
            r = one(sid + "_" + tag, lv, base, "triplet")
            if r:
                r["sid"] = sid
                out.append(r)
    return out


def auc1(v, y, a="1", b="0.3"):
    p = np.sort(v[y == a]); n = np.sort(v[y == b])
    if len(p) == 0 or len(n) == 0: return np.nan
    allv = np.concatenate([p, n]); r = np.argsort(np.argsort(allv)) + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def per_feature_auc(name, rows):
    y = np.array([r["lvl"] for r in rows])
    G = np.stack([r["gs"] for r in rows]); gap = np.stack([r["gap"] for r in rows])
    print(f"\n--- {name}: 每维 pooled AUC(1.0>0.3), 只列 |AUC-0.5|>0.15 ---")
    for i, nm in enumerate(NAMES):
        a = auc1(G[:, i], y)
        if abs(a - 0.5) > 0.15:
            print(f"   gs[{i:2d}] {nm:7s} AUC {a:.2f}   | 档均值 "
                  f"{np.mean(G[y=='1',i]):.3f}/{np.mean(G[y=='0.6',i]):.3f}/{np.mean(G[y=='0.3',i]):.3f}")
    for i in range(3):
        a = auc1(gap[:, i], y)
        print(f"   gap[{i}]  相对均值差 AUC {a:.2f}   | 档均值 "
              f"{np.mean(gap[y=='1',i]):.3f}/{np.mean(gap[y=='0.6',i]):.3f}/{np.mean(gap[y=='0.3',i]):.3f}")


def build_feats(rows, which):
    G = np.stack([r["gs"] for r in rows]); gap = np.stack([r["gap"] for r in rows])
    if which == "gs": return G
    if which == "gs_z":
        return (G - G.mean(1, keepdims=True)) / (G.std(1, keepdims=True) + 1e-9)
    if which == "gs_mean":                       # 逐图除以图像均值 (照明/增益归一)
        m = G[:, [0, 1, 2]].mean(1, keepdims=True)
        return G / (m + 1e-6)
    if which == "gap": return gap
    if which == "gs_z_gap":
        Z = (G - G.mean(1, keepdims=True)) / (G.std(1, keepdims=True) + 1e-9)
        return np.concatenate([Z, gap], 1)
    if which == "gs_mean_gap":
        m = G[:, [0, 1, 2]].mean(1, keepdims=True)
        return np.concatenate([G / (m + 1e-6), gap], 1)
    raise KeyError(which)


def fit_probe(rows_tr, rows_te, which, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr = build_feats(rows_tr, which), np.array([r["lvl"] for r in rows_tr])
    Xte, yte = build_feats(rows_te, which), np.array([r["lvl"] for r in rows_te])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    ord_err = float(np.abs([LEVELS.index(p) - LEVELS.index(t) for p, t in zip(pred, yte)]).mean())
    return acc, ord_err


def main():
    b21 = os.path.join(ROOT, "synth_rrw_v21_0913")
    bin_ = os.path.join(ROOT, "synth_rrw_v21_indomain")
    btr = os.path.join(ROOT, "train_triplets_0913")
    bma = os.path.join(ROOT, "train_triplets_match_0915")
    te, ind = load_synth_pool(b21), load_synth_pool(bin_)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    tid = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(btr, "T", "*.png")))[:n]
    mid = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(bma, "T", "*.png")))[:n]
    tr = load_triplet_pool(btr, tid)
    ma = load_triplet_pool(bma, mid)

    for nm, rows in [("v2.1", te), ("indomain", ind), ("train_0913", tr), ("train_match", ma)]:
        per_feature_auc(nm, rows)

    print("\n===== 跨域线性探针 (训练池拟合 -> 目标域测试) =====")
    print(f"{'feat':14s} {'->v2.1':>16s} {'->indomain':>16s}")
    for w in ["gs", "gs_z", "gs_mean", "gap", "gs_z_gap", "gs_mean_gap"]:
        a1, e1 = fit_probe(tr, te, w)
        a2, e2 = fit_probe(tr, ind, w)
        print(f"{w:14s}  acc {a1:.3f} (ordE {e1:.2f})  acc {a2:.3f} (ordE {e2:.2f})")
    print("-- match 池训练 --")
    for w in ["gs", "gs_mean_gap"]:
        a1, e1 = fit_probe(ma, te, w)
        print(f"{w:14s}  ->v2.1 acc {a1:.3f} (ordE {e1:.2f})")

    print("\n===== LOO-by-scene 域内 (12 场景) =====")
    for nm, rows in [("v2.1", te), ("indomain", ind)]:
        for w in ["gs", "gs_mean_gap"]:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sids = sorted(set(r["sid"] for r in rows)); y = np.array([r["lvl"] for r in rows])
            X = build_feats(rows, w); pred = np.empty(len(rows), dtype=object)
            idx = {s: [i for i, r in enumerate(rows) if r["sid"] == s] for s in sids}
            for s in sids:
                tr_i = np.concatenate([idx[t] for t in sids if t != s]); te_i = idx[s]
                sc = StandardScaler().fit(X[tr_i])
                clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X[tr_i]), y[tr_i])
                pred[te_i] = clf.predict(sc.transform(X[te_i]))
            print(f"  {nm:9s} {w:12s} acc {float((pred==y).mean()):.3f}")


if __name__ == "__main__":
    main()
