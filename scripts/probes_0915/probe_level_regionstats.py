#!/usr/bin/env python3
"""档位信号探针: 全局统计 vs 掩膜双区域相对对比度 (CPU, 只读).

对每个 (scene, level):
  - film mask (alpha<0.95, 排除 band), naked mask (alpha>=0.95)
  - 全局 std (per-channel std 均值)
  - film/naked 区域: 原始 std、高通 rms (I - boxmean(k))
  - 比值 r_var = V_f/V_n, r_hp = rms_f/rms_n (尺度不变的'膜内 vs 膜外'相对量)
输出:
  1) 各档 mean±scene-std 的表 (看 SNR: 跨档差 vs 跨场景散)
  2) 池化 AUC(跨场景, 只用该一维特征排序 1.0/0.6/0.3)
  3) LOO-by-scene 多分类 logistic (特征组对比: 全局21维 vs 双区域相对特征)
  4) 跨域: 训练池拟合 -> v2.1 测试 (同两特征组)
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import uniform_filter
from PIL import Image

ROOT = "/DataDisk/zmy_code/TintedGlass_Enhance"
LEVELS = ["1", "0.6", "0.3"]
FILM_TAU = 0.95
BAND_TAU = 0.02


def load_img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0


def load_alpha(p):
    return np.load(p).astype(np.float32)


def masks(a):
    band = (np.abs(np.diff(a, axis=0, prepend=a[:1])) + np.abs(np.diff(a, axis=1, prepend=a[:, :1]))) > BAND_TAU
    naked = (a >= FILM_TAU) & ~band
    film = (a < FILM_TAU) & ~band
    return film, naked


def masked_stats(x, m):
    """x (3,H,W) 或 (1,H,W), m (H,W) bool -> (per-ch std 均值, per-ch hp-rms 均值)."""
    if m.sum() < 50:
        return np.nan, np.nan
    stds, rms = [], []
    for c in range(x.shape[0]):
        v = x[c][m]
        stds.append(v.std())
        hp = x[c] - uniform_filter(x[c], size=17, mode="nearest")
        rms.append(np.sqrt((hp[m] ** 2).mean()))
    return float(np.mean(stds)), float(np.mean(rms))


def global_stats(x):
    """模型 level 分支用的 21 维全局统计 (复刻 _global_stats)。"""
    B = 1
    v = x.reshape(3, -1)
    qs = np.quantile(v, [0.1, 0.5, 0.9], axis=1)          # 3x3
    lum = v.mean(0)
    lq = np.quantile(lum, [0.05, 0.95])
    return np.concatenate([v.mean(1), v.std(1), qs.ravel(),
                           [v[0].mean() - v[1].mean(), v[1].mean() - v[2].mean()],
                           [lum.std(), lq[1] - lq[0],
                            (lum < 0.1).mean(), (lum > 0.9).mean()]])


def scene_feats(sid, lvl, base):
    I = load_img(os.path.join(base, "levels", f"alpha_{lvl}", sid + ".png"))
    A = load_alpha(os.path.join(base, "alpha", f"{sid}_a{lvl}.npy"))
    fm, nm = masks(A)
    g_std = float(I.std(axis=(1, 2)).mean())
    sf, hf = masked_stats(I, fm)
    sn, hn = masked_stats(I, nm)
    gs = global_stats(I)
    pair = np.array([sf, sn, sf / (sn + 1e-9), hf, hn, hf / (hn + 1e-9),
                     hf - hn, (hf - hn) / (hf + hn + 1e-9),
                     sf - sn, (sf - sn) / (sf + sn + 1e-9)])
    return dict(sid=sid, lvl=lvl, g_std=g_std, f_std=sf, n_std=sn,
                r_var=sf / (sn + 1e-9), r_hp=hf / (hn + 1e-9),
                f_hp=hf, n_hp=hn, pair=pair, gs=gs,
                cov=float((A < FILM_TAU).mean()))


def load_pool_triplet(base, ids, tags=("l100", "l060", "l030"), lvls=("1", "0.6", "0.3")):
    """train_triplets_*: I/{sid}_{tag}.png, alpha/{sid}_{tag}.npy."""
    rows = []
    for sid in ids:
        for tag, lv in zip(tags, lvls):
            I = load_img(os.path.join(base, "I", f"{sid}_{tag}.png"))
            A = load_alpha(os.path.join(base, "alpha", f"{sid}_{tag}.npy"))
            fm, nm = masks(A)
            sf, hf = masked_stats(I, fm)
            sn, hn = masked_stats(I, nm)
            gs = global_stats(I)
            rows.append(dict(sid=sid, lvl=lv, g_std=float(I.std(axis=(1, 2)).mean()),
                             f_std=sf, n_std=sn, r_var=sf / (sn + 1e-9),
                             r_hp=hf / (hn + 1e-9), f_hp=hf, n_hp=hn,
                             gs=gs, cov=float((A < FILM_TAU).mean()),
                             pair=np.array([sf, sn, sf / (sn + 1e-9), hf, hn,
                                            hf / (hn + 1e-9), hf - hn,
                                            (hf - hn) / (hf + hn + 1e-9),
                                            sf - sn, (sf - sn) / (sf + sn + 1e-9)])))
    return rows


def load_test_pool(base):
    ids = sorted(os.path.splitext(os.path.basename(f))[0]
                 for f in glob.glob(os.path.join(base, "gt", "*.png")))
    return [scene_feats(sid, lv, base) for sid in ids for lv in LEVELS]


def auc(x, pos, neg):
    """P(x_pos > x_neg) 近似 (排序法)。"""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def report(name, rows):
    print(f"\n===== {name} (n_scene≈{len(set(r['sid'] for r in rows))}) =====")
    lv = np.array([r["lvl"] for r in rows])
    for f in ["g_std", "r_var", "r_hp"]:
        v = np.array([r[f] for r in rows], float)
        line = []
        for L in LEVELS:
            m = lv == L
            line.append(f"{L}: {np.nanmean(v[m]):.3f}±{np.nanstd(v[m]):.3f}")
        # 池化 AUC (1.0 vs 0.3)
        a13 = auc(v, v[lv == "1"], v[lv == "0.3"])
        a16 = auc(v, v[lv == "1"], v[lv == "0.6"])
        a63 = auc(v, v[lv == "0.6"], v[lv == "0.3"])
        print(f"  {f:6s} " + " | ".join(line) + f"  || pooled AUC 1>0.3 {a13:.2f} 1>0.6 {a16:.2f} 0.6>0.3 {a63:.2f}")
    # 场景内配对排序 (1.0 > 0.6 > 0.3) 一致率
    for f in ["g_std", "r_hp"]:
        d = {r["sid"]: {} for r in rows}
        for r in rows:
            d[r["sid"]][r["lvl"]] = r[f]
        ok = 0
        for s, dd in d.items():
            if all(k in dd for k in LEVELS) and dd["1"] > dd["0.6"] > dd["0.3"]:
                ok += 1
        print(f"  {f:6s} 场景内完全配对排序正确: {ok}/{len(d)}")


def loo_logit(rows, feat_key, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sids = sorted(set(r["sid"] for r in rows))
    y = np.array([r["lvl"] for r in rows])
    X = np.stack([r[feat_key] if feat_key != "gs" else r["gs"] for r in rows])
    # scene-major 顺序: 每场景 3 档
    idx = {s: [i for i, r in enumerate(rows) if r["sid"] == s] for s in sids}
    pred = np.empty(len(rows), dtype=object)
    for s in sids:
        tr = np.concatenate([idx[t] for t in sids if t != s])
        te = idx[s]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    acc = float((pred == y).mean())
    conf = {}
    for t, p in zip(y, pred):
        conf[(t, p)] = conf.get((t, p), 0) + 1
    return acc, conf


def cross_domain(rows_tr, rows_te, feat_key, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    key = (lambda r: r["gs"]) if feat_key == "gs" else (lambda r: r[feat_key])
    Xtr = np.stack([key(r) for r in rows_tr]); ytr = np.array([r["lvl"] for r in rows_tr])
    Xte = np.stack([key(r) for r in rows_te]); yte = np.array([r["lvl"] for r in rows_te])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean())


def main():
    base21 = os.path.join(ROOT, "synth_rrw_v21_0913")
    basein = os.path.join(ROOT, "synth_rrw_v21_indomain")
    base_tr = os.path.join(ROOT, "train_triplets_0913")
    base_ma = os.path.join(ROOT, "train_triplets_match_0915")

    te = load_test_pool(base21)
    ind = load_test_pool(basein)
    tr_ids = sorted(os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(base_tr, "T", "*.png")))
    ma_ids = sorted(os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(base_ma, "T", "*.png")))
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    tr = load_pool_triplet(base_tr, tr_ids[:n_sub])
    ma = load_pool_triplet(base_ma, ma_ids[:n_sub])

    for name, rows in [("v2.1 (测试)", te), ("indomain (测试)", ind),
                       ("train_triplets_0913", tr), ("train_triplets_match_0915", ma)]:
        report(name, rows)

    print("\n===== LOO-by-scene 档位多分类 acc (12 场景; chance 0.333) =====")
    for name, rows in [("v2.1", te), ("indomain", ind)]:
        a_g, c_g = loo_logit(rows, "gs")
        a_p, c_p = loo_logit(rows, "pair")
        print(f"  {name}: 全局21维 {a_g:.3f} | 双区域相对特征(10维) {a_p:.3f} | chance 0.333")
        print(f"     pair 混淆: {dict(sorted(c_p.items()))}")

    print("\n===== 跨域: 用 train_triplets_0913 拟合 -> v2.1/indomain 测试 =====")
    for feat in ["gs", "pair"]:
        a21 = cross_domain(tr, te, feat)
        ain = cross_domain(tr, ind, feat)
        print(f"  {feat:4s}: ->v2.1 {a21:.3f} | ->indomain {ain:.3f}")
    a21 = cross_domain(ma, te, "gs"); ain = cross_domain(ma, te, "pair")
    print(f"  match池训练 ->v2.1: 全局 {a21:.3f} | 双区域 {ain:.3f}")


if __name__ == "__main__":
    main()
