#!/usr/bin/env python3
"""GPU 只读探针: 用现有 ckpt (fac_taps_long) 的特征做档位判决归因.

提取 (每图): feat GAP/GMP(128+128) | 21 维原始统计 | GT 掩膜条件化 GAP(film/naked) | 模型 p/ĉ
分析: (1) 复现模型档位 acc; (2) 跨域线性探针: 各特征组拟合自训练池, 测 v2.1/indomain;
      (3) 输入消融: 把 21 维统计 / GAP+GMP 替换成训练池均值 -> 看 acc 变化.
用法: python3 /tmp/probe_fac_features.py --gpu 3 --n-train 200 --out /tmp/probe_fac_feats.npz
"""
import argparse, os, sys, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/DataDisk/zmy_code/TintedGlass_Enhance"
RDNET = os.path.join(ROOT, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)
from models.arch.rdnet_atap import FullNet_NLP_ATap_Fac, load_official_atap  # noqa

LEVELS = ["1", "0.6", "0.3"]
TAGS = ("l100", "l060", "l030")


def load_img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0


def masks_np(a):
    band = (np.abs(np.diff(a, axis=0, prepend=a[:1])) + np.abs(np.diff(a, axis=1, prepend=a[:, :1]))) > 0.02
    return (a < 0.95) & ~band, (a >= 0.95) & ~band


def build(gpu, ckpt):
    net = FullNet_NLP_ATap_Fac(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                               num_subnet=4, loss_col=4, inject=True, raw_encoder=False)
    sd = torch.load(ckpt, map_location="cpu")["icnn"]
    net.load_state_dict(sd, strict=False)
    return net.to(f"cuda:{gpu}").eval()


@torch.no_grad()
def extract(net, I, Amask):
    """I (3,H,W) np; Amask (H,W) film 掩膜(1=膜内). 返回 feature dict."""
    x = torch.from_numpy(I)[None].cuda()
    x_base, x_stem = net.baseball(x)
    x_stem = net.baseball_adapter[0](x_stem)
    c3b = net.baseball_adapter[4](x_base[3])
    feat = net._fuse(net.alpha_c3, x_stem, c3b)
    alpha, m, g, p = net._fac_heads(feat, x)
    gap = F.adaptive_avg_pool2d(feat, 1).flatten(1)[0]
    gmp = F.adaptive_max_pool2d(feat, 1).flatten(1)[0]
    stats = net._global_stats(x)[0]
    hf, wf = feat.shape[-2:]
    fm = torch.from_numpy(Amask.astype(np.float32))[None, None].cuda()
    fm = F.interpolate(fm, size=(hf, wf), mode="area")[0, 0] > 0.6
    nm_ = ~fm
    fv = feat[0].permute(1, 2, 0)                       # hf,wf,128
    gfilm = fv[fm].mean(0) if fm.sum() > 10 else torch.zeros_like(gap)
    gnaked = fv[nm_].mean(0) if nm_.sum() > 10 else torch.zeros_like(gap)
    pm_full = m[0, 0][None, None]
    pm = F.interpolate(pm_full, size=(hf, wf), mode="area")[0, 0] > 0.5   # 模型 m̂ @h/2
    gfilm_m = fv[pm].mean(0) if pm.sum() > 10 else torch.zeros_like(gap)
    g_nak_m = fv[~pm].mean(0) if (~pm).sum() > 10 else torch.zeros_like(gap)
    return dict(gap=gap.cpu().numpy(), gmp=gmp.cpu().numpy(), stats=stats.float().cpu().numpy(),
                gfilm=gfilm.cpu().numpy(), gnaked=gnaked.cpu().numpy(),
                gfilm_m=gfilm_m.cpu().numpy(), gnaked_m=g_nak_m.cpu().numpy(),
                m_iou=float(((pm & fm).float().sum() / ((pm | fm).float().sum() + 1e-6))),
                p=p[0].float().cpu().numpy(), c=float((p * net.alpha_levels).sum()),
                m=m.mean().item(), a=alpha.mean().item())


def collect(net, base, kind, ids, lvls=LEVELS, resize=None):
    out = []
    for sid in ids:
        for lv in lvls:
            if kind == "synth":
                I = load_img(os.path.join(base, "levels", f"alpha_{lv}", sid + ".png"))
                A = np.load(os.path.join(base, "alpha", f"{sid}_a{lv}.npy")).astype(np.float32)
            else:
                tag = TAGS[LEVELS.index(lv)]
                I = load_img(os.path.join(base, "I", f"{sid}_{tag}.png"))
                A = np.load(os.path.join(base, "alpha", f"{sid}_{tag}.npy")).astype(np.float32)
            if resize:
                t = torch.from_numpy(I)[None]
                I = F.interpolate(t, size=resize, mode="area")[0].numpy()
                tA = torch.from_numpy(A)[None, None]
                A = F.interpolate(tA, size=resize, mode="area")[0, 0].numpy()
            fm, _ = masks_np(A)
            d = extract(net, I, fm)
            d["sid"] = sid; d["lvl"] = lv
            out.append(d)
    return out


def to_mat(rows, keys):
    return np.stack([np.concatenate([r[k] for k in keys]) for r in rows])


def probe(rows_tr, rows_te, keys, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr = to_mat(rows_tr, keys); ytr = np.array([r["lvl"] for r in rows_tr])
    Xte = to_mat(rows_te, keys); yte = np.array([r["lvl"] for r in rows_te])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=8000, C=C).fit(sc.transform(Xtr), ytr)
    pr = clf.predict(sc.transform(Xte))
    acc = float((pr == yte).mean())
    oe = float(np.mean([abs(LEVELS.index(a) - LEVELS.index(b)) for a, b in zip(pr, yte)]))
    return acc, oe


def model_acc(rows):
    y = np.array([r["lvl"] for r in rows])
    pr = np.array([LEVELS[int(r["p"].argmax())] for r in rows])
    acc = float((pr == y).mean())
    oe = float(np.mean([abs(LEVELS.index(a) - LEVELS.index(b)) for a, b in zip(pr, y)]))
    cmean = {lv: float(np.mean([r["c"] for r in rows if r["lvl"] == lv])) for lv in LEVELS}
    return acc, oe, cmean


def ablate(net, base, ids):
    """输入消融: 21 维统计或 GAP+GMP 置换为训练池均值后, 模型档位判决变化."""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "pilot_ckpt/fac_taps_long/ckpt_final.pth"))
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--out", default="/tmp/probe_fac_feats.npz")
    a = ap.parse_args()
    net = build(a.gpu, a.ckpt)

    b21 = os.path.join(ROOT, "synth_rrw_v21_0913")
    bin_ = os.path.join(ROOT, "synth_rrw_v21_indomain")
    btr = os.path.join(ROOT, "train_triplets_0913")
    ids21 = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(b21 + "/gt/*.png"))
    idsin = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(bin_ + "/gt/*.png"))
    idstr = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(btr + "/T/*.png"))[:a.n_train]

    print(f"[probe] v2.1 {len(ids21)} 场景, indomain {len(idsin)}, 训练池样本 {len(idstr)} ...", flush=True)
    te = collect(net, b21, "synth", ids21)
    ind = collect(net, bin_, "synth", idsin)
    tr = collect(net, btr, "triplet", idstr)
    te224 = collect(net, b21, "synth", ids21, resize=(224, 224))

    for nm, rows in [("v2.1", te), ("indomain", ind), ("v2.1@224", te224)]:
        acc, oe, cm = model_acc(rows)
        print(f"[模型] {nm:9s} 档位 acc {acc:.3f} (序误差 {oe:.2f})  ĉ均值 " +
              " ".join(f"{k}:{v:.3f}" for k, v in cm.items()), flush=True)

    print("\n=== 跨域线性探针 (训练池 200 场景拟合) ===")
    groups = [("stats21", ["stats"]), ("gap+gmp(256)", ["gap", "gmp"]),
              ("gap+gmp+stats(277)", ["gap", "gmp", "stats"]),
              ("mask-pool film/naked(256)", ["gfilm", "gnaked"]),
              ("mask-pool+diff(384)", ["gfilm", "gnaked"]),
              ("stats+maskpool", ["stats", "gfilm", "gnaked"])]
    for nm, keys in groups:
        a1, e1 = probe(tr, te, keys)
        a2, e2 = probe(tr, ind, keys)
        a3, e3 = probe(tr, te224, keys)
        print(f"  {nm:26s} ->v2.1 {a1:.3f}(E{e1:.2f})  ->indomain {a2:.3f}(E{e2:.2f})  ->v2.1@224 {a3:.3f}(E{e3:.2f})")

    # 掩膜条件化差分特征 (显式相对量): gfilm-gnaked 与其相对版, 单独成组
    def relmat(rows):
        X = []
        for r in rows:
            gf, gn = r["gfilm"], r["gnaked"]
            X.append(np.concatenate([gf - gn, gf / (np.abs(gn) + 1e-3), r["stats"]]))
        return np.stack(X)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ytr = np.array([r["lvl"] for r in tr])
    yte = np.array([r["lvl"] for r in te])
    yin = np.array([r["lvl"] for r in ind])

    def relmat2(rows):
        X = []
        for r in rows:
            gf, gn = r["gfilm_m"], r["gnaked_m"]
            X.append(np.concatenate([gf - gn, gf / (np.abs(gn) + 1e-3), r["stats"]]))
        return np.stack(X)
    Xtr2, Xte2, Xin2 = relmat2(tr), relmat2(te), relmat2(ind)
    sc2 = StandardScaler().fit(Xtr2)
    clf2 = LogisticRegression(max_iter=8000, C=0.3).fit(sc2.transform(Xtr2), ytr)
    p1 = clf2.predict(sc2.transform(Xte2)); p2 = clf2.predict(sc2.transform(Xin2))
    e1 = np.mean([abs(LEVELS.index(a)-LEVELS.index(b)) for a,b in zip(p1,yte)])
    print(f"  {'m̂-pool diff+stats':26s} ->v2.1 {float((p1==yte).mean()):.3f}(E{e1:.2f})  ->indomain {float((p2==yin).mean()):.3f}   [m̂ IoU v2.1 {np.mean([r['m_iou'] for r in te]):.3f}]")
    def relmat3(rows):
        X = []
        for r in rows:
            gf, gn = r["gfilm_m"], r["gnaked_m"]
            X.append(np.concatenate([gf - gn, np.abs(gf - gn), r["stats"]]))
        return np.stack(X)
    Xtr3, Xte3, Xin3 = relmat3(tr), relmat3(te), relmat3(ind)
    sc3 = StandardScaler().fit(Xtr3)
    clf3 = LogisticRegression(max_iter=8000, C=0.3).fit(sc3.transform(Xtr3), ytr)
    q1 = clf3.predict(sc3.transform(Xte3)); q2 = clf3.predict(sc3.transform(Xin3))
    print(f"  {'m̂-pool |diff|+stats':26s} ->v2.1 {float((q1==yte).mean()):.3f}  ->indomain {float((q2==yin).mean()):.3f}")
    Xtr, Xte, Xin = relmat(tr), relmat(te), relmat(ind)
    ytr = np.array([r["lvl"] for r in tr]); yte = np.array([r["lvl"] for r in te]); yin = np.array([r["lvl"] for r in ind])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=8000, C=0.3).fit(sc.transform(Xtr), ytr)
    p1 = clf.predict(sc.transform(Xte)); p2 = clf.predict(sc.transform(Xin))
    print(f"  {'film-naked diff+stats':26s} ->v2.1 {float((p1==yte).mean()):.3f}  ->indomain {float((p2==yin).mean()):.3f}")

    np.savez(a.out, tr=[{k: v for k, v in r.items()} for r in tr],
             te=[{k: v for k, v in r.items()} for r in te],
             ind=[{k: v for k, v in r.items()} for r in ind])
    print("saved", a.out)


if __name__ == "__main__":
    main()
