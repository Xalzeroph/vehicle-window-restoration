#!/usr/bin/env python3
"""S1 组件① 分解式 α 头 G1 评测 (方案 §2.6 / §9 口径)。

测试集 = v2.1 12 场景 × 3 档 (固定不动)。区域口径同 analyze_alpha_sat.py:
  naked = α_GT≥0.95; band = |∇α_GT|>0.02; film = 其余。

指标 (门限见 §2.6):
  - naked 锚定率: naked 区 α̂ 均值 (≥0.90)
  - film spread: film 区 α̂(lv1.0) − α̂(lv0.3) (≥0.40, GT 0.42)
  - 级别内 corr: 各档 film 区 corr(α̂, α_GT) (≥0.7)
  - mask IoU / boundary-F1(±4px): GT m_lab=1[α_GT<0.95] vs m̂>0.5 (≥0.85 / ≥0.6)
  - 档位 acc (argmax p, 序 [1.0,0.6,0.3]) (≥0.70) + ECE
taps/raw 两配置同头同监督, 对照即 §2.4 特征源判决。

用法:
  python3 scripts/eval_iar.py --encoder taps --ckpt pilot_ckpt/fac_taps/ckpt_final.pth --gpu 0
  python3 scripts/eval_iar.py --encoder raw  --ckpt pilot_ckpt/fac_raw/ckpt_final.pth  --gpu 1
输出: fac_pred_{encoder}/alpha_{lv:g}/{stem}_a.npy + _m.npy (fp16);
      eval_iar_{encoder}.csv (逐图行)
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)

from models.arch.rdnet_atap import FullNet_NLP_ATap_Fac  # noqa: E402

LEVELS = [1.0, 0.6, 0.3]
NAKED_TAU = 0.95
BAND_TAU = 0.02
BND_TOL = 4  # boundary-F1 容差 (±4px)


def load(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float32)


def grad_mag(a):
    gx = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
    gy = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
    return gx + gy


def edges(b):
    """二值图的 4-邻域形态学边界 (float [H,W] -> bool [H,W])。"""
    t = torch.from_numpy(b[None, None].astype(np.float32))
    mx = F.max_pool2d(t, 3, 1, 1)
    mn = -F.max_pool2d(-t, 3, 1, 1)
    return ((mx - mn) > 0.5)[0, 0].numpy()


def dilate(e, r):
    t = torch.from_numpy(e[None, None].astype(np.float32))
    return F.max_pool2d(t, 2 * r + 1, 1, r)[0, 0].numpy() > 0.5


def boundary_f1(gt_b, pr_b, tol=BND_TOL):
    if gt_b.sum() == 0 or pr_b.sum() == 0:
        return np.nan, np.nan, np.nan
    gtd, prd = dilate(gt_b, tol), dilate(pr_b, tol)
    prec = (pr_b & gtd).sum() / pr_b.sum()
    rec = (gt_b & prd).sum() / gt_b.sum()
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def build(encoder, ckpt, device):
    net = FullNet_NLP_ATap_Fac(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                               num_subnet=4, loss_col=4, inject=True,
                               raw_encoder=(encoder == "raw"))
    sd = torch.load(ckpt, map_location="cpu")["icnn"]
    miss, unex = net.load_state_dict(sd, strict=False)
    print(f"[fac-{encoder}] load {ckpt}: missing={len(miss)} unexpected={len(unex)}")
    return net.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["taps", "raw"], required=True)
    ap.add_argument("--ckpt", default=None, help="默认 pilot_ckpt/fac_{encoder}/ckpt_final.pth")
    ap.add_argument("--ts", default=os.path.join(PROJ, "synth_rrw_v21_0913"))
    ap.add_argument("--out-root", default=None, help="默认 fac_pred_{encoder}")
    ap.add_argument("--tag", default="", help="CSV 后缀: eval_iar_{encoder}{tag}.csv")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.ckpt is None:
        args.ckpt = os.path.join(PROJ, "pilot_ckpt", f"fac_{args.encoder}", "ckpt_final.pth")
    elif not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(PROJ, args.ckpt)
    out_root = args.out_root or os.path.join(PROJ, f"fac_pred_{args.encoder}")
    if not os.path.isabs(out_root):
        out_root = os.path.join(PROJ, out_root)

    device = torch.device(f"cuda:{args.gpu}")
    net = build(args.encoder, args.ckpt, device)
    stems = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(args.ts, "gt", "*.png")))

    per_lv = {}   # lv -> dict(lists)
    pool = {k: ([], []) for k in ("naked", "film")}   # 跨档像素池 (pred, gt)
    rows = []
    for lv in LEVELS:
        outd = os.path.join(out_root, f"alpha_{lv:g}")
        os.makedirs(outd, exist_ok=True)
        P, G, M, ML, P_post, R = [], [], [], [], [], []
        for stem in stems:
            i = load(os.path.join(args.ts, "levels", f"alpha_{lv:g}", stem + ".png"))
            x = torch.from_numpy(i.transpose(2, 0, 1) / 255.0)[None].to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                alpha, m, g, p = net.forward_alpha_only(x)
            ah = alpha.float()[0, 0].cpu().numpy()
            mh = m.float()[0, 0].cpu().numpy()
            ph = p.float()[0].cpu().numpy()
            gts = np.load(os.path.join(args.ts, "alpha", f"{stem}_a{lv:g}.npy")).astype(np.float32)
            np.save(os.path.join(outd, stem + "_a.npy"), ah.astype(np.float16))
            np.save(os.path.join(outd, stem + "_m.npy"), mh.astype(np.float16))
            P.append(ah); G.append(gts)
            m_lab = gts < NAKED_TAU
            pr_m = mh > 0.5
            inter = float((pr_m & m_lab).sum()); union = float((pr_m | m_lab).sum())
            prec, rec, f1 = boundary_f1(edges(m_lab), edges(pr_m))
            order = np.argsort(-ph)
            pred_idx = int(order[0])
            true_idx = LEVELS.index(lv)
            rows.append(dict(encoder=args.encoder, level=lv, scene=stem,
                             iou=inter / union if union > 0 else np.nan,
                             bnd_prec=prec, bnd_rec=rec, bnd_f1=f1,
                             lv_pred=pred_idx, lv_true=true_idx,
                             p_max=float(ph.max()),
                             a_mae=float(np.abs(ah - gts).mean())))
            P_post.append(ph); R.append(pred_idx == true_idx)
            band = grad_mag(gts) > BAND_TAU
            naked = gts >= NAKED_TAU
            film = ~(band | naked)
            ML.append((film, naked, m_lab))
            M.append(mh)
            print(f"[fac-{args.encoder}] lv={lv:g} {stem}: IoU={inter / union:.3f} "
                  f"bF1={f1 if np.isfinite(f1) else float('nan'):.3f} "
                  f"p={ph.argmax()}(true {true_idx}) aMAE={np.abs(ah - gts).mean():.3f}")

        # 区域聚合 (同 analyze_alpha_sat: 像素级跨场景)
        stats = {}
        for name, sel in (("naked", 1), ("film", 0)):
            pp = np.concatenate([a[sel_[sel]] for a, sel_ in zip(P, ML)])
            gg = np.concatenate([a[sel_[sel]] for a, sel_ in zip(G, ML)])
            corr = float(np.corrcoef(pp, gg)[0, 1]) if pp.std() > 1e-8 and gg.std() > 1e-8 else np.nan
            stats[name] = dict(pred=float(pp.mean()), gt=float(gg.mean()),
                               mae=float(np.abs(pp - gg).mean()), corr=corr)
            pool[name][0].append(pp); pool[name][1].append(gg)
        per_lv[lv] = dict(stats=stats, mae=float(np.mean([r["a_mae"] for r in rows if r["level"] == lv])),
                          iou=float(np.nanmean([r["iou"] for r in rows if r["level"] == lv])),
                          bf1=float(np.nanmean([r["bnd_f1"] for r in rows if r["level"] == lv])))

    # ---- 汇总 ----
    acc = float(np.mean([r["lv_true"] == r["lv_pred"] for r in rows]))
    p_max = np.array([r["p_max"] for r in rows])
    correct = np.array([r["lv_true"] == r["lv_pred"] for r in rows], float)
    ece = float(np.mean([abs(p_max[b].mean() - correct[b].mean())
                         for b in np.array_split(np.argsort(p_max), 5) if len(b)]))
    naked_all = float(np.concatenate(pool["naked"][0]).mean())   # 三档像素池
    spread = per_lv[1.0]["stats"]["film"]["pred"] - per_lv[0.3]["stats"]["film"]["pred"]
    spread_gt = per_lv[1.0]["stats"]["film"]["gt"] - per_lv[0.3]["stats"]["film"]["gt"]
    iou_all = float(np.nanmean([r["iou"] for r in rows]))
    bf1_all = float(np.nanmean([r["bnd_f1"] for r in rows]))

    print(f"\n======== G1 指标 [fac-{args.encoder}] (12 场景 × 3 档) ========")
    print(f"{'指标':34s}{'值':>9s}{'门限':>10s}  判定")
    def g(name, val, thr, ge=True, fmt="{:.3f}"):
        ok = (val >= thr) if ge else (val <= thr)
        print(f"{name:34s}{fmt.format(val):>9s}{fmt.format(thr):>10s}  {'PASS' if ok else 'FAIL'}")
        return ok
    g("naked 锚定率 (α_GT≥0.95 区 α̂ 均值)", naked_all, 0.90)
    g("film spread (lv1.0 − lv0.3)", spread, 0.40)
    for lv in LEVELS:
        g(f"film corr @档{lv:g}", per_lv[lv]["stats"]["film"]["corr"], 0.70)
    g("mask IoU (m̂>0.5)", iou_all, 0.85)
    g("boundary-F1 (≥0.6)", bf1_all, 0.60)
    g("档位 acc", acc, 0.70)
    print(f"\nfilm α̂: " + " / ".join(f"{per_lv[lv]['stats']['film']['pred']:.3f}" for lv in LEVELS)
          + "  (GT " + " / ".join(f"{per_lv[lv]['stats']['film']['gt']:.3f}" for lv in LEVELS) + ")")
    print(f"naked α̂: " + " / ".join(f"{per_lv[lv]['stats']['naked']['pred']:.3f}" for lv in LEVELS)
          + "  (GT " + " / ".join(f"{per_lv[lv]['stats']['naked']['gt']:.3f}" for lv in LEVELS) + ")")
    print(f"α MAE / IoU / bF1 逐档: "
          + " | ".join(f"{lv:g}: {per_lv[lv]['mae']:.3f}/{per_lv[lv]['iou']:.3f}/{per_lv[lv]['bf1']:.3f}"
                       for lv in LEVELS))
    print(f"档位 acc={acc:.3f} ECE={ece:.3f} (GT spread {spread_gt:.3f})")
    conf = np.zeros((3, 3), int)
    for r in rows:
        conf[r["lv_true"], r["lv_pred"]] += 1
    print("混淆 (行=真档 [1.0,0.6,0.3], 列=预测):\n" + str(conf))

    csv = os.path.join(PROJ, f"eval_iar_{args.encoder}{args.tag}.csv")
    keys = ["encoder", "level", "scene", "iou", "bnd_prec", "bnd_rec", "bnd_f1",
            "lv_pred", "lv_true", "p_max", "a_mae"]
    with open(csv, "w") as fp:
        fp.write(",".join(keys) + "\n")
        for r in rows:
            fp.write(",".join(f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"\n-> {csv}\n-> {out_root}/alpha_*/{{stem}}_a.npy|_m.npy")


if __name__ == "__main__":
    main()
