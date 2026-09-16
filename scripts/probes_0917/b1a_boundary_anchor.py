#!/usr/bin/env python3
"""批次1a：膜边界台阶锚探针（零训练，合成集）。

原理：膜边两侧 R 相同（p=0 不穿膜）+ T 连续 → I_裸 − I_膜 = (1−α)·T。
估计器（逐图）：
  mean / std  : α̂ = mean(I_膜带)/mean(I_裸带)、std 比（朴素）
  reg         : 跨边界配对回归 jump ~ I_裸 的斜率 → α̂ = 1 − slope（已知偏置：∝varT/(varT+varR)）
  ddiff       : 双差分（对配对间的 R 取差抵消）→ α̂ = 1 − d( jump )/d( I_裸 ) 的中位数
真值：膜侧边界带的 α 均值（GT α 图）。
用法: python3 scripts/probes_0917/b1a_boundary_anchor.py [--sets ...]
产物: eval/probe_b1a_boundary.csv + 控制台分档表
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETS = ["synth_rrw_v21_0913", "synth_rrw_v21_indomain", "synth_rrw_v22_scan"]
BARE_T, FILM_T2 = 0.995, 0.99      # 膜侧带用 dist(bare)∈[6,12]，见 §诊断


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--min-film-px", type=int, default=200)
    args = ap.parse_args()
    rows = []
    for s in args.sets.split(","):
        root = os.path.join(PROJ, "data/test_sets", s)
        scenes = sorted(os.listdir(os.path.join(root, "gt")))
        levels = sorted(os.listdir(os.path.join(root, "levels")),
                        key=lambda x: -float(x.split("_")[-1]))
        for lv in levels:
            key = lv.split("_")[-1]
            for sc in scenes:
                sid = os.path.splitext(sc)[0]
                A = np.load(os.path.join(root, "alpha", f"{sid}_a{key}.npy")).astype(np.float32)
                if A.ndim == 3:
                    A = A[:, :, 0]
                I = img(os.path.join(root, "levels", lv, sc))
                bare = A > BARE_T
                film = A < FILM_T2
                if bare.sum() < 50 or film.sum() < args.min_film_px:
                    rows.append(dict(set=s, level=lv, scene=sid, n_pairs=0, alpha_true=float("nan"),
                                     a_mean="", a_std="", a_reg="", a_ddiff="", a_cov="", note="无裸区/膜区过小"))
                    continue
                # 过渡带（生成器为 4-8px sigmoid；6-12px 处信号最强）：膜侧取 dist(bare)∈[6,12]
                from scipy.spatial import cKDTree
                b_all = np.stack(np.nonzero(bare), 1)
                f_all = np.stack(np.nonzero(~bare), 1)
                tree_b = cKDTree(b_all)
                d_bare, _ = tree_b.query(np.stack(np.nonzero(~bare), 1))
                dmap = np.full(A.shape, 1e9, np.float32)
                dmap[~bare] = d_bare
                film_b = (~bare) & (dmap >= 6) & (dmap <= 12)
                d_film, _ = cKDTree(f_all).query(b_all)
                bare_b = np.zeros(A.shape, bool)
                bare_b[b_all[d_film <= 6][:, 0], b_all[d_film <= 6][:, 1]] = True
                f_ys, f_xs = np.nonzero(film_b)
                b_ys, b_xs = np.nonzero(bare_b)
                if len(f_ys) < 30 or len(b_ys) < 30:
                    rows.append(dict(set=s, level=lv, scene=sid, n_pairs=len(f_ys), alpha_true=float("nan"),
                                     a_mean="", a_std="", a_reg="", a_ddiff="", note="边界带过小"))
                    continue
                If = I[f_ys, f_xs].mean(1)          # 膜侧亮度（三通道均值）
                Ib = I[b_ys, b_xs].mean(1)
                a_true = float(A[film_b].mean())
                # (a) 均值比 / 标准差比
                a_mean = float(If.mean() / max(Ib.mean(), 1e-6))
                a_std = float(If.std() / max(Ib.std(), 1e-6))
                # (b) 最近配对回归
                b_xy = np.stack([b_ys, b_xs], 1).astype(np.float32)
                idx = ((f_ys[:, None] - b_xy[None, :, 0]) ** 2 +
                       (f_xs[:, None] - b_xy[None, :, 1]) ** 2).argmin(1)
                Ib_n = Ib[idx]
                jump = Ib_n - If
                slope = float(np.polyfit(Ib_n, jump, 1)[0]) if Ib_n.std() > 1e-3 else np.nan
                a_reg = 1.0 - slope
                # (c) 双差分：按 Ib 分箱，箱内均值差
                order = np.argsort(Ib_n)
                k = max(len(order) // 8, 5)
                lo, hi = order[:k], order[-k:]
                d_jump = jump[hi].mean() - jump[lo].mean()
                d_Ib = Ib_n[hi].mean() - Ib_n[lo].mean()
                a_ddiff = 1.0 - d_jump / d_Ib if d_Ib > 1e-3 else np.nan
                # (d) 协方差矩估计：var(Ib)=vT+vR, var(If)=α²vT+vR, cov=αvT+vR
                #     ⇒ α̂ = (cov − var(If)) / (var(Ib) − cov)（cov(T,R)=0 时无偏）
                Ifp = I[f_ys, f_xs].mean(1)
                Ibp = I[b_ys, b_xs].mean(1)[idx]
                cov_b = float(np.cov(Ibp, Ifp)[0, 1])
                den = float(np.var(Ibp) - cov_b)
                a_cov = (cov_b - float(np.var(Ifp))) / den if abs(den) > 1e-6 else np.nan
                # (e) 电平修正：α̂ = (I_f − R)/(I_b − R)（R 用真值 / 用 plain 分离器 R̂）
                def ratio_corr(Rm):
                    Rf_ = Rm[f_ys, f_xs].mean(1).mean()
                    Rb_ = Rm[b_ys, b_xs].mean(1).mean()
                    den2 = Ib.mean() - Rb_
                    return (If.mean() - Rf_) / den2 if abs(den2) > 1e-6 else np.nan
                R_gt = img(os.path.join(root, "r", sc))
                a_corr_gt = ratio_corr(R_gt)
                rp = os.path.join(PROJ, "runs/pred/rdt_plain", s, lv, sid, "r_hat.png")
                a_corr_sep = ratio_corr(img(rp)) if os.path.exists(rp) else np.nan
                rows.append(dict(set=s, level=lv, scene=sid, n_pairs=len(f_ys), alpha_true=round(a_true, 3),
                                 a_mean=round(a_mean, 3), a_std=round(a_std, 3),
                                 a_reg=round(a_reg, 3) if a_reg == a_reg else "",
                                 a_ddiff=round(a_ddiff, 3) if a_ddiff == a_ddiff else "",
                                 a_cov=round(a_cov, 3) if a_cov == a_cov else "",
                                 a_corr_gt=round(a_corr_gt, 3) if a_corr_gt == a_corr_gt else "",
                                 a_corr_sep=round(a_corr_sep, 3) if a_corr_sep == a_corr_sep else "", note=""))
    ok = [r for r in rows if r["note"] == ""]
    print(f"有效样本 {len(ok)}/{len(rows)}")
    for col in ("a_mean", "a_std", "a_reg", "a_ddiff", "a_cov", "a_corr_gt", "a_corr_sep"):
        v = np.array([r[col] if r[col] != "" else np.nan for r in ok], float)
        m = ~np.isnan(v)
        print(f"  {col:7s}: 有效 {m.sum()}/{len(v)}  MAE {np.abs(v[m]-np.array([r['alpha_true'] for r in ok],float)[m]).mean():.3f}  "
              f"corr {np.corrcoef(v[m], np.array([r['alpha_true'] for r in ok],float)[m])[0,1]:+.3f}")
    # 分档
    print("\n分档（valid, MAE per estimator）")
    for s in sorted({r["set"] for r in rows}):
        for lv in sorted({r["level"] for r in rows if r["set"] == s}):
            sub = [r for r in ok if r["set"] == s and r["level"] == lv]
            if not sub:
                continue
            t = np.array([r["alpha_true"] for r in sub])
            out = []
            for col in ("a_mean", "a_std", "a_reg", "a_ddiff", "a_corr_gt", "a_corr_sep"):
                v = np.array([r[col] if r[col] != "" else np.nan for r in sub], float)
                m = ~np.isnan(v)
                out.append(f"{col[2:]}:{np.abs(v[m]-t[m]).mean():.3f}" if m.any() else f"{col[2]}:na")
            print(f"  {s[:20]:>20} {lv:>11} n={len(sub):2d} α真{t.mean():.3f}  " + "  ".join(out))
    out = os.path.join(PROJ, "eval/probe_b1a_boundary.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
