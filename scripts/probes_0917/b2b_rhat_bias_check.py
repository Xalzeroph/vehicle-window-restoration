#!/usr/bin/env python3
"""核查："R̂ 在裸带系统性偏亮"是否成立，以及机制（欠分离 vs 缩水）。

对 v2.1 / v23_lowscan 每个（档,场景）用 GT α 定义带（膜侧带=距裸区6-12px；裸侧带=距膜≤6px），算：
  1) 带内 mean(R̂) − mean(R_gt)（灰阶；逐场景符号一致性）
  2) 全图同量（判断是否只在裸带）
  3) 机制探针：per-pixel 误差 e=R̂−R 与 T、与 R 的相关（欠分离会把 T 漏进 R̂ → corr(e,T)>0）
产物: eval/probe_b2b_rhat_bias.csv + 控制台
"""
import csv, os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETS = ["synth_rrw_v21_0913", "synth_rrw_v23_lowscan"]


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def main():
    rows = []
    for s in SETS:
        root = os.path.join(PROJ, "data/test_sets", s)
        for lv in sorted(os.listdir(os.path.join(root, "levels")), key=lambda x: -float(x.split("_")[-1])):
            key = lv.split("_")[-1]
            for sc in sorted(os.listdir(os.path.join(root, "gt"))):
                sid = os.path.splitext(sc)[0]
                rp = os.path.join(PROJ, "runs/pred/rdt_plain", s, lv, sid, "r_hat.png")
                if not os.path.exists(rp):
                    continue
                A = np.load(os.path.join(root, "alpha", f"{sid}_a{key}.npy")).astype(np.float32)
                A = A[:, :, 0] if A.ndim == 3 else A
                Rh = img(rp).mean(2)
                Rg = img(os.path.join(root, "r", sc)).mean(2)
                T = img(os.path.join(root, "gt", sc)).mean(2)
                bare = A > 0.995
                if bare.sum() < 200:
                    continue
                b_all = np.stack(np.nonzero(bare), 1)
                f_all = np.stack(np.nonzero(~bare), 1)
                d_bare, _ = cKDTree(b_all).query(f_all)
                dm = np.full(A.shape, 1e9, np.float32)
                dm[~bare] = d_bare
                fb = (~bare) & (dm >= 6) & (dm <= 12)
                dbf, _ = cKDTree(f_all).query(b_all)
                bb = np.zeros(A.shape, bool)
                bb[b_all[dbf <= 6][:, 0], b_all[dbf <= 6][:, 1]] = True
                if fb.sum() < 30 or bb.sum() < 30:
                    continue
                e = Rh - Rg
                rows.append(dict(
                    set=s, level=lv, scene=sid,
                    bare_dev=round(float(e[bb].mean()), 2),
                    film_dev=round(float(e[fb].mean()), 2),
                    global_dev=round(float(e.mean()), 2),
                    corr_e_T=round(float(np.corrcoef(e.ravel(), T.ravel())[0, 1]), 3),
                    corr_e_R=round(float(np.corrcoef(e.ravel(), Rg.ravel())[0, 1]), 3),
                    R_gt_bare=round(float(Rg[bb].mean()), 1), R_hat_bare=round(float(Rh[bb].mean()), 1)))
    print(f"{'set':>22} {'level':>11} {'n':>3} | 裸带偏差 均值/符号 | 膜带偏差 | 全图偏差 | corr(e,T)")
    for s in SETS:
        for lv in sorted({r["level"] for r in rows if r["set"] == s}, key=lambda x: -float(x.split("_")[-1])):
            sub = [r for r in rows if r["set"] == s and r["level"] == lv]
            if not sub:
                continue
            d = np.array([r["bare_dev"] for r in sub])
            f = np.array([r["film_dev"] for r in sub])
            g = np.array([r["global_dev"] for r in sub])
            c = np.array([r["corr_e_T"] for r in sub])
            print(f"{s[:22]:>22} {lv:>11} {len(sub):>3} | {d.mean():+7.2f} 同号{(d > 0).mean():.0%} | {f.mean():+7.2f} | {g.mean():+7.2f} | {c.mean():+.2f}")
    allb = np.array([r["bare_dev"] for r in rows])
    print(f"\n合计：裸带偏差 均值 {allb.mean():+.2f} 灰阶，同号率 {(allb > 0).mean():.0%}（n={len(allb)}）")
    out = os.path.join(PROJ, "eval/probe_b2b_rhat_bias.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
