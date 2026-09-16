#!/usr/bin/env python3
"""批次1c：全膜候选族 + 自动选择探针（零训练，复用 plain 的 R̂/T̂ 预测）。

作法（每图）：
 - 取"全膜裁片"：α 图腐蚀后仍为膜区（距边界≥16px）的最大内接区域（居中裁 256×128）
 - 候选 T̂_i = (I − R̂)/α_i，α_i ∈ {0.75,0.6,0.45,0.3,0.2}（对齐 VLT 档）
 - oracle@k = 候选中最高 PSNR（量化"歧义里剩多少可分"）
 - 自动选择器（无参考）：① 熵最大 ② 梯度能量最大 ③ 亮度均值最接近 118（训练集 GT 先验）
   报告各自 PSNR / 悔值（oracle@k − 选中）/ 命中最优比例 / 命中前二比例
用法: python3 scripts/probes_0917/b1c_candidate_family.py
产物: eval/probe_b1c_candidates.csv
"""
import csv, os
import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALPHAS = [0.75, 0.6, 0.45, 0.3, 0.2]
PRIOR_T = 96.4          # 训练集 T 亮度均值（100 场景实测；无参考判据的合法先验）
SETS = ["synth_rrw_v21_0913", "synth_rrw_v22_scan"]


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def entropy(y):
    h, _ = np.histogram(y, bins=32, range=(0, 255))
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def grad_energy(y):
    gy, gx = np.gradient(y)
    return float((gy ** 2 + gx ** 2).mean())


def main():
    rows = []
    for s in SETS:
        root = os.path.join(PROJ, "data/test_sets", s)
        for lv in sorted(os.listdir(os.path.join(root, "levels")), key=lambda x: -float(x.split("_")[-1])):
            key = lv.split("_")[-1]
            for sc in sorted(os.listdir(os.path.join(root, "gt"))):
                sid = os.path.splitext(sc)[0]
                A = np.load(os.path.join(root, "alpha", f"{sid}_a{key}.npy")).astype(np.float32)
                A = A[:, :, 0] if A.ndim == 3 else A
                if (A < 0.9).sum() < 5000:
                    continue                                  # 无足够全膜区
                pred = os.path.join(PROJ, "runs/pred/rdt_plain", s, lv, sid)
                Rh = img(os.path.join(pred, "r_hat.png"))
                I = img(os.path.join(root, "levels", lv, sc))
                gt = img(os.path.join(root, "gt", sc))
                # 裁片：取 A<0.9 的质心为中心，256×128，且片内全为膜（距离边界≥16px）
                ys, xs = np.nonzero(A < 0.9)
                cy, cx = int(ys.mean()), int(xs.mean())
                y0, x0 = np.clip(cy - 64, 0, A.shape[0] - 128), np.clip(cx - 128, 0, A.shape[1] - 256)
                sl = (slice(y0, y0 + 128), slice(x0, x0 + 256))
                if (A[sl] < 0.9).mean() < 0.98:               # 裁片不"全膜"则跳过（无参照场景）
                    continue
                Tc = (I[sl] - Rh[sl]).clip(0, 255)
                gc = gt[sl]
                cands, metrics = [], []
                for a in ALPHAS:
                    c = (Tc / a).clip(0, 255)
                    y = c.mean(2)
                    cands.append(c)
                    metrics.append(dict(a=a, psnr=psnr(c, gc), ent=entropy(y),
                                        grad=grad_energy(y), mean=float(y.mean())))
                ps = np.array([m["psnr"] for m in metrics])
                # 对照：真 α 候选（分离"表示误差"与"歧义"）；真 R+真 α 候选（天花板）
                At = A[sl]
                c_gt = ((I[sl] - Rh[sl]) / np.maximum(At, 0.05)[..., None]).clip(0, 255)
                r_gt = np.asarray(Image.open(os.path.join(root, "r", sc)).convert("RGB"), np.float32)[sl]
                c_perf = ((I[sl] - r_gt) / np.maximum(At, 0.05)[..., None]).clip(0, 255)
                p_gtalpha = psnr(c_gt, gc); p_perfect = psnr(c_perf, gc)
                o = int(ps.argmax())
                picks = dict(oracle=o,
                             sel_ent=int(np.argmax([m["ent"] for m in metrics])),
                             sel_grad=int(np.argmax([m["grad"] for m in metrics])),
                             sel_mean=int(np.argmin([abs(m["mean"] - PRIOR_T) for m in metrics])))
                rows.append(dict(set=s, level=lv, scene=sid, a_true=round(float(A[sl].mean()), 3),
                                 oracle_psnr=round(float(ps[o]), 2), oracle_a=ALPHAS[o],
                                 sel_ent_psnr=round(float(ps[picks["sel_ent"]]), 2), sel_ent_a=ALPHAS[picks["sel_ent"]],
                                 sel_grad_psnr=round(float(ps[picks["sel_grad"]]), 2), sel_grad_a=ALPHAS[picks["sel_grad"]],
                                 sel_mean_psnr=round(float(ps[picks["sel_mean"]]), 2), sel_mean_a=ALPHAS[picks["sel_mean"]],
                                 gtalpha_psnr=round(float(p_gtalpha), 2), perfect_psnr=round(float(p_perfect), 2),
                                 cand_psnrs="/".join(f"{v:.1f}" for v in ps)))
    n = len(rows)
    psarr = lambda k: np.array([r[k] for r in rows], float)
    orc = psarr("oracle_psnr")
    print(f"样本 {n}（全膜裁片，k={len(ALPHAS)}）")
    print(f"oracle@k: 均值 {orc.mean():.2f} dB | 最低档(α=0.2) {np.mean([float(r['cand_psnrs'].split('/')[-1]) for r in rows]):.2f} | "
          f"α=0.45 {np.mean([float(r['cand_psnrs'].split('/')[2]) for r in rows]):.2f} | "
          f"真α候选 {psarr('gtalpha_psnr').mean():.2f} | 真R+真α {psarr('perfect_psnr').mean():.2f}")
    for key, name in (("sel_ent", "熵最大"), ("sel_grad", "梯度能量最大"), ("sel_mean", "均值逼近训练先验96.4")):
        v = psarr(f"{key}_psnr")
        hit = np.mean([ALPHAS.index(r[f"{key}_a"]) == ALPHAS.index(r["oracle_a"]) for r in rows])
        top2 = np.mean([abs(ALPHAS.index(r[f"{key}_a"]) - ALPHAS.index(r["oracle_a"])) <= 1 for r in rows])
        print(f"  {name:8s}: PSNR {v.mean():.2f} dB  悔值 {(orc - v).mean():+.2f} dB  命中最优 {hit:.0%}  命中前二 {top2:.0%}")
    out = os.path.join(PROJ, "eval/probe_b1c_candidates.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
