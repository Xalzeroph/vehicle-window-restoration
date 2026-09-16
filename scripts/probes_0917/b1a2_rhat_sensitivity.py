#!/usr/bin/env python3
"""批次1a 统计补救（2/2）：α̂ 对 R̂ 带内偏差的敏感性。

估计器 α̂ = (mean(I_膜带) − mean(R̂_膜带)) / (mean(I_裸带) − mean(R̂_裸带))，只吃 R̂ 在两个
边界带内的 *区域均值偏差*（非 PSNR）。本脚本量化：
  (1) 带内偏差 → α̂ 误差的相关性：dev_膜 = mean(R̂_膜带)−mean(R_gt_膜带)、dev_裸 同理；
  (2) 受控扰动：R̂ += δ（δ=±5,±10 灰阶，两带同加）重算 α̂ → 实测 dα̂/dδ，
      与理论对比（两带同加时 dα̂/dδ = (A−B)/B² = −(1−α)/B，B = I_裸带−R̂_裸带；
      仅膜带加 δ 时 = −1/B；仅裸带加 δ 时 = +α/B）。
取带代码自上向下复制自 b1a_boundary_anchor.py（未改动原脚本），条带定义一致。

用法: python3 scripts/probes_0917/b1a2_rhat_sensitivity.py
产物: eval/probe_b1a2_sensitivity.csv + 控制台
"""
import csv, os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_IN = os.path.join(PROJ, "eval/probe_b1a_boundary.csv")
CSV_OUT = os.path.join(PROJ, "eval/probe_b1a2_sensitivity.csv")
BARE_T, FILM_T2 = 0.995, 0.99          # 与 b1a 一致
DELTAS = (-10, -5, 5, 10)              # 灰阶（0-255 标度）


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


_mask_cache = {}


def bands(set_name, sid, lv):
    """复制自 b1a_boundary_anchor.py 的条带定义（裸区 = α>BARE_T；膜带 = dist(bare)∈[6,12]；
    裸带 = 裸像素距膜 ≤6px）。同一 (scene, level) 的 α 图在 v21_0913/v22_scan 间逐文件一致，
    故掩膜按 (scene,level) 缓存复用。"""
    key = (sid, lv)
    if key in _mask_cache:
        return _mask_cache[key]
    root = os.path.join(PROJ, "data/test_sets", set_name)
    keyv = lv.split("_")[-1]
    A = np.load(os.path.join(root, "alpha", f"{sid}_a{keyv}.npy")).astype(np.float32)
    if A.ndim == 3:
        A = A[:, :, 0]
    bare = A > BARE_T
    f_all = np.stack(np.nonzero(~bare), 1)
    b_all = np.stack(np.nonzero(bare), 1)
    d_bare, _ = cKDTree(b_all).query(f_all, workers=4)          # 膜像素到裸区距离
    dmap = np.full(A.shape, 1e9, np.float32)
    dmap[~bare] = d_bare
    film_b = (~bare) & (dmap >= 6) & (dmap <= 12)
    d_film, _ = cKDTree(f_all).query(b_all, workers=4)
    bare_b = np.zeros(A.shape, bool)
    sel = d_film <= 6
    bare_b[b_all[sel][:, 0], b_all[sel][:, 1]] = True
    f_ys, f_xs = np.nonzero(film_b)
    b_ys, b_xs = np.nonzero(bare_b)
    _mask_cache[key] = (f_ys, f_xs, b_ys, b_xs)
    return _mask_cache[key]


def main():
    rows = list(csv.DictReader(open(CSV_IN)))
    out = []
    for i, r in enumerate(rows):
        s, lv, sid = r["set"], r["level"], r["scene"]
        root = os.path.join(PROJ, "data/test_sets", s)
        f_ys, f_xs, b_ys, b_xs = bands(s, sid, lv)
        I = img(os.path.join(root, "levels", lv, f"{sid}.png"))
        Rg = img(os.path.join(root, "r", f"{sid}.png"))
        Rh = img(os.path.join(PROJ, "runs/pred/rdt_plain", s, lv, sid, "r_hat.png"))
        i_f = float(I[f_ys, f_xs].mean(1).mean()); i_b = float(I[b_ys, b_xs].mean(1).mean())
        rh_f = float(Rh[f_ys, f_xs].mean(1).mean()); rh_b = float(Rh[b_ys, b_xs].mean(1).mean())
        rg_f = float(Rg[f_ys, f_xs].mean(1).mean()); rg_b = float(Rg[b_ys, b_xs].mean(1).mean())
        a_true = float(r["alpha_true"])
        B = i_b - rh_b                                    # ≈ T（裸带透射电平）
        A_ = i_f - rh_f                                   # ≈ αT
        a_hat = A_ / B
        dev_f, dev_b = rh_f - rg_f, rh_b - rg_b           # 带内偏差（灰阶）
        err_s = a_hat - a_true
        # 扰动：两带同加 δ
        a_hat_d = {d: (A_ - d) / (B - d) for d in DELTAS}
        d_all = np.array([DELTAS[0], DELTAS[1], 0.0, DELTAS[2], DELTAS[3]])
        a_all = np.array([a_hat_d[DELTAS[0]], a_hat_d[DELTAS[1]], a_hat,
                          a_hat_d[DELTAS[2]], a_hat_d[DELTAS[3]]])
        slope_both = float(np.polyfit(d_all, a_all, 1)[0])
        slope_central = (a_hat_d[5] - a_hat_d[-5]) / 10.0
        row = dict(set=s, level=lv, scene=sid, alpha_true=a_true,
                   n_film=len(f_ys), n_bare=len(b_ys),
                   i_f=round(i_f, 2), i_b=round(i_b, 2),
                   rhat_f=round(rh_f, 2), rhat_b=round(rh_b, 2),
                   rgt_f=round(rg_f, 2), rgt_b=round(rg_b, 2),
                   dev_f=round(dev_f, 3), dev_b=round(dev_b, 3),
                   dev_diff=round(dev_f - a_true * dev_b, 3),
                   a_hat=round(a_hat, 4), err_signed=round(err_s, 4), err_abs=round(abs(err_s), 4),
                   a_hat_m10=round(a_hat_d[-10], 5), a_hat_m5=round(a_hat_d[-5], 5),
                   a_hat_p5=round(a_hat_d[5], 5), a_hat_p10=round(a_hat_d[10], 5),
                   slope_both=round(slope_both, 6), slope_central=round(slope_central, 6),
                   thy_slope_both=round((A_ - B) / B ** 2, 6),
                   thy_inv_B=round(1.0 / B, 6),
                   thy_slope_film_only=round(-1.0 / B, 6),
                   thy_slope_bare_only=round(a_hat / B, 6))
        out.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(rows)}", flush=True)

    # ---- 一致性：复算 a_hat vs CSV a_corr_sep ----
    dmax = max(abs(r["a_hat"] - float(c["a_corr_sep"])) for r, c in zip(out, rows))
    amax = max(abs(r["alpha_true"] - float(c["alpha_true"])) for r, c in zip(out, rows))
    print(f"\n[一致性] 复算 α̂ vs CSV a_corr_sep: max|Δ|={dmax:.4f}；α_true max|Δ|={amax:.4f}")

    err = np.array([r["err_signed"] for r in out]); eab = np.abs(err)
    df_ = np.array([r["dev_f"] for r in out]); db_ = np.array([r["dev_b"] for r in out])
    dd = np.array([r["dev_diff"] for r in out])
    rh_b_arr = np.array([r["rhat_b"] for r in out]); i_b = np.array([r["i_b"] for r in out])
    Bv = i_b - rh_b_arr; al = np.array([r["alpha_true"] for r in out])
    n = len(err)

    def corr(x, y, tag, ytag="err_signed"):
        rp, pp = pearsonr(x, y); rs, ps = spearmanr(x, y)
        print(f"  {tag:<28s} vs {ytag:<10s} Pearson r={rp:+.3f} (p={pp:.1e})  Spearman ρ={rs:+.3f}")

    print(f"\n================ (1) R̂ 带内偏差 → α̂ 误差（n={n}） ================")
    print(f"  典型量级: mean|dev_膜|={np.abs(df_).mean():.2f}   mean|dev_裸|={np.abs(db_).mean():.2f} 灰阶; "
          f"中位 T̂(=I_裸−R̂_裸)={np.median(Bv):.1f}; MAE={eab.mean():.3f}")
    corr(df_, err, "膜带偏差 dev_膜"); corr(db_, err, "裸带偏差 dev_裸")
    corr(df_, eab, "膜带偏差 dev_膜", "err_abs"); corr(db_, eab, "裸带偏差 dev_裸", "err_abs")
    corr(dd, err, "差分偏差 dev_膜−α·dev_裸 (靶)", "err_signed")
    # 线性回归 err ~ dev_膜 + dev_裸（理论系数 ≈ −1/T, +α/T）
    X = np.column_stack([np.ones(n), df_, db_])
    coef, *_ = np.linalg.lstsq(X, err, rcond=None)
    pred = X @ coef
    r2 = 1 - ((err - pred) ** 2).sum() / ((err - err.mean()) ** 2).sum()
    print(f"  OLS err ~ dev_膜+dev_裸: 系数 {coef[1]:+.5f}, {coef[2]:+.5f} "
          f"(理论 −1/T≈{(-1 / Bv).mean():+.5f}, +α/T≈{(al / Bv).mean():+.5f})  R²={r2:.3f}")
    print(f"  残差 MAE {np.abs(err - pred).mean():.3f} vs 原始 MAE {eab.mean():.3f}")

    print(f"\n================ (2) 受控扰动 dα̂/dδ（δ=±5,±10 灰阶，两带同加） ================")
    sl = np.array([r["slope_both"] for r in out]); th = np.array([r["thy_slope_both"] for r in out])
    invB = np.array([r["thy_inv_B"] for r in out])
    print(f"  实测斜率: mean {sl.mean():+.5f}  中位 {np.median(sl):+.5f}  "
          f"[{sl.min():+.5f}, {sl.max():+.5f}] (灰阶⁻¹)")
    print(f"  理论(两带): (A−B)/B² = −(1−α)/T̂: mean {th.mean():+.5f}  中位 {np.median(th):+.5f}")
    print(f"  实测 vs 理论(两带): max|Δ|={np.abs(sl - th).max():.2e}, corr={np.corrcoef(sl, th)[0, 1]:.6f}")
    print(f"  对照 1/T̂ (任务书预期 1/(I_裸−R̂)): 中位 {np.median(invB):+.5f} → |实测|/该值 中位 "
          f"{np.median(np.abs(sl)) / np.median(invB):.3f}（≈1−α 量级，符号为负）")
    print(f"  仅膜带加 δ 理论 −1/T̂: 中位 {np.median(-invB):+.5f}；仅裸带加 δ 理论 +α/T̂: 中位 {np.median(al / Bv):+.5f}")
    print(f"  效应折算: δ=+5 灰阶 → Δα̂ 中位 {np.median(np.array([r['a_hat_p5'] - r['a_hat'] for r in out])):+.5f}; "
          f"δ=−5 → {np.median(np.array([r['a_hat_m5'] - r['a_hat'] for r in out])):+.5f}")
    # 非线性/二阶：+10 与 −10 的对称性
    asym = np.array([(r["a_hat_p10"] - r["a_hat"]) + (r["a_hat_m10"] - r["a_hat"]) for r in out])
    print(f"  二阶不对称 |α̂(+10)+α̂(−10)−2α̂(0)|: 中位 {np.median(np.abs(asym)):.1e}（±10 灰阶内线性近似误差量级）")

    # 聚类稳健版（消伪重复）：按场景先平均成 24 点再算相关
    import collections
    byc = collections.defaultdict(list)
    for r in out:
        byc[r["scene"]].append(r)
    cd = np.array([np.mean([x["dev_f"] for x in v]) for v in byc.values()])
    cb = np.array([np.mean([x["dev_b"] for x in v]) for v in byc.values()])
    cx = np.array([np.mean([x["dev_diff"] for x in v]) for v in byc.values()])
    ce = np.array([np.mean([x["err_signed"] for x in v]) for v in byc.values()])
    print(f"\n[聚类稳健] 24 个场景点: corr(dev_膜,err)={pearsonr(cd, ce)[0]:+.3f}  "
          f"corr(dev_裸,err)={pearsonr(cb, ce)[0]:+.3f}  corr(差分,err)={pearsonr(cx, ce)[0]:+.3f}")

    with open(CSV_OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\n-> {os.path.relpath(CSV_OUT, PROJ)}")


if __name__ == "__main__":
    main()
