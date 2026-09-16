#!/usr/bin/env python3
"""批次1a 统计补救（1/2）：场景聚类 CI（消除伪重复）。

背景（P4 复核）：eval/probe_b1a_boundary.csv 的 168 行不是独立样本——
  * synth_rrw_v21_0913 与 synth_rrw_v22_scan 共用同一 gt/α 文件（同 seed 20260913，
    α_eff 一致，仅 R 池不同：r_rrw_strong vs wsoft），逐场景视为同一 cluster；
  * synth_rrw_v21_indomain 为独立 12 场景（u00..u11，src_t 各不相同）。
⇒ 正确 cluster 数 = 24（t00..t11 + u00..u11），本脚本同时报
   朴素（全部行独立）CI 作对比，并给 t 家族 12-cluster（df=11，t=2.201）分口径。

误差口径：e = |a_corr_sep − alpha_true|（逐行），cluster 内先平均，再跨 cluster 汇总。

用法: python3 scripts/probes_0917/b1a2_scene_clustered_ci.py
产物: eval/probe_b1a2_clustered_ci.csv（每 cluster 一行）+ 控制台表
"""
import csv, hashlib, os
import numpy as np
from scipy.stats import t as tdist

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_IN = os.path.join(PROJ, "eval/probe_b1a_boundary.csv")
CSV_OUT = os.path.join(PROJ, "eval/probe_b1a2_clustered_ci.csv")
FAM = {"synth_rrw_v21_0913": "t", "synth_rrw_v22_scan": "t", "synth_rrw_v21_indomain": "u"}
BINS = [("hi", 0.85, np.inf), ("mid", 0.5, 0.85), ("lo", -np.inf, 0.5)]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def verify_scene_identity():
    """审计聚类依据：v21_0913 与 v22_scan 的 gt/α 是否逐文件一致。"""
    base = os.path.join(PROJ, "data/test_sets")
    gt_same, a_same, a_tot = 0, 0, 0
    for i in range(12):
        f = f"t{i:02d}.png"
        gt_same += md5(os.path.join(base, "synth_rrw_v21_0913/gt", f)) == \
                   md5(os.path.join(base, "synth_rrw_v22_scan/gt", f))
        for lv in ("1", "0.6", "0.3"):
            a_tot += 1
            a_same += md5(os.path.join(base, "synth_rrw_v21_0913/alpha", f"t{i:02d}_a{lv}.npy")) == \
                      md5(os.path.join(base, "synth_rrw_v22_scan/alpha", f"t{i:02d}_a{lv}.npy"))
    # u 与 t 是否为同一场景：降采样灰度图跨集相关（身份场景应 r≈1）
    from PIL import Image
    def small(p):
        a = Image.open(p).convert("L")
        a = a.resize((128, 96))
        return np.asarray(a, np.float32).ravel()
    us = [small(os.path.join(base, "synth_rrw_v21_indomain/gt", f"u{i:02d}.png")) for i in range(12)]
    ts = [small(os.path.join(base, "synth_rrw_v21_0913/gt", f"t{i:02d}.png")) for i in range(12)]
    r_cross = max(np.corrcoef(u, v)[0, 1] for u in us for v in ts)
    print(f"[审计] v21_0913 vs v22_scan: gt 一致 {gt_same}/12, α 一致 {a_same}/{a_tot}")
    print(f"[审计] indomain(u) vs t 家族最大跨集相关 r={r_cross:+.3f}（远低于身份阈值 0.999）")
    return gt_same, a_same, a_tot, r_cross


def ci95(vals):
    v = np.asarray(vals, float)
    n = len(v)
    m = v.mean()
    s = v.std(ddof=1) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 1 else 0.0
    tc = tdist.ppf(0.975, n - 1) if n > 1 else np.nan
    half = tc * se if n > 1 else 0.0
    return dict(n=n, mean=m, sd=s, se=se, t=tc, df=n - 1, lo=m - half, hi=m + half, width=2 * half)


def report(tag, cluster_vals, naive_vals):
    c, nv = ci95(cluster_vals), ci95(naive_vals)
    print(f"\n{tag}")
    print(f"  聚类: n_cluster={c['n']:2d}  均值 {c['mean']:.3f}  sd {c['sd']:.3f}  "
          f"95%CI [{c['lo']:.3f},{c['hi']:.3f}] 宽 {c['width']:.3f}  (t={c['t']:.3f}, df={c['df']})")
    print(f"  朴素: n_row    ={nv['n']:3d}  均值 {nv['mean']:.3f}  sd {nv['sd']:.3f}  "
          f"95%CI [{nv['lo']:.3f},{nv['hi']:.3f}] 宽 {nv['width']:.3f}  (t={nv['t']:.3f}, df={nv['df']})")
    if nv["width"] > 1e-9:
        print(f"  加宽: 聚类/朴素 = {c['width'] / nv['width']:.2f}×")
    return c, nv


def main():
    verify_scene_identity()
    rows = list(csv.DictReader(open(CSV_IN)))
    for r in rows:
        r["e"] = abs(float(r["a_corr_sep"]) - float(r["alpha_true"]))
        r["es"] = float(r["a_corr_sep"]) - float(r["alpha_true"])
        r["fam"] = FAM[r["set"]]
        r["cluster"] = r["scene"]        # t00..t11 跨 v21/v22 合并；u00..u11 独立

    clusters = sorted({r["cluster"] for r in rows})
    print(f"样本 {len(rows)} 行 -> {len(clusters)} clusters（t 家族 12 + u 家族 12）")

    # ---- 每 cluster 一行 CSV ----
    out_rows = []
    for c in clusters:
        sub = [r for r in rows if r["cluster"] == c]
        alpha = np.array([float(r["alpha_true"]) for r in sub])
        e = np.array([r["e"] for r in sub])
        def per_set(s):
            x = [r["e"] for r in sub if r["set"] == s]
            return round(float(np.mean(x)), 4) if x else ""
        def per_bin(lo, hi):
            x = [r["e"] for r, a in zip(sub, alpha) if lo < a <= hi]
            return round(float(np.mean(x)), 4) if x else ""
        out_rows.append(dict(
            cluster=c, scene=c, sets="+".join(sorted({r["set"] for r in sub})),
            n_rows=len(sub), levels=",".join(sorted({r["level"] for r in sub})),
            alpha_true_mean=round(float(alpha.mean()), 3),
            e_mean=round(float(e.mean()), 4), e_max=round(float(e.max()), 4),
            es_mean=round(float(np.mean([r["es"] for r in sub])), 4),
            e_v21_0913=per_set("synth_rrw_v21_0913"), e_v22_scan=per_set("synth_rrw_v22_scan"),
            e_v21_indomain=per_set("synth_rrw_v21_indomain"),
            e_hi=per_bin(0.85, np.inf), e_mid=per_bin(0.5, 0.85), e_lo=per_bin(-np.inf, 0.5),
            corr_sep_gt=round(float(np.corrcoef([float(r["a_corr_sep"]) for r in sub], alpha)[0, 1]), 3),
        ))

    # ---- 汇总 ----
    def cluster_means(sub_rows):
        cs = sorted({r["cluster"] for r in sub_rows})
        return [float(np.mean([r["e"] for r in sub_rows if r["cluster"] == c])) for c in cs], cs

    t_rows = [r for r in rows if r["fam"] == "t"]
    u_rows = [r for r in rows if r["fam"] == "u"]
    print("\n================ 聚类 CI 汇总（估计器 a_corr_sep，误差 e=|α̂−α_true|） ================")
    cm_all, cs_all = cluster_means(rows)
    report(f"【全样本 24 clusters（若把 t/u 场景视同族会低估；实际 24 个独立场景）】", cm_all, [r["e"] for r in rows])
    cm_t, cs_t = cluster_means(t_rows)
    report("【t 家族 12 clusters（v21_0913+v22_scan, 132 行）——任务书 12-cluster 口径】", cm_t, [r["e"] for r in t_rows])
    cm_u, cs_u = cluster_means(u_rows)
    report("【u 家族 12 clusters（indomain, 36 行）】", cm_u, [r["e"] for r in u_rows])

    # ---- 对照：若 R̂=R_gt（a_corr_gt），带内偏差为零时的误差地板（同一聚类口径）----
    print("\n[对照] a_corr_gt（R 用真值）同口径：")
    for tag, sub in [("全样本24", rows), ("t家族12", t_rows), ("u家族12", u_rows)]:
        byc = {}
        for r in sub:
            byc.setdefault(r["cluster"], []).append(abs(float(r["a_corr_gt"]) - float(r["alpha_true"])))
        cmv = [float(np.mean(v)) for v in byc.values()]
        c = ci95(cmv)
        print(f"  {tag}: {len(cmv)} clusters 均值 {c['mean']:.3f} 95%CI [{c['lo']:.3f},{c['hi']:.3f}] "
              f"(t={c['t']:.3f}, df={c['df']})")

    # ---- α 分档 ----
    print("\n================ 按 α_true 分档（聚类 CI） ================")
    for name, rows_g in [("全样本24", rows), ("t家族12", t_rows), ("u家族12", u_rows)]:
        print(f"-- {name} --")
        for tag, lo, hi in BINS:
            bin_rows = [r for r in rows_g if lo < float(r["alpha_true"]) <= hi]
            if not bin_rows:
                print(f"  {tag:3s} (α_真∈({lo},{hi})): 无样本")
                continue
            cm, cs = cluster_means(bin_rows)
            c, nv = report(f"  {tag:3s} (α_真∈({lo},{hi})): rows={len(bin_rows)}, clusters={len(cs)}", cm, [r["e"] for r in bin_rows])

    out = os.path.join(PROJ, "eval/probe_b1a2_clustered_ci.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\n-> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
