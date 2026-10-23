#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""域消融 v1 聚合：6 条预登记对照（2026-10-23）。

输入
----
`eval/table_final_1019_dom_v1.csv`     15 场景全量（辅表）
`eval/table_final_1019_dom_v1_12.csv`  12 场景测试子集（主表，与 R-in19 可比）

统计口径（沿用上一轮已修好的实现）
--------------------------------
· 每臂先按**种子**求逐场景均值 → m[scene]；臂均值 = mean(m)
· 种子 std = 各 seed 的臂级均值之间的标准差
· 与参照臂配对：d[scene] = m_A − m_B，n = 共同场景数
  p = 2·(1 − CDF_t(|t|, n−1))；95% CI = Δ ± t₀.₉₇₅,n−1 · SE

预登记 6 条对照（不做全表两两比较）
--------------------------------
1. 域（室内 may）      : R-may   − R-apr
2. 域（第三域 jun）    : R-jun   − R-apr
3. 选取规则（may）     : R-may14 − R-may
4. 选取规则（apr）     : R-apr16 − R-apr
5. 组合（apr+may）     : R-aprmay − max(R-apr, R-may)
6. 组合（全三域）      : R-all   − max(R-apr, R-may, R-jun)
7. 域内上界            : R-in19  − R-apr
（7 = 6 条 + 域内上界，均按 Holm 校正后 MDE 判定）
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os

import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
EV = os.path.join(P, "eval")
OUT = os.path.join(EV, "dom_v1")
MDE_12 = 1.0 * math.sqrt(15.0 / 12.0)     # 1.118
MDE_15 = 1.0
METS = ("psnr", "hf_psnr", "grad_cos", "psnr_hole", "ssim")


def t_p(t, df):
    if df <= 0 or not np.isfinite(t):
        return float("nan"), float("nan")
    try:
        from scipy import stats
        return 2.0 * (1.0 - float(stats.t.cdf(abs(t), df))), float(stats.t.ppf(0.975, df))
    except Exception:
        return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))), 1.96


def load(fn):
    p = os.path.join(EV, fn)
    if not os.path.exists(p):
        return {}
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(p, encoding="utf-8")):
        arm = r["arm"]
        for pre in ("dom12_", "dom_"):
            if arm.startswith(pre):
                arm = arm[len(pre):]
                break
        if "_s" not in arm:
            continue
        base, seed = arm.rsplit("_s", 1)
        out[base][seed][r["scene"] + "__" + r["input"]] = r
    return out


def scene_mean(data, base, met):
    seeds = data.get(base, {})
    ks = set()
    for s in seeds.values():
        ks |= set(s.keys())
    out = {}
    for k in ks:
        v = [float(seeds[s][k][met]) for s in seeds if k in seeds[s] and met in seeds[s][k]]
        if v:
            out[k] = float(np.mean(v))
    return out


def paired(A, B, alpha=0.05):
    """配对 t：返回 Δ、**按 alpha 调整的** CI（默认 95%）、以及该 alpha 下的 MDE。

    2026-10-23 修：早期版本用「固定 MDE = 1.12/1.73 dB」与 CI 比大小，
    但 MDE 应随每个对比的 SE 变化。正确做法是把 α 调到 Bonferroni 水平后
    直接看**调整后 CI 是否排除 0**；MDE_i = t_{1−α/2}(df) · SE_i 只作参考报出。
    """
    ks = sorted(set(A) & set(B))
    d = [A[k] - B[k] for k in ks]
    n = len(d)
    if n < 3:
        return None
    m = float(np.mean(d)); sd = float(np.std(d, ddof=1)); se = sd / math.sqrt(n)
    t = m / se if se > 0 else 0.0
    p, tc95 = t_p(t, n - 1)
    try:
        from scipy import stats
        tc = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    except Exception:
        tc = tc95
    return dict(n=n, d=m, sd=sd, se=se, t=t, p=p,
                lo=m - tc95 * se, hi=m + tc95 * se,
                lo_adj=m - tc * se, hi_adj=m + tc * se,
                mde=tc * se, alpha=alpha,
                win=sum(1 for x in d if x > 0))


def verdict(st, is_ref=False):
    if is_ref:
        return "参照"
    if st is None:
        return "数据不足"
    if st["lo_adj"] > 0:
        return "✅ 胜出"
    if st["hi_adj"] < 0:
        return "❌ 更差"
    if st["lo"] <= 0 <= st["hi"]:
        return "无可检出"
    return ("↑ 正方向但未过校正" if st["lo"] > 0 else "↓ 负方向但未过校正")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    d15 = load("table_final_1019_dom_v1.csv")
    d12 = load("table_final_1019_dom_v1_12.csv")
    print(f"15 场景表: {len(d15)} 个臂  12 场景表: {len(d12)} 个臂")
    print(f"  15: {sorted(d15)}")
    print(f"  12: {sorted(d12)}")
    if not d12:
        raise SystemExit("缺 12 场景表；先跑 table_final_1019.py --tag dom_v1_12")
    if "R-apr" not in d12:
        raise SystemExit("缺参照臂 R-apr")

    # ---- 臂级汇总 ----
    rows = []
    for base in sorted(d12, key=lambda x: (x != "R-apr", x)):
        rec = {"arm": base, "n_seeds": len(d12[base])}
        for met in METS:
            sm = scene_mean(d12, base, met)
            rec[met] = float(np.mean(list(sm.values()))) if sm else float("nan")
        seed_means = []
        for s, sc in d12[base].items():
            seed_means.append(float(np.mean([float(x["psnr"]) for x in sc.values()])))
        rec["psnr_seed_std"] = (float(np.std(seed_means, ddof=1))
                               if len(seed_means) > 1 else 0.0)
        rows.append(rec)
    by = {r["arm"]: r for r in rows}

    # ---- 6+1 条预登记对照（主表 = 12 场景）----
    ref_A = scene_mean(d12, "R-apr", "psnr")
    contrasts = []

    def add(label, a, b, note=""):
        if a not in d12 or b not in d12:
            contrasts.append(dict(label=label, a=a, b=b, st=None, note="缺臂"))
            return
        st = paired(scene_mean(d12, a, "psnr"), scene_mean(d12, b, "psnr"))
        contrasts.append(dict(label=label, a=a, b=b, st=st, note=note))

    add("① 域·室内 may", "R-may", "R-apr")
    add("② 域·第三域 jun", "R-jun", "R-apr")
    add("③ 选取规则·may", "R-may14", "R-may")
    add("④ 选取规则·apr", "R-apr16", "R-apr")
    # 组合：与「所包含单域里最好的那个」比
    def best_of(cands):
        vs = [(c, by[c]["psnr"]) for c in cands if c in by]
        return max(vs, key=lambda x: x[1])[0] if vs else None

    b2 = best_of(["R-apr", "R-may"])
    if b2:
        add(f"⑤ 组合 apr+may（vs {b2}）", "R-aprmay", b2)
    b_mj = best_of(["R-may", "R-jun"])
    if b_mj:
        add(f"⑥ 组合 may+jun（vs {b_mj}）", "R-mayjun", b_mj)
    b3 = best_of(["R-apr", "R-may", "R-jun"])
    if b3:
        add(f"⑦ 组合 全三域（vs {b3}）", "R-all", b3)
    add("⑧ 域内上界 1019×3场景", "R-in19", "R-apr")

    # Bonferroni 调整 α（按实际可算的对照数）
    n_cmp = len([c for c in contrasts if c["st"]])
    alpha_adj = 0.05 / max(n_cmp, 1)
    for c in contrasts:
        if c["st"] is None:
            continue
        st = paired(scene_mean(d12, c["a"], "psnr"),
                    scene_mean(d12, c["b"], "psnr"), alpha=alpha_adj)
        c["st"] = st
    mde = MDE_12   # 仅作参考基线；真正的判据是调整后 CI 是否排除 0

    md = [f"# 域消融 v1 汇总（主表 12 场景；{n_cmp} 条对照，Bonferroni α={alpha_adj:.4f}）\n",
          "## 臂级结果（12 场景测试子集）\n",
          "| 臂 | 种子 | PSNR | 种子std | HF-PSNR | 梯度余弦 | 洞区PSNR | SSIM |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['arm']} | {r['n_seeds']} | {r['psnr']:.3f} | {r['psnr_seed_std']:.3f} | "
                  f"{r['hf_psnr']:.2f} | {r['grad_cos']:.3f} | {r['psnr_hole']:.2f} | "
                  f"{r['ssim']:.4f} |")

    md += ["\n## 预登记对照（配对 t，12 场景）\n",
           "| 对照 | 比较 | ΔPSNR | 95% CI | 校正后 CI | 该对比 MDE | t | p | 胜/负 | 判定 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in contrasts:
        st = c["st"]
        if st is None:
            md.append(f"| {c['label']} | {c['a']} − {c['b']} | — | — | — | — | — | — | — | "
                      f"{c.get('note','缺')} |")
            continue
        md.append(f"| {c['label']} | {c['a']} − {c['b']} | **{st['d']:+.3f}** | "
                  f"[{st['lo']:+.3f}, {st['hi']:+.3f}] | "
                  f"[{st['lo_adj']:+.3f}, {st['hi_adj']:+.3f}] | "
                  f"{st['mde']:.3f} | {st['t']:+.2f} | {st['p']:.4f} | "
                  f"{st['win']}/{st['n']-st['win']} | {verdict(st)} |")

    md.append(f"\n> 判定：**Bonferroni 调整后 CI 排除 0** 才算胜出/更差；"
              f"该对比 MDE = t(1−α_adj/2, df)·SE，随每条对比的离散度变化。"
              f"「胜/负」= 12 场景中 Δ>0 的个数。")
    txt = "\n".join(md)
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(dict(arms=rows, contrasts=contrasts, mde12=MDE_12, mde_holm=mde,
                   n_contrasts=n_cmp),
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "a", "b", "d_psnr", "ci95_lo", "ci95_hi",
                    "ci_adj_lo", "ci_adj_hi", "mde", "t", "p", "win", "n", "verdict"])
        for c in contrasts:
            st = c["st"]
            if st is None:
                w.writerow([c["label"], c["a"], c["b"]] + [""] * 10 + [c.get("note", "")])
            else:
                w.writerow([c["label"], c["a"], c["b"], f"{st['d']:.4f}",
                            f"{st['lo']:.4f}", f"{st['hi']:.4f}",
                            f"{st['lo_adj']:.4f}", f"{st['hi_adj']:.4f}",
                            f"{st['mde']:.4f}", f"{st['t']:.3f}", f"{st['p']:.5f}",
                            st["win"], st["n"], verdict(st)])
    print("\n" + txt)
    print(f"\n写入 {OUT}/summary.{{md,csv,json}}")

    # 辅表：15 场景
    if d15:
        md2 = ["## 辅表：15 场景全量（与既有 M=20.901 / 官方 20.05 衔接）\n",
               "| 臂 | 种子 | PSNR | 种子std | 洞区PSNR |", "|---|---|---|---|---|"]
        for base in sorted(d15, key=lambda x: (x != "R-apr", x)):
            sm = scene_mean(d15, base, "psnr")
            hsm = scene_mean(d15, base, "psnr_hole")
            sms = []
            for s, sc in d15[base].items():
                sms.append(float(np.mean([float(x["psnr"]) for x in sc.values()])))
            std = float(np.std(sms, ddof=1)) if len(sms) > 1 else 0.0
            md2.append(f"| {base} | {len(d15[base])} | {np.mean(list(sm.values())):.3f} | "
                       f"{std:.3f} | {np.mean(list(hsm.values())):.2f} |")
        open(os.path.join(OUT, "summary_15.md"), "w", encoding="utf-8").write(
            "\n".join(md2) + "\n")
        print(f"写入 {OUT}/summary_15.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
