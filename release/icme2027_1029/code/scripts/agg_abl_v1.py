#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""损失消融 v1 聚合（2026-10-21）。

输入
----
`eval/table_final_1019_abl_v1.csv`（`table_final_1019.py --tag abl_v1` 产出，逐 场景×臂）
`runs/pred/real_pair_1019_abl_<arm>/`（算去反射率用）

输出
----
`eval/abl_v1/summary.{md,csv,json}`

统计口径（红队修订版）
--------------------
· 每臂先按**种子求逐场景均值**，得 m[scene]；臂均值 = mean(m)
· 种子 std = 各 seed 的臂级均值之间的标准差（衡量训练随机性）
· 与 C0 比较：d[scene] = m_A − m_C0，做 15 场景**配对 t 检验**，给 Δ、sd、t、p、**95% CI**
· 分级：CI 下界 ≥ +1.0 dB → 胜出；CI 全落 ±1.0 → 无可检出效应；CI 上界 ≤ −1.0 → 更差
  （1.0 dB = 5 臂 Holm 校正后的 MDE；n=15、配对 sd≈1.0 时 MDE@80% 为 0.77 dB）
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
EV = os.path.join(P, "eval")
OUT = os.path.join(EV, "abl_v1")
PRED = os.path.join(P, "runs/pred")
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
MDE = 1.0
BASE = "C0"


def t_crit_and_p(t, df):
    """双侧 t 检验 p 值 + t 临界值（无 scipy 时退化为正态近似）。

    注意：p = 2·(1 − CDF(|t|))，曾漏掉 `1 −` 导致输出成 0.98 这类错误值。
    """
    if df <= 0 or not np.isfinite(t):
        return float("nan"), float("nan")
    try:
        from scipy import stats
        p = 2.0 * (1.0 - float(stats.t.cdf(abs(t), df)))
        return p, float(stats.t.ppf(0.975, df))
    except Exception:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        return p, 1.96


def removal_rate(arm):
    """去反射率 = 1 − mean|T̂−GT| / mean|I−GT|（逐对 L1，越高越接近真值）。"""
    root = os.path.join(PRED, f"real_pair_1019_{arm}")
    if not os.path.isdir(root):
        return None
    ri, rt = [], []
    for sc in sorted(os.listdir(PAIR)):
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        G = cv2.imread(os.path.join(d, "GT.png"))
        if G is None:
            continue
        for f in sorted(os.listdir(d)):
            if not (f.startswith("I_") and f.endswith(".png")):
                continue
            stem = os.path.splitext(f)[0]
            I = cv2.imread(os.path.join(d, f))
            p = os.path.join(root, f"{sc}__{stem}", "t_hat.png")
            if I is None or not os.path.exists(p):
                continue
            T = cv2.imread(p)
            if T.shape != G.shape:
                T = cv2.resize(T, (G.shape[1], G.shape[0]))
            ri.append(float(np.abs(I.astype(np.float64) - G).mean()))
            rt.append(float(np.abs(T.astype(np.float64) - G).mean()))
    if not ri:
        return None
    return 1.0 - float(np.mean(rt)) / float(np.mean(ri))


METRICS = [("psnr", "PSNR↑"), ("hf_psnr", "HF-PSNR↑"), ("grad_cos", "梯度余弦↑"),
           ("psnr_hole", "洞区 PSNR↑")]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cp = os.path.join(EV, "table_final_1019_abl_v1.csv")
    if not os.path.exists(cp):
        raise SystemExit(f"缺 {cp}；先跑 table_final_1019.py --tag abl_v1")
    rows = list(csv.DictReader(open(cp, encoding="utf-8")))
    print(f"读入 {len(rows)} 行，臂: {sorted(set(r['arm'] for r in rows))}")

    # 收集：base -> seed -> scene -> row
    data = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        arm = r["arm"]
        # 权威表里的臂名带前缀（abl_C0_s101），先剥掉再切种子
        if arm.startswith("abl_"):
            arm = arm[4:]
        if "_s" not in arm:
            continue
        base, seed = arm.rsplit("_s", 1)
        key = r["scene"] + "__" + r["input"]
        data[base][seed][key] = r

    def scene_mean(base, metric):
        """该臂按种子求逐场景均值 → {scene: value}"""
        seeds = data[base]
        scenes = set()
        for s in seeds.values():
            scenes |= set(s.keys())
        out = {}
        for k in scenes:
            v = [float(seeds[s][k][metric]) for s in seeds if k in seeds[s]]
            if v:
                out[k] = float(np.mean(v))
        return out

    bases = sorted(data.keys(), key=lambda x: (x != BASE, x))
    base_scenes = {b: scene_mean(b, "psnr") for b in bases}
    if BASE not in base_scenes or not base_scenes[BASE]:
        raise SystemExit(f"缺基线 {BASE}")

    recs = []
    for b in bases:
        rec = {"arm": b, "n_seeds": len(data[b])}
        # 种子级臂均值（用于种子 std）
        seed_arm_mean = {}
        for s, sc_rows in data[b].items():
            seed_arm_mean[s] = float(np.mean([float(x["psnr"]) for x in sc_rows.values()]))
        rec["psnr"] = float(np.mean(list(seed_arm_mean.values())))
        rec["psnr_seed_std"] = float(np.std(list(seed_arm_mean.values()), ddof=1)) \
            if len(seed_arm_mean) > 1 else 0.0
        for m, _ in METRICS[1:]:
            sm = scene_mean(b, m)
            rec[m] = float(np.mean(list(sm.values()))) if sm else float("nan")
        rr = []
        for s in data[b]:
            v = removal_rate(f"abl_{b}_s{s}")
            if v is not None:
                rr.append(v)
        rec["removal"] = float(np.mean(rr)) if rr else float("nan")

        # 与基线配对
        cs = base_scenes[BASE]
        common = sorted(set(base_scenes[b]) & set(cs))
        d = [base_scenes[b][k] - cs[k] for k in common]
        n = len(d)
        mean = float(np.mean(d))
        sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 0 else float("nan")
        if se == 0 or not np.isfinite(se):
            t, p, tc = 0.0, 1.0, float("nan")   # 自比（d 全 0）或退化
        else:
            t = mean / se
            p, tc = t_crit_and_p(t, n - 1)
        rec.update(d_psnr=mean, d_sd=sd, d_t=t, d_p=p,
                   d_ci_lo=mean - (tc * se if np.isfinite(tc) else 0.0),
                   d_ci_hi=mean + (tc * se if np.isfinite(tc) else 0.0),
                   n_pairs=n, win=sum(1 for x in d if x > 0))
        # 分级（按 CI 与 MDE；并区分方向）
        if b == BASE:
            rec["verdict"] = "基线"
        elif rec["d_ci_lo"] >= MDE:
            rec["verdict"] = "✅ 胜出"
        elif rec["d_ci_hi"] <= -MDE:
            rec["verdict"] = "❌ 更差"
        elif rec["d_ci_lo"] <= 0 <= rec["d_ci_hi"]:
            rec["verdict"] = "无可检出效应"
        elif rec["d_ci_lo"] > 0:
            rec["verdict"] = "↑ 有正方向但未达 MDE"
        else:
            rec["verdict"] = "↓ 有负方向但未达 MDE"
        recs.append(rec)

    # ---- 输出 ----
    keys = ["arm", "n_seeds", "psnr", "psnr_seed_std", "d_psnr", "d_ci_lo", "d_ci_hi",
            "d_t", "d_p", "win", "hf_psnr", "grad_cos", "psnr_hole", "removal", "verdict"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    json.dump(recs, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    md = [f"# 损失消融 v1 汇总（{len(recs)} 臂；基线 {BASE}；MDE={MDE} dB）\n",
          "| 臂 | 种子 | PSNR | 种子std | **ΔPSNR** | **95% CI** | t | p | 胜/负 | HF-PSNR | 梯度余弦 | 洞区 | 去反射率 | 判定 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        md.append(
            f"| {r['arm']} | {r['n_seeds']} | {r['psnr']:.3f} | {r['psnr_seed_std']:.3f} | "
            f"**{r['d_psnr']:+.3f}** | [{r['d_ci_lo']:+.3f}, {r['d_ci_hi']:+.3f}] | "
            f"{r['d_t']:+.2f} | {r['d_p']:.3f} | {r['win']}/{r['n_pairs']-r['win']} | "
            f"{r['hf_psnr']:.2f} | {r['grad_cos']:.3f} | {r['psnr_hole']:.2f} | "
            f"{r['removal']*100 if not math.isnan(r['removal']) else float('nan'):.1f}% | {r['verdict']} |")
    md.append(f"\n> 判定规则：CI 下界 ≥ +{MDE} dB → 胜出；CI 全落 ±{MDE} → 无可检出效应；"
              f"CI 上界 ≤ −{MDE} → 更差。MDE 来自 n=15、配对 sd≈1.0、5 臂 Holm 校正。")
    txt = "\n".join(md)
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print("\n" + txt)
    print(f"\n写入 {OUT}/summary.{{md,csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
