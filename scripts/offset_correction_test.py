#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零 GPU 验证：apr 的偏置是"常数偏亮"，可否用训练域自估的常数校正回来（2026-10-23）。

背景（已实测）
------------
apr 臂输出的最优集级仿射是 `a≈0.98（≈1）、b≈−12`，即**纯加性偏亮约 12 灰阶**；
而 apr 训练对的 I−T 亮度差只有 4.2，1019 需要 16.4 → 模型学不到"该变暗多少"。

本脚本测三件事
------------
1. **oracle 常数校正**：对 apr 输出统一减 k，扫 k，看 PSNR 峰值与增益（上界）
2. **可部署估计**：只用训练域统计（apr 的 mean(I)−mean(T)=4.2）而**不看 GT**，
   推出 k_deploy = (target_pct − apr_pct)·mean(I_1019)，看能拿到多少
   （target_pct 用各候选：训练池 7.0% / 1019 的 14.1% 作对照）
3. **是否只对 apr 有效**：同法套到 may / mayjun / in19，看增益是否随"原本就校准好"而变小
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PR = os.path.join(P, "runs/pred")
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")

ARMS = [("R-apr", "real_pair_1019_dom_R-apr_s101"),
        ("R-apr", "real_pair_1019_dom_R-apr_s202"),
        ("R-apr", "real_pair_1019_dom_R-apr_s303"),
        ("R-may", "real_pair_1019_dom_R-may_s101"),
        ("R-mayjun", "real_pair_1019_dom_R-mayjun_s101"),
        ("R-jun", "real_pair_1019_dom_R-jun_s101"),
        ("R-in19", "real_pair_1019_dom_R-in19_s101")]

KS = list(range(-20, 21, 1))          # 统一减 k 灰阶（负=加亮）


def psnr(a, g):
    mse = [float(np.mean((a[..., c].astype(np.float64) - g[..., c].astype(np.float64)) ** 2))
           for c in range(3)]
    return float(np.mean([99.0 if m <= 1e-12 else 10 * np.log10(255.0 ** 2 / m) for m in mse]))


scenes = sorted(x for x in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, x)))
GT, I19 = {}, {}
for sc in scenes:
    d = os.path.join(PAIR, sc)
    GT[sc] = cv2.imread(os.path.join(d, "GT.png"), cv2.IMREAD_COLOR)
    I19[sc] = cv2.imread(os.path.join(d, "I_000000.png"), cv2.IMREAD_COLOR)
I19_luma = float(np.mean([cv2.cvtColor(I19[s], cv2.COLOR_BGR2GRAY).mean() for s in scenes]))
GT_luma = float(np.mean([cv2.cvtColor(GT[s], cv2.COLOR_BGR2GRAY).mean() for s in scenes]))
need_pct = (I19_luma - GT_luma) / I19_luma

print("=" * 84)
print("1019 输入/真值亮度")
print("=" * 84)
print(f"  I 亮度 {I19_luma:.1f}   GT 亮度 {GT_luma:.1f}   需要的暗化 {I19_luma-GT_luma:.1f} "
      f"灰阶（{100*need_pct:.1f}%）")

# 训练域统计（用来做"可部署"估计，不看 GT）
APR_PCT = 0.032      # 已实测：apr 的 mean(I−T)/mean(I)
POOL_PCT = 0.070     # 三域加权

print()
print("=" * 84)
print("各臂：oracle 常数校正的扫描结果")
print("=" * 84)
print(f"  {'臂':<10}{'raw':>8}{'最优k':>7}{'校正后':>9}{'增益':>8}{'可部署k':>9}{'部署后':>9}{'部署增益':>10}")
print("  " + "-" * 76)
out = {}
for arm, pred_dir in ARMS:
    root = os.path.join(PR, pred_dir)
    if not os.path.isdir(root):
        print(f"  {arm:<10} 缺 {pred_dir}")
        continue
    imgs = {}
    for sc in scenes:
        p = os.path.join(root, f"{sc}__I_000000", "t_hat.png")
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            g = GT[sc]
            if im.shape != g.shape:
                im = cv2.resize(im, (g.shape[1], g.shape[0]))
            imgs[sc] = im
    if not imgs:
        continue
    raw = float(np.mean([psnr(imgs[s], GT[s]) for s in imgs]))
    curve = {}
    for k in KS:
        curve[k] = float(np.mean([psnr(np.clip(imgs[s].astype(np.float64) - k, 0, 255), GT[s])
                                  for s in imgs]))
    kbest = max(curve, key=curve.get)
    # 可部署：k = (pool_pct − apr_pct) * I19_luma，其中 I19_luma 是**输入**统计（非 GT）
    kdep = int(round((POOL_PCT - APR_PCT) * I19_luma))
    dep = curve.get(kdep, float("nan"))
    out.setdefault(arm, []).append((raw, kbest, curve[kbest], kdep, dep))
    print(f"  {arm:<10}{raw:>8.3f}{kbest:>+7d}{curve[kbest]:>9.3f}{curve[kbest]-raw:>+8.3f}"
          f"{kdep:>+9d}{dep:>9.3f}{dep-raw:>+10.3f}")

print()
print("=" * 84)
print("多种子臂取均值")
print("=" * 84)
print(f"  {'臂':<10}{'raw':>8}{'最优k':>7}{'校正后':>9}{'增益':>8}{'部署k':>7}{'部署后':>9}{'部署增益':>10}")
print("  " + "-" * 76)
summary = {}
for arm, vals in out.items():
    a = np.array(vals, dtype=float).mean(axis=0)
    summary[arm] = dict(raw=a[0], k=int(round(a[1])), corrected=a[2],
                        gain=a[2] - a[0], k_dep=int(round(a[3])), dep=a[4],
                        dep_gain=a[4] - a[0])
    print(f"  {arm:<10}{a[0]:>8.3f}{int(round(a[1])):>+7d}{a[2]:>9.3f}{a[2]-a[0]:>+8.3f}"
          f"{int(round(a[3])):>+7d}{a[4]:>9.3f}{a[4]-a[0]:>+10.3f}")

print()
print("=" * 84)
print("结论判据")
print("=" * 84)
if "R-apr" in summary:
    s = summary["R-apr"]
    print(f"  apr 最优常数 k = {s['k']:+d} 灰阶，增益 {s['gain']:+.3f} dB")
    print(f"  （与先前仿射分解给出的 b ≈ −12 是否一致：{'是' if abs(s['k'] + 12) <= 3 else '否'}）")
    print(f"  可部署常数 k = {s['k_dep']:+d}，增益 {s['dep_gain']:+.3f} dB")
    verdict = "成立" if s["gain"] >= 1.0 else ("部分成立" if s["gain"] >= 0.5 else "不成立")
    print(f"  → 「apr 的亏空主要是常数偏亮」: {verdict}")
    if s["dep_gain"] >= 0.8 * s["gain"]:
        print(f"  → 「可用训练域统计自估 k，无需 GT」: 成立（部署增益达上界 "
              f"{s['dep_gain']/s['gain']*100:.0f}%）")
    else:
        print(f"  → 「可用训练域统计自估 k」: 不足（仅达上界 "
              f"{s['dep_gain']/max(s['gain'],1e-6)*100:.0f}%），需更强估计")
for a in ("R-may", "R-in19"):
    if a in summary:
        print(f"  {a} 增益 {summary[a]['gain']:+.3f} dB（原本校准好的臂应增益小 → 交叉验证）")

os.makedirs(os.path.join(P, "eval/queue"), exist_ok=True)
json.dump(dict(I19_luma=I19_luma, GT_luma=GT_luma, need_pct=need_pct,
               apr_pct=APR_PCT, pool_pct=POOL_PCT, summary=summary),
          open(os.path.join(P, "eval/queue/offset_test.json"), "w"),
          ensure_ascii=False, indent=1)
print("\n写入 eval/queue/offset_test.json")
