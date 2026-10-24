#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apr 为什么会害：机制量化为「训练域的去反射暗化量」与性能的相关（2026-10-23）。

假设
----
去反射本质上是"把反射叠加的亮度去掉"，所以训练域教会模型**暗多少**。
若测试域需要的暗化量与训练域差很多，模型就系统性偏亮/偏暗 → PSNR 掉。

测的量（每个训练臂，按实际参与训练的对数与域构成加权）
    I_luma, T_luma, darkening = I_luma − T_luma（灰阶），dark_pct = darkening / I_luma
测试域需要：1019 的 darkening / dark_pct
然后算 |dark_pct_train − dark_pct_1019| 与 PSNR 的 Spearman 相关。
"""
from __future__ import annotations

import collections
import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")
MAY14 = os.path.join(P, "data/train_sets/pairs_may")
T19 = os.path.join(P, "data/test_sets/real_pair_1019")

# 12 场景主表的 PSNR（多种子取均值）
PSNR = {"R-apr": 20.203, "R-may": 22.547, "R-jun": 19.889, "R-mayjun": 23.242,
        "R-all": 22.043, "R-aprmay": 21.238, "R-aprjun": 21.193, "R-in19": 26.416,
        "R-may14": 20.944, "R-apr16": 19.595}

# 各臂的域构成（与训练实际一致）
COMPOS = {
    "R-apr":    {"apr": 126},
    "R-may":    {"may": 130},
    "R-jun":    {"jun": 44},
    "R-mayjun": {"may": 130, "jun": 44},
    "R-all":    {"apr": 126, "may": 130, "jun": 44},
    "R-aprmay": {"apr": 126, "may": 130},
    "R-aprjun": {"apr": 126, "jun": 44},
}


def luma_stats(i_dir, t_dir, cases=None):
    names = sorted(f for f in os.listdir(i_dir) if f.endswith(".png"))
    if cases is not None:
        names = [f for f in names if f[:-4] in cases]
    il, tl = [], []
    for f in names:
        raw = f[:-4].split("__")[1] if "__" in f else ""
        pass
    # 直接逐图算
    for f in names:
        I = cv2.imread(os.path.join(i_dir, f))
        T = cv2.imread(os.path.join(t_dir, f))
        if I is None or T is None:
            continue
        if I.shape[:2] != T.shape[:2]:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))
        il.append(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).mean())
        tl.append(cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).mean())
    return float(np.mean(il)), float(np.mean(tl)), len(il)


print("=" * 92)
print("各域的去反射「暗化量」（I 亮度 − T 亮度）")
print("=" * 92)
m = json.load(open(os.path.join(POOL, "meta.json"), encoding="utf-8"))
by = collections.defaultdict(set)
for case, v in m.items():
    by[str(v.get("domain"))].add(case)

dom_stats = {}
for dom in ("apr", "may", "jun"):
    # 用 meta 里的 case 集合过滤 I 目录
    il, tl = [], []
    for case in sorted(by[dom]):
        I = cv2.imread(os.path.join(POOL, "I", case + ".png"))
        T = cv2.imread(os.path.join(POOL, "T", case + ".png"))
        if I is None or T is None:
            continue
        if I.shape[:2] != T.shape[:2]:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))
        il.append(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).mean())
        tl.append(cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).mean())
    il, tl = float(np.mean(il)), float(np.mean(tl))
    dom_stats[dom] = dict(n=len(il if isinstance(il, list) else []), I=il, T=tl,
                          dark=il - tl, pct=(il - tl) / il)
    print(f"  {dom:<6} n={len(by[dom]):>4}  I={il:6.1f}  T={tl:6.1f}  "
          f"暗化={il-tl:5.1f} 灰阶  ({100*(il-tl)/il:4.1f}%)")

# may14（手挑）
i14 = os.path.join(MAY14, "I"); t14 = os.path.join(MAY14, "T")
if os.path.isdir(i14):
    il, tl = [], []
    for f in sorted(os.listdir(i14)):
        if not f.endswith(".png"):
            continue
        I = cv2.imread(os.path.join(i14, f)); T = cv2.imread(os.path.join(t14, f))
        if I is None or T is None:
            continue
        if I.shape[:2] != T.shape[:2]:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))
        il.append(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).mean())
        tl.append(cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).mean())
    il, tl = float(np.mean(il)), float(np.mean(tl))
    dom_stats["may14"] = dict(I=il, T=tl, dark=il - tl, pct=(il - tl) / il)
    print(f"  may14  n={len(os.listdir(i14)):>4}  I={il:6.1f}  T={tl:6.1f}  "
          f"暗化={il-tl:5.1f} 灰阶  ({100*(il-tl)/il:4.1f}%)")

# 1019 需要多少
il, tl = [], []
for sc in sorted(os.listdir(T19)):
    d = os.path.join(T19, sc)
    if not os.path.isdir(d):
        continue
    T = cv2.imread(os.path.join(d, "GT.png"))
    I = cv2.imread(os.path.join(d, "I_000000.png"))
    if T is None or I is None:
        continue
    il.append(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).mean())
    tl.append(cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).mean())
il19, tl19 = float(np.mean(il)), float(np.mean(tl))
need = (il19 - tl19) / il19
print(f"  {'1019':<6} n={len(il):>4}  I={il19:6.1f}  T={tl19:6.1f}  "
      f"暗化={il19-tl19:5.1f} 灰阶  ({100*need:4.1f}%)   ← 测试域需要的量")

print()
print("=" * 92)
print("训练臂的加权暗化比例 vs PSNR")
print("=" * 92)
print(f"  {'臂':<10}{'对数':>5}{'暗化比例':>9}{'与1019差':>10}{'|差|':>8}{'PSNR':>8}")
print("  " + "-" * 50)
rows = []
for arm, comp in COMPOS.items():
    tot = sum(comp.values())
    pct = sum(n * dom_stats[d]["pct"] for d, n in comp.items()) / tot
    gap = abs(pct - need)
    rows.append((arm, tot, pct, pct - need, gap, PSNR[arm]))
    print(f"  {arm:<10}{tot:>5}{100*pct:>8.1f}%{100*(pct-need):>+9.1f}pp"
          f"{100*gap:>7.1f}pp{PSNR[arm]:>8.3f}")

# may14 单独
if "may14" in dom_stats:
    pct = dom_stats["may14"]["pct"]
    rows.append(("R-may14", 14, pct, pct - need, abs(pct - need), PSNR["R-may14"]))
    print(f"  {'R-may14':<10}{14:>5}{100*pct:>8.1f}%{100*(pct-need):>+9.1f}pp"
          f"{100*abs(pct-need):>7.1f}pp{PSNR['R-may14']:>8.3f}")

# Spearman
def spearman(x, y):
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = ((rx - ry) ** 2).sum()
    return 1 - 6 * d2 / (n * (n * n - 1))

gaps = [r[4] for r in rows]
perf = [r[5] for r in rows]
print()
print(f"  全部 {len(rows)} 臂：Spearman(|暗化比例差|, PSNR) = {spearman(gaps, perf):+.3f}")
sub = [r for r in rows if r[0] != "R-jun"]
print(f"  去掉 jun（仅 44 对/2 场景，欠训）{len(sub)} 臂："
      f"Spearman = {spearman([r[4] for r in sub], [r[5] for r in sub]):+.3f}")
print(f"\n  1019 需要的暗化比例 = {100*need:.1f}%")

json.dump(dict(dom_stats=dom_stats, need=need, rows=rows),
          open("/tmp/dark_analysis.json", "w"), ensure_ascii=False, indent=1, default=str)
print("\n写入 /tmp/dark_analysis.json")
