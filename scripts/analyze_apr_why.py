#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为什么 apr 同为室外却最差：把各域与 1019 的可测统计量摆出来（2026-10-23）。

要检验的候选解释
--------------
H1 色调/曝光错配：apr 是 4 月室外、1019 是 10 月室外，GT 的整体亮度/色偏不同
   → 训练把输出拉向 4 月的"正确色调"，在 10 月上就是系统性偏差
H2 改写幅度错配：apr 的 Ω 占比大（需改写 20% 画面），1019 只需 8%
   → 学到"大幅改写"，在 1019 上过冲
H3 反射强度错配：|I−T| 差异
H4 域内多样性：场景数/帧数（已知 apr 126/6、may 130/5、jun 44/2）

测的量
------
每个域（含 1019 测试集）统计：n、Ω 中位、T 的 luma 均值/标准差、T 的逐通道均值（色偏）、
I 的 luma、|I−T| 中位、弱反射帧占比。
"""
from __future__ import annotations

import collections
import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")


def stats_dir(i_dir, t_dir, om_dir, label, limit=None):
    if not os.path.isdir(i_dir):
        print(f"  {label:<22} 缺目录")
        return None
    names = sorted(f for f in os.listdir(i_dir) if f.endswith(".png"))
    if limit:
        names = names[:limit]
    tl, ts, tc, il, it, om = [], [], [], [], [], []
    for f in names:
        T = cv2.imread(os.path.join(t_dir, f))
        I = cv2.imread(os.path.join(i_dir, f))
        if T is None or I is None:
            continue
        if T.shape[:2] != I.shape[:2]:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))
        Tg = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
        Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        tl.append(Tg.mean()); ts.append(Tg.std())
        tc.append([T[..., c].mean() for c in range(3)])       # B,G,R
        il.append(Ig.mean())
        it.append(np.abs(I.astype(np.float64) - T.astype(np.float64)).mean())
        p = os.path.join(om_dir, f)
        if os.path.exists(p):
            m = cv2.imread(p, 0)
            if m is not None:
                om.append((m > 127).mean())
    if not tl:
        print(f"  {label:<22} 无有效样本")
        return None
    tc = np.array(tc)
    med = lambda v: float(np.median(v))
    print(f"  {label:<22}{len(tl):>5}{med(om) if om else float('nan'):>8.3f}"
          f"{med(tl):>8.1f}{med(ts):>7.1f}"
          f"{tc[:,0].mean():>7.1f}{tc[:,1].mean():>7.1f}{tc[:,2].mean():>7.1f}"
          f"{med(il):>8.1f}{med(it):>8.2f}")
    return dict(n=len(tl), omega=med(om) if om else None, T_luma=med(tl),
                T_std=med(ts), T_B=float(tc[:, 0].mean()), T_G=float(tc[:, 1].mean()),
                T_R=float(tc[:, 2].mean()), I_luma=med(il), I_minus_T=med(it))


print("=" * 96)
print("各域训练样本 vs 1019 测试集的图像统计量（[0,255]）")
print("=" * 96)
hdr = (f"  {'数据集':<22}{'n':>5}{'Ω中位':>8}{'T亮度':>8}{'T标准差':>7}"
       f"{'T_B':>7}{'T_G':>7}{'T_R':>7}{'I亮度':>8}{'|I-T|':>8}")
print(hdr)
print("  " + "-" * 92)

res = {}
m = json.load(open(os.path.join(POOL, "meta.json"), encoding="utf-8"))
by = collections.defaultdict(list)
for case, v in m.items():
    by[str(v.get("domain"))].append(case)

for dom in ("apr", "may", "jun"):
    # 该域的子集：临时按名字过滤
    sub = set(by[dom])
    tl, ts, tc, il, it, om = [], [], [], [], [], []
    for case in sorted(sub):
        T = cv2.imread(os.path.join(POOL, "T", case + ".png"))
        I = cv2.imread(os.path.join(POOL, "I", case + ".png"))
        mk = cv2.imread(os.path.join(POOL, "omega", case + ".png"), 0)
        if T is None or I is None:
            continue
        if T.shape[:2] != I.shape[:2]:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))
        Tg = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
        Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        tl.append(Tg.mean()); ts.append(Tg.std())
        tc.append([T[..., c].mean() for c in range(3)])
        il.append(Ig.mean())
        it.append(np.abs(I.astype(np.float64) - T.astype(np.float64)).mean())
        if mk is not None:
            om.append((mk > 127).mean())
    if not tl:
        continue
    tc = np.array(tc)
    med = lambda v: float(np.median(v))
    print(f"  {dom:<22}{len(tl):>5}{med(om):>8.3f}{med(tl):>8.1f}{med(ts):>7.1f}"
          f"{tc[:,0].mean():>7.1f}{tc[:,1].mean():>7.1f}{tc[:,2].mean():>7.1f}"
          f"{med(il):>8.1f}{med(it):>8.2f}")
    res[dom] = dict(n=len(tl), omega=med(om), T_luma=med(tl), T_std=med(ts),
                    T_B=float(tc[:, 0].mean()), T_G=float(tc[:, 1].mean()),
                    T_R=float(tc[:, 2].mean()), I_luma=med(il), I_minus_T=med(it))

# 1019 测试集
T19 = os.path.join(P, "data/test_sets/real_pair_1019")
tl, ts, tc, il, it = [], [], [], [], []
for sc in sorted(os.listdir(T19)):
    d = os.path.join(T19, sc)
    if not os.path.isdir(d):
        continue
    T = cv2.imread(os.path.join(d, "GT.png"))
    I = cv2.imread(os.path.join(d, "I_000000.png"))
    if T is None or I is None:
        continue
    Tg = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
    Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
    tl.append(Tg.mean()); ts.append(Tg.std())
    tc.append([T[..., c].mean() for c in range(3)])
    il.append(Ig.mean())
    it.append(np.abs(I.astype(np.float64) - T.astype(np.float64)).mean())
tc = np.array(tc)
med = lambda v: float(np.median(v))
print(f"  {'1019 测试集':<22}{len(tl):>5}{float('nan'):>8}"
      f"{med(tl):>8.1f}{med(ts):>7.1f}"
      f"{tc[:,0].mean():>7.1f}{tc[:,1].mean():>7.1f}{tc[:,2].mean():>7.1f}"
      f"{med(il):>8.1f}{med(it):>8.2f}")
res["1019"] = dict(n=len(tl), T_luma=med(tl), T_std=med(ts),
                   T_B=float(tc[:, 0].mean()), T_G=float(tc[:, 1].mean()),
                   T_R=float(tc[:, 2].mean()), I_luma=med(il), I_minus_T=med(it))

print()
print("=" * 96)
print("与 1019 的偏离（绝对值，越小越接近测试集）")
print("=" * 96)
print(f"  {'域':<8}{'ΔT亮度':>9}{'ΔT标准差':>10}{'ΔT_B':>8}{'ΔT_G':>8}{'ΔT_R':>8}"
      f"{'ΔI亮度':>9}{'Δ|I-T|':>9}{'ΔΩ':>8}")
print("  " + "-" * 76)
t19 = res["1019"]
for dom in ("apr", "may", "jun"):
    if dom not in res:
        continue
    r = res[dom]
    print(f"  {dom:<8}{abs(r['T_luma']-t19['T_luma']):>9.1f}"
          f"{abs(r['T_std']-t19['T_std']):>10.1f}"
          f"{abs(r['T_B']-t19['T_B']):>8.1f}{abs(r['T_G']-t19['T_G']):>8.1f}"
          f"{abs(r['T_R']-t19['T_R']):>8.1f}"
          f"{abs(r['I_luma']-t19['I_luma']):>9.1f}"
          f"{abs(r['I_minus_T']-t19['I_minus_T']):>9.2f}"
          f"{abs((r['omega'] or 0)-(0)):>8.3f}")
json.dump(res, open("/tmp/domain_stats.json", "w"), ensure_ascii=False, indent=1)
print("\n写入 /tmp/domain_stats.json")
