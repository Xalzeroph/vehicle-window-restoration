#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：级联是否学到去反射 + 训练/测试反射强度是否匹配（2026-10-21）。

要回答用户的两个质疑
------------------
1. "连去反射都没学会" → 级联在 1019 上的**实际改动量**有多大？
   若 |Δ| ≈ 0，说明它学成了 near-identity，根本没在去反射。
2. "在什么数据上微调的" → 训练数据（OpenRR + 300 实拍）的**反射强度分布**
   与 1019 测试集差多少？若训练侧反射显著更弱，模型必然欠去反射。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def pct(v, q):
    return float(np.percentile(v, q))


print("=" * 74)
print("① 级联在 1019 上的实际改动量 |Δ| = |T̂_naf − T̂_base|（[0,255] 灰度级）")
print("=" * 74)
for tag in ("naf_l1", "naf_l1_hf", "naf_l1_gc", "naf_l1_hf_gc"):
    d = []
    for sc in sorted(os.listdir(os.path.join(PRED, f"real_pair_1019_{tag}"))):
        a = read(os.path.join(PRED, f"real_pair_1019_{tag}", sc, "t_hat.png"))
        b = read(os.path.join(PRED, "real_pair_1019_rdnet_zs", sc, "t_hat.png"))
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            a = cv2.resize(a, (b.shape[1], b.shape[0]))
        d.append(float(np.abs(a - b).mean()))
    if d:
        print(f"  {tag:<14} |Δ|均值 {np.mean(d):6.2f} 级  中位 {np.median(d):6.2f}  "
              f"max {np.max(d):6.2f}   （满量程 255）")

print()
print("=" * 74)
print("② 训练数据 vs 1019 测试集的反射强度")
print("=" * 74)

# 训练侧：OpenRR meta 自带的 R_abs_mean / R_abs_p95
om = os.path.join(P, "data/train_sets/train_pairs_opensrc_0920/meta.json")
r_open = []
if os.path.exists(om):
    for x in json.load(open(om, encoding="utf-8")):
        if str(x.get("src", "")).lower() == "sir2":
            continue
        if x.get("R_abs_mean") is not None:
            r_open.append(float(x["R_abs_mean"]))
if r_open:
    print(f"  OpenRR-5k (训练, n={len(r_open)})   R_abs_mean: "
          f"p10 {pct(r_open,10):5.2f}  中位 {pct(r_open,50):5.2f}  p90 {pct(r_open,90):6.2f}")

# 训练侧：300 实拍配对
rm = os.path.join(P, "data/train_sets/real_pairs_1020/meta.json")
r_real = []
if os.path.exists(rm):
    m = json.load(open(rm, encoding="utf-8"))
    r_real = [float(v["R_abs_mean"]) for v in m.values() if "R_abs_mean" in v]
if r_real:
    print(f"  实拍 300 对 (训练, n={len(r_real)})   R_abs_mean: "
          f"p10 {pct(r_real,10):5.2f}  中位 {pct(r_real,50):5.2f}  p90 {pct(r_real,90):6.2f}")

# 测试侧：1019 的 |I−GT| 逐对统计（与上面 R_abs_mean 同口径）
t19 = []
for sc in sorted(os.listdir(PAIR)):
    d = os.path.join(PAIR, sc)
    if not os.path.isdir(d):
        continue
    G = read(os.path.join(d, "GT.png"))
    if G is None:
        continue
    for f in sorted(os.listdir(d)):
        if f.startswith("I_") and f.endswith(".png"):
            I = read(os.path.join(d, f))
            if I is not None:
                t19.append(float(np.abs(I - G).mean()))
if t19:
    print(f"  1019 测试 (n={len(t19)})            |I−GT| 均值: "
          f"p10 {pct(t19,10):5.2f}  中位 {pct(t19,50):5.2f}  p90 {pct(t19,90):6.2f}")

print()
print("=" * 74)
print("③ 各臂实际去掉了多少反射（用 |T̂ − T| 相对 |I − T| 的下降）")
print("=" * 74)
print(f"{'臂':<14}{'|I−GT|':>9}{'|T̂−GT|':>9}{'降幅%':>8}{'PSNR':>8}")
print("-" * 48)
base_res = None
for arm, sub in (("identity", None), ("rdnet_zs", "real_pair_1019_rdnet_zs"),
                 ("ft_s2main", "real_pair_1019_ft_s2main"),
                 ("ft_mixbest", "real_pair_1019_ft_mixbest"),
                 ("naf_l1", "real_pair_1019_naf_l1")):
    ri, rt, ps = [], [], []
    for sc in sorted(os.listdir(PAIR)):
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        G = read(os.path.join(d, "GT.png"))
        if G is None:
            continue
        for f in sorted(os.listdir(d)):
            if not (f.startswith("I_") and f.endswith(".png")):
                continue
            stem = os.path.splitext(f)[0]
            I = read(os.path.join(d, f))
            if I is None:
                continue
            if arm == "identity":
                T_hat = I
            else:
                p = os.path.join(PRED, sub, f"{sc}__{stem}", "t_hat.png")
                if not os.path.exists(p):
                    continue
                T_hat = read(p)
                if T_hat.shape != G.shape:
                    T_hat = cv2.resize(T_hat, (G.shape[1], G.shape[0]))
            ri.append(float(np.abs(I - G).mean()))
            rt.append(float(np.abs(T_hat - G).mean()))
            mse = [float(np.mean((T_hat[..., c] - G[..., c]) ** 2)) for c in range(3)]
            ps.append(float(np.mean([99 if m <= 1e-12 else 10*np.log10(255**2/m) for m in mse])))
    if ri:
        a, b = np.mean(ri), np.mean(rt)
        print(f"{arm:<14}{a:>9.2f}{b:>9.2f}{(1-b/a)*100:>7.1f}%{np.mean(ps):>8.2f}")
