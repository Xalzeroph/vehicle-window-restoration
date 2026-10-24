#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apr 为什么最差：把误差分解为「色调」与「结构」（2026-10-23，零 GPU）。

思路
----
若 apr 的问题主要是"把输出色调拉向 4 月室外"，那么**用最优仿射把输出对齐到 GT 的色调后**，
它的 PSNR 应当明显追上 may。分解方式：

  raw PSNR          : 原样
  oracle 逐图仿射    : 每张图、每通道各拟合 (a,b)，P' = a·P + b 最小化 ‖P'−GT‖²（上界，不可部署）
  数据集级仿射       : 所有图共用一组 (a,b)（较弱的 oracle）
  tone-free 后的 Δ  : 与 may 的差距还剩多少

仿射解（逐通道，闭式）：
  a = cov(P,G)/var(P)，b = mean(G) − a·mean(P)
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PR = os.path.join(P, "runs/pred")
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")

ARMS = [("R-apr", "dom_R-apr_s101"), ("R-apr", "dom_R-apr_s202"), ("R-apr", "dom_R-apr_s303"),
        ("R-may", "dom_R-may_s101"),
        ("R-jun", "dom_R-jun_s101"),
        ("R-mayjun", "dom_R-mayjun_s101"),
        ("R-aprmay", "dom_R-aprmay_s101"),
        ("R-aprjun", "dom_R-aprjun_s101"),
        ("R-all", "dom_R-all_s101"),
        ("R-may14", "dom_R-may14_s101"), ("R-may14", "dom_R-may14_s202"),
        ("R-may14", "dom_R-may14_s303"),
        ("R-apr16", "dom_R-apr16_s101"),
        ("R-in19", "dom_R-in19_s101")]


def load_case(sc):
    d = os.path.join(PAIR, sc)
    G = cv2.imread(os.path.join(d, "GT.png"), cv2.IMREAD_COLOR)
    return G


def psnr(a, g):
    mse = [float(np.mean((a[..., c].astype(np.float64) - g[..., c].astype(np.float64)) ** 2))
           for c in range(3)]
    return float(np.mean([99.0 if m <= 1e-12 else 10 * np.log10(255.0 ** 2 / m) for m in mse]))


def affine_per_image(Pt, G):
    """逐图逐通道最优 (a,b)。返回拟合后的图。"""
    out = np.empty_like(Pt, dtype=np.float64)
    for c in range(3):
        p = Pt[..., c].astype(np.float64).ravel()
        g = G[..., c].astype(np.float64).ravel()
        vp = p.var()
        a = (np.cov(p, g, bias=True)[0, 1] / vp) if vp > 1e-9 else 1.0
        b = g.mean() - a * p.mean()
        out[..., c] = a * Pt[..., c] + b
    return np.clip(out, 0, 255)


def affine_dataset(imgs_p, imgs_g):
    """所有图共用一组 (a,b)。返回 (a,b) 与按此变换后的每图。"""
    coef = []
    for c in range(3):
        p = np.concatenate([x[..., c].astype(np.float64).ravel() for x in imgs_p])
        g = np.concatenate([x[..., c].astype(np.float64).ravel() for x in imgs_g])
        vp = p.var()
        a = (np.cov(p, g, bias=True)[0, 1] / vp) if vp > 1e-9 else 1.0
        b = g.mean() - a * p.mean()
        coef.append((a, b))
    out = []
    for x in imgs_p:
        y = np.empty(x.shape, dtype=np.float64)
        for c in range(3):
            a, b = coef[c]
            y[..., c] = a * x[..., c] + b
        out.append(np.clip(y, 0, 255))
    return coef, out


print("=" * 88)
print("误差分解：色调 vs 结构（1019 全 15 场景）")
print("=" * 88)
print(f"  {'臂':<10}{'raw':>8}{'+逐图仿射':>10}{'增益':>7}{'+集级仿射':>10}{'增益':>7}"
      f"{'tone-free 后差距':>16}")
print("  " + "-" * 84)

scenes = sorted(x for x in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, x)))
GT = {sc: load_case(sc) for sc in scenes}
base_tonefree = None
rows = {}
for arm, pred_dir in ARMS:
    root = os.path.join(PR, "real_pair_1019_" + pred_dir)
    if not os.path.isdir(root):
        print(f"  {arm:<10} 缺 {pred_dir}")
        continue
    P_, G_, keys = [], [], []
    for sc in scenes:
        p = os.path.join(root, f"{sc}__I_000000", "t_hat.png")
        if not os.path.exists(p):
            continue
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        g = GT[sc]
        if im.shape != g.shape:
            im = cv2.resize(im, (g.shape[1], g.shape[0]))
        P_.append(im); G_.append(g); keys.append(sc)
    if not P_:
        continue
    raw = float(np.mean([psnr(a, b) for a, b in zip(P_, G_)]))
    pix = float(np.mean([psnr(affine_per_image(a, b), b) for a, b in zip(P_, G_)]))
    coef, Pds = affine_dataset(P_, G_)
    ds = float(np.mean([psnr(a, b) for a, b in zip(Pds, G_)]))
    rows.setdefault(arm, []).append((raw, pix, ds, coef))
    print(f"  {arm:<10}{raw:>8.3f}{pix:>10.3f}{pix-raw:>+7.3f}{ds:>10.3f}{ds-raw:>+7.3f}")

# 汇总（多种子臂取均值）
print()
print("  " + "-" * 84)
print(f"  {'臂(均值)':<10}{'raw':>8}{'逐图仿射':>10}{'增益':>7}{'集级仿射':>10}{'增益':>7}")
print("  " + "-" * 84)
summary = {}
coefs = {}
for arm, vals in rows.items():
    a = np.array([v[:3] for v in vals]).mean(axis=0)
    summary[arm] = dict(raw=float(a[0]), per_image=float(a[1]), dataset=float(a[2]))
    coefs[arm] = vals[0][3]      # 每通道 (gain, offset)
    print(f"  {arm:<10}{a[0]:>8.3f}{a[1]:>10.3f}{a[1]-a[0]:>+7.3f}"
          f"{a[2]:>10.3f}{a[2]-a[0]:>+7.3f}")

print()
print("=" * 88)
print("集级最优仿射系数（P' = a·P + b，逐通道）—— 用以判断偏置是加性还是增益")
print("=" * 88)
print(f"  {'臂':<10}{'a_B':>8}{'b_B':>8}{'a_G':>8}{'b_G':>8}{'a_R':>8}{'b_R':>8}"
      f"{'|b|均值':>9}{'|a-1|均值':>10}")
print("  " + "-" * 78)
for arm, cf in coefs.items():
    aB, bB = cf[0]; aG, bG = cf[1]; aR, bR = cf[2]
    mb = np.mean([abs(bB), abs(bG), abs(bR)])
    ma = np.mean([abs(aB - 1), abs(aG - 1), abs(aR - 1)])
    print(f"  {arm:<10}{aB:>8.3f}{bB:>+8.2f}{aG:>8.3f}{bG:>+8.2f}{aR:>8.3f}{bR:>+8.2f}"
          f"{mb:>9.2f}{ma:>10.3f}")
summary["_coef"] = coefs

print()
print("=" * 88)
print("关键比较：tone-free 之后，apr 与 may 的差距还剩多少？")
print("=" * 88)
if "R-apr" in summary and "R-may" in summary:
    A, M = summary["R-apr"], summary["R-may"]
    for k, lab in (("raw", "原始"), ("per_image", "逐图仿射后"), ("dataset", "集级仿射后")):
        gap = M[k] - A[k]
        print(f"  {lab:<12} apr {A[k]:7.3f}  may {M[k]:7.3f}  差距 {gap:+.3f} dB")
    gain = (A["per_image"] - A["raw"])
    print(f"\n  apr 靠色调仿射可回收 {gain:+.3f} dB；原始差距 {M['raw']-A['raw']:+.3f} dB")
    print(f"  → 色调可解释的比例 ≈ {max(0.0, gain)/max(1e-6, M['raw']-A['raw'])*100:.0f}%")

json.dump(summary, open("/tmp/tone_decomp.json", "w"), ensure_ascii=False, indent=1)
print("\n写入 /tmp/tone_decomp.json")
