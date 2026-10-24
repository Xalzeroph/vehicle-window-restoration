#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建「暗化剂量」训练集（E2 修正版，2026-10-23，纯 CPU）。

动机
----
已实测：apr 臂的输出最优常数校正 = **减亮 11 灰阶**，回收 +1.27 dB；而 may/in19 几乎不需要。
原因是 apr 的反射**只加结构、不加净亮度**（mean(I)−mean(T) = 4.2 灰阶 = 3.2%），
而 1019 的反射净加 16.4 灰阶（14.1%）→ 模型学不到"该变暗多少"，输出整体偏亮。

做法
----
对 apr 的每一对，把 **T 统一下移 k**，使该对的暗化比例达到目标 `target_pct`：

    k = target_pct · mean_luma(I) − (mean_luma(I) − mean_luma(T))
    T' = clip(T − k, 0, 255)

I 不动（输入不变），于是 (I, T') 教会模型"要去掉更多亮度"。
k 可为负（此时为加亮）。

剂量点
------
`--targets 0.08,0.11,0.14`（默认三档；14% ≈ 1019 需要的 14.1%，但**仅作剂量一端**，
不是唯一候选 —— 三档一起才能看剂量-反应曲线）。
另建 `--targets 0.032`（= apr 原样）作为对照等价物（但原样已有 R-apr，故可省）。

产出
----
`data/train_sets/real_pairs_1020_apr_dark{08,11,14}/`（I/T/R/omega + meta.json，含 C_I）
  · R 依 (I, T') 重算：R = I − g·T'（与原 builder 同口径，g 用 robust_gain）
  · omega 沿用原掩膜（Ω 由 (I,T) 决定，本变换是全局常数平移，不影响结构）
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import cv2
import numpy as np
import sys

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
from eval_protocol import robust_gain      # noqa: E402

SRC = os.path.join(P, "data/train_sets/real_pairs_1020_apr")
SUBS = ("I", "T", "R", "omega")


def luma(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def c_i_of(I_bgr, om):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(om.astype(np.uint8), ker) > 0) & (~om)
    if not (om.any() and ring.any()):
        return 0.0
    lI = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(lI[ring].mean() - lI[om].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--targets", default="0.08,0.11,0.14")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.src, "meta.json"), encoding="utf-8"))
    print(f"源 {args.src}: {len(meta)} 对")

    for tgt in [float(x) for x in args.targets.split(",")]:
        tag = f"{int(round(tgt*100)):02d}"
        out = os.path.join(P, f"data/train_sets/real_pairs_1020_apr_dark{tag}")
        if not args.dry_run:
            for s in SUBS:
                os.makedirs(os.path.join(out, s), exist_ok=True)
        new_meta = {}
        ks, pcts = [], []
        for case, v in sorted(meta.items()):
            I = cv2.imread(os.path.join(args.src, "I", case + ".png"))
            T = cv2.imread(os.path.join(args.src, "T", case + ".png"))
            om_u8 = cv2.imread(os.path.join(args.src, "omega", case + ".png"), 0)
            if I is None or T is None or om_u8 is None:
                continue
            li, lt = luma(I), luma(T)
            cur = (li - lt) / li if li > 0 else 0.0
            k = tgt * li - (li - lt)
            T2 = np.clip(T.astype(np.float64) - k, 0, 255)
            li2, lt2 = li, luma(T2.astype(np.uint8))
            pct2 = (li2 - lt2) / li2 if li2 > 0 else 0.0
            g = robust_gain(I, T2)
            R = I.astype(np.float64) - T2 * g
            om = om_u8 > 127
            if not args.dry_run:
                cv2.imwrite(os.path.join(out, "I", case + ".png"), I)
                cv2.imwrite(os.path.join(out, "T", case + ".png"), T2.astype(np.uint8))
                cv2.imwrite(os.path.join(out, "R", case + ".png"),
                            np.clip(R + 128.0, 0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(out, "omega", case + ".png"), (om * 255).astype(np.uint8))
            m = dict(v)
            m.update(domain="apr_dark" + tag, k=round(float(k), 2),
                     dark_pct_orig=round(cur, 4), dark_pct_new=round(pct2, 4),
                     C_I=round(c_i_of(I, om), 2))
            new_meta[case] = m
            ks.append(k); pcts.append(pct2)
        if not args.dry_run and new_meta:
            json.dump(new_meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        if ks:
            print(f"  target {100*tgt:.0f}% -> {out}")
            print(f"    {len(new_meta)} 对  k 中位 {np.median(ks):+.1f} 灰阶  "
                  f"实际暗化比例 中位 {100*np.median(pcts):.1f}%")
    if args.dry_run:
        print("\n(dry-run，未落盘)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
