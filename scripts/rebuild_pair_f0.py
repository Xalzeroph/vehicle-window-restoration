#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建配对集：I 帧一律取**每场景第一帧 f0**（2026-10-20，用户指定）。

为什么改：原先把 I 取在 GT 前 30/60/90 帧（1–3 秒前），那时反射已淡到几乎为 0，
导致"输入≈真值"、identity 白赢。f0 是每个视频开头，反射应最强。

输出覆盖 data/test_sets/real_pair_1019/{scene}/
    GT.png      # 人工标注的无反射帧（原尺寸裁 1299x858，再 resize 到 1280x832）
    I_000000.png
    meta.json   # 含 gt_frame / i_frame / 两者帧号距离 / 尺寸

同时输出 reflection 强度量化：每场景 INPUT 与 GT 的逐通道稳健增益 |g-1|，
用来定量回答"这次 I 帧到底带了多少反射"。
"""
from __future__ import annotations

import json
import os
import re
import shutil

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
RAW = os.path.join(P, "data/raw_1019")
OUT = os.path.join(P, "data/test_sets/real_pair_1019")
TARGET = (1280, 832)   # (W, H)


def robust_gain(a, b, keep=0.7, iters=3):
    A = a.astype(np.float64); B = b.astype(np.float64)
    g = np.ones(3)
    for c in range(3):
        x = A[..., c].ravel(); y = B[..., c].ravel()
        for _ in range(iters):
            den = float((y * y).sum())
            gc = float((x * y).sum() / den) if den > 1e-9 else 1.0
            r = np.abs(x - gc * y)
            k = r <= np.percentile(r, keep * 100)
            x, y = x[k], y[k]
            if len(x) < 100:
                break
        g[c] = gc
    return g


def luma(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def main() -> int:
    # 清空旧配对集，避免残留 I_001784.png 之类混进来
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    scenes = sorted(s for s in os.listdir(RAW)
                    if s.startswith("VID_20261019_") and os.path.isdir(os.path.join(RAW, s)))
    print(f"重建配对集（I 一律取 f0）：{len(scenes)} 个场景 -> {OUT}\n")

    rows = []
    print(f"{'场景':<30}{'GT帧':>8}{'I帧':>8}{'|g-1|':>9}{'luma比':>9}{'g[B,G,R]':>22}")
    print("-" * 92)
    for sc in scenes:
        src = os.path.join(RAW, sc)
        lb = os.path.join(src, "label.txt")
        gt = None
        if os.path.exists(lb):
            m = re.search(r"^gt_frame:\s*(\d+)", open(lb, encoding="utf-8").read(), re.M)
            if m:
                gt = int(m.group(1))
        if gt is None:
            print(f"[跳过] {sc}: 无 gt_frame")
            continue

        gsrc = os.path.join(src, f"frame_{gt:06d}.jpg")
        isrc = os.path.join(src, "frame_000000.jpg")
        if not os.path.exists(gsrc):
            print(f"[跳过] {sc}: GT 帧文件不存在")
            continue
        if not os.path.exists(isrc):
            print(f"[跳过] {sc}: f0 文件不存在")
            continue

        G = cv2.imread(gsrc)
        I = cv2.imread(isrc)
        if G is None or I is None:
            print(f"[跳过] {sc}: 读图失败")
            continue

        # 量化反射强度（在 resize 前算，口径与之前一致）
        g = robust_gain(I, G)
        dev = float(np.abs(g - 1).mean())
        lr = luma(I) / max(luma(G), 1e-6)

        G = cv2.resize(G, TARGET, interpolation=cv2.INTER_AREA)
        I = cv2.resize(I, TARGET, interpolation=cv2.INTER_AREA)

        d = os.path.join(OUT, sc)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "GT.png"), G)
        cv2.imwrite(os.path.join(d, "I_000000.png"), I)

        meta = {"scene": sc, "gt_frame": gt, "i_frame": 0, "i_file": "I_000000.png",
                "gt_file": "GT.png", "dist_frames": gt, "dist_sec": round(gt / 30.0, 1),
                "size": list(TARGET),
                "input_vs_gt": {"gain_dev": round(dev, 4),
                                "gain_bgr": [round(float(x), 4) for x in g],
                                "luma_ratio": round(lr, 4)},
                "note": "I 取每场景第一帧 f0（反射最强处）；GT 为人工标注无反射帧"}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        rows.append(meta)
        print(f"{sc:<30}{gt:>8}{0:>8}{dev:>9.4f}{lr:>9.3f}"
              f"{str([round(float(x),3) for x in g]):>22}")

    print("-" * 92)
    if rows:
        devs = [r["input_vs_gt"]["gain_dev"] for r in rows]
        lrs = [r["input_vs_gt"]["luma_ratio"] for r in rows]
        print(f"共 {len(rows)} 对；|g-1| 中位 {np.median(devs):.4f} "
              f"(P90 {np.percentile(devs,90):.4f})；luma比 中位 {np.median(lrs):.3f}")
        with open(os.path.join(OUT, "input_strength.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print(f"写入 {OUT}/input_strength.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
