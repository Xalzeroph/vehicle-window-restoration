#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DWT 分解可视化 —— **全局统一色标版**（2026-10-20）。

上一版的问题：每个子带各自用 pct1-99 拉伸 → 每个格子看起来都"有内容"，
但**看不出哪个子带能量大**，且 LL1（几乎全分辨率的低频）看起来最亮，
与能量占比（LL3 占 72%）矛盾，极易误读。

本版改为：**同一张源图（I / GT / R）的全部子带共用一个线性比例尺**，
其上限取该源图所有子带系数的 99.5 分位。这样：
  - 格子越亮 = 该子带系数越大；
  - 可跨子带、跨尺度直接比较能量强弱。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/dwt_viz_global"
LEVELS = 3


def dwt_subs(gray, levels=LEVELS):
    out, cur = [], gray.astype(np.float32)
    for _ in range(levels):
        h, w = cur.shape
        cur = cur[: h // 2 * 2, : w // 2 * 2]
        a, b = cur[:, 0::2], cur[:, 1::2]
        lo, hi = (a + b) / np.sqrt(2.0), (a - b) / np.sqrt(2.0)
        c, d = lo[0::2, :], lo[1::2, :]
        LL, LH = (c + d) / np.sqrt(2.0), (c - d) / np.sqrt(2.0)
        c2, d2 = hi[0::2, :], hi[1::2, :]
        HL, HH = (c2 + d2) / np.sqrt(2.0), (c2 - d2) / np.sqrt(2.0)
        out.append({"LL": LL, "LH": LH, "HL": HL, "HH": HH})
        cur = LL
    return out


def render_row(subs, tag, w):
    """用**该源图全部子带的同一比例尺**渲染一行。"""
    allabs = np.concatenate([np.abs(s[b]).ravel() for s in subs
                             for b in ("LL", "LH", "HL", "HH")])
    scale = float(np.percentile(allabs, 99.5))
    scale = max(scale, 1e-6)
    cells = []
    for lvl, s in enumerate(subs, 1):
        for b in ("LL", "LH", "HL", "HH"):
            a = np.abs(s[b])
            v = np.clip(a / scale, 0, 1) * 255.0
            img = v.astype(np.uint8)
            h, iw = img.shape
            ch = max(2, int(round(w * h / iw)))
            img = cv2.resize(img, (w, ch), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img, (0, 0), (w, 18), (0, 0, 0), -1)
            cv2.putText(img, f"{tag} {b}{lvl}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
            cells.append((lvl, img))
    # 按级分组，视觉上分块
    imgs = [c[1] for c in cells]
    hmin = min(i.shape[0] for i in imgs)
    return np.hstack([i[:hmin] for i in imgs])


def gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im.astype(np.float32) if im is not None else None


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    # 只为 3 个代表场景出图（低/中/高重叠），并统一说明色标
    picks = ["VID_20261019_170406", "VID_20261019_164257", "VID_20261019_165126"]
    for sc in picks:
        if not os.path.isdir(os.path.join(PAIR, sc)):
            continue
        I = gray(os.path.join(PAIR, sc, "I_000000.png"))
        G = gray(os.path.join(PAIR, sc, "GT.png"))
        if I is None or G is None:
            continue
        R = I - G
        rows = [render_row(dwt_subs(x), t, 230)
                for t, x in (("I", I), ("GT", G), ("R=I-GT", R))]
        hmin = min(r.shape[0] for r in rows)
        sheet = np.vstack([r[:hmin] for r in rows])
        title = np.zeros((38, sheet.shape[1], 3), np.uint8)
        cv2.putText(title,
                    f"{sc}  DWT L1-L3  |coeff| with ONE shared scale per row "
                    f"(99.5 pct) -> brighter = larger coefficient",
                    (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
        out = np.vstack([title, sheet])
        p = os.path.join(OUT, f"{sc}__dwt_global.jpg")
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {sc} -> {out.shape[1]}x{out.shape[0]}  {p}")
    print(f"\n输出目录: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
