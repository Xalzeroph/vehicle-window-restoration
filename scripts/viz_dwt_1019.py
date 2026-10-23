#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DWT 分解结构可视化（2026-10-20）。

目的：把 `I`、`GT`、`R = I − GT` 三张图的 3 级 DWT 子带分别画出来，
     直观回答"反射长在哪个尺度/哪个方向"，而不只是看能量数字。

输出（每个场景各一张）：
  <repo>/_review_1019/dwt_viz/<scene>__dwt.jpg
    第 1 行 = I 的子带    第 2 行 = GT 的子带    第 3 行 = R 的子带
    列 = LL / LH / HL / HH（每级各画一份，用 |系数| 归一化显示）

另出一张汇总图 dwt_summary.jpg（3 个代表场景并排）。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/dwt_viz"
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


def to_u8(x, pct=(1, 99)):
    """把系数图按分位数拉伸到 0-255，负值取绝对值显示结构。"""
    a = np.abs(x).astype(np.float64)
    lo, hi = np.percentile(a, pct)
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    v = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
    return v.astype(np.uint8)


def cell(img, text, w):
    h, iw = img.shape[:2]
    ch = max(2, int(round(w * h / iw)))
    s = cv2.resize(img, (w, ch), interpolation=cv2.INTER_NEAREST)
    s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(s, (0, 0), (w, 20), (0, 0, 0), -1)
    cv2.putText(s, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 255), 1, cv2.LINE_AA)
    return s


def gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im.astype(np.float32) if im is not None else None


def viz_scene(sc, w=240):
    I = gray(os.path.join(PAIR, sc, "I_000000.png"))
    G = gray(os.path.join(PAIR, sc, "GT.png"))
    if I is None or G is None:
        return None
    R = I - G
    rows = []
    for name, img in (("I", I), ("GT", G), ("R=I-GT", R)):
        subs = dwt_subs(img)
        cells = []
        for lvl, d in enumerate(subs, 1):
            for b in ("LL", "LH", "HL", "HH"):
                cells.append(cell(to_u8(d[b]), f"{name} {b}{lvl}", w))
        rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    # 统一行高（子带尺寸不同，取最小高度对齐）
    hmin = min(r.shape[0] for r in rows)
    sheet = np.vstack([r[:hmin] for r in rows])
    title = np.zeros((34, sheet.shape[1], 3), np.uint8)
    cv2.putText(title, f"{sc}   DWT level1..3  (|coeff|, pct1-99)   "
                       f"rows: I / GT / R=I-GT", (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
    return np.vstack([title, sheet])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    made = []
    for sc in scenes:
        s = viz_scene(sc)
        if s is None:
            continue
        p = os.path.join(OUT, f"{sc}__dwt.jpg")
        cv2.imwrite(p, s, [cv2.IMWRITE_JPEG_QUALITY, 90])
        made.append(p)
        print(f"  {sc} -> {s.shape[1]}x{s.shape[0]}")

    # 汇总：取重叠度最低/最高的两个 + 一个中等，做并排
    try:
        ov = json.load(open("/DataDisk/zmy_code/TintedGlass_Enhance/eval/"
                            "real_pair_1019_overlap.json", encoding="utf-8"))
        srt = sorted(ov, key=lambda r: r["overlap_cos"])
        picks = [srt[0]["scene"], srt[len(srt) // 2]["scene"], srt[-1]["scene"]]
        subs = []
        for sc in picks:
            s = viz_scene(sc, w=190)
            if s is not None:
                subs.append(s)
        if subs:
            hmin = min(x.shape[0] for x in subs)
            sm = np.hstack([x[:hmin] for x in subs])
            p = os.path.join(OUT, "dwt_summary.jpg")
            cv2.imwrite(p, sm, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"\n汇总图（低/中/高重叠）: {picks} -> {p}")
    except Exception as e:
        print(f"[跳过汇总图] {e}")

    print(f"\n共 {len(made)} 张 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
