#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rdt_s2v2_main_s202 大图对比（2026-10-21）。

目的：目视回答"v2_s202 的 GC 更高是真的结构更对，还是把不确定边缘抹掉了"。
做法：只留 5 个臂（GT / INPUT / 官方 / v2_s202 / mix_b2），格子放大到 560px；
      每个场景三行：全图、放大裁块、裁块的**梯度幅值**（直接看结构在不在）。
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
OUT = os.path.join(P, "_review_1019/v2s202")

ARMS = [("GT", None), ("INPUT", "INPUT"),
        ("rdnet_zs", "real_pair_1019_rdnet_zs"),
        ("v2_s202", "real_pair_1019_v2_s202"),
        ("mix_b2", "real_pair_1019_ft_mixbest")]
# v2_s202 明显赢官方的场景 + 一个典型场景
SCENES = [("VID_20261019_170033", "I_000000"), ("VID_20261019_171126", "I_000000"),
          ("VID_20261019_165126", "I_000000")]
CROP, CW = 300, 560


def rd(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def gm(x):
    g = gray(x)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.sqrt(gx * gx + gy * gy)
    lo, hi = np.percentile(m, 2), np.percentile(m, 98)
    return (np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def cell(img, text, w, lh=26):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, iw = img.shape[:2]
    ch = max(2, int(round(w * h / iw)))
    s = cv2.resize(img, (w, ch), interpolation=cv2.INTER_AREA)
    c = np.zeros((ch + lh, w, 3), np.uint8); c[:] = (16, 16, 16)
    c[lh:] = s
    cv2.putText(c, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return c


def bar(t, w, color=(0, 255, 0)):
    b = np.zeros((26, w, 3), np.uint8)
    cv2.putText(b, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return b


def main() -> int:
    # 逐场景指标（复用上一步的 CSV）
    mp = {}
    cp = os.path.join(P, "eval/v2s202_psnr_gc.csv")
    if os.path.exists(cp):
        for r in csv.DictReader(open(cp, encoding="utf-8")):
            mp[r["scene"]] = r

    for sc, stem in SCENES:
        d = os.path.join(PAIR, sc)
        gt = rd(os.path.join(d, "GT.png"))
        inp = rd(os.path.join(d, f"{stem}.png"))
        if gt is None or inp is None:
            continue
        H, W = gt.shape[:2]
        # 金字塔窗口：GT 梯度能量最集中处
        m = np.sqrt(cv2.Sobel(gray(gt), cv2.CV_32F, 1, 0) ** 2 +
                    cv2.Sobel(gray(gt), cv2.CV_32F, 0, 1) ** 2).astype(np.float64)
        ii = cv2.integral(m)
        cc = min(CROP, H, W)
        sums = ii[cc:, cc:] - ii[:-cc, cc:] - ii[cc:, :-cc] + ii[:-cc, :-cc]
        y0, x0 = np.unravel_index(int(np.argmax(sums)), sums.shape)
        y0, x0 = int(y0), int(x0)

        r = mp.get(sc, {})
        rowf, rowz, rowg = [], [], []
        for name, sub in ARMS:
            if name == "GT":
                img = gt
            elif name == "INPUT":
                img = inp
            else:
                p = os.path.join(PRED, sub, f"{sc}__{stem}", "t_hat.png")
                img = rd(p) if os.path.exists(p) else None
                if img is None:
                    continue
                if img.shape[:2] != gt.shape[:2]:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            tag = name
            if name not in ("GT", "INPUT") and r:
                tag = f"{name}  {float(r.get(f'psnr_{name}', 0)):.2f}dB  GC{float(r.get(f'gc_{name}', 0)):.3f}"
            rowf.append(cell(img, tag, CW))
            cz = img[y0:y0 + cc, x0:x0 + cc]
            rowz.append(cell(cz, tag, CW))
            rowg.append(cell(gm(cz), "|grad| " + name, CW))

        if not rowf:
            continue
        wmin = min(x.shape[1] for x in rowf)
        sheet = np.vstack([
            bar(f"{sc}  FULL FRAME", rowf[0].shape[1] * len(rowf)),
            np.hstack(rowf),
            bar(f"ZOOM  ({x0},{y0})  {cc}px  —— GT 梯度能量最集中处", rowf[0].shape[1] * len(rowf)),
            np.hstack(rowz),
            bar("|GRADIENT| of the same crop  —— 看结构还在不在", rowf[0].shape[1] * len(rowf)),
            np.hstack(rowg),
        ])
        p = os.path.join(OUT, f"big_{sc}.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"  {sc}: {sheet.shape[1]}x{sheet.shape[0]} -> {p}")
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
