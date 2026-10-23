#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""级联 NAFNet 的目视对比（2026-10-21）：GT / INPUT / base(官方) / naf_l1。

要回答：+0.37 dB PSNR 与 LapVar 78%→103% 是真细节，还是加了"通用纹理"？
所以给 全图 / 放大裁块 / 梯度幅值 三行。
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
OUT = os.path.join(P, "_review_1019/naf")
ARMS = [("GT", None), ("INPUT", "INPUT"),
        ("base 官方", "real_pair_1019_rdnet_zs"),
        ("cascade l1", "real_pair_1019_naf_l1")]
SCENES = ["VID_20261019_165126", "VID_20261019_170033", "VID_20261019_164257"]
CROP, CW = 300, 620


def rd(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def gm(x):
    g = gray(x)
    m = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    lo, hi = np.percentile(m, 2), np.percentile(m, 98)
    return (np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def cell(img, t, w, lh=26):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, iw = img.shape[:2]
    ch = max(2, int(round(w * h / iw)))
    s = cv2.resize(img, (w, ch), interpolation=cv2.INTER_AREA)
    c = np.zeros((ch + lh, w, 3), np.uint8); c[:] = (16, 16, 16)
    c[lh:] = s
    cv2.putText(c, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return c


def bar(t, w):
    b = np.zeros((26, w, 3), np.uint8)
    cv2.putText(b, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return b


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    mp = {}
    cp = os.path.join(P, "eval/table_final_1019_naf.csv")
    if os.path.exists(cp):
        for r in csv.DictReader(open(cp, encoding="utf-8")):
            mp.setdefault(r["scene"], {})[r["arm"]] = r

    for sc in SCENES:
        d = os.path.join(PAIR, sc)
        gt = rd(os.path.join(d, "GT.png"))
        inp = rd(os.path.join(d, "I_000000.png"))
        if gt is None or inp is None:
            continue
        H, W = gt.shape[:2]
        g = gray(gt)
        m = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
        ii = cv2.integral(m.astype(np.float64))
        cc = min(CROP, H, W)
        sums = ii[cc:, cc:] - ii[:-cc, cc:] - ii[cc:, :-cc] + ii[:-cc, :-cc]
        y0, x0 = np.unravel_index(int(np.argmax(sums)), sums.shape)

        rf, rz, rg = [], [], []
        for name, sub in ARMS:
            if name == "GT":
                img = gt
            elif name == "INPUT":
                img = inp
            else:
                p = os.path.join(PRED, sub, f"{sc}__I_000000", "t_hat.png")
                img = rd(p) if os.path.exists(p) else None
                if img is None:
                    continue
                if img.shape[:2] != gt.shape[:2]:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            tag = name
            arm_key = {"base 官方": "rdnet_zs", "cascade l1": "naf_l1"}.get(name)
            if arm_key and sc in mp and arm_key in mp[sc]:
                rr = mp[sc][arm_key]
                tag = f"{name} {float(rr['psnr']):.2f}dB GC{float(rr['grad_cos']):.3f} HF{float(rr['hf_psnr']):.1f}"
            rf.append(cell(img, tag, CW))
            cz = img[y0:y0 + cc, x0:x0 + cc]
            rz.append(cell(cz, tag, CW))
            rg.append(cell(gm(cz), "|grad| " + name, CW))

        if not rf:
            continue
        wsum = rf[0].shape[1] * len(rf)
        sheet = np.vstack([bar(f"{sc}  FULL FRAME", wsum), np.hstack(rf),
                           bar(f"ZOOM ({x0},{y0}) {cc}px", wsum), np.hstack(rz),
                           bar("|GRADIENT| of the crop —— 看加的是真细节还是通用纹理", wsum),
                           np.hstack(rg)])
        p = os.path.join(OUT, f"cmp_{sc}.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"  {sc}: {sheet.shape[1]}x{sheet.shape[0]} -> {p}")
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
