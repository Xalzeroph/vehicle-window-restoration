#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rdt_s2v2_main_s202 推理对比图（2026-10-21）。

为什么单独出这个臂的图
-------------------
它是**唯一在干净测试集上 GC 超过官方 RDNet 的臂**（0.616 vs 0.611），
但 PSNR 低 1.05 dB、Laplacian 只有 GT 的 53% —— 需要目视回答：
    "GC 更高"是结构真的保对了，还是把不确定的边缘抹掉、只留少数方向明确的？
图里同时给**全图**与**强结构放大裁块**，并逐场景标 PSNR / GC。

输出
----
_review_1019/v2s202/rows_<scene>.jpg   每场景：全图行 + 裁块行（含 GT/INPUT/各臂）
_review_1019/v2s202/grid_all.jpg       15 场景总览
eval/fig_1019/v2s202_psnr_gc.csv       逐场景 PSNR / GC
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
EVAL = os.path.join(P, "eval")

# 顺序：GT / INPUT / 官方 / 三个 s2v2 / mix / s2main / lightrr
ARMS = [("GT", None),
        ("INPUT", "INPUT"),
        ("rdnet_zs", "real_pair_1019_rdnet_zs"),
        ("v2_s202", "real_pair_1019_v2_s202"),
        ("v2_s101", "real_pair_1019_v2_s101"),
        ("v2_s303", "real_pair_1019_v2_s303"),
        ("mix_b2", "real_pair_1019_ft_mixbest"),
        ("s2main", "real_pair_1019_ft_s2main"),
        ("lightrr", "real_pair_1019_lightrr")]

CROP = 260
FULL_W = 300
CROP_W = 300


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def psnr(a, b):
    v = []
    for c in range(3):
        mse = float(np.mean((a[..., c].astype(np.float64) - b[..., c].astype(np.float64)) ** 2))
        v.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


def grad_cos(a_g, b_g, topk=0.15):
    ax, ay = np.gradient(a_g); bx, by = np.gradient(b_g)
    ma, mb = np.hypot(ax, ay), np.hypot(bx, by)
    m = (ma >= np.percentile(ma, (1 - topk) * 100)) & (mb >= np.percentile(mb, (1 - topk) * 100))
    if m.sum() < 100:
        return float("nan")
    dot = (ax * bx + ay * by)[m]
    return float(np.mean(dot / (ma * mb)[m].clip(1e-6)))


def cell(img, text, w, label_h=22):
    h, iw = img.shape[:2]
    ch = max(2, int(round(w * h / iw)))
    s = cv2.resize(img, (w, ch), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ch + label_h, w, 3), np.uint8)
    canvas[:] = (18, 18, 18)
    canvas[label_h:] = s
    cv2.putText(canvas, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    pairs = []
    for sc in sorted(os.listdir(PAIR)):
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                pairs.append((sc, os.path.splitext(f)[0]))
    print(f"配对 {len(pairs)} 组")

    rows_csv = []
    grid_rows = []
    for sc, stem in pairs:
        d = os.path.join(PAIR, sc)
        gt = read(os.path.join(d, "GT.png"))
        inp = read(os.path.join(d, f"{stem}.png"))
        if gt is None or inp is None:
            continue

        # 选裁块：GT 上梯度能量最集中的窗口（结构最丰富处，GC 的判据区）
        g = gray(gt)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag = cv2.blur(np.sqrt(gx * gx + gy * gy), (CROP, CROP))
        H, W = mag.shape
        m = cv2.copyMakeBorder(mag, 0, max(0, CROP - H), 0, max(0, CROP - W),
                               cv2.BORDER_CONSTANT, value=0)
        _, _, _, maxloc = cv2.minMaxLoc(m[:H, :W])
        cy, cx = maxloc[1], maxloc[0]
        y0 = int(np.clip(cy - CROP // 2, 0, max(0, H - CROP)))
        x0 = int(np.clip(cx - CROP // 2, 0, max(0, W - CROP)))

        cells_full, cells_crop = [], []
        rec = {"scene": sc, "input": stem}
        for name, sub in ARMS:
            if name == "GT":
                img = gt
            elif name == "INPUT":
                img = inp
            else:
                p = os.path.join(PRED, sub, f"{sc}__{stem}", "t_hat.png")
                if not os.path.exists(p):
                    continue
                img = read(p)
                if img is None:
                    continue
                if img.shape[:2] != gt.shape[:2]:
                    img = cv2.resize(img, (gt.shape[1], gt.shape[0]),
                                     interpolation=cv2.INTER_AREA)
            if name not in ("GT", "INPUT"):
                rec[f"psnr_{name}"] = round(psnr(img, gt), 3)
                rec[f"gc_{name}"] = round(grad_cos(gray(img), gray(gt)), 4)
            tag = name
            if name not in ("GT", "INPUT"):
                tag = f"{name} {rec.get(f'psnr_{name}', 0):.2f}dB GC{rec.get(f'gc_{name}', 0):.3f}"
            cells_full.append(cell(img, tag, FULL_W))
            cells_crop.append(cell(img[y0:y0 + CROP, x0:x0 + CROP], tag, CROP_W))

        if not cells_full:
            continue
        rows_csv.append(rec)

        bar1 = np.zeros((24, cells_full[0].shape[1] * len(cells_full), 3), np.uint8)
        cv2.putText(bar1, f"{sc}  FULL FRAME", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        row_full = np.hstack(cells_full)
        bar2 = np.zeros((24, cells_crop[0].shape[1] * len(cells_crop), 3), np.uint8)
        cv2.putText(bar2, f"ZOOM  x0={x0} y0={y0} {CROP}px  (GT 梯度能量最集中处)",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        row_crop = np.hstack(cells_crop)

        hmin = min(row_full.shape[1], row_crop.shape[1])
        img = np.vstack([bar1, row_full[:, :hmin], bar2, row_crop[:, :hmin]])
        cv2.imwrite(os.path.join(OUT, f"rows_{sc}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        grid_rows.append(row_full[:, :hmin])          # 总览：直接复用全图行
        print(f"  {sc}  crop=({x0},{y0})  "
              f"v2_s202 {rec.get('psnr_v2_s202', 0):.2f}dB "
              f"GC{rec.get('gc_v2_s202', 0):.3f} | "
              f"rdnet {rec.get('psnr_rdnet_zs', 0):.2f} "
              f"GC{rec.get('gc_rdnet_zs', 0):.3f}")

    # 汇总 CSV
    if rows_csv:
        keys = ["scene", "input"] + [k for k in rows_csv[0] if k.startswith(("psnr_", "gc_"))]
        cp = os.path.join(EVAL, "v2s202_psnr_gc.csv")
        with open(cp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows_csv)
        print(f"\n写入 {cp}")

        # 均值
        print(f"\n{'臂':<12}{'PSNR':>8}{'GC':>8}")
        print("-" * 28)
        for name in [n for n, s in ARMS if n not in ("GT", "INPUT")]:
            ps = [r.get(f"psnr_{name}") for r in rows_csv if r.get(f"psnr_{name}")]
            gs = [r.get(f"gc_{name}") for r in rows_csv if r.get(f"gc_{name}")]
            if ps:
                print(f"{name:<12}{np.mean(ps):>8.2f}{np.nanmean(gs):>8.3f}")

        # 逐场景胜负：v2_s202 vs rdnet_zs 的 GC
        win = sum(1 for r in rows_csv
                  if r.get("gc_v2_s202", 0) > r.get("gc_rdnet_zs", 0))
        print(f"\nv2_s202 的 GC 高于官方的场景数: {win}/{len(rows_csv)}")

    # 总览拼图（行 = 场景，列 = 臂）
    if grid_rows:
        wmin = min(r.shape[1] for r in grid_rows)
        grid = np.vstack([r[:, :wmin] for r in grid_rows])
        p = os.path.join(OUT, "grid_all.jpg")
        cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\n总览 {grid.shape[1]}x{grid.shape[0]} -> {p}")
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
