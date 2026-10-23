#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10-19 配对集：多模型对比图 + 汇总表（2026-10-20）。

图（每张一行一个场景，列 = GT / INPUT / 各模型输出）
--------------------------------------------------
  eval/fig_1019/grid_all.jpg        15 场景 × 全臂（缩略，整体鸟瞰）
  eval/fig_1019/case_<scene>.jpg    逐场景大图（便于细看反射残留）
  eval/fig_1019/metrics_bar.png     PSNR/SSIM 柱状图

表
--
  eval/fig_1019/table_1019.md       最终表格（Markdown，可直接贴进文档）
"""
from __future__ import annotations

import csv
import glob
import json
import os

import cv2
import numpy as np

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR_DIR = os.path.join(PROJ, "data/test_sets/real_pair_1019")
PRED_ROOT = os.path.join(PROJ, "runs/pred")
EVAL_DIR = os.path.join(PROJ, "eval")
OUT = os.path.join(EVAL_DIR, "fig_1019")

# 展示顺序与显示名
ARMS = [
    ("identity", "INPUT (identity)"),
    ("rdnet_zs", "RDNet official"),
    ("ft_s2main", "Ours ft_s2main"),
    ("ft_mixbest", "Ours ft_mixbest"),
    ("lightrr", "LightRR (CVPR26)"),
]


def log(*a):
    print(*a, flush=True)


def read_img(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


def cell(img, text, w, label_h=26):
    h, iw = img.shape[:2]
    ch = max(2, int(round(w * h / iw)))
    s = cv2.resize(img, (w, ch), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ch + label_h, w, 3), np.uint8)
    canvas[:] = (24, 24, 24)
    canvas[label_h:] = s
    cv2.putText(canvas, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def find_pred(arm: str, scene: str, stem: str):
    p = os.path.join(PRED_ROOT, f"real_pair_1019_{arm}", f"{scene}__{stem}", "t_hat.png")
    return p if os.path.exists(p) else None


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    pairs = []
    for sc in sorted(os.listdir(PAIR_DIR)):
        d = os.path.join(PAIR_DIR, sc)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                pairs.append((sc, os.path.splitext(f)[0]))
    log(f"配对 {len(pairs)} 组")

    # 只展示场景的"离 GT 最近"的那个输入（stem 排序最后 = 距离最小）
    by_scene = {}
    for sc, stem in pairs:
        by_scene.setdefault(sc, []).append(stem)
    reps = [(sc, sorted(v)[-1]) for sc, v in sorted(by_scene.items())]
    log(f"代表帧：{len(reps)} 个场景（每场景取最接近 GT 的输入）")

    avail = [(k, n) for k, n in ARMS
             if k == "identity" or find_pred(k, reps[0][0], reps[0][1])]
    log(f"可用臂：{[n for _, n in avail]}")

    # ---- 鸟瞰大图 ----
    cw = 260
    rows, used = [], []
    for sc, stem in reps:
        gt = read_img(os.path.join(PAIR_DIR, sc, "GT.png"))
        inp = read_img(os.path.join(PAIR_DIR, sc, f"{stem}.png"))
        if gt is None or inp is None:
            continue
        cells = [cell(gt, f"GT {sc[-6:]}", cw), cell(inp, "INPUT", cw)]
        got = False
        for k, _ in avail:
            if k == "identity":
                continue
            p = find_pred(k, sc, stem)
            if p:
                cells.append(cell(read_img(p), k, cw))
                got = True
        if not got:
            continue
        rows.append(json_cat(cells))
        used.append(sc)
    if rows:
        grid = np.vstack([np.hstack(r) for r in rows])
        p = os.path.join(OUT, "grid_all.jpg")
        cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        log(f"  鸟瞰图 {grid.shape[1]}x{grid.shape[0]} -> {p}")

    # ---- 逐场景大图 ----
    for sc, stem in reps:
        gt = read_img(os.path.join(PAIR_DIR, sc, "GT.png"))
        inp = read_img(os.path.join(PAIR_DIR, sc, f"{stem}.png"))
        if gt is None or inp is None:
            continue
        W = 520
        cells = [cell(gt, f"GT  (gt frame)", W), cell(inp, f"INPUT  {stem}", W)]
        for k, _ in avail:
            if k == "identity":
                continue
            p = find_pred(k, sc, stem)
            if p:
                cells.append(cell(read_img(p), k, W))
        if len(cells) <= 2:
            continue
        n = len(cells)
        cols = min(n, 3)
        rws = (n + cols - 1) // cols
        hgt = cells[0].shape[0]
        wid = cells[0].shape[1]
        canvas = np.zeros((rws * hgt, cols * wid, 3), np.uint8)
        canvas[:] = (16, 16, 16)
        for i, c in enumerate(cells):
            r, q = divmod(i, cols)
            canvas[r * hgt:(r + 1) * hgt, q * wid:(q + 1) * wid] = c
        cv2.imwrite(os.path.join(OUT, f"case_{sc}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"  逐场景图 -> {OUT}/case_*.jpg")

    # ---- 汇总表 ----
    summ = {}
    sp = os.path.join(EVAL_DIR, "real_pair_1019_summary.json")
    if os.path.exists(sp):
        summ.update(json.load(open(sp, encoding="utf-8")))
    for k, _ in ARMS:
        f = os.path.join(EVAL_DIR, f"real_pair_1019_{k}_summary.json")
        if os.path.exists(f):
            summ[k] = json.load(open(f, encoding="utf-8"))
    mx = os.path.join(EVAL_DIR, "real_pair_1019_mix_arms.json")
    if os.path.exists(mx):
        d = json.load(open(mx, encoding="utf-8"))
        b = d.get("best")
        if b and b in d["arms"]:
            summ["mix_best_name"] = b
            summ["ft_mixbest"] = {**d["arms"][b], "n": d["arms"][b]["n"]}

    order = [k for k, _ in ARMS if k in summ]
    lines = ["| 模型 | n | PSNR↑ | SSIM↑ | PSNR(gm)↑ | ΔPSNR vs INPUT |",
             "|---|---|---|---|---|---|"]
    base = summ["identity"]["psnr"] if "identity" in summ else None
    name = dict(ARMS)
    for k in order:
        s = summ[k]
        d = f"{s['psnr']-base:+.2f}" if base is not None else "-"
        extra = ""
        if k == "ft_mixbest" and "mix_best_name" in summ:
            extra = f" ({summ['mix_best_name']})"
        lines.append(f"| {name.get(k,k)}{extra} | {s['n']} | {s['psnr']:.2f} | "
                     f"{s['ssim']:.4f} | {s['psnr_gm']:.2f} | {d} |")
    tbl = "\n".join(lines)
    open(os.path.join(OUT, "table_1019.md"), "w", encoding="utf-8").write(
        "# 10-19 配对集（n=%d）多模型对比\n\n%s\n" % (summ.get("identity", {}).get("n", 0), tbl))
    log("\n" + tbl)

    # ---- 柱状图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = order
        ps = [summ[k]["psnr"] for k in ks]
        ss = [summ[k]["ssim"] for k in ks]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        lbl = [name.get(k, k) for k in ks]
        ax[0].bar(range(len(ks)), ps, color="#4C72B0")
        ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels(lbl, rotation=20, ha="right")
        ax[0].set_ylabel("PSNR (dB)"); ax[0].set_title("PSNR on real_pair_1019")
        ax[0].grid(axis="y", alpha=.3)
        ax[1].bar(range(len(ks)), ss, color="#DD8452")
        ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels(lbl, rotation=20, ha="right")
        ax[1].set_ylabel("SSIM"); ax[1].set_title("SSIM on real_pair_1019")
        ax[1].grid(axis="y", alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "metrics_bar.png"), dpi=140)
        log(f"  柱状图 -> {OUT}/metrics_bar.png")
    except Exception as e:
        log(f"  [跳过柱状图] {e}")
    return 0


def json_cat(cells):
    return cells


if __name__ == "__main__":
    raise SystemExit(main())
