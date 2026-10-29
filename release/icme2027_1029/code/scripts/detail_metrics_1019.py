#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""细节保真指标对比：Laplacian var vs 更干净的口径（2026-10-20）。

Laplacian var 的问题
------------------
它把三件事混在一起：
  · 真实细节的锐度
  · **反射层自带的边缘**（INPUT 比 GT 高 33% 就是这个原因）
  · 整体对比度
所以它是"边缘能量"而非"保真度"，**不是越高越好**，只能对 GT 看相对位置。

本脚本给三个更干净的替代口径，并排对照：
  1. **HF-PSNR**：只在**高通带**上算 PSNR —— 直接问"高频对不对"，而不是"高频多不多"
  2. **梯度余弦相似度**：∇T̂ 与 ∇T 在强梯度处的方向一致性（结构是否保对）
  3. **Laplacian var**（现有口径，留作对照）
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
EVAL = os.path.join(P, "eval")

ARMS = {"rdnet_zs": "real_pair_1019_rdnet_zs",
        "ft_s2main": "real_pair_1019_ft_s2main",
        "ft_mixbest": "real_pair_1019_ft_mixbest",
        "lightrr": "real_pair_1019_lightrr"}
SIG_LP = 2.0   # 高通的高斯尺度


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def gray(x):
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def highpass(g):
    return g - cv2.GaussianBlur(g, (0, 0), SIG_LP)


def psnr_gray(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def psnr_rgb(a, b):
    return float(np.mean([psnr_gray(a[..., c], b[..., c]) for c in range(3)]))


def lapvar(x):
    # 注意：OpenCV 5.0 不支持 float32 输入 + CV_64F 输出的 Laplacian 组合，
    # 必须用 CV_32F（实测报 "Unsupported combination of source format (=5), destination (=6)"）
    return float(cv2.Laplacian(gray(x), cv2.CV_32F).var(dtype=np.float64))


def grad_cos(a_g, b_g, topk=0.15):
    ax, ay = np.gradient(a_g)
    bx, by = np.gradient(b_g)
    ma = np.hypot(ax, ay); mb = np.hypot(bx, by)
    m = (ma >= np.percentile(ma, (1 - topk) * 100)) & \
        (mb >= np.percentile(mb, (1 - topk) * 100))
    if m.sum() < 100:
        return float("nan")
    dot = (ax * bx + ay * by)[m]
    den = (ma * mb)[m].clip(1e-6)
    return float(np.mean(dot / den))


def main() -> int:
    pairs = []
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
                    pairs.append((sc, os.path.splitext(f)[0], I, G))
    print(f"配对 {len(pairs)} 组\n")

    gt_g = [gray(G) for _, _, _, G in pairs]
    gt_hp = [highpass(g) for g in gt_g]
    gt_lap = float(np.mean([lapvar(G) for _, _, _, G in pairs]))
    inp_lap = float(np.mean([lapvar(I) for _, _, I, _ in pairs]))
    print(f"GT     Laplacian var = {gt_lap:6.1f}   （参照）")
    print(f"INPUT  Laplacian var = {inp_lap:6.1f}   = {inp_lap/gt_lap*100:.0f}% of GT\n")

    rows = {}
    hdr = (f"{'臂':<12}{'PSNR':>7}{'SSIM':>7}{'HF-PSNR':>9}{'梯度余弦':>9}"
           f"{'LapVar':>8}{'%ofGT':>7}")
    print(hdr); print("-" * len(hdr))

    # identity 基线
    for name, preds in [("identity", [I for _, _, I, _ in pairs])] + [
            (arm, None) for arm in ARMS]:
        if preds is None:
            sub = ARMS[name]
            preds = []
            for sc, stem, I, G in pairs:
                p = os.path.join(PRED, sub, f"{sc}__{stem}", "t_hat.png")
                if not os.path.exists(p):
                    continue
                Pr = read(p)
                if Pr is None:
                    continue
                if Pr.shape[:2] != I.shape[:2]:
                    Pr = cv2.resize(Pr, (I.shape[1], I.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                preds.append(Pr)
        if len(preds) != len(pairs):
            print(f"[跳过] {name}: 预测数不匹配")
            continue

        ps, ss, hf, gc, lv = [], [], [], [], []
        for (sc, stem, I, G), Pr in zip(pairs, preds):
            ps.append(psnr_rgb(Pr, G))
            ss.append(float(np.mean([ssim_fn(Pr[..., c], G[..., c], data_range=255)
                                     for c in range(3)])))
            hf.append(psnr_gray(highpass(gray(Pr)), highpass(gray(G))))
            gc.append(grad_cos(gray(Pr), gray(G)))
            lv.append(lapvar(Pr))
        rows[name] = {"psnr": round(float(np.mean(ps)), 3),
                      "ssim": round(float(np.mean(ss)), 4),
                      "hf_psnr": round(float(np.mean(hf)), 3),
                      "grad_cos": round(float(np.nanmean(gc)), 4),
                      "lapvar": round(float(np.mean(lv)), 1),
                      "lapvar_pct_of_gt": round(float(np.mean(lv)) / gt_lap * 100, 1)}
        r = rows[name]
        print(f"{name:<12}{r['psnr']:>7.2f}{r['ssim']:>7.4f}{r['hf_psnr']:>9.2f}"
              f"{r['grad_cos']:>9.3f}{r['lapvar']:>8.1f}{r['lapvar_pct_of_gt']:>6.0f}%")

    json.dump({"gt_lapvar": round(gt_lap, 1), "input_lapvar": round(inp_lap, 1),
               "arms": rows},
              open(os.path.join(EVAL, "detail_metrics_1019.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n写入 {EVAL}/detail_metrics_1019.json")
    print("\n判读：")
    print("  HF-PSNR 高 = 高频内容与 GT 一致（既不过平也不乱加）")
    print("  梯度余弦 接近 1 = 保住了正确的结构方向；接近 0 = 结构与 GT 无关")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
