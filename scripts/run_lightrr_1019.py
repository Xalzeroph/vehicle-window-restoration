#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 LightRR 独立环境里跑 1019 配对集推理（2026-10-20）。

为什么单独一支：LightRR 需要 torch 2.4.0 + mamba_ssm + CUDA 扩展，
装在自己的 conda env（/DataDisk/zmy_code/LightRR_env/lightrr），
与 xreflection 环境不兼容，所以必须分进程跑。

本脚本只负责"读配对集 -> 推理 -> 存预测 + 逐对指标 CSV"，
出图与跨模型汇总由 xreflection 环境的 make_1019_figures.py 完成。

口径与其它臂一致：配对集已统一 1280x832（32 的倍数），故 pad 为 0；
PSNR/SSIM 逐通道算再取通道均值，data_range=255。
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

LIGHTRR_REPO = "/home/zmy/mycode/LightRR-CVPR26"
PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR_DIR = os.path.join(PROJ, "data/test_sets/real_pair_1019")
ARM = "lightrr"
PRED_ROOT = os.path.join(PROJ, "runs/pred", f"real_pair_1019_{ARM}")
EVAL_DIR = os.path.join(PROJ, "eval")

sys.path.insert(0, LIGHTRR_REPO)
sys.path.insert(0, os.path.join(PROJ, "scripts"))


def log(*a):
    print(*a, flush=True)


def bw_psnr(a, b):
    v = []
    for c in range(a.shape[-1]):
        mse = float(np.mean((a[..., c].astype(np.float64) - b[..., c].astype(np.float64)) ** 2))
        v.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


def bw_ssim(a, b):
    return float(np.mean([ssim_fn(a[..., c], b[..., c], data_range=255)
                          for c in range(a.shape[-1])]))


def gain_match(q, g):
    q = np.asarray(q, np.float64); g = np.asarray(g, np.float64)
    s = float((q * g).sum() / max((q * q).sum(), 1e-8))
    return np.clip(q * s, 0, 255)


def main() -> int:
    dev = "cuda"
    from lightrr_infer import LightRRInfer

    ckpt = os.path.join(LIGHTRR_REPO, "weights/lightrr/lightrr_woNature.pt")
    log("加载 LightRR ...")
    model = LightRRInfer(ckpt, device=dev)
    log(f"  ckpt={model.ckpt_info}  params={model.n_params/1e6:.2f} M")

    pairs = []
    for sc in sorted(os.listdir(PAIR_DIR)):
        d = os.path.join(PAIR_DIR, sc)
        if not os.path.isdir(d):
            continue
        gt = cv2.cvtColor(cv2.imread(os.path.join(d, "GT.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                im = cv2.cvtColor(cv2.imread(os.path.join(d, f)),
                                  cv2.COLOR_BGR2RGB).astype(np.float32)
                pairs.append((sc, os.path.splitext(f)[0], im, gt))
    log(f"配对 {len(pairs)} 组")

    rows = []
    for i, (sc, stem, im, gt) in enumerate(pairs, 1):
        x = torch.from_numpy(im.transpose(2, 0, 1) / 255.0)[None].float().to(dev)
        with torch.no_grad():
            y = model(x)                      # (1,3,H,W) in [0,1]，已裁回原尺寸
        pred = y[0].permute(1, 2, 0).cpu().numpy() * 255.0
        p = bw_psnr(pred, gt); s = bw_ssim(pred, gt)
        pg = bw_psnr(gain_match(pred, gt), gt)
        rows.append({"arm": ARM, "scene": sc, "input": stem,
                     "psnr": round(p, 4), "ssim": round(s, 4), "psnr_gm": round(pg, 4)})
        d = os.path.join(PRED_ROOT, f"{sc}__{stem}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "t_hat.png"),
                    cv2.cvtColor(np.clip(pred, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i % 10 == 0:
            log(f"  {i}/{len(pairs)}  PSNR(so far)={np.mean([r['psnr'] for r in rows]):.2f}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    cp = os.path.join(EVAL_DIR, f"real_pair_1019_{ARM}.csv")
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summ = {"n": len(rows),
            "psnr": round(float(np.mean([r["psnr"] for r in rows])), 3),
            "ssim": round(float(np.mean([r["ssim"] for r in rows])), 4),
            "psnr_gm": round(float(np.mean([r["psnr_gm"] for r in rows])), 3)}
    log(f"-> {ARM}: PSNR {summ['psnr']:.2f}  SSIM {summ['ssim']:.4f}  "
        f"PSNR(gm) {summ['psnr_gm']:.2f}")
    sp = os.path.join(EVAL_DIR, f"real_pair_1019_{ARM}_summary.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    log(f"写入 {cp} / {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
