#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量评估 mix_* 六个臂在 1019 配对集上的表现，挑出最佳（2026-10-20）。

复用 eval_1019_multi.py 的模型构建与推理口径（直接 import，避免两份实现漂移）。
只报 PSNR/SSIM（raw + gm），并输出汇总 JSON；预测图仅对最佳臂保留（省磁盘）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "/DataDisk/zmy_code/TintedGlass_Enhance/scripts")
from eval_1019_multi import (  # noqa: E402
    PRED_ROOT, EVAL_DIR, build_rdnet, load_img, load_rdnet,
    predict_rdnet, bw_psnr, bw_ssim, gain_match, PAIR_DIR,
)

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
ROOT = os.path.join(PROJ, "runs/ckpt")


def log(*a):
    print(*a, flush=True)


def pairs():
    out = []
    for sc in sorted(os.listdir(PAIR_DIR)):
        d = os.path.join(PAIR_DIR, sc)
        if not os.path.isdir(d):
            continue
        gt = load_img(os.path.join(d, "GT.png"))
        if gt is None:
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                im = load_img(os.path.join(d, f))
                if im is not None:
                    out.append((sc, os.path.splitext(f)[0], im, gt))
    return out


def main() -> int:
    dev = torch.device("cuda:0")
    P = pairs()
    log(f"配对 {len(P)} 组")

    net, net_c = build_rdnet("both")
    net.to(dev).eval(); net_c.to(dev).eval()

    arms = sorted(d for d in os.listdir(ROOT) if d.startswith("mix_"))
    res = {}
    for a in arms:
        ck = os.path.join(ROOT, a, "ckpt_final.pth")
        if not os.path.exists(ck):
            log(f"[跳过] {a}: 无 ckpt_final.pth")
            continue
        log(f"\n=== {a} ===")
        load_rdnet(net, ck, "both")
        net.to(dev).eval(); net_c.to(dev).eval()
        t0 = time.time()
        ps, ss, pg = [], [], []
        for i, (sc, stem, im, gt) in enumerate(P, 1):
            pred = predict_rdnet(net, net_c, im, dev)
            ps.append(bw_psnr(pred, gt))
            ss.append(bw_ssim(pred, gt))
            pg.append(bw_psnr(gain_match(pred, gt), gt))
            if i % 14 == 0:
                log(f"  {i}/{len(P)}  PSNR={np.mean(ps):.2f}")
        res[a] = {"n": len(P),
                  "psnr": round(float(np.mean(ps)), 3),
                  "ssim": round(float(np.mean(ss)), 4),
                  "psnr_gm": round(float(np.mean(pg)), 3),
                  "sec": round(time.time() - t0, 1),
                  "ckpt": ck}
        log(f"  -> {a}: PSNR {res[a]['psnr']:.2f}  SSIM {res[a]['ssim']:.4f}")

    best = max(res, key=lambda k: res[k]["psnr"]) if res else None
    log("\n" + "=" * 56)
    log(f"{'arm':<22}{'PSNR':>8}{'SSIM':>9}{'PSNR(gm)':>10}")
    log("-" * 56)
    for a, r in sorted(res.items(), key=lambda kv: -kv[1]["psnr"]):
        log(f"{a:<22}{r['psnr']:>8.2f}{r['ssim']:>9.4f}{r['psnr_gm']:>10.2f}"
            + ("   <== best" if a == best else ""))
    sp = os.path.join(EVAL_DIR, "real_pair_1019_mix_arms.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump({"arms": res, "best": best}, fh, ensure_ascii=False, indent=1)
    log(f"\n最佳: {best}")
    log(f"写入 {sp}")
    if best:
        log(f"BEST_CKPT={res[best]['ckpt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
