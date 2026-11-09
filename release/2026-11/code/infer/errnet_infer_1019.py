#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal inference wrapper for ERRNet (CVPR 2019). ASCII-only.

Verified recipe (from models/errnet_model.py + models/vgg.py):
    in_channels = 3 + 1472            (--hyper)
    hypercolumn = Vgg19 features at indices [2, 7, 12, 21, 30]  -> 64+128+256+512+512 = 1472
    each feature bilinear-upsampled to (H, W)
    x = cat([img, *hypercolumn], dim=1)     # 1475 channels
    y = net_i(x)                           # DRNet(1475, 3, 256, 13, se_reduction=8, pyramid=True)

Input convention: RGB in [0, 1]  -- their dataset applies plain ToTensor(), NO ImageNet Normalize.

We load `models/arch/default.py` and `models/vgg.py`-equivalent code by importing the arch file
standalone: `from models import arch` would run models/__init__.py -> errnet_model.py ->
util/util.py, which executes os.popen("stty size") at import time and dies in a non-TTY.
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F

ERRNET = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/baselines/ERRNet"
P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")

VGG_INDICES = [2, 7, 12, 21, 30]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def load_arch():
    spec = ilu.spec_from_file_location("_errnet_arch", os.path.join(ERRNET, "models/arch/default.py"))
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(ckpt, dev):
    arch = load_arch()
    net = arch.DRNet(3 + 1472, 3, 256, 13, norm=None, res_scale=0.1,
                     se_reduction=8, bottom_kernel_size=1, pyramid=True)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck["icnn"] if isinstance(ck, dict) and "icnn" in ck else ck
    net.load_state_dict(sd, strict=True)
    net.eval().to(dev)

    from torchvision import models as tvm
    vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features.eval().to(dev)
    for p in vgg.parameters():
        p.requires_grad_(False)
    print("[load] ERRNet strict=True OK (%d keys) + vgg19 features" % len(sd))
    return net, vgg


@torch.no_grad()
def run_one(net, vgg, bgr, dev, mod=32, max_side=1024):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h0, w0 = rgb.shape[:2]
    scale = 1.0
    if max(h0, w0) > max_side:
        scale = max_side / float(max(h0, w0))
        rgb = cv2.resize(rgb, (int(round(w0 * scale)), int(round(h0 * scale))),
                         interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(dev)
    h, w = x.shape[-2:]
    ph = (mod - h % mod) % mod
    pw = (mod - w % mod) % mod
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")

    feats, cur = [], x
    for i in range(VGG_INDICES[-1]):
        cur = vgg[i](cur)
        if (i + 1) in VGG_INDICES:
            feats.append(F.interpolate(cur.detach(), size=x.shape[-2:], mode="bilinear", align_corners=False))
    inp = torch.cat([x] + feats, dim=1)

    y = net(inp)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = y[:, :, :h, :w].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    if scale != 1.0:
        y = cv2.resize(y, (w0, h0), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    return rgb, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/baselines/ckpt/ERRNet/errnet_060_00463920.pt")
    ap.add_argument("--pair", default=os.path.join(P, "data/test_sets/real_pair_1019"))
    ap.add_argument("--name", default="errnet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, vgg = build(args.ckpt, dev)

    scenes = [s for s in sorted(os.listdir(args.pair)) if os.path.isdir(os.path.join(args.pair, s))]
    if args.limit:
        scenes = scenes[:args.limit]
    outdir = os.path.join(P, "runs/pred", "real_pair_1019_" + args.name)
    tot, n = 0.0, 0
    for sc in scenes:
        ip = os.path.join(args.pair, sc)
        gp = os.path.join(ip, "GT.png")
        ins = [f for f in sorted(os.listdir(ip)) if f.startswith("I_") and f.endswith(".png")]
        if not ins:
            continue
        I = cv2.imread(os.path.join(ip, ins[0]))
        _, y = run_one(net, vgg, I, dev, max_side=args.max_side)
        sub = sc + "__" + os.path.splitext(ins[0])[0]
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
        cv2.imwrite(os.path.join(outdir, sub, "t_hat.png"), (y[:, :, ::-1] * 255.0).round().astype(np.uint8))
        if os.path.exists(gp):
            G = cv2.imread(gp)
            if G.shape[:2] != y.shape[:2]:
                G = cv2.resize(G, (y.shape[1], y.shape[0]))
            p = psnr(y[:, :, ::-1] * 255.0, G)
            tot += p; n += 1
            print("  %-24s PSNR %.3f" % (sc[:24], p), flush=True)
    print("[done] mean PSNR = %.3f over %d scenes -> %s" % (tot / max(n, 1), n, outdir))


if __name__ == "__main__":
    main()
