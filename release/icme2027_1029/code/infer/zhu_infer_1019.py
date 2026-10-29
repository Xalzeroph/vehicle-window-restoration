#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal inference wrapper for Zhu et al. CVPR 2024 (RRW / RemoVal). ASCII-only.

Official pipeline (from their testing.py):
    sparse = RefDet(I)                 # reflection "detection" prior
    T_hat  = NAFNet_wDetHead(I, sparse)

Checkpoints (raw OrderedDict, strict=True):
    ckpt/RD.pth  -> RefDet(backbone='efficientnet-b3')   keys: feat_net.*
    ckpt/RR.pth  -> NAFNet_wDetHead(...)                keys: intro/ending/encoders.*

Input convention: images in [0,1]; RefDet applies ImageNet normalization internally
(ConstantNormalize), NAFNet pads to padder_size internally and crops back.

Output layout matches the other arms: runs/pred/real_pair_1019_<name>/<scene>/t_hat.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import cv2
import torch

REMO = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/Reflection_RemoVal_CVPR2024-main"
P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
sys.path.insert(0, REMO)
sys.path.insert(0, os.path.join(REMO, "networks"))
os.chdir(REMO)

from networks.NAFNet_arch import NAFNet_wDetHead  # noqa: E402
from networks.network_RefDet import RefDet  # noqa: E402


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def build(dev):
    net = NAFNet_wDetHead(img_channel=3, width=32, middle_blk_num=1,
                          enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                          concat=True, merge_manner=0)
    net_Det = RefDet(backbone="efficientnet-b3", proj_planes=16, pred_planes=32,
                     use_pretrained=False, fix_backbone=False, has_se=False,
                     num_of_layers=6, expansion=4)
    sd_r = torch.load(os.path.join(REMO, "ckpt/RR.pth"), map_location="cpu")
    sd_d = torch.load(os.path.join(REMO, "ckpt/RD.pth"), map_location="cpu")
    net.load_state_dict(sd_r, strict=True)
    net_Det.load_state_dict(sd_d, strict=True)
    net.eval().to(dev)
    net_Det.eval().to(dev)
    print("[load] strict=True OK  (RR %d keys, RD %d keys)" % (len(sd_r), len(sd_d)))
    return net, net_Det


@torch.no_grad()
def run_one(net, net_Det, bgr, dev):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(dev)
    sparse = net_Det(x)
    out = net(x, sparse)
    out = out.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    return rgb, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=os.path.join(P, "data/test_sets/real_pair_1019"))
    ap.add_argument("--name", default="removal_zhu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    ap.add_argument("--out-root", default=os.path.join(P, "runs/pred"))
    args = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, net_Det = build(dev)

    scenes = [s for s in sorted(os.listdir(args.pair)) if os.path.isdir(os.path.join(args.pair, s))]
    if args.limit:
        scenes = scenes[:args.limit]
    outdir = os.path.join(args.out_root, "real_pair_1019_" + args.name)
    os.makedirs(outdir, exist_ok=True)

    tot, n = 0.0, 0
    for sc in scenes:
        ip = os.path.join(args.pair, sc)
        gp = os.path.join(ip, "GT.png")
        ins = [f for f in sorted(os.listdir(ip)) if f.startswith("I_") and f.endswith(".png")]
        if not ins:
            print("  [skip] %s (no I_*.png)" % sc)
            continue
        I = cv2.imread(os.path.join(ip, ins[0]))
        _, out = run_one(net, net_Det, I, dev)
        sub = sc + "__" + os.path.splitext(ins[0])[0]   # match the repo convention <scene>__<input>
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
        cv2.imwrite(os.path.join(outdir, sub, "t_hat.png"), (out[:, :, ::-1] * 255.0).round().astype(np.uint8))
        if os.path.exists(gp):
            G = cv2.imread(gp)
            if G.shape[:2] != out.shape[:2]:
                G = cv2.resize(G, (out.shape[1], out.shape[0]))
            p = psnr(out[:, :, ::-1] * 255.0, G)
            tot += p; n += 1
            print("  %-24s PSNR %.3f" % (sc[:24], p), flush=True)
        else:
            print("  %-24s (no GT)" % sc[:24], flush=True)
    print("[done] mean PSNR = %.3f over %d scenes -> %s" % (tot / max(n, 1), n, outdir))


if __name__ == "__main__":
    main()
