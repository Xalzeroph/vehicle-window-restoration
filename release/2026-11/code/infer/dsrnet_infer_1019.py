#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal inference wrapper for DSRNet (ICCV 2023, Tianjin Univ / Guo Xiaojie group).

Verified from XReflection-main:
    xreflection/models/dsrnet_model.py:
        self.vgg   = Vgg19().eval()                      (xreflection/losses/vgg.py)
        out_t, out_r, out_rr = self.net_g(inp, self.vgg(inp))
    options/train_dsrnet.yml  network_g:
        type: DSRNet, width: 64, middle_blk_num: 12,
        enc_blk_nums: [2,2,4,8], dec_blk_nums: [2,2,2,2], lrm_blk_nums: [2,4]
    checkpoint (Lightning):
        state_dict['net_g.*'] (1118) + state_dict['vgg.*'] (32)

Input convention: RGB in [0, 1].
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F

XR = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/XReflection-main"
CKPT = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/baselines/ckpt/dsr-25.8915.ckpt"
P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
sys.path.insert(0, XR)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def build(dev):
    from xreflection.archs.dsrnet_arch import DSRNet
    from xreflection.losses.vgg import Vgg19

    net = DSRNet(in_channels=3, out_channels=3, width=64, middle_blk_num=12,
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2], lrm_blk_nums=[2, 4])
    vgg = Vgg19()
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    nsd = {k[len("net_g."):]: v for k, v in sd.items() if k.startswith("net_g.")}
    vsd = {k[len("vgg."):]: v for k, v in sd.items() if k.startswith("vgg.")}
    net.load_state_dict(nsd, strict=True)
    try:
        vgg.load_state_dict(vsd, strict=True)
        print("[load] vgg weights from ckpt OK (%d)" % len(vsd))
    except Exception as e:  # noqa: BLE001
        print("[warn] vgg load fallback (pretrained): %s" % e)
    net.eval().to(dev)
    vgg.eval().to(dev)
    for p in vgg.parameters():
        p.requires_grad_(False)
    print("[load] DSRNet strict=True OK (net_g %d keys)" % len(nsd))
    return net, vgg


@torch.no_grad()
def run_one(net, vgg, bgr, dev, mod=32, max_side=768):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h0, w0 = rgb.shape[:2]
    scale = 1.0
    if max(h0, w0) > max_side:
        scale = max_side / float(max(h0, w0))
        rgb = cv2.resize(rgb, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(dev)
    h, w = x.shape[-2:]
    ph = (mod - h % mod) % mod
    pw = (mod - w % mod) % mod
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    out = net(x, vgg(x))
    y = out[0] if isinstance(out, (tuple, list)) else out
    y = y[:, :, :h, :w].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    if scale != 1.0:
        y = cv2.resize(y, (w0, h0), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    return rgb, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=os.path.join(P, "data/test_sets/real_pair_1019"))
    ap.add_argument("--name", default="dsrnet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=768)
    args = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, vgg = build(dev)

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
