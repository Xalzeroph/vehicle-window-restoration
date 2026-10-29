#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal inference wrapper for IBCLN (CVPR 2020). ASCII-only.

Verified recipe (from models/IBCLN_model.py):
    netG_T = define_G(3*3=9, 3, ngf=64, 'gen_drop', norm='instance')
    netG_R = define_G(9, 3, 64, 'gen_drop', 'instance')
    init:  T0 = I ;  R0 = 0.1 * ones ;  h/c states = zeros(B, ngf*4, H//4, W//4)
    for i in range(3):
        T, t_h, t_c, _, _ = netG_T(cat([I, T, R], 1), t_h, t_c)
        R, r_h, r_c, _, _ = netG_R(cat([I, T, R], 1), r_h, r_c)
    out = clamp(T, 0, 1)

Input convention: RGB in [0, 1] (their datasets use plain ToTensor(), no Normalize).

Output layout: runs/pred/real_pair_1019_<name>/<scene>__<input>/t_hat.png
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

IBCLN = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/baselines/IBCLN"
CKPT_DIR = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/baselines/ckpt/IBCLN"
P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
sys.path.insert(0, IBCLN)

NGF = 64
N_ITER = 3


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def load_networks():
    """Load models/networks.py standalone to avoid models/__init__ side effects."""
    try:
        from models import networks  # noqa: F401
        return networks
    except Exception as e:  # noqa: BLE001
        print("[warn] `from models import networks` failed (%s); using importlib" % e)
        for p in (IBCLN,):
            if p not in sys.path:
                sys.path.insert(0, p)
        spec = ilu.spec_from_file_location("_ibcln_networks", os.path.join(IBCLN, "models/networks.py"))
        m = ilu.module_from_spec(spec)
        sys.modules["_ibcln_networks"] = m
        spec.loader.exec_module(m)
        return m


def build(dev):
    networks = load_networks()
    g_t = networks.define_G(3 * 3, 3, NGF, "gen_drop", "instance")
    g_r = networks.define_G(3 * 3, 3, NGF, "gen_drop", "instance")
    sd_t = torch.load(os.path.join(CKPT_DIR, "final_net_G_T.pth"), map_location="cpu")
    sd_r = torch.load(os.path.join(CKPT_DIR, "final_net_G_R.pth"), map_location="cpu")
    g_t.load_state_dict(sd_t, strict=True)
    g_r.load_state_dict(sd_r, strict=True)
    g_t.eval().to(dev)
    g_r.eval().to(dev)
    print("[load] IBCLN G_T/G_R strict=True OK (%d / %d keys)" % (len(sd_t), len(sd_r)))
    return g_t, g_r


@torch.no_grad()
def run_one(g_t, g_r, bgr, dev, mod=32):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(dev)
    h0, w0 = x.shape[-2:]
    ph = (mod - h0 % mod) % mod
    pw = (mod - w0 % mod) % mod
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    B, C, H, W = x.shape

    T = x.clone()
    R = torch.full_like(x, 0.1)
    z = lambda: torch.zeros((B, NGF * 4, H // 4, W // 4), device=dev)  # noqa: E731
    t_h, t_c, r_h, r_c = z(), z(), z(), z()

    for _ in range(N_ITER):
        T, t_h, t_c, _, _ = g_t(torch.cat((x, T, R), 1), t_h, t_c)
        R, r_h, r_c, _, _ = g_r(torch.cat((x, T, R), 1), r_h, r_c)

    y = T[:, :, :h0, :w0].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    return rgb, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=os.path.join(P, "data/test_sets/real_pair_1019"))
    ap.add_argument("--name", default="ibcln")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=2048)
    args = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g_t, g_r = build(dev)

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
        hh, ww = I.shape[:2]
        scale = 1.0
        if max(hh, ww) > args.max_side:
            scale = args.max_side / float(max(hh, ww))
            I = cv2.resize(I, (int(round(ww * scale)), int(round(hh * scale))), interpolation=cv2.INTER_AREA)
        _, y = run_one(g_t, g_r, I, dev)
        if scale != 1.0:
            y = cv2.resize(y, (ww, hh), interpolation=cv2.INTER_CUBIC).clip(0, 1)
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
