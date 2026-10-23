#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""级联 NAFNet 在 1019 上的推理（2026-10-21）。

流程
----
    T̂ = clamp(T̂_base + NAFNet(I, T̂_base), 0, 1)
`T̂_base` 直接读已算好的官方臂预测（`runs/pred/real_pair_1019_rdnet_zs/`），
不重跑 266 M 的 RDNet。

用法
----
  python3 scripts/eval_naf_1019.py --ckpt runs/ckpt/naf_l1/ckpt_final.pth --name naf_l1
  python3 scripts/eval_naf_1019.py --ckpt ... --name naf_v2base --base-arm real_pair_1019_v2_s202
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "RDNet"))

from models.arch.NAFNET import NAFNet_refine   # noqa: E402

PAIR = os.path.join(PROJ, "data/test_sets/real_pair_1019")
PRED = os.path.join(PROJ, "runs/pred")


def read_rgb(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-arm", default="real_pair_1019_rdnet_zs")
    ap.add_argument("--pairs", default=PAIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--enc-last", type=int, default=28)
    args = ap.parse_args()

    ck_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJ, args.ckpt)
    if not os.path.exists(ck_path):
        raise SystemExit(f"ckpt 不存在: {ck_path}")
    dev = torch.device(args.device)

    net = NAFNet_refine(img_channel=6, width=args.width, middle_blk_num=1,
                        enc_blk_nums=[1, 1, 1, args.enc_last],
                        dec_blk_nums=[1, 1, 1, 1],
                        global_residual=False).to(dev).eval()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    miss, unex = net.load_state_dict(sd, strict=False)
    print(f"[naf] ckpt={os.path.basename(ck_path)} missing={len(miss)} "
          f"unexpected={len(unex)} iter={ck.get('iter', '-') if isinstance(ck, dict) else '-'}")
    if len(miss) or len(unex):
        print(f"[naf] 警告：键不完全匹配 miss={miss[:3]} unex={unex[:3]}")

    base_root = os.path.join(PRED, args.base_arm)
    if not os.path.isdir(base_root):
        raise SystemExit(f"base 臂预测不存在: {base_root}")

    pairs = []
    for sc in sorted(os.listdir(args.pairs)):
        d = os.path.join(args.pairs, sc)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                stem = os.path.splitext(f)[0]
                if os.path.exists(os.path.join(base_root, f"{sc}__{stem}", "t_hat.png")):
                    pairs.append((sc, stem))
    print(f"[naf] 配对 {len(pairs)} 组，base={args.base_arm}")

    out_root = os.path.join(PRED, f"real_pair_1019_{args.name}")
    psnrs = []
    for i, (sc, stem) in enumerate(pairs, 1):
        I = read_rgb(os.path.join(args.pairs, sc, f"{stem}.png"))
        B = read_rgb(os.path.join(base_root, f"{sc}__{stem}", "t_hat.png"))
        G = cv2.imread(os.path.join(args.pairs, sc, "GT.png"), cv2.IMREAD_COLOR)
        if I is None or B is None or G is None:
            continue
        if B.shape[:2] != I.shape[:2]:
            B = cv2.resize(B, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(dev)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                delta = net(t(I), t(B))
            out = (torch.from_numpy(np.ascontiguousarray(B.transpose(2, 0, 1)))[None].to(dev)
                   + delta.float()).clamp(0, 1)
        P = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0)
        # 快速 PSNR（与主表同口径：逐通道后取均值）
        Gf = G.astype(np.float64)
        Ps = np.clip(P, 0, 255).astype(np.float64)
        ms = [float(np.mean((Ps[..., 2 - c] - Gf[..., c]) ** 2)) for c in range(3)]
        psnrs.append(float(np.mean([99.0 if m <= 1e-12 else 10 * np.log10(255.0 ** 2 / m)
                                    for m in ms])))
        d = os.path.join(out_root, f"{sc}__{stem}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "t_hat.png"),
                    cv2.cvtColor(np.clip(P, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i % 5 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}  PSNR(sofar)={np.mean(psnrs):.3f}", flush=True)

    info = {"arm": args.name, "ckpt": ck_path, "base_arm": args.base_arm,
            "n_pairs": len(pairs), "psnr": round(float(np.mean(psnrs)), 4) if psnrs else None}
    with open(os.path.join(out_root, "arm_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    print(f"[done] PSNR={info['psnr']} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
