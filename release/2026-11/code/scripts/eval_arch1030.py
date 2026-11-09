#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARCH_H1（2026-10-30）单臂推理：与训练端**同一份模块构造**，只写预测图供
`scripts/table_final_1019.py` 打分（不在这里算指标，避免口径漂移）。

用法（服务器，项目根；先看 nvidia-smi 选卡并显式钉卡）:
  PY=/DataDisk/miniconda/envs/xreflection/bin/python3
  # 主臂
  CUDA_VISIBLE_DEVICES=4 $PY -u scripts/eval_arch1030.py \\
      --ckpt runs/arch1030/A_h1_s101/ckpt_final.pth --name arch1030_h1_s101
  # 机制诊断版：同一 ckpt、子带门强制 g≡1（不重训、不据此选模型）
  CUDA_VISIBLE_DEVICES=4 $PY -u scripts/eval_arch1030.py \\
      --ckpt runs/arch1030/A_h1_s101/ckpt_final.pth --name arch1030_h1_s101_g1 --gate-identity
  然后 $PY scripts/table_final_1019.py

注意：`--gate-identity` 只用于回答"效果是否真的经过子带算子"，它**不是**一个可选的模型；
预登记里它不能替代主臂，也不能拿它去挑模型。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_1019_multi import PAIR_DIR, PRED_ROOT, bw_psnr, load_img   # noqa: E402


def detect_mona(ck):
    peft = ck.get("peft", {}) if isinstance(ck, dict) else {}
    mode = peft.get("mode", "") if isinstance(peft, dict) else ""
    if mode in ("both", "ffn", "none"):
        return mode
    sd = ck.get("icnn", ck)
    return "both" if any("mona1.gamma" in k or "mona2.gamma" in k for k in sd) else "none"


@torch.no_grad()
def predict(net, net_c, img, dev):
    x = torch.from_numpy(img.transpose(2, 0, 1) / 255.0)[None].to(dev)
    pad = (0, (-x.shape[-1]) % 32, 0, (-x.shape[-2]) % 32)
    xp = F.pad(x, pad, mode="reflect") if any(pad) else x
    code = net_c(xp)
    with torch.cuda.amp.autocast():
        _, out = net(xp, code, prompt=True)
    t = out[-1][0, 3:].float().clamp(0, 1)
    H, W = img.shape[:2]
    return t[:, :H, :W].permute(1, 2, 0).cpu().numpy() * 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="臂名（写进 runs/pred/real_pair_1019_<name>）")
    ap.add_argument("--pairs", default=PAIR_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mona", default="auto", choices=["auto", "none", "both", "ffn"])
    ap.add_argument("--gate-identity", action="store_true",
                    help="诊断：强制 g≡1（同一 ckpt，用于判断效果是否经过子带算子）")
    args = ap.parse_args()

    ck_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJ, args.ckpt)
    if not os.path.exists(ck_path):
        raise SystemExit(f"ckpt 不存在: {ck_path}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and ck.get("arch") not in (None, "subband1030"):
        raise SystemExit(f"ckpt arch={ck.get('arch')} 不是 subband1030，拒绝用本脚本推理")
    mona = detect_mona(ck) if args.mona == "auto" else args.mona
    print(f"[arch1030-eval] {args.name} ckpt={ck_path} mona={mona} gate_identity={args.gate_identity}",
          flush=True)

    import finetune_arch1030 as ft                       # 触发 os.chdir(RDNet) + sys.path
    net, net_c = ft.build_subband1030(mona=mona, grad_ckpt=False)
    sd = ck.get("icnn", ck)
    miss, unex = net.load_state_dict(sd, strict=False)
    if miss or unex:
        raise SystemExit(f"权重未完整加载: missing={miss[:6]} unexpected={unex[:6]}")
    net.subband_gate_identity = bool(args.gate_identity)
    dev = torch.device(args.device)
    net.to(dev).eval()
    net_c.to(dev).eval()

    sds = net.state_dict()
    mk = [k for k in sds if "mona1.gamma" in k or "mona2.gamma" in k]
    fp = round(float(sum(sds[k].float().sum() for k in mk)), 6) if mk else float("nan")
    sg = float(net.subband.out_conv.weight.float().abs().mean())
    print(f"[arch1030-eval] mona 指纹={fp}  subband|out_conv.w|mean={sg:.3e}", flush=True)

    pairs = []
    for sc in sorted(os.listdir(args.pairs)):
        d = os.path.join(args.pairs, sc)
        if not os.path.isdir(d):
            continue
        gt = load_img(os.path.join(d, "GT.png"))
        if gt is None:
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                im = load_img(os.path.join(d, f))
                if im is not None:
                    pairs.append((sc, os.path.splitext(f)[0], im, gt))
    print(f"[arch1030-eval] 配对 {len(pairs)} 组", flush=True)

    out_root = os.path.join(PRED_ROOT, f"real_pair_1019_{args.name}")
    psnrs = []
    for i, (sc, stem, im, gt) in enumerate(pairs, 1):
        pred = predict(net, net_c, im, dev)
        psnrs.append(bw_psnr(pred, gt))
        d = os.path.join(out_root, f"{sc}__{stem}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "t_hat.png"),
                    cv2.cvtColor(np.clip(pred, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i % 5 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}  PSNR(sofar)={np.mean(psnrs):.3f}", flush=True)

    info = {"arm": args.name, "ckpt": ck_path, "mona": mona, "arch": "subband1030",
            "gate_identity": bool(args.gate_identity), "mona_fingerprint": fp,
            "n_pairs": len(pairs), "psnr_float_screen_only": round(float(np.mean(psnrs)), 4),
            "pred_dir": out_root,
            "note": "屏幕 PSNR 仅供 sanity；权威数字一律用 table_final_1019.py 读存盘 PNG"}
    with open(os.path.join(out_root, "arm_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    print(f"\n[done] -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
