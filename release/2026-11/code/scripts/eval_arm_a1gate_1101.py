#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 单臂推理（2026-11-01）：在 real_pair_1019 上跑"基线 + 有界残差倍率门"的 ckpt，
写 runs/pred/real_pair_1019_<name>/<scene>__<stem>/t_hat.png，供 scripts/table_final_1019.py 出分。

为什么必须另起入口
------------------
表口径要求预测图是 uint8 PNG、RGB->BGR 写盘、clamp 顺序与 `predict_rdnet` 一致。
带门的臂若直接用 `eval_arm_1019.py`，门权重根本不会被加载（`load_state_dict(strict=False)`
静默丢掉 `a1_gate_*`），打出来的分是"无门基线"——本项目已经因同类静默失败吃过亏。
本入口的硬闸（任一不满足直接抛异常，绝不出分）：
  1. ckpt 必须含 `a1_gate_sd` 与 `a1_gate_cfg`；
  2. 门参数量必须等于 `param_report(cfg)['n_total']`（默认 1025）；
  3. `net.load_state_dict(icnn)` 的 missing / unexpected 必须都为 0（含 mona）；
  4. 记录门配置指纹到 arm_info.json，证明不是随机初始化。

用法
----
  CUDA_VISIBLE_DEVICES=7 python3 scripts/eval_arm_a1gate_1101.py \
      --ckpt runs/a1_1101/gate_s101/ckpt_final.pth --name a1_s101
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

from eval_1019_multi import PAIR_DIR, PRED_ROOT, bw_psnr, build_rdnet, load_img  # noqa: E402
from rdnet_a1gate_1101 import (  # noqa: E402
    load_gate_from_ckpt, param_report, predict_with_gate,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="带门的 ckpt（必须含 a1_gate_sd）")
    ap.add_argument("--name", required=True, help="臂名（写 runs/pred/real_pair_1019_<name>）")
    ap.add_argument("--pairs", default=PAIR_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mona", default="both", choices=["none", "both", "ffn"])
    args = ap.parse_args()

    ck_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJ, args.ckpt)
    if not os.path.exists(ck_path):
        raise SystemExit(f"[a1] ckpt 不存在: {ck_path}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "a1_gate_sd" not in ck:
        raise SystemExit("[a1] ckpt 不含 a1_gate_sd：这是无门基线，拒绝按带门臂打分")
    if "a1_gate_cfg" not in ck:
        raise SystemExit("[a1] ckpt 不含 a1_gate_cfg：无法确定门结构")
    cfg = dict(ck["a1_gate_cfg"])
    rep = param_report(cfg)

    dev = torch.device(args.device)
    net, net_c = build_rdnet(args.mona)
    gate = load_gate_from_ckpt(ck, device=None)
    n_gate = sum(p.numel() for p in gate.parameters())
    if n_gate != rep["n_total"]:
        raise SystemExit(f"[a1] 门参数量 {n_gate} != 期望 {rep['n_total']}（cfg={cfg}）")

    sd = ck["icnn"]
    miss, unex = net.load_state_dict(sd, strict=False)
    if miss or unex:
        raise SystemExit(f"[a1] net 权重未完整加载：missing={miss[:5]} unexpected={unex[:5]}")
    net.to(dev).eval()
    net_c.to(dev).eval()
    gate.to(dev).eval()

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
    if not pairs:
        raise SystemExit(f"[a1] {args.pairs} 下没有可用配对；拒绝在空面板上出分")

    out_root = os.path.join(PRED_ROOT, f"real_pair_1019_{args.name}")
    os.makedirs(out_root, exist_ok=True)
    psnrs = []
    for i, (sc, stem, im, gt) in enumerate(pairs, 1):
        pred = predict_with_gate(net, net_c, im, dev, gate)
        psnrs.append(bw_psnr(pred, gt))
        d = os.path.join(out_root, f"{sc}__{stem}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "t_hat.png"),
                    cv2.cvtColor(np.clip(pred, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i % 5 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}  PSNR(sofar)={np.mean(psnrs):.3f}", flush=True)

    rec = dict(arm=args.name, ckpt=ck_path, mona=args.mona, arch="rdnet+a1gate",
               n_pairs=len(pairs), psnr=round(float(np.mean(psnrs)), 4),
               pred_dir=out_root, n_gate_param=n_gate, a1_gate_cfg=cfg,
               gate_param_report=rep, ckpt_iter=ck.get("iter"))
    with open(os.path.join(out_root, "arm_info.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print(f"\n[done] PSNR={rec['psnr']:.3f}  -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
