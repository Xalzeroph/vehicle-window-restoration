#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用单臂推理：给任意 ckpt 在 1019 配对集上跑预测，供 table_final_1019.py 打分（2026-10-20）。

设计要点
--------
1. **mona 模式从 ckpt 自动读**（`ck["peft"]["mode"]`），不靠人指定 —— 手工传错会导致
   静默退回 plain 而数字失真（本项目已踩过）。
2. 复用 `eval_1019_multi.py` 的 build/predict/load，保证与既有四臂**同一口径**。
3. 只写预测图，不重复算全指标（指标统一由 table_final_1019.py 出），避免两处实现漂移。

用法
----
  python3 scripts/eval_arm_1019.py --ckpt runs/ckpt/rdt_s2v2_main_s101/ckpt_3000.pth \
      --name v2_s101_3k
  python3 scripts/eval_arm_1019.py --ckpt official --name rdnet_zs     # 官方权重
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

from eval_1019_multi import (  # noqa: E402
    PAIR_DIR, PRED_ROOT, build_rdnet, bw_psnr, load_rdnet, load_img,
    predict_rdnet,
)

OFFICIAL = "/DataDisk/zmy_code/RDNet-main/rdnet.pth"


def detect_mona(ck_path: str) -> str:
    """从 ckpt 的 peft 字段判断 mona 模式；官方权重无 peft → none。"""
    try:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"读 ckpt 失败 {ck_path}: {e}")
    if not isinstance(ck, dict):
        return "none"
    peft = ck.get("peft", {})
    mode = peft.get("mode", "") if isinstance(peft, dict) else str(peft)
    if mode in ("both", "ffn"):
        return mode
    # 兜底：按 state_dict 里是否有 mona 键判断
    sd = ck.get("icnn", ck)
    if any("mona1.gamma" in k or "mona2.gamma" in k for k in sd):
        return "both"
    return "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt 路径，或 'official'")
    ap.add_argument("--name", required=True, help="臂名（写进 runs/pred/real_pair_1019_<name>）")
    ap.add_argument("--pairs", default=PAIR_DIR, help="配对集目录")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mona", default="auto", choices=["auto", "none", "both", "ffn"])
    ap.add_argument("--arch", default="mona", choices=["mona", "clean1029"])
    args = ap.parse_args()

    ck_path = OFFICIAL if args.ckpt == "official" else (
        args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJ, args.ckpt))
    if not os.path.exists(ck_path):
        raise SystemExit(f"ckpt 不存在: {ck_path}")

    mona = args.mona if args.mona != "auto" else detect_mona(ck_path)
    print(f"[arm] {args.name}  ckpt={ck_path}  mona={mona}", flush=True)

    dev = torch.device(args.device)
    if args.arch == "clean1029":                      # 2026-10-30: 队友新架构
        import os as _os, sys as _sys
        _os.environ["TINTED_ARCH"] = "clean1029"
        _sys.path.insert(0, _os.path.join(PROJ, "RDNet_clean1029"))
        _sys.path.insert(0, _os.path.join(PROJ, "scripts"))
        import finetune_rdnet_s2 as _ft
        net, net_c = _ft.build_clean1029()
        ck = torch.load(ck_path, map_location="cpu")
        sd_ = ck.get("icnn", ck.get("state_dict", ck))
        miss, unex = net.load_state_dict(sd_, strict=False)
        print(f"[arm] clean1029 载入: missing={len(miss)} unexpected={len(unex)}", flush=True)
        mona = "none"
    else:
        net, net_c = build_rdnet(mona)
        load_rdnet(net, ck_path, mona)
    net.to(dev).eval()
    net_c.to(dev).eval()

    # 指纹（用 mona 参数，不用冻结主干 —— 后者三臂会完全相同）
    sd = net.state_dict()
    mk = [k for k in sd if "mona1.gamma" in k or "mona2.gamma" in k]
    fp = round(float(sum(sd[k].float().sum() for k in mk)), 6) if mk else float("nan")
    print(f"[arm] mona 指纹 = {fp}", flush=True)

    # 收集配对
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
    print(f"[arm] 配对 {len(pairs)} 组", flush=True)

    out_root = os.path.join(PRED_ROOT, f"real_pair_1019_{args.name}")
    psnrs = []
    for i, (sc, stem, im, gt) in enumerate(pairs, 1):
        pred = predict_rdnet(net, net_c, im, dev)
        psnrs.append(bw_psnr(pred, gt))
        d = os.path.join(out_root, f"{sc}__{stem}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "t_hat.png"),
                    cv2.cvtColor(np.clip(pred, 0, 255).astype(np.uint8),
                                 cv2.COLOR_RGB2BGR))
        if i % 5 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}  PSNR(sofar)={np.mean(psnrs):.3f}", flush=True)

    info = {"arm": args.name, "ckpt": ck_path, "mona": mona,
            "mona_fingerprint": fp, "n_pairs": len(pairs),
            "psnr": round(float(np.mean(psnrs)), 4), "pred_dir": out_root}
    with open(os.path.join(out_root, "arm_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    print(f"\n[done] PSNR={info['psnr']:.3f}  -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
