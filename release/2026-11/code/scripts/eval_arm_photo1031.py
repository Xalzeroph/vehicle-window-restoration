#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I1 单臂推理（2026-10-31）：在 1019 配对集上跑 **含光度适配器** 的 ckpt，供 table_final_1019.py 打分。

为什么必须另起一个入口（不能直接用 `eval_arm_1019.py`）
--------------------------------------------------------
`eval_arm_1019.py` 用 `eval_1019_multi.build_rdnet/load_rdnet` 构造**无 photo 的基线网**，
`load_state_dict(strict=False)` 会**静默丢掉** ckpt 里所有 `photo.*` 权重 —— 于是"带适配器
的臂"会被当成基线打分，数字看起来完全正常。本项目已因同类静默失败吃过亏
（`eval_1019_multi.py` 里"ckpt 不含 mona 权重但 --mona both"的检查就是为此加的）。

本入口的硬闸（任一不满足直接抛异常，绝不出分）
---------------------------------------------
1. ckpt 必须含 `photo.*` 权重（否则说明拿错 ckpt）；
2. photo key 数 == 网络 photo key 数，且一个都不能 missing；
3. 非新增（既非 mona 也非 photo）的 key 必须 missing=0、unexpected=0；
4. mona 侧沿用原口径断言；
5. 记录 photo 参数指纹（证明适配器真被载入，而不是随机初始化）。

口径：预测图写法、PNG 存盘、PSNR 实现全部复用 `eval_1019_multi`（`predict_rdnet` /
`bw_psnr`），与既有各臂**同一口径**；本脚本不重复算全指标（统一由 table_final_1019.py 出）。

用法
----
  CUDA_VISIBLE_DEVICES=5 python3 scripts/eval_arm_photo1031.py \\
      --ckpt runs/photo1031/<arm>/ckpt_final.pth --name p1_s101
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
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, HERE)

from eval_1019_multi import (  # noqa: E402
    PAIR_DIR, PRED_ROOT, bw_psnr, load_img, predict_rdnet,
)

OFFICIAL = "/DataDisk/zmy_code/RDNet-main/rdnet.pth"


def detect_mona(ck_path: str) -> str:
    """从 ckpt 的 peft 字段判断 mona 模式；兜底看 state_dict 里有没有 mona 键。"""
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        return "none"
    peft = ck.get("peft", {})
    mode = peft.get("mode", "") if isinstance(peft, dict) else str(peft)
    if mode in ("both", "ffn"):
        return mode
    sd = ck.get("icnn", ck)
    return "both" if any("mona1.gamma" in k or "mona2.gamma" in k for k in sd) else "none"


def build_photo(mona: str = "both"):
    """构造 FullNet_NLP_Photo + Mona + 冻结分类器（cwd 需在 RDNet 内才能载 focal.pth）。"""
    sys.path.insert(0, RDNET)
    cwd = os.getcwd()
    os.chdir(RDNET)
    try:
        from models.arch.rdnet_photo1031 import FullNet_NLP_Photo
        from models.arch.classifier import PretrainedConvNext
        from rdnet_peft import patch_focalnet
        net = FullNet_NLP_Photo(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                                num_subnet=4, loss_col=4, num_classes=1000, drop_path=0,
                                save_memory=True, inter_supv=True, head_init_scale=None,
                                kernel_size=3, photo_enabled=True)
        if mona != "none":
            n = patch_focalnet(net, mona, verbose=False)
            print(f"[arm] patch_focalnet mode={mona}: {n} blocks", flush=True)
        net_c = PretrainedConvNext("convnext_small_in22k")
        ck2 = torch.load(os.path.join(PROJ, "pretrained/cls_model.pth"), map_location="cpu")
        net_c.load_state_dict(ck2.get("icnn", ck2.get("model", ck2)), strict=False)
        net_c.eval().requires_grad_(False)
    finally:
        os.chdir(cwd)
    return net, net_c


def load_photo(net, path: str, mona: str) -> dict:
    """载 ckpt 并做硬闸；返回 photo 指纹信息（写进 arm_info.json）。"""
    from rdnet_peft import is_mona_name, mona_params, mona_state_dict_keys
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("icnn", ck) if isinstance(ck, dict) else ck
    miss, unex = net.load_state_dict(sd, strict=False)

    new = lambda k: is_mona_name(k) or k.startswith("photo.")          # noqa: E731
    miss_core = [k for k in miss if not new(k)]
    miss_ph = [k for k in miss if k.startswith("photo.")]
    miss_mo = [k for k in miss if is_mona_name(k)]
    n_ph_net = len([k for k in net.state_dict() if k.startswith("photo.")])
    n_ph_sd = len([k for k in sd if k.startswith("photo.")])
    print(f"[arm] ckpt={os.path.basename(path)} missing={len(miss)} unexpected={len(unex)} "
          f"| photo net={n_ph_net} ckpt={n_ph_sd} missing={len(miss_ph)} "
          f"| mona missing={len(miss_mo)}", flush=True)

    if n_ph_sd == 0:
        raise RuntimeError("ckpt 不含 photo.* 权重：本入口会静默退回无适配器基线，拒绝出分")
    if n_ph_sd != n_ph_net or miss_ph:
        raise RuntimeError(f"photo 权重未完整加载 ckpt={n_ph_sd} net={n_ph_net} "
                           f"missing={len(miss_ph)}")
    if miss_core or unex:
        raise RuntimeError(f"非新增权重不一致 missing={miss_core[:6]} unexpected={unex[:6]}")
    if mona != "none":
        if len(mona_state_dict_keys(sd)) == 0:
            raise RuntimeError(f"ckpt 不含 mona 权重但 --mona {mona}：会静默退回 plain")
        if len(mona_state_dict_keys(sd)) != len(mona_params(net)) or miss_mo:
            raise RuntimeError(f"mona 未完整加载 miss={len(miss_mo)}")

    ph = net.photo
    fp = round(float(sum(p.detach().float().sum() for p in ph.parameters())), 6)
    norms = {k: round(float(p.detach().float().norm()), 6) for k, p in ph.named_parameters()}
    print(f"[arm] photo 指纹 Σ={fp}  非零参数 "
          f"{sum(1 for p in ph.parameters() if float(p.detach().abs().max()) > 0)}/"
          f"{len(list(ph.parameters()))}", flush=True)
    return dict(photo_sum=fp, photo_norm=norms, n_photo_keys=n_ph_sd, iter=ck.get("iter")
                if isinstance(ck, dict) else None,
                arch=ck.get("arch") if isinstance(ck, dict) else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt 路径（不支持 'official'：官方权重无适配器）")
    ap.add_argument("--name", required=True, help="臂名（写进 runs/pred/real_pair_1019_<name>）")
    ap.add_argument("--pairs", default=PAIR_DIR, help="配对集目录")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mona", default="auto", choices=["auto", "none", "both", "ffn"])
    args = ap.parse_args()

    ck_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJ, args.ckpt)
    if not os.path.exists(ck_path):
        raise SystemExit(f"ckpt 不存在: {ck_path}")
    mona = args.mona if args.mona != "auto" else detect_mona(ck_path)
    print(f"[arm] {args.name}  ckpt={ck_path}  mona={mona}", flush=True)

    dev = torch.device(args.device)
    net, net_c = build_photo(mona)
    fp_info = load_photo(net, ck_path, mona)
    net.to(dev).eval()
    net_c.to(dev).eval()

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
        raise SystemExit(f"[arm] {args.pairs} 下没有可用配对（需要 <scene>/GT.png + <scene>/I_*.png）——"
                         f"拒绝在空面板上出分")

    out_root = os.path.join(PRED_ROOT, f"real_pair_1019_{args.name}")
    os.makedirs(out_root, exist_ok=True)
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

    rec = {"arm": args.name, "ckpt": ck_path, "mona": mona, "arch": "photo1031",
           "n_pairs": len(pairs), "psnr": round(float(np.mean(psnrs)), 4),
           "pred_dir": out_root, **fp_info}
    with open(os.path.join(out_root, "arm_info.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print(f"\n[done] PSNR={rec['psnr']:.3f}  -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
