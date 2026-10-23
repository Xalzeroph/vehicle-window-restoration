#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10-19 配对集：多模型统一推理 + 评测 + 对比图（2026-10-20）。

模型臂
------
  identity   : 输入原样（平凡基线，必须报，否则绝对数字无法解读）
  rdnet_zs   : RDNet 官方权重 /DataDisk/zmy_code/RDNet-main/rdnet.pth
  ft_s2main  : 我们微调 rdt_s2_main_s101/ckpt_10000.pth（mona PEFT）
  ft_mixbest : 我们微调 mix_* 里表现最好的一臂（--mix-ckpt 指定，可不给）
  lightrr    : 由 run_lightrr_1019.py 在 LightRR 环境里产出，本脚本只读它的预测

口径（**所有臂完全一致**）
-------------------------
- 配对集已统一 resize 到 1280x832（32 的倍数），故不做任何额外 align/resize；
- 指标：PSNR / SSIM，逐通道算再取通道均值（与本仓 util 口径一致），data_range=255；
- 只报 raw（不做增益匹配）为主，同时报 gm（最优乘性匹配）作参考；
- SSIM 用 skimage structural_similarity 全通道 mean。

输出
----
eval/real_pair_1019_<arm>.csv          逐对指标
eval/real_pair_1019_summary.json       各臂汇总
runs/pred/real_pair_1019_<arm>/<scene>__<stem>/t_hat.png   预测（供出图）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
RDNET_DIR = os.path.join(PROJ, "RDNet")
PAIR_DIR = os.path.join(PROJ, "data/test_sets/real_pair_1019")
PRED_ROOT = os.path.join(PROJ, "runs/pred")
EVAL_DIR = os.path.join(PROJ, "eval")

CKPT = {
    "rdnet_zs": "/DataDisk/zmy_code/RDNet-main/rdnet.pth",
    "ft_s2main": os.path.join(PROJ, "runs/ckpt/rdt_s2_main_s101/ckpt_10000.pth"),
}


def log(*a):
    print(*a, flush=True)


def load_img(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) if im is not None else None


def bw_psnr(a, b):
    v = []
    for c in range(a.shape[-1]):
        mse = float(np.mean((a[..., c].astype(np.float64) - b[..., c].astype(np.float64)) ** 2))
        v.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


def bw_ssim(a, b):
    return float(np.mean([ssim_fn(a[..., c], b[..., c], data_range=255)
                          for c in range(a.shape[-1])]))


def gain_match(q, g):
    q = np.asarray(q, np.float64)
    g = np.asarray(g, np.float64)
    s = float((q * g).sum() / max((q * q).sum(), 1e-8))
    return np.clip(q * s, 0, 255)


# ---------------- RDNet ----------------
def build_rdnet(mona: str = "none"):
    sys.path.insert(0, RDNET_DIR)
    cwd = os.getcwd()
    os.chdir(RDNET_DIR)
    from models.arch.RDnet_ import FullNet_NLP
    from models.arch.classifier import PretrainedConvNext
    from rdnet_peft import patch_focalnet
    net = FullNet_NLP(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                      loss_col=4, num_classes=1000, drop_path=0, save_memory=True,
                      inter_supv=True, head_init_scale=None, kernel_size=3)
    if mona != "none":
        n = patch_focalnet(net, mona, verbose=False)
        log(f"  patch_focalnet mode={mona}: {n} blocks")
    ck_path = os.path.join(PROJ, "pretrained/cls_model.pth")
    net_c = PretrainedConvNext("convnext_small_in22k")
    ck2 = torch.load(ck_path, map_location="cpu")
    sd2 = ck2.get("icnn", ck2.get("model", ck2))
    net_c.load_state_dict(sd2, strict=False)
    net_c.eval().requires_grad_(False)
    os.chdir(cwd)
    return net, net_c


def load_rdnet(net, path, mona: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("icnn", ck) if isinstance(ck, dict) else ck
    miss, unex = net.load_state_dict(sd, strict=False)
    log(f"  ckpt={os.path.basename(path)} missing={len(miss)} unexpected={len(unex)} "
        f"iter={ck.get('iter', '-') if isinstance(ck, dict) else '-'}")
    if mona != "none":
        from rdnet_peft import is_mona_name, mona_params, mona_state_dict_keys
        n_sd = len(mona_state_dict_keys(sd))
        n_net = len(mona_params(net))
        miss_mona = [k for k in miss if is_mona_name(k)]
        if n_sd == 0:
            raise RuntimeError(f"ckpt 不含 mona 权重但 --mona {mona}：会静默退回 plain")
        if n_sd != n_net or miss_mona:
            raise RuntimeError(f"mona 未完整加载 ckpt={n_sd} net={n_net} miss={len(miss_mona)}")
        log(f"  mona={mona} 完整加载 {n_net} keys")
    return net


@torch.no_grad()
def predict_rdnet(net, net_c, img, dev):
    x = torch.from_numpy(img.transpose(2, 0, 1) / 255.0)[None].to(dev)
    pad = (0, (-x.shape[-1]) % 32, 0, (-x.shape[-2]) % 32)
    xp = F.pad(x, pad, mode="reflect") if any(pad) else x
    code = net_c(xp)
    with torch.cuda.amp.autocast():
        _, out = net(xp, code, prompt=True)
    t = out[-1][0, 3:].float().clamp(0, 1)
    H, W = img.shape[:2]
    t = t[:, :H, :W].permute(1, 2, 0).cpu().numpy() * 255.0
    return t


# ---------------- 主流程 ----------------
def scenes_pairs():
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-ckpt", default="", help="mix_* 最佳臂的 ckpt 路径（可选）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", default="identity,rdnet_zs,ft_s2main",
                    help="逗号分隔：identity,rdnet_zs,ft_s2main,ft_mixbest")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    dev = torch.device(args.device)
    os.makedirs(EVAL_DIR, exist_ok=True)
    pairs = scenes_pairs()
    log(f"配对 {len(pairs)} 组，臂 {arms}，dev={dev}")

    # 预加载 RDNet 需要的臂
    rd_arms = [a for a in arms if a in ("rdnet_zs", "ft_s2main", "ft_mixbest")]
    nets = {}
    # 臂 -> ckpt 显式映射。
    # 两个必须避免的坑（本次都踩过）：
    #  1) `CKPT.get(a) or args.mix_ckpt` 会让传了 --mix-ckpt 的所有臂都回退到同一 ckpt；
    #  2) **复用同一个 net 实例**换 ckpt 重新 load_state_dict(strict=False) 时，
    #     patch_focalnet 后的 key 覆盖会静默失效（实测三个臂指纹完全相同）。
    #     → 每个臂必须 build 一个**全新**的 net。
    arm_ckpt = dict(CKPT)
    if args.mix_ckpt:
        arm_ckpt["ft_mixbest"] = args.mix_ckpt

    loaded = {}
    for a in rd_arms:
        mona = "none" if a == "rdnet_zs" else "both"
        path = arm_ckpt.get(a)
        if not path or not os.path.exists(path):
            log(f"[跳过臂] {a}: 权重不存在 ({path})")
            continue
        log(f"加载 {a} (mona={mona}) <- {path}")
        net, net_c = build_rdnet(mona)          # ★ 每臂全新实例
        load_rdnet(net, path, mona)
        net.to(dev).eval(); net_c.to(dev).eval()
        loaded[a] = {"mona": mona, "ckpt": path, "net": net, "net_c": net_c}

    # 指纹校验：必须用 **mona 参数**做指纹。
    # 用 stem_comp 之类冻结主干做指纹会三臂完全相同（主干不参与微调），
    # 会误判成"加载了同一权重"。实测 mona1.gamma 之和：
    #   official(ckpt 无 mona) = 0.000192, ft_s2main = 0.336574, ft_mixbest = 0.529241
    fp = {}
    for a, info in loaded.items():
        sd = info["net"].state_dict()
        mona_keys = [k for k in sd if "mona1.gamma" in k or "mona2.gamma" in k]
        if mona_keys:
            s = float(sum(sd[k].float().sum() for k in mona_keys))
        else:
            s = float("nan")
        fp[a] = round(s, 6)
    log("\n[臂指纹校验 —— mona 参数之和]")
    for a, info in loaded.items():
        log(f"  {a:<12} ckpt={info['ckpt'].split('/')[-2]:<24} mona_sum={fp[a]:.6f}")
    vals = [v for v in fp.values() if v == v]
    if len(set(vals)) < len(vals):
        raise RuntimeError(f"不同臂的 mona 指纹相同 {fp} -> 仍有 bug，拒绝继续")

    summary = {}
    for a in arms:
        if a in loaded:
            info = loaded[a]
            net, net_c = info["net"], info["net_c"]
            log(f"[{a}] 使用 {info['ckpt']}")
        rows, t0 = [], time.time()
        for i, (sc, stem, im, gt) in enumerate(pairs, 1):
            if a == "identity":
                pred = im
            else:
                pred = predict_rdnet(net, net_c, im, dev)
            p_raw = bw_psnr(pred, gt)
            p_gm = bw_psnr(gain_match(pred, gt), gt)
            s_raw = bw_ssim(pred, gt)
            rows.append({"arm": a, "scene": sc, "input": stem,
                         "psnr": round(p_raw, 4), "ssim": round(s_raw, 4),
                         "psnr_gm": round(p_gm, 4)})
            d = os.path.join(PRED_ROOT, f"real_pair_1019_{a}", f"{sc}__{stem}")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "t_hat.png"),
                        cv2.cvtColor(np.clip(pred, 0, 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))
            if i % 10 == 0:
                log(f"  [{a}] {i}/{len(pairs)}")
        el = time.time() - t0
        cp = os.path.join(EVAL_DIR, f"real_pair_1019_{a}.csv")
        with open(cp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        summary[a] = {"n": len(rows),
                      "psnr": round(float(np.mean([r["psnr"] for r in rows])), 3),
                      "ssim": round(float(np.mean([r["ssim"] for r in rows])), 4),
                      "psnr_gm": round(float(np.mean([r["psnr_gm"] for r in rows])), 3),
                      "sec_per_img": round(el / max(len(rows), 1), 3)}
        log(f"  -> {a}: PSNR {summary[a]['psnr']:.2f}  SSIM {summary[a]['ssim']:.4f}  "
            f"PSNR(gm) {summary[a]['psnr_gm']:.2f}  {summary[a]['sec_per_img']:.2f}s/img")

    sp = os.path.join(EVAL_DIR, "real_pair_1019_summary.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    log(f"\n写入 {sp}")
    log("\n" + "=" * 62)
    log(f"{'arm':<12}{'n':>4}{'PSNR':>9}{'SSIM':>9}{'PSNR(gm)':>11}")
    log("-" * 62)
    for a, s in summary.items():
        log(f"{a:<12}{s['n']:>4}{s['psnr']:>9.2f}{s['ssim']:>9.4f}{s['psnr_gm']:>11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
