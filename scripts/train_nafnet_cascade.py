#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""级联 NAFNet 微调（stage-2 refiner）—— 2026-10-21。

思路
----
冻结 RDNet 得到 `T̂_base`（已缓存），再训一个 NAFNet 学**增量**：
    T̂ = clamp(T̂_base + NAFNet(I, T̂_base), 0, 1)
这样 base 的去反射能力被完整保留，refiner 只做修正 —— 可归因（base vs cascade）。

四臂损失对照（预登记）
--------------------
  A0  l1                  : L1 —— 纯级联基线
  A1  l1+hf               : + 1.0·L_HF —— 高频内容监督（主假设）
  A2  l1+gc               : + 0.1·L_gc —— 梯度方向余弦（对照）
  A3  l1+hf+gc            : 两者叠加
（全部附带 L_res=0.02·|T̂−T̂_base|₁ 作级联稳定项，四臂一视同仁）

两个必须记住的实现细节
--------------------
1. **`NAFNet_refine(global_residual=True)` 有 bug**：其 `x + inp[:3]` 切的是 **batch 维**，
   bs≠3 直接报维度错（实测 bs=3 能跑、bs=4 失败）。
   → 用 `global_residual=False`，残差**在外部加**。
2. **GCF 不能防糊**：输出变糊时 ∇T̂ 方向仍对，cos≈1。
   所以 GCF 只作"去反射方向"约束，细节交给 L_HF。这也是默认给 0.1 而非主权重的理由。

用法
----
  python3 scripts/train_nafnet_cascade.py --arm l1+hf --gpu 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "RDNet"))

from gcf_loss import grad_cosine_loss          # noqa: E402
from models.arch.NAFNET import NAFNet_refine   # noqa: E402

CACHE = os.path.join(PROJ, "data/train_sets/stage2_cache_1021")
EPS = 1e-3
SIG_LP = 2.0
KSIZE = 17          # 与 OpenCV float 路径的 σ=2 高斯核尺寸一致
ARMS = {"l1": (1.0, 0.0, 0.0), "l1+hf": (1.0, 1.0, 0.0),
        "l1+gc": (1.0, 0.0, 0.1), "l1+hf+gc": (1.0, 1.0, 0.1)}
LAM_RES = 0.02

_GP = {}


def gauss_kernel(dev, dt):
    """17×17 各向同性高斯核（σ=2），形状 (1,1,17,17)。

    注意：PyTorch 的 **reflect padding 对 4D 输入必须一次填最后两个维度**
    （填 (8,8) 只填 W 会报 NotImplementedError），所以这里用单个 2D 卷积核，
    padding 用 4 元组。
    """
    key = (dev, dt)
    if key in _GP:
        return _GP[key]
    r = torch.arange(KSIZE, device=dev, dtype=dt) - (KSIZE - 1) / 2
    k1 = torch.exp(-(r ** 2) / (2 * SIG_LP ** 2))
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1)
    k2 = k2 / k2.sum()
    _GP[key] = k2.view(1, 1, KSIZE, KSIZE)
    return _GP[key]


def highpass(x):
    """灰度高通：x - Gaussian(σ=2)。x: (B,C,H,W) -> 同形状（逐通道）。"""
    k = gauss_kernel(x.device, x.dtype)
    c = x.shape[1]
    p = KSIZE // 2
    y = F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"),
                 k.repeat(c, 1, 1, 1), groups=c)
    return x - y


def charb(a, b):
    return torch.sqrt((a - b) ** 2 + EPS ** 2).mean()


class CacheDS(Dataset):
    def __init__(self, root, size=256, seed=0):
        self.items = json.load(open(os.path.join(root, "meta.json"), encoding="utf-8"))
        assert self.items, f"空缓存: {root}"
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _rd(p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, i):
        it = self.items[i]
        I, T, B = self._rd(it["I"]), self._rd(it["T"]), self._rd(it["base"])
        h, w = I.shape[:2]
        # 短边不足时先上采样 —— 否则 `s=min(size,h,w)` 会逐样本不同，
        # DataLoader collate 报 "Trying to resize storage that is not resizable"
        # （实测 2300 张里有 2 张 openrr 短边只有 181/236 px，导致四臂全崩）
        if min(h, w) < self.size:
            sc = self.size / min(h, w) * 1.02
            nh, nw = int(round(h * sc)), int(round(w * sc))
            I, T, B = (cv2.resize(x, (nw, nh), interpolation=cv2.INTER_LINEAR)
                       for x in (I, T, B))
            h, w = nh, nw
        s = self.size
        y0 = int(self.rng.integers(0, h - s + 1)); x0 = int(self.rng.integers(0, w - s + 1))
        I, T, B = (x[y0:y0 + s, x0:x0 + s] for x in (I, T, B))
        if self.rng.random() < 0.5:
            I, T, B = (x[:, ::-1].copy() for x in (I, T, B))
        chw = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).clone()
        return chw(I), chw(T), chw(B)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    w_t, w_hf, w_gc = ARMS[args.arm]
    tag = args.arm.replace("+", "_")
    out = args.out or os.path.join(PROJ, f"runs/ckpt/naf_{tag}")
    os.makedirs(out, exist_ok=True)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)

    net = NAFNet_refine(img_channel=6, width=32, middle_blk_num=1,
                        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                        global_residual=False).to(dev)
    n = sum(p.numel() for p in net.parameters())
    ds = CacheDS(args.cache, size=args.size, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.1 + 0.45 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))
    scaler = torch.cuda.amp.GradScaler()

    print(f"[naf] arm={args.arm}  loss w_t={w_t} w_hf={w_hf} w_gc={w_gc} w_res={LAM_RES}")
    print(f"[naf] params {n/1e6:.2f}M  data {len(ds)}  bs={args.batch} "
          f"size={args.size}  iters={args.iters}  out={out}", flush=True)

    logf = open(os.path.join(out, "train_log.csv"), "w")
    logf.write("iter,L_t,L_hf,L_gc,L_res,total,sec\n")
    it, t0 = 0, time.time()
    net.train()
    while it < args.iters:
        for I, T, B in dl:
            if it >= args.iters:
                break
            I = I.to(dev, non_blocking=True); T = T.to(dev, non_blocking=True)
            B = B.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast():
                delta = net(I, B)
                T_hat = (B + delta).clamp(0, 1)
                L_t = w_t * charb(T_hat, T)
                L_res = LAM_RES * (T_hat - B).abs().mean()
                L_hf = L_gc = torch.zeros((), device=dev)
                if w_hf > 0:
                    go = T_hat.mean(1, keepdim=True)
                    gt = T.mean(1, keepdim=True)
                    L_hf = w_hf * charb(highpass(go), highpass(gt))
                if w_gc > 0:
                    L_gc = w_gc * grad_cosine_loss(T_hat.float(), T.float())
                total = L_t + L_hf + L_gc + L_res
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(opt); scaler.update(); sched.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                sec = time.time() - t0
                g = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)
                print(f"[naf] it {it:5d} L_t={g(L_t):.4f} L_hf={g(L_hf):.4f} "
                      f"L_gc={g(L_gc):.4f} L_res={g(L_res):.4f} "
                      f"total={g(total):.4f} {sec:.0f}s", flush=True)
                logf.write(f"{it},{g(L_t):.5f},{g(L_hf):.5f},{g(L_gc):.5f},"
                           f"{g(L_res):.5f},{g(total):.5f},{sec:.1f}\n")
                logf.flush()
            if it % 2500 == 0 and it > 0:
                torch.save({"model": net.state_dict(), "args": vars(args), "iter": it},
                           os.path.join(out, f"ckpt_{it}.pth"))

    torch.save({"model": net.state_dict(), "args": vars(args), "iter": it},
               os.path.join(out, "ckpt_final.pth"))
    logf.close()
    print(f"[naf] done -> {out}/ckpt_final.pth  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
