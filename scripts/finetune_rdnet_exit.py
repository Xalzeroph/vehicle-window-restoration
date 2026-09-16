#!/usr/bin/env python3
"""S2: 物理出口臂——冻结 plain 主干，只训出口头（E=α̃ 除法 / G=自由增益）。

T̂ = (1−β)·T̂_head + β·T̂_phys
  E: T̂_phys = (I − R̂)/α̃,  α̃ = α_min + (1−α_min)·sigmoid(h) ∈ [α_min, 1]（有界=裸区保护）
  G: T̂_phys = (I − R̂)·g,   g = g_max·sigmoid(h)
分段训练（P2 审核修订：sigmoid 零初始化 ≠ 0，且 β=0 会让出口梯度消失）：
  stage1 (iter < --stage1) β≡1 → 出口自身先学；stage2 β 逐像素可学（init≈1）
主干全冻结（防 R̂ 退化）；损失 = Charbonnier(T̂, T)，无 L_recons（与除法出口冲突）。

用法:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/finetune_rdnet_exit.py --arm E --seed 20260915 \
      --out runs/ckpt/exit_E_s20260915
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)
from models.arch.RDnet_ import FullNet_NLP              # noqa: E402
from models.arch.classifier import PretrainedConvNext  # noqa: E402

from finetune_rdnet_tint import TintDataset, charbonnier, _seed_worker  # noqa: E402


class ExitHead(nn.Module):
    def __init__(self, arm, alpha_min=0.05, g_max=20.0, beta_init=4.0):
        super().__init__()
        self.arm, self.alpha_min, self.g_max = arm, alpha_min, g_max
        self.body = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1))
        nn.init.zeros_(self.body[-1].bias)
        with torch.no_grad():
            self.body[-1].bias[1] = beta_init      # β 起始 ≈1（stage2 从这里往下学）

    def forward(self, I, R_hat, T_head):
        out = self.body(torch.cat([I, R_hat, T_head], 1))
        h_main, h_beta = out[:, :1], out[:, 1:2]
        if self.arm == "E":
            main = self.alpha_min + (1 - self.alpha_min) * torch.sigmoid(h_main)   # α̃
            phys = (I - R_hat) / main
        else:
            main = self.g_max * torch.sigmoid(h_main)                             # g
            phys = (I - R_hat) * main
        beta = torch.sigmoid(h_beta)
        return beta, main, phys


def build_plain(base_ckpt, save_memory=True):
    net = FullNet_NLP(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                      loss_col=4, num_classes=1000, drop_path=0, save_memory=save_memory,
                      inter_supv=True, head_init_scale=None, kernel_size=3)
    ck = torch.load(base_ckpt, map_location="cpu")
    net.load_state_dict(ck.get("icnn", ck), strict=False)
    net_c = PretrainedConvNext("convnext_small_in22k")
    ck2 = torch.load(os.path.join(PROJ, "pretrained/cls_model.pth"), map_location="cpu")
    net_c.load_state_dict(ck2.get("icnn", ck2.get("model", ck2)), strict=False)
    net.eval().requires_grad_(False)
    net_c.eval().requires_grad_(False)
    return net, net_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["E", "G"], required=True)
    ap.add_argument("--base", default="runs/ckpt/rdt_plain/ckpt_final.pth")
    ap.add_argument("--triplets", default="data/train_sets/train_triplets_rdt_0915")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--stage1", type=int, default=3000, help="β≡1 只训出口的 iter 数")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seg-iters", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.out if os.path.isabs(args.out) else os.path.join(PROJ, args.out)
    os.makedirs(out, exist_ok=True)
    base = args.base if os.path.isabs(args.base) else os.path.join(PROJ, args.base)
    trip = args.triplets if os.path.isabs(args.triplets) else os.path.join(PROJ, args.triplets)

    dev = torch.device(f"cuda:{args.gpu}")
    net, net_c = build_plain(base)
    net.to(dev); net_c.to(dev)
    head = ExitHead(args.arm).to(dev)
    n_tr = sum(p.numel() for p in head.parameters())
    print(f"[exit] arm={args.arm} base={os.path.relpath(base, PROJ)} head={n_tr/1e3:.0f}k 参数, 主干全冻结", flush=True)

    ds = TintDataset(trip)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.05 + 0.95 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)) / 2)

    logf = open(os.path.join(out, "train_log.csv"), "w")
    logf.write("iter,L_t,mean_main,mean_beta,sec\n")
    segs = [(224, args.batch), (256, max(6, args.batch // 2)), (320, max(4, args.batch // 3))]
    it, t0 = 0, time.time()
    while it < args.iters:
        s, b = segs[(it // args.seg_iters) % len(segs)]
        ds.size = s
        dl = DataLoader(ds, batch_size=b, shuffle=True, num_workers=4, drop_last=True,
                        pin_memory=True, worker_init_fn=_seed_worker)
        for I, T, R, A in dl:
            if it >= args.iters:
                break
            I = I.to(dev, non_blocking=True); T = T.to(dev, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast():
                code = net_c(I)
                _, cols = net(I, code, prompt=True)
                out_last = cols[-1].float()
                R_hat, T_head = out_last[:, :3].clamp(0, 1), out_last[:, 3:].clamp(0, 1)
            beta, main, phys = head(I, R_hat, T_head)
            if it < args.stage1:
                beta = torch.ones_like(beta)
            T_hat = ((1 - beta) * T_head + beta * phys).clamp(0, 1)
            loss = charbonnier(T_hat, T).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                sec = time.time() - t0
                mm, mb = float(main.detach().mean()), float(beta.detach().mean())
                print(f"[exit] it {it:5d} L_t={float(loss.detach()):.5f} main={mm:.3f} β={mb:.3f} {sec:.0f}s", flush=True)
                logf.write(f"{it},{float(loss.detach()):.5f},{mm:.4f},{mb:.4f},{sec:.1f}\n")
                logf.flush()
    torch.save({"head": head.state_dict(), "args": vars(args)}, os.path.join(out, "ckpt_final.pth"))
    print(f"done -> {os.path.relpath(out, PROJ)}/ckpt_final.pth  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
