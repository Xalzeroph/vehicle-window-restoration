#!/usr/bin/env python3
"""Pilot 微调入口: 方案 A (αTAP) vs 方案 B (零结构改动技巧包)。

协议 (docs/2026.9.9_模型分析与训练入口落地 + 9.13 评审修正):
  - 初始化: 官方 rdnet.pth; 冻结 FocalNet backbone (A/B 同待遇, 12GB 可行性)
  - 数据: data/train_sets/train_pairs_pilot (I/T/R uint8 + alpha float16), 224², amp
  - 阶段 1: 只训 α 头 (L_α); 阶段 2: β 线性 ramp 0→1, 全可训练参数, L_T+L_R+L_α
  - 方案 A: inject=True  → prompt 拼接 α̂ 统计 / P3 硬减法 T0=I−R̂+Δ̂ / 除法出口
  - 方案 B: inject=False → 官方参数化; L_T 用显式 w=clamp((0.3/α_GT)²,1,10) 加权;
            α 头仍训练 (L_α) 但不注入, 供 α̂ MAE 报告
  - 权重 (两臂相同): L_R 权重同 w; L_α = log 域 L1 × T_GT × (1+10·band) ×
    (1+κ_naked·[α_GT≥0.95]); 饱和掩膜 (I≥254) 排除 T/R 损失
    (κ_naked 默认 2.0: round 2 修 α̂ 高区压缩, 原始 pilot 等价 κ=0)

用法:
  python3 scripts/finetune_alpha_tap.py --arm A --gpu 0 --iters 1000 --batch 4
  python3 scripts/finetune_alpha_tap.py --arm B --gpu 1 --iters 1000 --batch 4
  # α 头饱和测试 (只训 α 头, 可用 --init 从 pilot ckpt 热启动):
  python3 scripts/finetune_alpha_tap.py --arm A --alpha-only --iters 8000 \
      --out runs/ckpt/A_sat --init runs/ckpt/A_r3/ckpt_final.pth --gpu 0
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)

from models.arch.rdnet_atap import (FullNet_NLP_ATap, FullNet_NLP_ATap_Code,  # noqa: E402
                                    load_official_atap)
from models.arch.classifier import PretrainedConvNext  # noqa: E402

RDNET_CKPT = os.path.join(PROJ, "pretrained", "rdnet.pth")
ALPHA_REF = 0.3
KAPPA_B = 10.0
BAND_TAU = 0.02
NAKED_TAU = 0.95


class PairDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "I")))
        assert self.ids, f"空数据集: {root}"

    def __len__(self):
        return len(self.ids)

    def _img(self, sub, sid):
        return np.asarray(Image.open(os.path.join(self.root, sub, sid + ".png")).convert("RGB"),
                          np.float32).transpose(2, 0, 1) / 255.0

    def __getitem__(self, i):
        sid = self.ids[i]
        I = torch.from_numpy(self._img("I", sid).copy())
        T = torch.from_numpy(self._img("T", sid).copy())
        R = torch.from_numpy(self._img("R", sid).copy())
        A = torch.from_numpy(np.load(os.path.join(self.root, "alpha", sid + ".npy")).astype(np.float32)[None])
        return I, T, R, A


def charbonnier(d, eps=1e-3):
    return torch.sqrt(d * d + eps * eps)


def grad_mag(a):
    gx = F.pad((a[..., :, 1:] - a[..., :, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((a[..., 1:, :] - a[..., :-1, :]).abs(), (0, 0, 0, 1))
    return gx + gy


def load_ckpt_into(net, path):
    """pilot ckpt (inject 同构) → 同构模型; 兼容 prompt 首层 6→9 列加宽。"""
    sd = dict(torch.load(path, map_location="cpu")["icnn"])
    if net.inject and "prompt.0.weight" in sd and sd["prompt.0.weight"].shape[1] == 6:
        w, b = sd.pop("prompt.0.weight"), sd.pop("prompt.0.bias")
        miss, unex = net.load_state_dict(sd, strict=False)
        with torch.no_grad():
            net.prompt[0].weight[:, :6].copy_(w)
            net.prompt[0].bias.copy_(b)
    else:
        miss, unex = net.load_state_dict(sd, strict=False)
    print(f"[init] load {path}: missing={len(miss)} unexpected={len(unex)}")


def build_model(arm, device, init=None, code_cond=False):
    inject = arm == "A"
    cls = FullNet_NLP_ATap_Code if code_cond else FullNet_NLP_ATap
    net = cls(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
              num_subnet=4, loss_col=4, inject=inject)
    load_official_atap(net, RDNET_CKPT)
    if init:
        load_ckpt_into(net, init)
    for n, p in net.named_parameters():
        p.requires_grad_(not n.startswith("baseball."))
    net = net.to(device)
    net_c = PretrainedConvNext("convnext_small_in22k").to(device)
    net_c.load_state_dict(torch.load(os.path.join(RDNET, "pretrained/cls_model.pth"),
                                     map_location="cpu")["icnn"])
    net_c.eval()
    for p in net_c.parameters():
        p.requires_grad_(False)
    return net, net_c


def alpha_params(net):
    names = ("alpha_c3.", "alpha_head.", "alpha_refine.")
    return {n for n, _ in net.named_parameters() if n.startswith(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--pairs", default=os.path.join(PROJ, "data/train_sets/train_pairs_pilot"))
    ap.add_argument("--out", default=None, help="默认 {PROJ}/runs/ckpt/{arm}")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stage1-frac", type=float, default=0.3)
    ap.add_argument("--ramp-frac", type=float, default=0.4, help="β 0→1 占 iters 比例")
    ap.add_argument("--lam-alpha", type=float, default=1.0)
    ap.add_argument("--kappa-naked", type=float, default=2.0,
                    help="L_α 对无膜区(α_GT>=0.95)的加权倍数, 对抗 α̂ 高区压缩")
    ap.add_argument("--alpha-only", action="store_true",
                    help="饱和测试: 全程只训 α 头 (单列前向, 跳过 subnet/解码器), 不进入阶段 2")
    ap.add_argument("--code-cond", action="store_true",
                    help="α 头 v2: classifier code 注入全局档位先验 (FullNet_NLP_ATap_Code)")
    ap.add_argument("--init", default=None, help="初始 ckpt (默认官方 rdnet.pth; 可续热启动)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    out = args.out or os.path.join(PROJ, "runs/ckpt", args.arm)
    if not os.path.isabs(out):      # 脚本 chdir 到 RDNet, 相对路径锚定项目根
        out = os.path.join(PROJ, out)
    os.makedirs(out, exist_ok=True)
    for k in ("init", "pairs"):                     # 脚本 chdir 到 RDNet, 相对路径锚定项目根
        v = getattr(args, k)
        if v and not os.path.isabs(v):
            setattr(args, k, os.path.join(PROJ, v))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    ds = PairDataset(args.pairs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    print(f"[arm {args.arm}] data={len(ds)}  iters={args.iters} batch={args.batch} -> {out}")

    net, net_c = build_model(args.arm, device, args.init, args.code_cond)
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in net.parameters())
    print(f"[arm {args.arm}] trainable {n_tr/1e6:.1f}M / total {n_all/1e6:.1f}M (冻结 backbone)")

    a_names = alpha_params(net)
    if args.stage1_frac > 0:
        for n, p in net.named_parameters():
            if p.requires_grad and n not in a_names:
                p.requires_grad_(False)

    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    stage1 = args.iters if args.alpha_only else int(args.iters * args.stage1_frac)
    ramp_len = max(1, int(args.iters * args.ramp_frac))
    if args.alpha_only:
        print(f"[arm {args.arm}] alpha-only 饱和模式: 全程只训 α 头 ({args.iters} iter)")
    logf = open(os.path.join(out, "train_log.csv"), "w")
    logf.write("iter,beta,L_T,L_R,L_alpha,alpha_hat_mean,alpha_mae,delta_abs,sec\n")
    t0 = time.time()
    it = 0
    run = {}
    while it < args.iters:
        for I, T, R, A in dl:
            if it >= args.iters:
                break
            if it == stage1 and stage1 > 0:
                for n, p in net.named_parameters():
                    if not n.startswith("baseball."):
                        p.requires_grad_(True)
                opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
                print(f"--- iter {it}: stage 2, 全部可训练参数解锁 ---")
            beta = 0.0 if it < stage1 else min(1.0, (it - stage1) / ramp_len)
            I, T, R, A = I.to(device), T.to(device), R.to(device), A.to(device)

            with torch.no_grad():
                code = net_c(I)

            with torch.cuda.amp.autocast():
                if it < stage1:
                    alpha = net.forward_alpha_only(I, code) if args.code_cond \
                        else net.forward_alpha_only(I)
                    r_hat = t_hat = delta = None
                else:
                    cols, aux = net(I, code, beta=beta)
                    r_hat, t_hat = cols[-1][:, :3], cols[-1][:, 3:]
                    alpha, delta = aux["alpha"], aux["delta"]

                sat = (I.amax(1, keepdim=True) >= 254.5 / 255).float()
                m = 1.0 - sat
                w = torch.clamp((ALPHA_REF / A) ** 2, 1.0, 10.0)
                band = (grad_mag(A) > BAND_TAU).float()
                naked = (A >= NAKED_TAU).float()
                w_a = (T.mean(1, keepdim=True) * (1.0 + KAPPA_B * band)
                       * (1.0 + args.kappa_naked * naked))
                L_alpha = (((torch.log(alpha.clamp(min=1e-3)) - torch.log(A.clamp(min=1e-3))).abs()
                            * w_a).sum() / w_a.sum().clamp(min=1.0))
                if it < stage1:
                    loss = args.lam_alpha * L_alpha
                    L_T = L_R = torch.zeros(())
                else:
                    dT = charbonnier(t_hat - T)
                    if args.arm == "A":
                        L_T = (dT * m).sum() / m.sum().clamp(min=1.0)
                    else:
                        L_T = (dT * w * m).sum() / (w * m).sum().clamp(min=1.0)
                    L_R = (charbonnier(r_hat - R) * w * m).sum() / (w * m).sum().clamp(min=1.0)
                    loss = L_T + L_R + args.lam_alpha * L_alpha

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                amae = (alpha.float() - A).abs().mean().item()
                run = dict(alpha_hat_mean=alpha.float().mean().item(), alpha_mae=amae,
                           delta_abs=delta.abs().mean().item() if delta is not None else 0.0,
                           L_T=L_T.item(), L_R=L_R.item(), L_alpha=L_alpha.item())
            if it % args.log_every == 0 or it == args.iters - 1:
                sec = time.time() - t0
                print(f"[{args.arm}] it {it:4d} beta={beta:.2f} "
                      f"L_T={run['L_T']:.4f} L_R={run['L_R']:.4f} L_a={run['L_alpha']:.4f} "
                      f"a_hat={run['alpha_hat_mean']:.3f} MAE={run['alpha_mae']:.3f} "
                      f"|d|={run['delta_abs']:.3f} {sec:.0f}s")
                logf.write(f"{it},{beta:.3f},{run['L_T']:.5f},{run['L_R']:.5f},{run['L_alpha']:.5f},"
                           f"{run['alpha_hat_mean']:.4f},{run['alpha_mae']:.4f},{run['delta_abs']:.4f},{sec:.1f}\n")
                logf.flush()
            it += 1
    logf.close()

    ck = {"icnn": net.state_dict(), "arm": args.arm, "args": vars(args)}
    torch.save(ck, os.path.join(out, "ckpt_final.pth"))
    print(f"done [{args.arm}] -> {out}/ckpt_final.pth  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
