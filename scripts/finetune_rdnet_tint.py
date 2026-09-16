#!/usr/bin/env python3
"""S1: RDNet 贴膜域微调（分离器线）——冻 FocalNet backbone，训其余参数。

数据: data/train_sets/train_triplets_rdt_0915（I/T/R/alpha, 5 档）
损失: Σ_列 (w_col/Σw)·[ Charbonnier(T̂,T)·w(α) + 1.5·Charbonnier(R̂,R) ] + 0.2·|I−(T̂+R̂)|
      w(α) = clamp((0.3/α)², 1, 10)  ← 逆衰减加权（低档样本才激活）
模型: FullNet_NLP + 官方 rdnet.pth；prompt code 由冻结的 cls_model（ConvNeXt）给出
显存: 冻 backbone bs8@224² ≈ 6.3GB（GPU6 实测, AMP）

用法:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/finetune_rdnet_tint.py \
      --out runs/ckpt/rdt_freeze --iters 30000 --batch 8 --gpu 0
"""
import argparse, csv, json, os, sys, time
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
from models.arch.RDnet_ import FullNet_NLP              # noqa: E402
from models.arch.classifier import PretrainedConvNext  # noqa: E402

EPS = 1e-3


def charbonnier(a, b):
    return torch.sqrt((a - b) ** 2 + EPS ** 2)


def _seed_worker(worker_id):
    # fork 出的 worker 共享父进程 numpy 状态 → 各 worker 增强序列相同；用 torch 已种好的
    # worker 种子重种 numpy（torch.initial_seed 每 worker/每 epoch 唯一）
    np.random.seed(torch.initial_seed() % (2 ** 32))


class TintDataset(Dataset):
    """一场景一种：随机档 + 随机裁剪(224-320²) + 翻转。"""

    def __init__(self, root, size_min=224, size_max=320):
        self.root = root
        self.size = size_min          # 批内统一: 训练循环定期改（避免 collate 形状不一致）
        with open(os.path.join(root, "meta.json")) as fp:
            meta = json.load(fp)
        self.levels = meta["levels"]
        self.tags = [f"l{round(float(lv) * 100):03d}" for lv in self.levels]
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "T")))
        assert self.ids, f"空数据集: {root}"
        self.smin, self.smax = size_min, size_max

    def __len__(self):
        return len(self.ids)

    def _img(self, sub, name):
        return np.asarray(Image.open(os.path.join(self.root, sub, name)).convert("RGB"),
                          np.float32).transpose(2, 0, 1) / 255.0

    def __getitem__(self, i):
        sid = self.ids[i]
        j = np.random.randint(len(self.tags))
        I = self._img("I", f"{sid}_{self.tags[j]}.png")
        T = self._img("T", sid + ".png")
        R = self._img("R", sid + ".png")
        A = np.load(os.path.join(self.root, "alpha", f"{sid}_{self.tags[j]}.npy")).astype(np.float32)
        if A.ndim == 2:
            A = A[None]
        s = self.size
        H, W = I.shape[-2:]
        y0 = np.random.randint(0, H - s + 1); x0 = np.random.randint(0, W - s + 1)
        I, T, R, A = (x[..., y0:y0 + s, x0:x0 + s] for x in (I, T, R, A))
        if np.random.rand() < 0.5:
            I, T, R, A = (x[..., ::-1].copy() for x in (I, T, R, A))
        c = lambda x: torch.from_numpy(np.ascontiguousarray(x))    # noqa: E731
        return c(I), c(T), c(R), c(A)


def build(gpu=0, pretrained_ckpt=None, save_memory=True):
    net = FullNet_NLP(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                      loss_col=4, num_classes=1000, drop_path=0, save_memory=save_memory,
                      inter_supv=True, head_init_scale=None, kernel_size=3)
    ck = torch.load(pretrained_ckpt or os.path.join(PROJ, "pretrained/rdnet.pth"),
                    map_location="cpu")
    sd = ck.get("icnn", ck)
    miss, unex = net.load_state_dict(sd, strict=False)
    print(f"[rdt] rdnet.pth: missing={len(miss)} unexpected={len(unex)}")
    net_c = PretrainedConvNext("convnext_small_in22k")
    ck2 = torch.load(os.path.join(PROJ, "pretrained/cls_model.pth"), map_location="cpu")
    sd2 = ck2.get("icnn", ck2.get("model", ck2))
    miss2, unex2 = net_c.load_state_dict(sd2, strict=False)
    print(f"[rdt] cls_model.pth: missing={len(miss2)} unexpected={len(unex2)}")
    net_c.eval().requires_grad_(False)
    return net, net_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", default=os.path.join(PROJ, "data/train_sets/train_triplets_rdt_0915"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze", choices=["backbone", "none"], default="backbone",
                    help="backbone=冻 FocalNet；none=全参微调")
    ap.add_argument("--lam-wlow", type=float, default=1.0,
                    help="逆衰减加权开关（>0=开）：w(α)=clamp((wlow_ref/α)²,1,wlow_cap)，T/R 共用")
    ap.add_argument("--wlow-ref", type=float, default=0.3,
                    help="w(α) 锚点：α≥ref 处 w=1（0.3=原 pilot 配方，训练档位下几乎不激活）")
    ap.add_argument("--wlow-cap", type=float, default=10.0)
    ap.add_argument("--lam-r", type=float, default=1.5)
    ap.add_argument("--lam-recons", type=float, default=0.2)
    ap.add_argument("--save-memory", action="store_true", default=True)
    ap.add_argument("--resume", default="", help="ckpt 路径，续训（恢复 net/opt/scaler/iter）")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seg-iters", type=int, default=2000, help="裁剪尺寸/批大小切换周期")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.out or os.path.join(PROJ, "runs/ckpt/rdt_freeze")
    if not os.path.isabs(out):
        out = os.path.join(PROJ, out)
    os.makedirs(out, exist_ok=True)

    dev = torch.device(f"cuda:{args.gpu}")
    net, net_c = build(args.gpu, save_memory=args.save_memory)
    if args.freeze == "backbone":
        for n, p in net.named_parameters():
            if n.startswith("baseball."):
                p.requires_grad_(False)
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in net.parameters())
    print(f"[rdt] trainable {n_tr/1e6:.1f}M / {n_all/1e6:.1f}M  (freeze={args.freeze})")
    net.to(dev); net_c.to(dev)

    ds = TintDataset(os.path.join(PROJ, args.triplets))
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.1 + 0.45 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))
    scaler = torch.cuda.amp.GradScaler()
    start_it = 0
    if args.resume:
        ck_r = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(ck_r["icnn"])
        if "opt" in ck_r:
            opt.load_state_dict(ck_r["opt"])
        if "scaler" in ck_r:
            scaler.load_state_dict(ck_r["scaler"])
        start_it = int(ck_r.get("iter", 0))
        sched.last_epoch = start_it
        print(f"[rdt] resume {args.resume} @ iter {start_it}", flush=True)
    logf = open(os.path.join(out, "train_log.csv"), "w")
    logf.write("iter,L_t,L_r,L_recons,total,sec\n")
    print(f"[rdt] 数据 {len(ds)} 场景 × {len(ds.tags)} 档; batch={args.batch}（320² 段自动降档）")

    # 裁剪尺寸与批大小绑定：320²@bs8 在 10GB 卡 OOM（实测 256² bs8≈7.5G）
    segs = [(224, args.batch), (256, min(args.batch, 6)), (320, min(args.batch, 4))]
    it, t0, run = start_it, time.time(), {}
    while it < args.iters:
        s, b = segs[(it // args.seg_iters) % len(segs)]   # 定期换裁剪尺寸 + 批大小
        ds.size = s
        dl = DataLoader(ds, batch_size=b, shuffle=True, num_workers=4, drop_last=True,
                        pin_memory=True, worker_init_fn=_seed_worker)
        for I, T, R, A in dl:
            if it >= args.iters:
                break
            I = I.to(dev, non_blocking=True); T = T.to(dev, non_blocking=True)
            R = R.to(dev, non_blocking=True); A = A.to(dev, non_blocking=True)
            with torch.no_grad():
                code = net_c(I)
            with torch.cuda.amp.autocast():
                _, x_img_out = net(I, code, prompt=True)
                L_t = L_r = torch.zeros((), device=dev)
                wsum = 0.0
                for i, out_col in enumerate(x_img_out):
                    R_hat, T_hat = out_col[:, :3], out_col[:, 3:]
                    if args.lam_wlow > 0:
                        w_alpha = ((args.wlow_ref / A.clamp(min=0.02)) ** 2).clamp(1.0, args.wlow_cap)
                        lt = (charbonnier(T_hat, T) * w_alpha).mean()
                        lr_i = (charbonnier(R_hat, R) * w_alpha).mean()
                    else:
                        lt = charbonnier(T_hat, T).mean()
                        lr_i = charbonnier(R_hat, R).mean()
                    wc = float(i + 1); wsum += wc
                    L_t = L_t + lt * wc
                    L_r = L_r + args.lam_r * lr_i * wc
                L_t = L_t / wsum; L_r = L_r / wsum
                R_hat, T_hat = x_img_out[-1][:, :3], x_img_out[-1][:, 3:]
                L_recons = args.lam_recons * (I - T_hat - R_hat).abs().mean()
                total = L_t + L_r + L_recons
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(opt); scaler.update(); sched.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                sec = time.time() - t0
                run = dict(L_t=float(L_t.detach()), L_r=float(L_r.detach()),
                           L_recons=float(L_recons.detach()), total=float(total.detach()))
                print(f"[rdt] it {it:5d} L_t={run['L_t']:.4f} L_r={run['L_r']:.4f} "
                      f"L_recons={run['L_recons']:.4f} total={run['total']:.4f} {sec:.0f}s",
                      flush=True)
                logf.write(",".join([str(it)] + [f"{run[k]:.5f}" for k in
                                                 ("L_t", "L_r", "L_recons", "total")]
                                    + [f"{sec:.1f}"]) + "\n")
                logf.flush()
            if it % args.save_every == 0:
                torch.save({"icnn": net.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "args": vars(args), "iter": it},
                           os.path.join(out, f"ckpt_{it}.pth"))
    torch.save({"icnn": net.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "args": vars(args), "iter": it},
               os.path.join(out, "ckpt_final.pth"))
    print(f"done -> {out}/ckpt_final.pth  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
