#!/usr/bin/env python3
"""M1 掩码门控探针（2026-09-21 轮）：冻结官方 RDNet，只训掩码头。

架构:  T̂ = I − m̂ ⊙ D
  D = 官方 decoder 残差 = I − 官方输出（冻结白拿）
  m̂ = MaskHead(FocalNet 四级特征)（唯一可训，末层 bias=−4 → 初始≈0）

数据: data/train_sets/train_mask_0921/{I,T,mask}（Ω 标签；无洞帧 mask 全零）
损失: BCE + Dice（纯掩码监督，探针口径）
预算: 三档同预算 10k iter、bs8、AdamW 1e-3、seed101（容量阶梯 --width 32/96/192）

用法:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/finetune_rdnet_mask.py --width 32 --out runs/ckpt/mask_w32
"""
import argparse, csv, json, math, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ORIG_CWD = os.getcwd()
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "RDNet"))
os.chdir(os.path.join(PROJ, "RDNet"))
from finetune_rdnet_peft import build               # noqa: E402

DATA = os.path.join(PROJ, "data/train_sets/train_mask_0921")
HOLD = ("VID_20260426_173604", "VID_20260426_173827")   # 预登记留出场景


class MaskHead(nn.Module):
    """U-Net 式多尺度掩码解码器；width 控制容量（三档）。"""

    def __init__(self, in_chs, width=32):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, width, 1) for c in in_chs])
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                          nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            for _ in range(len(in_chs) - 1)])
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, feats, out_hw):
        x = None
        for i in range(len(feats) - 1, -1, -1):
            p = self.proj[i](feats[i])
            if x is not None:
                x = F.interpolate(x, size=p.shape[-2:], mode='bilinear', align_corners=False)
                x = self.blocks[i](torch.cat([x, p], 1))
            else:
                x = self.blocks[0](torch.cat([p, p], 1)) if len(feats) == 1 else p
        x = F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)
        return self.head(x)


class MaskDataset(Dataset):
    def __init__(self, split='train', size=320):
        with open(os.path.join(DATA, 'meta.json')) as fp:
            meta = json.load(fp)
        self.ids = [k for k, v in meta.items()
                    if (v['scene'] in HOLD) == (split == 'holdout')]
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        k = self.ids[i]
        rng = np.random.default_rng(torch.initial_seed() % (2 ** 32) + i)
        def rd(sub, gray=False):
            im = cv2.imread(os.path.join(DATA, sub, k + '.png'), 0 if gray else 1)
            return im
        I = rd('I'); T = rd('T'); M = rd('mask', True)
        s = self.size
        h, w = I.shape[:2]
        y0 = int(rng.integers(0, h - s + 1)); x0 = int(rng.integers(0, w - s + 1))
        I = I[y0:y0 + s, x0:x0 + s]; T = T[y0:y0 + s, x0:x0 + s]; M = M[y0:y0 + s, x0:x0 + s]
        if rng.random() < 0.5:
            I, T, M = I[:, ::-1].copy(), T[:, ::-1].copy(), M[:, ::-1].copy()
        c = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float() / 255.  # noqa: E731
        return c(I), c(T), torch.from_numpy(np.ascontiguousarray(M[None])).float() / 255.


def dice_loss(p, t, eps=1.0):
    p = torch.sigmoid(p)
    num = 2 * (p * t).sum()
    return 1 - (num + eps) / (p.sum() + t.sum() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--iters', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--out', default=None)
    ap.add_argument('--log-every', type=int, default=250)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.out or os.path.join(PROJ, f'runs/ckpt/mask_w{args.width}')
    if not os.path.isabs(out):
        out = os.path.join(PROJ, out)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(f'cuda:{args.gpu}')

    net, net_c = build(save_memory=True, mona='none', grad_ckpt=False)
    net.eval().requires_grad_(False)
    net.to(dev); net_c.to(dev)
    feats = {}
    h = net.baseball.register_forward_hook(lambda m, i, o: feats.update(base=o[0]))
    # 探测特征通道（跑一次占位前向）
    with torch.no_grad():
        x0 = torch.zeros(1, 3, 256, 256, device=dev)
        net(x0, net_c(x0), prompt=True)
    in_chs = [f.shape[1] for f in feats['base']]
    head = MaskHead(in_chs, width=args.width).to(dev)
    n_head = sum(p.numel() for p in head.parameters())
    print(f'[m1] FocalNet 特征通道 {in_chs} | MaskHead width={args.width} '
          f'{n_head/1e6:.2f}M | 官方冻结 {sum(p.numel() for p in net.parameters())/1e6:.1f}M', flush=True)

    ds_tr = MaskDataset('train', size=320)
    ds_ho = MaskDataset('holdout', size=320)
    print(f'[m1] 训练 {len(ds_tr)} 帧 / 留出 {len(ds_ho)} 帧（{HOLD}）', flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 200) * (0.05 + 0.95 * 0.5 *
                        (1 + math.cos(math.pi * min(i, args.iters) / args.iters))))
    dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    logf = open(os.path.join(out, 'train_log.csv'), 'w')
    logf.write('iter,bce,dice,iou_ho,sec\n')
    it, t0, run = 0, time.time(), {}
    head.train()
    while it < args.iters:
        for I, T, M in dl:
            if it >= args.iters:
                break
            I = I.to(dev); M = M.to(dev)
            with torch.no_grad():
                code = net_c(I)
                _, outs = net(I, code, prompt=True)
                D = I - outs[-1][:, 3:]                      # 官方残差
            logits = head(feats['base'], I.shape[-2:])
            bce = F.binary_cross_entropy_with_logits(logits, M)
            dl_ = dice_loss(logits, M)
            loss = bce + dl_
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
            it += 1
            if it % args.log_every == 0:
                head.eval()
                ious = []
                with torch.no_grad():
                    for I2, T2, M2 in DataLoader(torch.utils.data.Subset(ds_ho, range(min(40, len(ds_ho)))),
                                                 batch_size=2, num_workers=0):
                        I2 = I2.to(dev); M2 = M2.to(dev)
                        net(I2, net_c(I2), prompt=True)      # 先跑官方前向以更新 hook 特征
                        lg = head(feats['base'], I2.shape[-2:])
                        pr = (torch.sigmoid(lg) > 0.5).float()
                        inter = (pr * M2).sum().item(); union = ((pr + M2) > 0).sum().item()
                        ious.append(inter / max(union, 1))
                head.train()
                iou_ho = float(np.mean(ious)) if ious else 0.0
                sec = time.time() - t0
                print(f'[m1] it {it:5d} bce={bce.item():.4f} dice={dl_.item():.3f} '
                      f'IoU_holdout={iou_ho:.3f} {sec:.0f}s', flush=True)
                logf.write(f'{it},{bce.item():.5f},{dl_.item():.5f},{iou_ho:.4f},{sec:.0f}\n')
                logf.flush()
    torch.save(dict(head=head.state_dict(), width=args.width, in_chs=in_chs,
                    iter=it, holdout=HOLD), os.path.join(out, 'ckpt_final.pth'))
    print(f'[m1] done {it} iter -> {out}/ckpt_final.pth', flush=True)


if __name__ == '__main__':
    main()
