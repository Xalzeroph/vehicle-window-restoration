#!/usr/bin/env python3
"""M1 可辨识性上界探针（Opus 定的决定性实验）：像素直连小 U-Net 预测 Ω。

不接冻结主干（排除"FocalNet 特征信息不足"），直接从 I 预测 mask，同数据同预算。
判定（预登记）：训练域 IoU ≥0.80 → 掩码线有救（加高分辨率分支重跑）；≤0.65 → 单帧不可辨识，线封。
用法: CUDA_VISIBLE_DEVICES=1 python3 scripts/probe_mask_direct_0921.py
"""
import json, math, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, 'data/train_sets/train_mask_0921')
HOLD = ('VID_20260426_173604', 'VID_20260426_173827')
OUT = os.path.join(PROJ, 'runs/ckpt/mask_direct')


class Conv(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class DirectUNet(nn.Module):
    def __init__(self, w=48):
        super().__init__()
        self.e1 = Conv(3, w); self.e2 = Conv(w, 2 * w); self.e3 = Conv(2 * w, 4 * w); self.e4 = Conv(4 * w, 8 * w)
        self.d3 = Conv(8 * w + 4 * w, 4 * w); self.d2 = Conv(4 * w + 2 * w, 2 * w); self.d1 = Conv(2 * w + w, w)
        self.head = nn.Conv2d(w, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.avg_pool2d(e1, 2)); e3 = self.e3(F.avg_pool2d(e2, 2))
        e4 = self.e4(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode='bilinear', align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], 1))
        return self.head(d1)


class DS(Dataset):
    def __init__(self, split='train', size=320):
        meta = json.load(open(os.path.join(DATA, 'meta.json')))
        self.ids = [k for k, v in meta.items() if v['scene'] != 'synth' and (v['scene'] in HOLD) == (split == 'hold')]
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        k = self.ids[i]
        rng = np.random.default_rng(torch.initial_seed() % (2 ** 32) + i)
        I = cv2.imread(os.path.join(DATA, 'I', k + '.png')); M = cv2.imread(os.path.join(DATA, 'mask', k + '.png'), 0)
        s = self.size; h, w = I.shape[:2]
        y0, x0 = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
        I, M = I[y0:y0 + s, x0:x0 + s], M[y0:y0 + s, x0:x0 + s]
        if rng.random() < 0.5:
            I, M = I[:, ::-1].copy(), M[:, ::-1].copy()
        return (torch.from_numpy(np.ascontiguousarray(I[:, :, ::-1].transpose(2, 0, 1))).float() / 255.,
                torch.from_numpy(np.ascontiguousarray(M[None])).float() / 255.)


def dice(p, t, eps=1.0):
    p = torch.sigmoid(p)
    return 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def iou_full(model, ds, dev, n=20):
    ious = []
    for i in range(min(n, len(ds))):
        k = ds.ids[i]
        I = cv2.imread(os.path.join(DATA, 'I', k + '.png')); M = cv2.imread(os.path.join(DATA, 'mask', k + '.png'), 0)
        h, w = I.shape[:2]
        pad = cv2.copyMakeBorder(I, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REFLECT)
        x = torch.from_numpy(pad[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            pr = (torch.sigmoid(model(x))[0, 0, :h, :w] > 0.5).cpu().numpy()
        gt = (M > 127)
        ious.append(float((pr & gt).sum()) / max(float((pr | gt).sum()), 1))
    return float(np.median(ious))


def main():
    dev = torch.device('cuda:0')
    torch.manual_seed(101); np.random.seed(101)
    os.makedirs(OUT, exist_ok=True)
    model = DirectUNet(w=48).to(dev)
    print(f'[direct] U-Net w=48 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
    ds_tr, ds_ho = DS('train'), DS('hold')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(i, 10000) / 10000))))
    dl = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    logf = open(os.path.join(OUT, 'train_log.csv'), 'w'); logf.write('iter,bce,dice,iou_ho,sec\n')
    it, t0 = 0, time.time()
    while it < 10000:
        for I, M in dl:
            if it >= 10000: break
            I, M = I.to(dev), M.to(dev)
            lg = model(I)
            bce = F.binary_cross_entropy_with_logits(lg, M); dl_ = dice(lg, M)
            (bce + dl_).backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step(); it += 1
            if it % 1000 == 0:
                model.eval(); ho = iou_full(model, ds_ho, dev); tr = iou_full(model, ds_tr, dev); model.train()
                sec = time.time() - t0
                print(f'[direct] it {it} bce={bce.item():.4f} dice={dl_.item():.3f} IoU_train={tr:.3f} IoU_hold={ho:.3f} {sec:.0f}s', flush=True)
                logf.write(f'{it},{bce.item():.5f},{dl_.item():.5f},{ho:.4f},{sec:.0f}\n'); logf.flush()
    torch.save(dict(model=model.state_dict()), os.path.join(OUT, 'ckpt_final.pth'))
    print('[direct] done', flush=True)


if __name__ == '__main__':
    main()
