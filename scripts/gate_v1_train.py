#!/usr/bin/env python3
"""门控 v1 训练：两级（1/4 分辨率块决策 + 全分辨率软边）（2026-10-08）
用法: CUDA_VISIBLE_DEVICES=2 python3 scripts/gate_v1_train.py --train apr --out runs/ckpt/gate_v1_apr --iters 3000
      CUDA_VISIBLE_DEVICES=2 python3 scripts/gate_v1_train.py --train apr,clean --out runs/ckpt/gate_v1_apr_clean
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = 32


class Gate(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(11, ch, 3, 1, 1), nn.ReLU(inplace=True),   # I3+P3+|P-I|3+mean1+std1
                                nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True))
        self.down = nn.Sequential(nn.Conv2d(ch, ch*2, 3, 2, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(ch*2, ch*2, 3, 1, 1), nn.ReLU(inplace=True))
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(ch*2, ch, 3, 1, 1), nn.ReLU(inplace=True))
        self.head1 = nn.Conv2d(ch, 1, 3, 1, 1)
        self.fine = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, I, P):
        d = P - I
        gray = I.mean(1, keepdim=True)
        m = F.avg_pool2d(gray, 4, 4)                       # 1/4 分辨率（与 I/P 池化后一致）
        s = F.avg_pool2d(gray*gray, 4, 4) - m*m
        x = torch.cat([F.avg_pool2d(I, 4, 4), F.avg_pool2d(P, 4, 4), F.avg_pool2d(d.abs(), 4, 4),
                       m, s.clamp(min=0).sqrt()], 1)
        f = self.c1(x); f = self.up(self.down(f))
        g1 = torch.sigmoid(self.head1(f))                       # 1/4 分辨率（块决策）
        gU = F.interpolate(g1, size=I.shape[-2:], mode='bilinear', align_corners=False)
        gx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        gy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        gx = F.pad(gx, (0, 1, 0, 0)); gy = F.pad(gy, (0, 0, 0, 1))
        g = torch.sigmoid(self.fine(torch.cat([gU, gray, gx + gy], 1)))   # 全分辨率（软边）
        return g1, g


def _pad(im, min_side, mult=BS, fill=0):
    h, w = im.shape[:2]
    H = max(min_side, ((h + mult - 1)//mult)*mult)
    W = max(min_side, ((w + mult - 1)//mult)*mult)
    if (H, W) == (h, w):
        return im
    if im.ndim == 2:
        return cv2.copyMakeBorder(im, 0, H-h, 0, W-w, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(im, 0, H-h, 0, W-w, cv2.BORDER_REFLECT)


def load_set(root, names, min_side=256):
    xs = []
    for n in names:
        for f in sorted(glob.glob(os.path.join(root, n, '*.npz'))):
            z = np.load(f)
            I, P, T = (_pad(z[k], min_side) for k in ('I', 'P', 'T'))
            bh, bw = I.shape[0]//BS, I.shape[1]//BS
            gs, mk = z['gstar'], z['mask']
            gs = cv2.copyMakeBorder(gs, 0, bh-gs.shape[0], 0, bw-gs.shape[1], cv2.BORDER_REPLICATE)
            mk = cv2.copyMakeBorder(mk, 0, bh-mk.shape[0], 0, bw-mk.shape[1], cv2.BORDER_REPLICATE)
            xs.append((I, P, T, gs, mk))
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='apr'); ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=3000); ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--crop', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam-lab', type=float, default=2.0); ap.add_argument('--lam-harm', type=float, default=1.0)
    ap.add_argument('--lam-tv', type=float, default=0.02)
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    data = load_set(os.path.join(PROJ, 'data/gate_v1'), a.train.split(','))
    print(f'[gate] 训练样本 {len(data)} 张（{a.train}）', flush=True)
    net = Gate().to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    rng = np.random.default_rng(20261008)
    for it in range(a.iters + 1):
        idx = rng.integers(0, len(data), a.bs)
        I = np.zeros((a.bs, 3, a.crop, a.crop), np.float32); P = np.zeros_like(I); T = np.zeros_like(I)
        for k, i in enumerate(idx):
            Ii, Pi, Ti, _, _ = data[i]
            H, W = Ii.shape[:2]
            ch, cw = min(a.crop, H), min(a.crop, W)
            y0 = int(rng.integers(0, (H - ch)//BS + 1))*BS; x0 = int(rng.integers(0, (W - cw)//BS + 1))*BS
            I[k] = Ii[y0:y0+ch, x0:x0+cw].transpose(2, 0, 1)[:, :ch//BS*BS, :cw//BS*BS]/255.
            P[k] = Pi[y0:y0+ch, x0:x0+cw].transpose(2, 0, 1)[:, :ch//BS*BS, :cw//BS*BS]/255.
            T[k] = Ti[y0:y0+ch, x0:x0+cw].transpose(2, 0, 1)[:, :ch//BS*BS, :cw//BS*BS]/255.
        I = torch.from_numpy(I).to(dev); P = torch.from_numpy(P).to(dev); T = torch.from_numpy(T).to(dev)
        # 块级标签（从 npz 现算更省事：这里用 crop 后逐块复算）
        def blk(im):
            return F.avg_pool2d(im, BS, BS)
        g1, g = net(I, P)
        Ig = I + g*(P - I)
        L_rec = (Ig - T).abs().mean()
        L_harm = torch.relu((Ig - T).abs() - (I - T).abs()).mean()
        # 标签监督：逐像素闭式 g*，池化到块级与 1/4 级两路
        a_ = (I - T); b_ = (P - I)
        num = -(a_*b_).sum(1, keepdim=True); den = (b_*b_).sum(1, keepdim=True) + 1e-6
        gstar_f = (num/den).clamp(0, 1)
        msk_f = (den > 1e-4).float()
        gs_blk, ms_blk = blk(gstar_f), blk(msk_f)
        gs_q, ms_q = F.avg_pool2d(gstar_f, 4, 4), F.avg_pool2d(msk_f, 4, 4)
        L_lab = ((ms_blk*((blk(g) - gs_blk)**2)).sum()/(ms_blk.sum() + 1e-6)
                 + (ms_q*((g1 - gs_q)**2)).sum()/(ms_q.sum() + 1e-6))
        L_tv = (g[:, :, :, 1:] - g[:, :, :, :-1]).abs().mean() + (g[:, :, 1:, :] - g[:, :, :-1, :]).abs().mean()
        loss = L_rec + a.lam_lab*L_lab + a.lam_harm*L_harm + a.lam_tv*L_tv
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            print(f'[gate] it {it:5d} rec {L_rec.item():.4f} lab {L_lab.item():.4f} '
                  f'harm {L_harm.item():.4f} gmean {g.mean().item():.3f}', flush=True)
    os.makedirs(os.path.join(PROJ, a.out), exist_ok=True)
    torch.save(dict(sd=net.state_dict(), train=a.train, iters=a.iters), os.path.join(PROJ, a.out, 'gate.pth'))
    print(f'[gate] saved {a.out}/gate.pth', flush=True)


if __name__ == '__main__':
    main()
