#!/usr/bin/env python3
"""门控 v1 建集：缓存官方 P + 算块级 g* 标签（2026-10-08）
用法: CUDA_VISIBLE_DEVICES=2 python3 scripts/gate_v1_build.py --sets apr,may,clean --n-clean 64
产物: data/gate_v1/<set>/<case>.npz  (I,P,T uint8; gstar/mask/r_obs/luma 块级)
"""
import argparse, json, os, sys
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PROJ, 'RDNet'))
os.chdir(os.path.join(PROJ, 'RDNet'))
from finetune_rdnet_peft import build      # noqa: E402
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
from eval_protocol import ls_scale, apply_gm  # noqa: E402

B = '/home/zmy/mycode/datasets'
SETDIR = {'apr': f'{PROJ}/data/test_sets/real_pair_apr_0917',
          'may': f'{PROJ}/data/test_sets/real_pair_may_0917'}
VOC = f'{PROJ}/data/external/VOC2012/VOC2012_s320'
CAP, BS = 512, 32


def pad32(x):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REFLECT), h, w


def infer(net, net_c, dev, I):
    Ip, h, w = pad32(I)
    x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32)/255.).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        code = net_c(x); _, outs = net(x, code, prompt=True); T = outs[-1][:, 3:]
    out = T[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1][:h, :w]
    return (out*255).round().astype(np.uint8)


def fit_cap(im, cap=CAP):
    if max(im.shape[:2]) <= cap:
        return im
    sc = cap/max(im.shape[:2]); sz = (int(im.shape[1]*sc), int(im.shape[0]*sc))
    return cv2.resize(im, sz, interpolation=cv2.INTER_AREA)


def labels(I, P, T):
    """块级 g*（增益对齐口径，隔离全局曝光）+ mask + r_obs + luma"""
    h, w = T.shape[:2]; nh, nw = h//BS, w//BS
    g = float(ls_scale(I, T)); Ig = apply_gm(I, g)
    gs = np.full((nh, nw), np.nan, np.float32); ms = np.zeros((nh, nw), np.float32)
    ro = np.zeros((nh, nw), np.float32); lu = np.zeros((nh, nw), np.float32)
    for by in range(nh):
        for bx in range(nw):
            sl = np.s_[by*BS:(by+1)*BS, bx*BS:(bx+1)*BS]
            tb = T[sl].astype(np.float64); ib = Ig[sl].astype(np.float64); pb = P[sl].astype(np.float64)
            a = (ib - tb).ravel(); b = (pb - ib).ravel(); bb = float((b*b).sum())
            lu[by, bx] = tb.mean()
            ro[by, bx] = np.abs(ib - tb).max(axis=2).mean()/(tb.mean()+1e-9)
            if bb > 1e-6:
                gs[by, bx] = np.clip(-(a*b).sum()/bb, 0.0, 1.0)
                ms[by, bx] = 1.0
    return gs, ms, ro, lu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='apr,may,clean'); ap.add_argument('--n-clean', type=int, default=64)
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}')
    net, net_c = build(save_memory=True, mona='none', grad_ckpt=False); net.to(dev).eval(); net_c.to(dev)
    root = os.path.join(PROJ, 'data/gate_v1')
    for sname in a.sets.split(','):
        outd = os.path.join(root, sname); os.makedirs(outd, exist_ok=True)
        cases = []
        if sname == 'clean':
            fl = sorted(f for f in os.listdir(VOC) if os.path.splitext(f)[1].lower() in ('.jpg', '.png'))[:a.n_clean]
            for f in fl:
                im = cv2.imread(os.path.join(VOC, f))
                if im is not None:
                    cases.append((os.path.splitext(f)[0], im, im))
        else:
            d = SETDIR[sname]
            for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
                T = cv2.imread(f'{d}/{sc}/GT.png')
                if T is None: continue
                for f in sorted(x for x in os.listdir(f'{d}/{sc}') if x.startswith('I_')):
                    I = cv2.imread(f'{d}/{sc}/{f}')
                    if I is not None and I.shape == T.shape:
                        cases.append((f'{sc}__{f[2:-4]}', I, T))
        n = 0
        for case, I, T in cases:
            I, T = fit_cap(I), fit_cap(T)
            P = infer(net, net_c, dev, I)
            gs, ms, ro, lu = labels(I, P, T)
            np.savez_compressed(os.path.join(outd, f'{case}.npz'), I=I, P=P, T=T,
                                gstar=gs, mask=ms, r_obs=ro, luma=lu, bs=BS)
            n += 1
            if n % 20 == 0: print(f'  [{sname}] {n}', flush=True)
        print(f'[{sname}] done {n} -> {outd}', flush=True)


if __name__ == '__main__':
    main()
