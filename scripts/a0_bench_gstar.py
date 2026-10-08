#!/usr/bin/env python3
"""A0：公共 SIRR 基准上的免伤门控前提判决（零训练，2026-10-08）

判决问题：E3/E4 现象（干净区被伤 + 门控有 headroom）在标准基准上是否同样存在？
做法：官方 RDNet 推理 → 逐图存预测 → 图像级 PSNR/LPIPS + 逐 64x64 块算
      r_obs（E3 口径，增益对齐）、g*（把 T 投影到 I→P 线段）、ΔPSNR、oracle 上界。
用法：CUDA_VISIBLE_DEVICES=N python3 scripts/a0_bench_gstar.py --sets real20 --gpu 0
输出：eval/a0_blocks_<sets>.csv、eval/a0_img_<sets>.csv、runs/pred/bench_<set>_official/
"""
import argparse, csv, os, sys
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, 'RDNet'))
os.chdir(os.path.join(PROJ, 'RDNet'))
from finetune_rdnet_peft import build                      # noqa: E402
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
from eval_protocol import ls_scale, apply_gm               # noqa: E402
import pyiqa                                               # noqa: E402

B = '/home/zmy/mycode/datasets'
SETS = {
    'real20':      (f'{B}/real/real20/blended',                f'{B}/real/real20/transmission_layer'),
    'objects200':  (f'{B}/SIR2/SolidObjectDataset/solid200/blended', f'{B}/SIR2/SolidObjectDataset/solid200/transmission_layer'),
    'postcard199': (f'{B}/SIR2/postcard/postcard199/blended',  f'{B}/SIR2/postcard/postcard199/transmission_layer'),
    'wild55':      (f'{B}/SIR2/Wildscene/wild55/blended',      f'{B}/SIR2/Wildscene/wild55/transmission_layer'),
    'nature20':    (f'{B}/nature/nature20/blended',            f'{B}/nature/nature20/transmission'),
}
BS = 64
ROBS_EDGES = [0, .05, .1, .2, .3, .5, .8, 1.2, 9]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def pad32(x):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REFLECT), h, w


def infer(net, net_c, dev, I):
    Ip, h, w = pad32(I)
    x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        code = net_c(x)
        _, outs = net(x, code, prompt=True)
        T_hat = outs[-1][:, 3:]
    out = T_hat[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1][:h, :w]
    return (out * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--save-pred', type=int, default=1)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.gpu}')
    net, net_c = build(save_memory=True, mona='none', grad_ckpt=False)
    net.to(dev).eval(); net_c.to(dev)
    lp = pyiqa.create_metric('lpips', device=dev)

    brows, irows = [], []
    for sname in args.sets.split(','):
        idir, tdir = SETS[sname]
        outd = os.path.join(PROJ, f'runs/pred/bench_{sname}_official')
        os.makedirs(outd, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(idir)):
            stem = os.path.splitext(fn)[0]
            tf = next((f'{stem}{e}' for e in ('.png', '.jpg', '.jpeg')
                       if os.path.exists(os.path.join(tdir, f'{stem}{e}'))), None)
            if tf is None:
                continue
            I = cv2.imread(os.path.join(idir, fn)); T = cv2.imread(os.path.join(tdir, tf))
            if I is None or T is None or I.shape != T.shape:
                continue
            if max(I.shape[:2]) > 1024:
                sc = 1024 / max(I.shape[:2]); sz = (int(I.shape[1]*sc), int(I.shape[0]*sc))
                I = cv2.resize(I, sz, interpolation=cv2.INTER_AREA)
                T = cv2.resize(T, sz, interpolation=cv2.INTER_AREA)
            P = infer(net, net_c, dev, I)
            if args.save_pred:
                cv2.imwrite(os.path.join(outd, f'{stem}.png'), P)
            pI, pP = psnr(I, T), psnr(P, T)
            with torch.no_grad():
                def lpv(X):
                    t = torch.from_numpy(X[:, :, ::-1].astype(np.float32)/255.).permute(2, 0, 1)[None].to(dev)
                    u = torch.from_numpy(T[:, :, ::-1].astype(np.float32)/255.).permute(2, 0, 1)[None].to(dev)
                    return float(lp(t, u))
                lI, lP = lpv(I), lpv(P)
            irows.append(dict(set=sname, case=stem, psnr_I=round(pI, 2), psnr_P=round(pP, 2),
                              dpsnr=round(pP - pI, 2), lpips_I=round(lI, 4), lpips_P=round(lP, 4),
                              lpips_worse=int(lP > lI), psnr_worse=int(pP < pI)))
            g = float(ls_scale(I, T)); Ig = apply_gm(I, g)
            h, w = T.shape[:2]
            for y in range(0, h - BS + 1, BS):
                for x in range(0, w - BS + 1, BS):
                    tb = T[y:y+BS, x:x+BS].astype(np.float64)
                    ibr = I[y:y+BS, x:x+BS].astype(np.float64)
                    ib = Ig[y:y+BS, x:x+BS].astype(np.float64)
                    pb = P[y:y+BS, x:x+BS].astype(np.float64)
                    a = (ib - tb).ravel(); bv = (pb - ib).ravel()
                    bb = float((bv * bv).sum())
                    a2 = (ibr - tb).ravel(); b2 = (pb - ibr).ravel()
                    bb2 = float((b2 * b2).sum())
                    gs = float(-(a * bv).sum() / bb) if bb > 1e-6 else np.nan
                    gs_raw = float(-(a2 * b2).sum() / bb2) if bb2 > 1e-6 else np.nan
                    gsc = min(1.0, max(0.0, gs)) if not np.isnan(gs) else np.nan
                    d_i = psnr(ibr, tb); d_p = psnr(pb, tb)
                    if not np.isnan(gsc):
                        Ig_b = ibr + gsc * (pb - ibr)
                        d_c = psnr(Ig_b, tb)
                    else:
                        d_c = d_i
                    brows.append(dict(set=sname, case=stem, by=y, bx=x,
                                      luma_gt=round(float(tb.mean()), 1),
                                      r_obs=round(float(np.abs(ib - tb).max(axis=2).mean() / (tb.mean() + 1e-9)), 4),
                                      gstar=round(gs, 3) if not np.isnan(gs) else '',
                                      gstar_raw=round(gs_raw, 3) if not np.isnan(gs_raw) else '',
                                      gstar_clip=round(gsc, 3) if not np.isnan(gsc) else '',
                                      dpsnr=round(d_p - d_i, 2),
                                      gain_bin=round(max(d_p, d_i) - d_i, 2),
                                      gain_cont=round(d_c - d_i, 2),
                                      head_gate=round(d_c - d_p, 3)))
            n += 1
            if n % 25 == 0:
                print(f'  [{sname}] {n} 张 ...', flush=True)
        print(f'[{sname}] done n={n}', flush=True)

    tag = args.sets.replace(',', '_')
    with open(os.path.join(PROJ, f'eval/a0_img_{tag}.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(irows[0].keys())); wr.writeheader(); wr.writerows(irows)
    with open(os.path.join(PROJ, f'eval/a0_blocks_{tag}.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(brows[0].keys())); wr.writeheader(); wr.writerows(brows)

    print(f'\n===== A0 汇总（{tag}）=====')
    for sname in args.sets.split(','):
        ir = [r for r in irows if r['set'] == sname]
        br = [r for r in brows if r['set'] == sname]
        if not ir:
            continue
        print(f'\n--- {sname}（{len(ir)} 图 / {len(br)} 块）---')
        print(f'  ΔPSNR 均值 {np.mean([r["dpsnr"] for r in ir]):+.2f} dB | '
              f'PSNR 变差图占比 {100*np.mean([r["psnr_worse"] for r in ir]):.0f}% | '
              f'LPIPS 变差图占比 {100*np.mean([r["lpips_worse"] for r in ir]):.0f}%')
        print(f'  二值门控 headroom（图级，相对 P）{np.mean([max(0.0, r["psnr_I"] - r["psnr_P"]) for r in ir]):+.2f} dB')
        print(f'  连续门控 headroom（块级，相对 P）{np.mean([r["head_gate"] for r in br]):+.2f} dB | '
              f'g*≤0 块占比 {100*np.mean([1 if (r["gstar_clip"] != "" and float(r["gstar_clip"]) <= 0) else 0 for r in br]):.0f}%')
        print(f'  {"r_obs 档":<14}{"块数":>7}{"g*中位":>9}{"ΔPSNR中位":>11}{"headroom(vs P)":>15}')
        for lo, hi in zip(ROBS_EDGES[:-1], ROBS_EDGES[1:]):
            sub = [r for r in br if lo <= r['r_obs'] < hi and r['gstar_clip'] != '']
            if len(sub) < 8:
                continue
            print(f'  [{lo:>5g},{hi:>5g}){len(sub):>7}'
                  f'{np.median([float(r["gstar_clip"]) for r in sub]):>9.2f}'
                  f'{np.median([r["dpsnr"] for r in sub]):>+11.2f}'
                  f'{np.median([r["head_gate"] for r in sub]):>+15.3f}')


if __name__ == '__main__':
    main()
