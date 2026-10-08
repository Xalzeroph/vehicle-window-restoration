#!/usr/bin/env python3
"""门控 v1 验收：按预登记判据评测（2026-10-08）
判据：may 上 no-harm 帧数 ≥9/11；apr 上块级 ΔPSNR（vs 纯 RDNet）≥ +1.0 dB
用法: CUDA_VISIBLE_DEVICES=2 python3 scripts/gate_v1_eval.py --ckpt runs/ckpt/gate_v1_apr/gate.pth --tag apr
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gate_v1_train import Gate                      # noqa: E402
import pyiqa                                        # noqa: E402
BS = 32


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--sets', default='may,apr'); ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}')
    net = Gate().to(dev); sd = torch.load(os.path.join(PROJ, a.ckpt), map_location='cpu')
    net.load_state_dict(sd['sd']); net.eval()
    lp = pyiqa.create_metric('lpips', device=dev)
    print(f"[eval] ckpt={a.ckpt} train={sd.get('train')}", flush=True)
    for sname in a.sets.split(','):
        rows = []
        for f in sorted(glob.glob(os.path.join(PROJ, 'data/gate_v1', sname, '*.npz'))):
            z = np.load(f)
            I, P, T = z['I'], z['P'], z['T']
            xI = torch.from_numpy(I.transpose(2, 0, 1)[None].astype(np.float32)/255.).to(dev)
            xP = torch.from_numpy(P.transpose(2, 0, 1)[None].astype(np.float32)/255.).to(dev)
            with torch.no_grad():
                _, g = net(xI, xP)
                O = (xI + g*(xP - xI)).clamp(0, 1)
            gnp = g[0, 0].cpu().numpy()
            Onp = (O[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
            ps = {'I': psnr(I, T), 'P': psnr(P, T), 'O': psnr(Onp, T)}
            lps = {}
            for k, X in (('I', I), ('P', P), ('O', Onp)):
                with torch.no_grad():
                    t1 = torch.from_numpy(X[:, :, ::-1].astype(np.float32)/255.).permute(2, 0, 1)[None].to(dev)
                    t2 = torch.from_numpy(T[:, :, ::-1].astype(np.float32)/255.).permute(2, 0, 1)[None].to(dev)
                    lps[k] = float(lp(t1, t2))
            h, w = T.shape[:2]; nh, nw = h//BS, w//BS
            dOP = []
            for by in range(nh):
                for bx in range(nw):
                    sl = np.s_[by*BS:(by+1)*BS, bx*BS:(bx+1)*BS]
                    dOP.append(psnr(Onp[sl], T[sl]) - psnr(P[sl], T[sl]))
            rows.append(dict(case=os.path.basename(f)[:-4], psnr_I=round(ps['I'], 2), psnr_P=round(ps['P'], 2),
                             psnr_O=round(ps['O'], 2), dpsnr_O_vs_P=round(ps['O'] - ps['P'], 2),
                             lpips_I=round(lps['I'], 4), lpips_P=round(lps['P'], 4), lpips_O=round(lps['O'], 4),
                             blk_dpsnr_O_vs_P=round(float(np.mean(dOP)), 3), g_mean=round(float(gnp.mean()), 3),
                             harm_O_vs_I=int(ps['O'] < ps['I']), lp_harm_O_vs_I=int(lps['O'] > lps['I']),
                             harm_O_vs_P=int(ps['O'] < ps['P'] - 0.01)))
        import csv
        with open(os.path.join(PROJ, f'eval/gate_v1_{sname}_{a.tag}.csv'), 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        n = len(rows)
        noharm = sum(1 for r in rows if r['harm_O_vs_I'] == 0 and r['lp_harm_O_vs_I'] == 0)
        noharm_I = sum(1 for r in rows if r['psnr_I'] >= r['psnr_I'])
        print(f'\n===== {sname}（n={n}）=====')
        print(f'  PSNR 均值   I {np.mean([r["psnr_I"] for r in rows]):.2f} | P {np.mean([r["psnr_P"] for r in rows]):.2f} | O {np.mean([r["psnr_O"] for r in rows]):.2f}')
        print(f'  LPIPS 均值  I {np.mean([r["lpips_I"] for r in rows]):.4f} | P {np.mean([r["lpips_P"] for r in rows]):.4f} | O {np.mean([r["lpips_O"] for r in rows]):.4f}')
        print(f'  O vs P：图级 ΔPSNR 均值 {np.mean([r["dpsnr_O_vs_P"] for r in rows]):+.2f} dB ；'
              f'块级 ΔPSNR 均值 {np.mean([r["blk_dpsnr_O_vs_P"] for r in rows]):+.2f} dB（预登记线 +1.00）')
        print(f'  O 相对 I 的 no-harm 帧数 {noharm}/{n}（预登记线 may ≥9/11；identity 基线 {n}/{n}）')
        print(f'  门控均值 {np.mean([r["g_mean"] for r in rows]):.3f} | O 比 P 差的帧数 {sum(r["harm_O_vs_P"] for r in rows)}/{n}')
        print(f'  [写出] eval/gate_v1_{sname}_{a.tag}.csv')


if __name__ == '__main__':
    main()
