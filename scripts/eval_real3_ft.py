#!/usr/bin/env python3
"""合成域微调 RDNet 在三套实拍配对集上的评测（2026-09-17 轮）。

对照臂：identity / identity_gm（只调曝光）/ const / 官方零样本 / 三个微调 ckpt
（rdt_plain、rdt_plain_s101、rdt_freeze）。判决门：6 月集上微调 vs 零样本 ΔPSNR ≥ +0.5dB 且 p<0.05。
用法: python3 scripts/eval_real3_ft.py
"""
import csv, json, os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from scipy.stats import wilcoxon

SETS = [('real_pair_0917', 'jun', 'T.png', 'I.png'), ('real_pair_apr_0917', 'apr', 'GT.png', None),
        ('real_pair_may_0917', 'may', 'GT.png', None)]
MODELS = {'zeroshot': None, 'ft_plain': 'plain', 'ft_plain_s101': 'plain_s101', 'ft_freeze': 'freeze'}
ZS = {'jun': 'runs/pred/real_pair_0917_rdnet', 'apr': 'runs/pred/apr_rdnet', 'may': 'runs/pred/may_rdnet'}
FTDIR = {'jun': 'real_pair_0917', 'apr': 'apr', 'may': 'may'}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def opt_gain(p, r):
    p = p.astype(np.float64).ravel(); r = r.astype(np.float64).ravel()
    d = float((p * p).sum())
    return float((p * r).sum() / d) if d > 1e-9 else 1.0


def pred_path(model, tag, scene, stem):
    if model == 'zeroshot':
        base = ZS[tag]
        return os.path.join(base, scene, 'ytmt_ucs_sirs_l.png') if tag == 'jun' else \
               os.path.join(base, f'{scene}__{stem}', 'ytmt_ucs_sirs_l.png')
    return os.path.join(f'runs/pred/ft_{model.split("ft_")[1]}_{FTDIR[tag]}', f'{scene}__{stem}', 'ytmt_ucs_sirs_l.png')


def main():
    rows = []
    for d, tag, gtname, iname in SETS:
        for sc in sorted(x for x in os.listdir(f'data/test_sets/{d}') if x.startswith('VID_')):
            GT = cv2.imread(f'data/test_sets/{d}/{sc}/{gtname}')
            if GT is None:
                continue
            ins = [iname] if iname else sorted(f for f in os.listdir(f'data/test_sets/{d}/{sc}') if f.startswith('I_'))
            for fn in ins:
                I = cv2.imread(f'data/test_sets/{d}/{sc}/{fn}')
                if I is None:
                    continue
                stem = fn[:-4]
                rec = dict(set=tag, scene=sc, stem=stem,
                           identity=round(psnr(I, GT), 2),
                           identity_gm=round(psnr(np.clip(I * opt_gain(I, GT), 0, 255).astype(np.uint8), GT), 2),
                           const=round(psnr(np.full_like(I, int(I.mean())), GT), 2))
                for m in MODELS:
                    p = pred_path(m, tag, sc, stem)
                    if not os.path.exists(p):
                        continue
                    R = cv2.resize(cv2.imread(p), (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_LINEAR)
                    rec[m] = round(psnr(R, GT), 2)
                    rec[m + '_gm'] = round(psnr(np.clip(R * opt_gain(R, GT), 0, 255).astype(np.uint8), GT), 2)
                    rec[m + '_ssim'] = round(float(ssim_fn(R, GT, channel_axis=2, data_range=255)), 4)
                rows.append(rec)
    os.makedirs('eval', exist_ok=True)
    out = 'eval/eval_real3_ft.csv'
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('set', 'scene', 'stem'), k))
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summ = {}
    arms = ['identity', 'identity_gm', 'const', 'zeroshot', 'zeroshot_gm', 'ft_plain', 'ft_plain_gm',
            'ft_plain_s101', 'ft_plain_s101_gm', 'ft_freeze', 'ft_freeze_gm']
    for tag in ['jun', 'apr', 'may']:
        sub = [r for r in rows if r['set'] == tag]
        if not sub:
            continue
        print(f'\n=== {tag} (n={len(sub)}) PSNR 中位 / SSIM 中位 ===')
        summ[tag] = {}
        for a in arms:
            v = [r[a] for r in sub if a in r and r[a] is not None]
            if not v:
                continue
            s = [r[a + '_ssim'] for r in sub if a + '_ssim' in r]
            summ[tag][a] = dict(psnr_median=float(np.median(v)), n=len(v),
                                ssim_median=(float(np.median(s)) if s else None))
            print(f'  {a:<18}{np.median(v):>8.2f}' + (f'{np.median(s):>10.4f}' if s else ''))
        for a in ['ft_plain_gm', 'ft_plain_s101_gm', 'ft_freeze_gm']:
            x = [r[a] for r in sub if a in r]; y = [r['zeroshot_gm'] for r in sub if a in r]
            if len(x) > 5:
                try:
                    _, pv = wilcoxon(x, y)
                except ValueError:
                    pv = float('nan')
                summ[tag][f'{a}_vs_zeroshot_gm'] = dict(delta=float(np.median(np.array(x) - np.array(y))), p=float(pv))
                print(f'  {a} vs zeroshot_gm: 中位差 {np.median(np.array(x) - np.array(y)):+.2f} dB  p={pv:.4f}')
    json.dump(summ, open('eval/eval_real3_ft_summary.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n写出 {out}（{len(rows)} 行）')


if __name__ == '__main__':
    main()
