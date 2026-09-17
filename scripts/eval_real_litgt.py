#!/usr/bin/env python3
"""实拍配对集 v2 评测：GT = 无反射带补光帧（明亮理想输出），输入 = 低光反射 / 正常光带反射 / 低光无反射。
RDNet zero-shot + 四基线（identity / const / identity_gm / blur_identity），按输入类别分组报。
"""
import argparse, csv, json, os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from scipy.stats import wilcoxon


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def grad_energy(x):
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())


def opt_gain(pred, ref):
    p = pred.astype(np.float64).ravel(); r = ref.astype(np.float64).ravel()
    den = float((p * p).sum())
    return float((p * r).sum() / den) if den > 1e-9 else 1.0


def matched_blur(img, target_ge):
    lo, hi = 0.0, 8.0
    for _ in range(16):
        mid = (lo + hi) / 2
        if grad_energy(cv2.GaussianBlur(img, (0, 0), mid)) > target_ge:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return cv2.GaussianBlur(img, (0, 0), s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='data/test_sets/real_pair_litgt_0917')
    ap.add_argument('--pred', default='runs/pred/litgt_rdnet')
    ap.add_argument('--out', default='eval/eval_real_litgt_rdnet.csv')
    args = ap.parse_args()

    rows = []
    for sc in sorted(d for d in os.listdir(args.set) if d.startswith('VID_')):
        sd = os.path.join(args.set, sc)
        GT = cv2.imread(os.path.join(sd, 'GT.png'))
        for fn in sorted(f for f in os.listdir(sd) if f.startswith('I_')):
            cat = fn[:-4][2:]
            I = cv2.imread(os.path.join(sd, fn))
            po = os.path.join(args.pred, f'{sc}__{fn[:-4]}', 'ytmt_ucs_sirs_l.png')
            if I is None or GT is None or not os.path.exists(po):
                print(f'{sc}/{cat}: 缺文件 -> 跳过'); continue
            R = cv2.resize(cv2.imread(po), (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_LINEAR)
            arms = {'identity': I,
                    'const': np.full_like(I, int(round(I.mean()))),
                    'identity_gm': np.clip(I * opt_gain(I, GT), 0, 255).astype(np.uint8),
                    'blur_identity': matched_blur(I, grad_energy(I) * grad_energy(R) / max(grad_energy(I), 1e-6)),
                    'rdnet': R,
                    'rdnet_gm': np.clip(R * opt_gain(R, GT), 0, 255).astype(np.uint8)}
            rec = dict(scene=sc, cat=cat, gain_in=round(opt_gain(I, GT), 3), gain_rdnet=round(opt_gain(R, GT), 3),
                       ge_ratio=round(grad_energy(R) / max(grad_energy(I), 1e-6), 3))
            for a, im in arms.items():
                rec[f'{a}_psnr'] = round(psnr(im, GT), 2)
                rec[f'{a}_ssim'] = round(float(ssim_fn(im, GT, channel_axis=2, data_range=255)), 4)
            rows.append(rec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    arms = ['identity', 'const', 'identity_gm', 'blur_identity', 'rdnet', 'rdnet_gm']
    summ = {}
    for cat in ['ALL'] + sorted(set(r['cat'] for r in rows)):
        sub = rows if cat == 'ALL' else [r for r in rows if r['cat'] == cat]
        if not sub:
            continue
        print(f'\n=== 输入类别: {cat}  (n={len(sub)}) ===')
        print(f'{"臂":<16}{"PSNR中位":>10}{"PSNR均值":>10}{"SSIM中位":>10}')
        summ[cat] = {}
        for a in arms:
            p = np.array([r[f'{a}_psnr'] for r in sub]); s = np.array([r[f'{a}_ssim'] for r in sub])
            summ[cat][a] = dict(psnr_median=float(np.median(p)), psnr_mean=float(p.mean()), ssim_median=float(np.median(s)))
            print(f'{a:<16}{np.median(p):>10.2f}{p.mean():>10.2f}{np.median(s):>10.4f}')
        tests = {}
        for x, y in [('rdnet', 'identity'), ('rdnet_gm', 'identity_gm'), ('rdnet_gm', 'blur_identity')]:
            pa = np.array([r[f'{x}_psnr'] for r in sub]); pb = np.array([r[f'{y}_psnr'] for r in sub])
            try:
                _, pv = wilcoxon(pa, pb)
            except ValueError:
                pv = float('nan')
            tests[f'{x}_vs_{y}'] = dict(median_delta=float(np.median(pa - pb)), p=float(pv))
            print(f'  {x:<12} vs {y:<14} 中位差 {np.median(pa - pb):+6.2f} dB  p={pv:.4f}')
        summ[cat]['tests'] = tests
    summ['n'] = len(rows)
    json.dump(summ, open(os.path.splitext(args.out)[0] + '_summary.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n写出 {args.out}')


if __name__ == '__main__':
    main()
