#!/usr/bin/env python3
"""实拍配对集评测（2026-09-17 轮）：RDNet zero-shot + 四基线，逐场景 PSNR/SSIM。

臂：
  identity       输出=输入
  const          输出=输入均值常数图
  identity_gm    输出=最优全局增益后的输入（"什么都不做但允许调曝光"下界）
  blur_identity  输出=与 RDNet 同梯度能量损失的高斯模糊输入（防"变糊换降噪被 PSNR 奖励"）
  rdnet          RDNet zero-shot（l 层），resize 回 GT 尺寸
  rdnet_gm       同上 + 最优全局增益（GT 与输入存在曝光差，需分离"亮度错"与"内容错"）
G-3：场景级配对检验（Wilcoxon signed-rank）＋逐场景表
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
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(cv2.magnitude(sx, sy).mean())


def opt_gain(pred, ref):
    p = pred.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    den = float((p * p).sum())
    return float((p * r).sum() / den) if den > 1e-9 else 1.0


def matched_blur(img, target_ge):
    """二分找 sigma 使模糊后梯度能量 ≈ target_ge。"""
    lo, hi = 0.0, 8.0
    for _ in range(18):
        mid = (lo + hi) / 2
        b = cv2.GaussianBlur(img, (0, 0), mid)
        if grad_energy(b) > target_ge:
            lo = mid
        else:
            hi = mid
    return cv2.GaussianBlur(img, (0, 0), (lo + hi) / 2), (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='data/test_sets/real_pair_0917')
    ap.add_argument('--pred', default='runs/pred/real_pair_0917_rdnet')
    ap.add_argument('--pred-file', default='ytmt_ucs_sirs_l.png')
    ap.add_argument('--out', default='eval/eval_real_pair_rdnet.csv')
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.set) if d.startswith('VID_'))
    rows = []
    for sc in scenes:
        T = cv2.imread(os.path.join(args.set, sc, 'T.png'))
        I = cv2.imread(os.path.join(args.set, sc, 'I.png'))
        pp = os.path.join(args.pred, sc, args.pred_file)
        if not (os.path.exists(pp) and T is not None and I is not None):
            print(f'{sc}: 缺文件 -> 跳过'); continue
        R = cv2.resize(cv2.imread(pp), (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)

        arms = {}
        arms['identity'] = I
        arms['const'] = np.full_like(I, int(round(I.mean())))
        gi = opt_gain(I, T)
        arms['identity_gm'] = np.clip(I * gi, 0, 255).astype(np.uint8)
        ge_ratio = grad_energy(R) / max(grad_energy(I), 1e-6)
        arms['blur_identity'], sig = matched_blur(I, grad_energy(I) * ge_ratio)
        arms['rdnet'] = R
        arms['rdnet_gm'] = np.clip(R * opt_gain(R, T), 0, 255).astype(np.uint8)

        rec = dict(scene=sc, blur_sigma=round(sig, 2), gain_input=round(gi, 3),
                   gain_rdnet=round(opt_gain(R, T), 3), ge_ratio_rdnet=round(ge_ratio, 3))
        for name, im in arms.items():
            rec[f'{name}_psnr'] = round(psnr(im, T), 2)
            rec[f'{name}_ssim'] = round(float(ssim_fn(im, T, channel_axis=2, data_range=255)), 4)
        rows.append(rec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = list(rows[0].keys())
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    arms = ['identity', 'const', 'identity_gm', 'blur_identity', 'rdnet', 'rdnet_gm']
    summ = {}
    print(f'\n{"臂":<16}{"PSNR中位":>10}{"PSNR均值":>10}{"SSIM中位":>10}{"SSIM均值":>10}')
    for a in arms:
        p = np.array([r[f'{a}_psnr'] for r in rows]); s = np.array([r[f'{a}_ssim'] for r in rows])
        summ[a] = dict(psnr_median=float(np.median(p)), psnr_mean=float(p.mean()),
                       ssim_median=float(np.median(s)), ssim_mean=float(s.mean()))
        print(f'{a:<16}{np.median(p):>10.2f}{p.mean():>10.2f}{np.median(s):>10.4f}{s.mean():>10.4f}')

    print('\n--- 场景级配对检验（Wilcoxon signed-rank, n=%d）---' % len(rows))
    tests = {}
    for a, b in [('rdnet', 'identity'), ('rdnet_gm', 'identity_gm'), ('rdnet_gm', 'blur_identity'),
                 ('identity_gm', 'blur_identity'), ('rdnet', 'const')]:
        pa = np.array([r[f'{a}_psnr'] for r in rows]); pb = np.array([r[f'{b}_psnr'] for r in rows])
        try:
            st, pv = wilcoxon(pa, pb)
        except ValueError:
            st, pv = float('nan'), float('nan')
        d = pa - pb
        tests[f'{a}_vs_{b}'] = dict(median_delta=float(np.median(d)), p=float(pv))
        print(f'  {a:<12} vs {b:<14} 中位差 {np.median(d):+6.2f} dB   p={pv:.4f}   '
              f'{"(n.s.)" if pv > 0.05 else ""}')
    summ['tests'] = tests
    summ['n_scenes'] = len(rows)
    json.dump(summ, open(os.path.splitext(args.out)[0] + '_summary.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n写出 {args.out} 与 summary.json')


if __name__ == '__main__':
    main()
