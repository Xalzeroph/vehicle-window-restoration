#!/usr/bin/env python3
"""E3：按局部 r_obs 分箱，量「R 主导」的剂量-反应（2026-09-17 定位轮）。

问题：反射越强（局部 r_obs 越高），(a) 输入相对 GT 的损伤有多大？(b) RDNet 的收益如何变化？
预期（命题为真）：r_obs 升高 → 损伤升高、RDNet 增益下降并转负。
零训练；用现成三套配对集 + 已有 RDNet 预测。输出 eval/eval_e3_robs_bins.csv
"""
import csv, glob, os
import cv2
import numpy as np

SETS = [('jun', 'data/test_sets/real_pair_0917', 'T.png', 'I.png', 'runs/pred/real_pair_0917_rdnet'),
        ('apr', 'data/test_sets/real_pair_apr_0917', 'GT.png', None, 'runs/pred/apr_rdnet'),
        ('may', 'data/test_sets/real_pair_may_0917', 'GT.png', None, 'runs/pred/may_rdnet')]
BS = 64


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def pred_path(tag, base_pred, sc, stem):
    return os.path.join(base_pred, sc, 'ytmt_ucs_sirs_l.png') if tag == 'jun' else \
        os.path.join(base_pred, f'{sc}__{stem}', 'ytmt_ucs_sirs_l.png')


rows = []
for tag, dset, gtn, iname, base_pred in SETS:
    for sc in sorted(x for x in os.listdir(dset) if x.startswith('VID_')):
        GT = cv2.imread(f'{dset}/{sc}/{gtn}')
        if GT is None:
            continue
        ins = [iname] if iname else sorted(f for f in os.listdir(f'{dset}/{sc}') if f.startswith('I_'))
        for fn in ins:
            I = cv2.imread(f'{dset}/{sc}/{fn}')
            pp = pred_path(tag, base_pred, sc, fn[:-4])
            if I is None or not os.path.exists(pp):
                continue
            R = cv2.resize(cv2.imread(pp), (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_LINEAR)
            i_, t_ = I.astype(np.float64), GT.astype(np.float64)
            g = float((i_ * t_).sum() / ((i_ * i_).sum() + 1e-9))
            D = np.abs(g * i_ - t_).max(axis=2)
            h, w = GT.shape[:2]
            for y in range(0, h - BS + 1, BS):
                for x in range(0, w - BS + 1, BS):
                    tb = t_[y:y + BS, x:x + BS]
                    if tb.mean() < 25:
                        continue
                    r_obs = float(D[y:y + BS, x:x + BS].mean() / (tb.mean() + 1e-9))
                    gt_b = GT[y:y + BS, x:x + BS]
                    dm = psnr(I[y:y + BS, x:x + BS], gt_b)
                    rd = psnr(R[y:y + BS, x:x + BS], gt_b)
                    rows.append(dict(set=tag, scene=sc, stem=fn[:-4], by=y, bx=x,
                                     r_obs=round(r_obs, 4), psnr_input=round(dm, 2),
                                     rdnet_gain=round(rd - dm, 2)))

os.makedirs('eval', exist_ok=True)
out = 'eval/eval_e3_robs_bins.csv'
with open(out, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

r = np.array([x['r_obs'] for x in rows])
pi = np.array([x['psnr_input'] for x in rows])
rg = np.array([x['rdnet_gain'] for x in rows])
edges = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 9]
print(f'总块数 {len(rows)}（三套合并）')
print(f'{"r_obs 区间":<14}{"块数":>7}{"输入PSNR中位":>13}{"RDNet增益中位":>14}{"增益<0 占比":>12}')
for a, b in zip(edges[:-1], edges[1:]):
    m = (r >= a) & (r < b)
    if m.sum() < 10:
        continue
    print(f'[{a:.2f},{b:.2f}){"":<4}{int(m.sum()):>7}{np.median(pi[m]):>13.2f}{np.median(rg[m]):>+14.2f}{100*(rg[m]<0).mean():>11.0f}%')
print(f'\n相关系数 r_obs vs RDNet增益: {np.corrcoef(r, rg)[0,1]:+.3f}')
print(f'写出 {out}')
