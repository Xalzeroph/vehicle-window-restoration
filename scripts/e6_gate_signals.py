#!/usr/bin/env python3
"""E6：单帧门控信号对比（2026-09-17 单帧让步前的最后一搏）。

问题：E5 用 RDNet 自己的 |R̂| 当门控信号，PSNR 达标率仅 63%。换用"只看输入"的信号能否达标？
信号：s0=|R̂|(基线) / s1=局部梯度能量 / s2=局部亮度 / s3=局部对比度 / s4=s1+s3 组合
门控：w = clip((s−lo)/(hi−lo),0,1)，输出 (1−w)·I + w·RDNet（s 逐像素高斯平滑）
判据（预登记）：存在信号使 PSNR 达标率 >= 80% 且 LPIPS 不劣率 >= 70%。
"""
import csv, os
import cv2
import numpy as np
import torch
import pyiqa

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet')}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def paths(tag, sc, stem):
    base = SETS[tag][2]
    d = f'{base}/{sc}' if tag == 'jun' else f'{base}/{sc}__{stem}'
    return f'{d}/ytmt_ucs_sirs_l.png', f'{d}/ytmt_ucs_sirs_r.png'


def to_t(b):
    return torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)


def signals(I, Rr):
    g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ge = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mu = cv2.GaussianBlur(g, (0, 0), 8)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 8) - mu ** 2, 0))
    r = np.abs(Rr.astype(np.float32)).mean(axis=2)
    out = {'s0_|R|': r, 's1_grad': ge, 's2_bright': mu, 's3_std': sd,
           's4_grad+std': ge / (ge.mean() + 1e-6) + sd / (sd.mean() + 1e-6)}
    return {k: cv2.GaussianBlur(v, (0, 0), 8) for k, v in out.items()}


def main():
    dev = torch.device('cuda:0')
    lpips = pyiqa.create_metric('lpips', device=dev)
    items = []
    for tag, (d, gtn, _) in SETS.items():
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            GT = cv2.imread(f'{d}/{sc}/{gtn}')
            if GT is None:
                continue
            for fn in (sorted(f for f in os.listdir(f'{d}/{sc}') if f.startswith('I')) or ['I.png']):
                I = cv2.imread(f'{d}/{sc}/{fn}')
                pl, pr = paths(tag, sc, fn[:-4])
                if I is None or not os.path.exists(pl) or not os.path.exists(pr):
                    continue
                wh = (GT.shape[1], GT.shape[0])
                R = cv2.resize(cv2.imread(pl), wh, interpolation=cv2.INTER_LINEAR)
                Rr = cv2.resize(cv2.imread(pr), wh, interpolation=cv2.INTER_LINEAR)
                items.append((I, R, GT, signals(I, Rr)))

    print(f'n={len(items)}；判据：PSNR 达标 >=80% 且 LPIPS 不劣 >=70%')
    print(f'{"信号":<14}{"分位":>8}{"ΔPSNR均值":>11}{"PSNR达标":>10}{"LPIPS中位差":>12}{"LPIPS不劣":>10}')
    rows = []
    with torch.no_grad():
        base_lp = [float(lpips(to_t(I).to(dev), to_t(GT).to(dev))) for I, R, GT, s in items]
        for name in ['s0_|R|', 's1_grad', 's2_bright', 's3_std', 's4_grad+std']:
            for q in [0.5, 0.7, 0.9]:
                los = np.array([np.percentile(s[name], q * 100) for _, _, _, s in items])
                his = np.array([np.percentile(s[name], min(99, q * 100 + 20)) for _, _, _, s in items])
                dP, LP = [], []
                for k, (I, R, GT, sg) in enumerate(items):
                    sm = sg[name]
                    w = (np.ones_like(sm) if his[k] <= los[k] else
                         np.clip((sm - los[k]) / (his[k] - los[k]), 0, 1))[..., None]
                    O = np.clip(I * (1 - w) + R * w, 0, 255).astype(np.uint8)
                    dP.append(psnr(O, GT) - psnr(I, GT))
                    LP.append(float(lpips(to_t(O).to(dev), to_t(GT).to(dev))) - base_lp[k])
                dP, LP = np.array(dP), np.array(LP)
                rows.append((name, q, dP.mean(), (dP >= 0).mean(), np.median(LP), (LP <= 0).mean()))
                flag = '  <== 达标' if (dP >= 0).mean() >= 0.8 and (LP <= 0).mean() >= 0.7 else ''
                print(f'{name:<14}{q:>8}{dP.mean():>+11.2f}{100*(dP>=0).mean():>9.0f}%{np.median(LP):>+12.4f}'
                      f'{100*(LP<=0).mean():>9.0f}%{flag}')
    with open('eval/eval_e6_gate_signals.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['signal', 'quantile', 'dPSNR_mean', 'psnr_pass', 'lpips_delta_median', 'lpips_pass'])
        w.writerows(rows)


if __name__ == '__main__':
    main()
