#!/usr/bin/env python3
"""E5：免伤门控（启发式版，零训练）——用 RDNet 自己的 R̂ 层做门控信号。

动机（E3/E4）：干净区 RDNet 主动伤害（−2.26 dB、72% 为负），但强反射区有收益；
oracle headroom +2.02 dB。故需要一个**只看输入**的门控：|R̂| 局部能量小 → 保持输入；大 → 用 RDNet。

门控：w = clip((s − lo)/(hi − lo), 0, 1)，s = 块内 mean|R̂|（或逐像素平滑版）
      输出 = (1−w)·I + w·RDNet
判据（预登记）：全帧 ΔPSNR ≥ 0 **且** LPIPS 不劣（相对输入）。
扫 (lo, hi) 网格，报每个配置的 PSNR/LPIPS 与"该动/不该动"的通过率。
"""
import csv, os
import cv2
import numpy as np
import torch
import pyiqa

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet')}
GRID = [(0, 0), (2, 8), (4, 12), (6, 16), (10, 24), (15, 35), (25, 50)]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def paths(tag, sc, stem):
    base = SETS[tag][2]
    d = f'{base}/{sc}' if tag == 'jun' else f'{base}/{sc}__{stem}'
    return f'{d}/ytmt_ucs_sirs_l.png', f'{d}/ytmt_ucs_sirs_r.png'


def to_t(bgr):
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def main():
    dev = torch.device('cuda:0')
    lpips = pyiqa.create_metric('lpips', device=dev)
    items = []
    for tag, (d, gtn, _) in SETS.items():
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            GT = cv2.imread(f'{d}/{sc}/{gtn}')
            if GT is None:
                continue
            ins = sorted(f for f in os.listdir(f'{d}/{sc}') if f.startswith('I')) or ['I.png']
            for fn in ins:
                I = cv2.imread(f'{d}/{sc}/{fn}')
                pl, pr = paths(tag, sc, fn[:-4])
                if I is None or not os.path.exists(pl) or not os.path.exists(pr):
                    continue
                R = cv2.resize(cv2.imread(pl), (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_LINEAR)
                Rr = cv2.resize(cv2.imread(pr), (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_LINEAR)
                s = np.abs(Rr.astype(np.float32)).mean(axis=2)
                s = cv2.GaussianBlur(s, (0, 0), 8)
                items.append((tag, sc, fn[:-4], I, R, GT, s))

    print(f'n={len(items)} 张图；门控 = (1−w)·输入 + w·RDNet，w = clip((s−lo)/(hi−lo),0,1)')
    print(f'{"lo,hi":>10}{"ΔPSNR均值":>11}{"ΔPSNR中位":>11}{"PSNR通过":>9}{"LPIPS 中位":>12}{"LPIPS 不劣":>11}{"w 均值":>8}')
    rows = []
    for lo, hi in GRID:
        dP, LP, wm = [], [], []
        with torch.no_grad():
            for tag, sc, stem, I, R, GT, s in items:
                w = (np.ones_like(s) if hi <= lo else np.clip((s - lo) / (hi - lo), 0, 1))[..., None]
                O = np.clip(I * (1 - w) + R * w, 0, 255).astype(np.uint8)
                dP.append(psnr(O, GT) - psnr(I, GT))
                LP.append(float(lpips(to_t(O).to(dev), to_t(GT).to(dev))) -
                          float(lpips(to_t(I).to(dev), to_t(GT).to(dev))))
                wm.append(float(w.mean()))
        dP, LP = np.array(dP), np.array(LP)
        rows.append((lo, hi, dP.mean(), np.median(dP), (dP >= 0).mean(), np.median(LP), (LP <= 0).mean(), np.mean(wm)))
        print(f'{f"{lo},{hi}":>10}{dP.mean():>+11.2f}{np.median(dP):>+11.2f}{100*(dP>=0).mean():>8.0f}%'
              f'{np.median(LP):>+12.4f}{100*(LP<=0).mean():>10.0f}%{np.mean(wm):>8.2f}')
    with open('eval/eval_e5_gate.csv', 'w', newline='') as fh:
        w_ = csv.writer(fh)
        w_.writerow(['lo', 'hi', 'dPSNR_mean', 'dPSNR_median', 'psnr_pass_frac', 'lpips_delta_median', 'lpips_pass_frac', 'w_mean'])
        w_.writerows(rows)


if __name__ == '__main__':
    main()
