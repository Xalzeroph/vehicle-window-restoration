#!/usr/bin/env python3
"""E7：两个决定性诊断（2026-09-17 新循环）。

A. oracle 门天花板：逐块用 GT 选最优 w（在 输入 与 RDNet输出 之间插值），
   算 PSNR/LPIPS 达标率的上限。判据：LPIPS 上限 <70% → 门控范式死。
B. 判据自洽性：逐块 (ΔPSNR, ΔLPIPS) 相关系数。判据：显著负相关 → 判据自相矛盾，须换干预。

块 128²（LPIPS 在小块上噪声大），三套实拍配对集。
"""
import csv, os
import cv2
import numpy as np
import torch
import pyiqa
from scipy.stats import pearsonr

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet')}
BS = 128
WS = [0.0, 0.25, 0.5, 0.75, 1.0]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def path(tag, sc, stem):
    base = SETS[tag][2]
    return f'{base}/{sc}/ytmt_ucs_sirs_l.png' if tag == 'jun' else f'{base}/{sc}__{stem}/ytmt_ucs_sirs_l.png'


def to_t(b):
    return torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)


def main():
    dev = torch.device('cuda:0')
    lp = pyiqa.create_metric('lpips', device=dev)
    rec = []
    with torch.no_grad():
        for tag, (d, gtn, _) in SETS.items():
            for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
                GT = cv2.imread(f'{d}/{sc}/{gtn}')
                if GT is None:
                    continue
                for fn in (sorted(f for f in os.listdir(f'{d}/{sc}') if f.startswith('I')) or ['I.png']):
                    I = cv2.imread(f'{d}/{sc}/{fn}')
                    pp = path(tag, sc, fn[:-4])
                    if I is None or not os.path.exists(pp):
                        continue
                    wh = (GT.shape[1], GT.shape[0])
                    R = cv2.resize(cv2.imread(pp), wh, interpolation=cv2.INTER_LINEAR)
                    h, w = GT.shape[:2]
                    for y in range(0, h - BS + 1, BS):
                        for x in range(0, w - BS + 1, BS):
                            gtb = GT[y:y + BS, x:x + BS]
                            if gtb.mean() < 15:
                                continue
                            ib, rb = I[y:y + BS, x:x + BS], R[y:y + BS, x:x + BS]
                            tg, ti, tr = to_t(gtb).to(dev), to_t(ib).to(dev), to_t(rb).to(dev)
                            li, lr = float(lp(ti, tg)), float(lp(tr, tg))
                            rows = []
                            for wv in WS:
                                o = np.clip(ib * (1 - wv) + rb * wv, 0, 255).astype(np.uint8)
                                rows.append((psnr(o, gtb), float(lp(to_t(o).to(dev), tg))))
                            p0 = psnr(ib, gtb)
                            dPs = [r[0] - p0 for r in rows]      # 各 w 的 ΔPSNR
                            dLs = [r[1] - li for r in rows]      # 各 w 的 ΔLPIPS
                            # 约束 oracle：在 ΔLPIPS<=0 的 w 里取最大 ΔPSNR
                            feas = [dPs[k] for k in range(len(WS)) if dLs[k] <= 0]
                            dP_con = max(feas) if feas else 0.0
                            rec.append(dict(set=tag, scene=sc, y=y, x=x,
                                            dP_rd=round(dPs[-1], 2), dL_rd=round(dLs[-1], 4),
                                            dP_con=round(dP_con, 2),
                                            n_feas=len(feas)))
    with open('eval/eval_e7_oracle_ceiling.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rec[0].keys()))
        w.writeheader(); w.writerows(rec)

    dPc = np.array([r['dP_con'] for r in rec])
    dPrd = np.array([r['dP_rd'] for r in rec]); dLrd = np.array([r['dL_rd'] for r in rec])
    print(f'块数 {len(rec)}（{BS}²，三套合并）')
    print('\n[A\'] 约束 oracle：逐块在 ΔLPIPS<=0 的 w 中取最大 ΔPSNR')
    print(f'   可获得增益：均值 {dPc.mean():+.2f} dB   中位 {np.median(dPc):+.2f} dB')
    print(f'   能做到 ΔPSNR>0 的块占比 {100*(dPc>0.05).mean():.0f}%')
    print(f'   （对照 纯RDNet w=1：均值 {dPrd.mean():+.2f} dB，ΔPSNR>=0 {100*(dPrd>=0).mean():.0f}%，ΔLPIPS<=0 {100*(dLrd<=0).mean():.0f}%）')
    print('\n[B] 判据自洽性（逐块 ΔPSNR vs ΔLPIPS，w=1）:')
    r, p = pearsonr(dPrd, dLrd)
    print(f'   pearson r = {r:+.3f}  (p={p:.1e})  → {"负相关：判据自相矛盾" if r < -0.1 else "无显著冲突"}')
    r2, _ = pearsonr(dP, dL)
    print(f'   （oracle 最优 w 下）r = {r2:+.3f}')


if __name__ == '__main__':
    main()
