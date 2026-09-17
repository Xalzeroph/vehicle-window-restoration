#!/usr/bin/env python3
"""从 OpenRR-5k 配对提取真实反射层 R = I − T（有符号，保留"反射压暗"的剪影部分），
统计强度/形态分布，并与现有 R 池、6 月实拍 R_eff 对照。

背景：现有池 `data/pools/r_train_expand_0918`（287 张，RRW 提取）缺"深色剪影"反射；
OpenRR-5k（NTIRE 2026 真实拍摄配对，CC-BY-NC-SA-4.0）配对逐像素对齐（实测位移 <=0.05px），
可直接相减提 R。用法: python3 scripts/extract_openrr_r.py [--n 5000] [--out data/pools/r_openrr_0917]
"""
import argparse, csv, glob, os
import cv2
import numpy as np

SRC = 'data/external/OpenRR-5k/unpacked/trian_5k'


def stats_r(R):
    """R: float32 HxWx3，有符号（I - T）。"""
    Rn = R.mean(axis=2)
    neg = Rn < -8
    pos = Rn > 8
    n_neg = int(neg.sum())
    n_pos = int(pos.sum())
    largest = 0.0
    if n_neg > 50:
        n, lab, st, _ = cv2.connectedComponentsWithStats(neg.astype(np.uint8), 8)
        if n > 1:
            largest = float(st[1:, cv2.CC_STAT_AREA].max()) / Rn.size
    tot = float(np.abs(Rn).sum()) + 1e-9
    return dict(abs_mean=float(np.abs(Rn).mean()), std=float(Rn.std()),
                neg_frac=float(n_neg / Rn.size), pos_frac=float(n_pos / Rn.size),
                neg_energy=float(np.abs(Rn[neg]).sum() / tot) if n_neg else 0.0,
                largest_dark_blob=largest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--out', default='data/pools/r_openrr_0917')
    ap.add_argument('--save', action='store_true', help='把 R 层存到 --out（uint8, +128 偏移）')
    ap.add_argument('--stats-out', default='data/external/OpenRR-5k/analysis/r_stats.csv')
    args = ap.parse_args()

    B = sorted(glob.glob(os.path.join(SRC, 'blended', '*.jpg')))[:args.n]
    T = sorted(glob.glob(os.path.join(SRC, 'transmission_layer', '*.jpg')))[:args.n]
    assert len(B) == len(T) and B
    if args.save:
        os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    rows = []
    for bp, tp in zip(B, T):
        I = cv2.imread(bp).astype(np.float32)
        G = cv2.imread(tp).astype(np.float32)
        if I.shape != G.shape:
            G = cv2.resize(G, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
        R = I - G
        st = stats_r(R)
        st['name'] = os.path.basename(bp)
        rows.append(st)
        if args.save:
            np.save(os.path.join(args.out, os.path.splitext(st['name'])[0] + '.npy'), R.astype(np.float16))
    with open(args.stats_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    a = {k: np.array([r[k] for r in rows]) for k in ('abs_mean', 'std', 'neg_frac', 'neg_energy', 'largest_dark_blob')}
    print(f'=== OpenRR-5k {len(rows)} 对，R = I − T（有符号）===')
    for k, v in a.items():
        print(f'  {k:<18} 中位 {np.median(v):7.3f}  p10 {np.percentile(v,10):7.3f}  p90 {np.percentile(v,90):7.3f}  max {v.max():7.3f}')
    print('  （以上均为 0-255 尺度）')
    print('  （对照）6 月实拍 R_eff: |R|均值中位 16.4   std 中位 21.9；'
          '现有训练池 r_train_expand_0918: 21.9 / 13.9')
    sil = (a['largest_dark_blob'] > 0.10) & (a['neg_energy'] > 0.25)
    print(f'  剪影特征图（最大暗块>10% 且 负能量>25%）: {int(sil.sum())}/{len(rows)} = {sil.mean()*100:.1f}%')
    print(f'  明细 {args.stats_out}')


if __name__ == '__main__':
    main()
