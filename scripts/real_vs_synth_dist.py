#!/usr/bin/env python3
"""实拍配对集 vs 合成测试集的**分布对比**（2026-09-17）。

口径：
  T 域统计：短边缩到 224 再中心裁 224²（沿用 scripts/domain_stats.py 约定）；
  r_obs：mean(|I − g·T|) / mean(T)，g 为 T→I 的最优全局增益（光照差归一到增益里，剩下的即反射）。
用法: python3 scripts/real_vs_synth_dist.py
"""
import glob, os
import cv2
import numpy as np


def load224(p):
    im = cv2.imread(p)
    h, w = im.shape[:2]
    sc = 224.0 / min(h, w)
    im = cv2.resize(im, (max(224, int(round(w * sc))), max(224, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    y0, x0 = (h - 224) // 2, (w - 224) // 2
    return im[y0:y0 + 224, x0:x0 + 224]


def t_stats(p):
    im = load224(p).astype(np.float32)
    g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    # 噪声：中值滤波残差的 MAD（稳健）
    noise = float(np.median(np.abs(g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32))))
    sat = float((im.max(axis=2) - im.min(axis=2)).mean())
    return dict(mean=float(g.mean()), p5=float(np.percentile(g, 5)), p95=float(np.percentile(g, 95)),
                contrast=float(g.std()), lap=float(np.abs(lap).mean()), noise=noise, sat=sat)


def r_obs(ip, tp, full=False):
    I = cv2.imread(ip); T = cv2.imread(tp)
    if I is None or T is None:
        return None
    if full:
        if I.shape != T.shape:
            T = cv2.resize(T, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        I, T = load224(ip), load224(tp)
        if I.shape != T.shape:
            T = cv2.resize(T, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
    i = I.astype(np.float64); t = T.astype(np.float64)
    g = float((i * t).sum() / ((i * i).sum() + 1e-9))  # 由 I 到 T 的增益
    return float(np.abs(g * i - t).mean() / (t.mean() + 1e-9))


def summarize(name, ts, rs):
    t = np.array(ts)
    print(f'\n--- {name} (n={len(ts)}) ---')
    for i, k in enumerate(['mean', 'p5', 'p95', 'contrast', 'lap', 'noise', 'sat']):
        v = t[:, i]
        print(f'  {k:<9} 中位 {np.median(v):7.2f}   p10 {np.percentile(v,10):7.2f}   p90 {np.percentile(v,90):7.2f}')
    if rs:
        r = np.array([x for x in rs if x is not None])
        print(f'  r_obs     中位 {np.median(r):.3f}   p10 {np.percentile(r,10):.3f}   p90 {np.percentile(r,90):.3f}   max {r.max():.3f}')
    return t, (np.array([x for x in rs if x is not None]) if rs else None)


def main():
    print('=== T 域统计（224² 中心裁口径）===')
    REAL = 'data/test_sets/real_pair_0917'
    scenes = sorted(d for d in os.listdir(REAL) if d.startswith('VID_'))
    real_ts = [t_stats(os.path.join(REAL, s, 'T.png')) for s in scenes]
    real_rs = [r_obs(os.path.join(REAL, s, 'I.png'), os.path.join(REAL, s, 'T.png'), full=True) for s in scenes]
    summarize('实拍 GT（6月 低光无反射，对齐后）', [list(d.values()) for d in real_ts], real_rs)

    for ds, lv in [('synth_rrw_v21_0913', ['alpha_1', 'alpha_0.6', 'alpha_0.3']),
                   ('synth_rrw_v23_lowscan', ['alpha_1', 'alpha_0.3', 'alpha_0.05'])]:
        base = f'data/test_sets/{ds}'
        gts = sorted(glob.glob(base + '/gt/*.png'))
        if not gts:
            continue
        ts = [t_stats(p) for p in gts]
        summarize(f'{ds} / gt（合成 T）', [list(d.values()) for d in ts], [])
        for L in lv:
            ips = sorted(glob.glob(f'{base}/levels/{L}/*.png'))
            if not ips:
                continue
            rs = [r_obs(ip, gts[i % len(gts)]) for i, ip in enumerate(ips)]
            r = np.array([x for x in rs if x is not None])
            print(f'  [{ds} {L}] r_obs 中位 {np.median(r):.3f}  p10 {np.percentile(r,10):.3f}  '
                  f'p90 {np.percentile(r,90):.3f}  max {r.max():.3f}')
    print('\n注：合成 r_obs 用 224² 裁片；实拍用全图（对齐后），两者口径差异见文末说明。')


if __name__ == '__main__':
    main()
