#!/usr/bin/env python3
"""构建 S2 开源直督索引 train_pairs_opensrc_0920（P3-3b）。

源：OpenRR-5k（data/external/OpenRR-5k/unpacked/trian_5k，5000 对）+ SIR2（SIR 子集 499 对）。
不拷贝大图，只产出索引 meta.json（pairs 列表 + 逐对 |R| 统计），训练脚本按清单加载。

产物: data/train_sets/train_pairs_opensrc_0920/meta.json
用法: python3 scripts/build_train_opensrc_0920.py [--workers 24]
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, 'data/train_sets/train_pairs_opensrc_0920')
SRCS = {
    'openrr': ('/DataDisk/zmy_code/TintedGlass_Enhance/data/external/OpenRR-5k/unpacked/trian_5k/blended',
               '/DataDisk/zmy_code/TintedGlass_Enhance/data/external/OpenRR-5k/unpacked/trian_5k/transmission_layer'),
    'sir2': ('/home/zmy/mycode/datasets/SIR2/SIR/blended',
             '/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer'),
}


def one(args):
    src, i_dir, t_dir, name, gt_name = args
    I = cv2.imread(os.path.join(i_dir, name))
    T = cv2.imread(os.path.join(t_dir, gt_name))
    if I is None or T is None or I.shape != T.shape:
        return None
    R = I.astype(np.float32) - T.astype(np.float32)
    rn = np.abs(R).mean(axis=2)
    return dict(src=src, name=os.path.splitext(name)[0], I=os.path.join(i_dir, name),
                T=os.path.join(t_dir, gt_name), size=list(I.shape[:2]),
                R_abs_mean=round(float(rn.mean()), 2), R_abs_p95=round(float(np.percentile(rn, 95)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for src, (i_dir, t_dir) in SRCS.items():
        i_names = sorted(f for f in os.listdir(i_dir) if not f.startswith('.'))
        for name in i_names:
            jobs.append((src, i_dir, t_dir, name, name))
    print(f'配对任务 {len(jobs)}')
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=32)):
            if r:
                rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f'  {i+1}/{len(jobs)}', flush=True)
    with open(os.path.join(OUT, 'meta.json'), 'w') as fp:
        json.dump(rows, fp, indent=1)
    for src in SRCS:
        sub = [r['R_abs_mean'] for r in rows if r['src'] == src]
        if sub:
            q = np.percentile(sub, [10, 25, 50, 75, 90])
            print(f'{src}: n={len(sub)} |R|均值分位 p10 {q[0]:.1f} / p25 {q[1]:.1f} / '
                  f'中位 {q[2]:.1f} / p75 {q[3]:.1f} / p90 {q[4]:.1f}')
    print(f'共 {len(rows)} 对 -> {OUT}/meta.json')


if __name__ == '__main__':
    main()
