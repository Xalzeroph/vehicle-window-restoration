#!/usr/bin/env python3
"""apr 配对 I/GT 的"同刻性"核查（Opus 2026-09-22 实测的事实复核）。

Ω 外量 (a) 中位 |luma(I)−luma(GT)|；(b) 逐通道 3×3 仿射最小二乘对齐后的残差。
预期复现：中位 31.6（19.6–47.4）；仿射残差 10–16（每场景）。若成立 → apr 配对为"内容级不同刻"，
"训练域 16/16"的解释须改为场景记忆+色调对齐。
用法: python3 scripts/probes_0921/check_pair_gt_offset_0922.py
"""
import csv, importlib.util, os
import cv2
import numpy as np

os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')
spec = importlib.util.spec_from_file_location('eg2', 'scripts/eval_g2_baseline.py')
eg2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg2)
D = 'data/test_sets/real_pair_apr_0917'
rows = []
for sc in sorted(x for x in os.listdir(D) if x.startswith('VID_')):
    T = cv2.imread(f'{D}/{sc}/GT.png')
    for fn in sorted(f for f in os.listdir(f'{D}/{sc}') if f.startswith('I')):
        I = cv2.imread(f'{D}/{sc}/{fn}')
        o = eg2.omega_of(I, T)
        if not o:
            continue
        Om = cv2.dilate(o['Om'].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        gI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gT = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
        m = ~Om
        d = np.abs(gI - gT)[m]
        # 3×3 仿射（含偏置）最小二乘：GT ≈ W·[I,1]
        X = np.concatenate([I.reshape(-1, 3).astype(np.float64), np.ones((I.shape[0] * I.shape[1], 1))],
                           axis=1)[m.reshape(-1)]
        Y = T.reshape(-1, 3).astype(np.float64)[m.reshape(-1)]
        W, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = np.abs(X @ W - Y).mean()
        rows.append(dict(scene=sc, frame=fn[:-4], med_out=round(float(np.median(d)), 1),
                         mean_out=round(float(d.mean()), 1), p90_out=round(float(np.percentile(d, 90)), 1),
                         affine_resid=round(float(res), 1)))
        print(f'{sc[-6:]}/{fn[:-4]:<10} Ω外|luma差| 中位 {np.median(d):5.1f} 均值 {d.mean():5.1f} '
              f'p90 {np.percentile(d, 90):5.1f} | 仿射残差 {res:5.1f}')
a = lambda k: float(np.median([r[k] for r in rows]))
print(f'\n汇总（{len(rows)} 帧）：Ω 外中位 {a("med_out"):.1f}（范围 '
      f'{min(r["med_out"] for r in rows):.1f}–{max(r["med_out"] for r in rows):.1f}）| '
      f'仿射残差中位 {a("affine_resid"):.1f}（{min(r["affine_resid"] for r in rows):.1f}–'
      f'{max(r["affine_resid"] for r in rows):.1f}）')
with open('eval/probe_pair_gt_offset_apr_0922.csv', 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('-> eval/probe_pair_gt_offset_apr_0922.csv')
