#!/usr/bin/env python3
"""2026-09-22 更正轮的两项事实复核（CPU，读已存预测/配对图）。

背景：用户更正"全部批次 GT = 视频结尾黑布遮满全反射层段"（去反射语义）。
本脚本量化两件事：
  A. apr 配对 I/GT 的**系统色调差**（分区域 signed mean of T−I、增益 g）——检验 GT 是否"只是 I 去掉反射"。
  B. apr 16 帧的区域表（亮结构反射 / 固定暗结构 / 人影洞 Ω / 全图），臂 = 官方原生 / S2 s101@10k
     / 全局仿射基线（per-frame lstsq I→T，含 gain+offset，用 T 拟合 = 上界对照）。
用法: python3 scripts/probes_0921/check_gt_tone_regions_0922.py
产物: eval/probe_gt_tone_regions_0922.csv（逐帧）+ 屏显汇总
"""
import csv
import importlib.util
import os

import cv2
import numpy as np

os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')
spec = importlib.util.spec_from_file_location('eg2', 'scripts/eval_g2_baseline.py')
eg2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg2)
G = eg2.G
D = 'data/test_sets/real_pair_apr_0917'
ARMS = {'off': 'runs/pred/apr_rdnet_native', 's2': 'runs/pred/apr_s101_apr10k'}

agg = {a: {k: [] for k in ('br', 'dk', 'om', 'gl')} for a in list(ARMS) + ['aff']}
tone = {k: [] for k in ('om', 'ring', 'br', 'dk', 'gl', 'g')}
rows = []

for sc in sorted(x for x in os.listdir(D) if x.startswith('VID_')):
    T = cv2.imread(f'{D}/{sc}/GT.png')
    for fn in sorted(f for f in os.listdir(f'{D}/{sc}') if f.startswith('I')):
        I = cv2.imread(f'{D}/{sc}/{fn}')
        o = eg2.omega_of(I, T)
        if not o:
            continue
        Om = o['Om']
        Omd = cv2.dilate(Om.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        ring = cv2.dilate(Om.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~Om
        gI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gT = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
        Il = (np.clip(I, 0, 255) / 255.) ** G
        Tl = (np.clip(T, 0, 255) / 255.) ** G
        valid = Tl.mean(2) > (10 / 255.) ** G
        ratio = np.where(valid[..., None], Il / np.where(Tl > 1e-8, Tl, np.nan), np.nan)
        s = float(np.nanquantile(ratio.mean(2), 0.05))
        R = (I.astype(np.float64) - (np.clip(s * Tl, 0, 1)) ** (1 / G) * 255).mean(2)
        stR = cv2.boxFilter(np.abs(cv2.Laplacian(R.astype(np.float32), cv2.CV_32F)), -1, (9, 9))
        br = (R > 12) & (stR > 1) & (~Omd)
        dk = (gI < np.percentile(gI, 15)) & (np.abs(R) < 8) & (~Omd)
        m = ~Omd
        for nm, reg in (('om', Om), ('ring', ring), ('br', br), ('dk', dk), ('gl', np.ones_like(Om, bool))):
            tone[nm].append(float((gT - gI)[reg].mean()))
        gm = float(np.median(gT[m]) / max(np.median(gI[m]), 1e-6))
        tone['g'].append(float(s))
        # 全局仿射 I→T（per-frame 最小二乘，全图，含 gain+offset）——"全局色调对齐"上界对照
        Xf = np.concatenate([I.reshape(-1, 3).astype(np.float64), np.ones((I.shape[0] * I.shape[1], 1))], axis=1)
        Yf = T.reshape(-1, 3).astype(np.float64)
        W, *_ = np.linalg.lstsq(Xf[m.reshape(-1)], Yf[m.reshape(-1)], rcond=None)
        A_aff = np.clip(Xf @ W, 0, 255).reshape(I.shape).astype(np.uint8)
        imgs = {a: cv2.imread(f'{root}/{sc}__{fn[:-4]}/ytmt_ucs_sirs_l.png') for a, root in ARMS.items()}
        imgs['aff'] = A_aff
        row = dict(scene=sc[-6:], frame=fn[:-4], g_med=round(gm, 3), s_omega=round(s, 3),
                   T_minus_I_om=round(tone['om'][-1], 1), T_minus_I_ring=round(tone['ring'][-1], 1))
        for a in agg:
            P = imgs[a]
            if P is None:
                continue
            if P.shape != I.shape:
                P = cv2.resize(P, (I.shape[1], I.shape[0]))
            gP = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY).astype(np.float64)
            for nm, reg in (('br', br), ('dk', dk), ('om', Om), ('gl', np.ones_like(Om, bool))):
                agg[a][nm].append(float(np.abs(gP - gT)[reg].mean()))
            row[f'{a}_dP_om'] = round(float((gP - gI)[Om].mean()), 1)
            row[f'{a}_err_om'] = round(float(np.abs(gP - gT)[Om].mean()), 1)
            row[f'{a}_err_gl'] = round(float(np.abs(gP - gT).mean()), 1)
        rows.append(row)

print(f'== A. GT 相对 I 的色调差（apr {len(rows)} 帧，signed mean T−I）')
for nm, label in (('om', '人影洞Ω'), ('ring', '环带'), ('br', '亮反射'), ('dk', '固定暗结构'), ('gl', '全图')):
    print(f'  {label:<10}{np.mean(tone[nm]):+7.1f}')
print(f'  增益 g_med 中位 {np.median(tone["g"]):.2f}（omega 定式 s 中位）')
print('== B. |P−T|（均值；aff=全局仿射 I→T 上界对照）')
print(f'  {"臂":<6}{"亮反射":>9}{"暗结构":>9}{"Ω":>9}{"全图":>9}')
for a in agg:
    print(f'  {a:<6}' + ''.join(f'{np.mean(agg[a][k]):>9.1f}' for k in ('br', 'dk', 'om', 'gl')))

with open('eval/probe_gt_tone_regions_0922.csv', 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print('-> eval/probe_gt_tone_regions_0922.csv')
