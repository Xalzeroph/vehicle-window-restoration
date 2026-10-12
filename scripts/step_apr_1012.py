#!/usr/bin/env python3
"""apr 光度轨迹：分离"曝光台阶"与"布遮挡"（2026-10-12 判决版）

动机
----
前面三版脚本都有各自的循环：
  scan_*    锚点=I 帧本身 → "最优帧"必为 I 帧
  verdict_* 锚点=I 帧均值 → 只能答"与 I 是否一致"，不能答"能否换帧"
  direct_*  锚点=现行 GT.png → GT.png 自身 g≡1 必夺冠
本版**完全不用 GT、不用 I 作为锚**，只用帧间关系给整段标定光度轨迹。

口径
----
场景静止 → 任取两帧 t、ref，其**逐像素比值**的中位数就是相机光度状态之比：
      s(t) = median_pixels( frame_t / frame_ref )      逐通道
这是对"曝光×白平衡"的稳健估计，不受布遮挡影响（布只影响被遮住的像素，
中位数对遮挡鲁棒）。取 ref = 首帧，s(0)=1。

输出
----
  s(t) 轨迹  → 台阶在哪一帧、阶跃多大
  台阶判据   ：相邻帧 |Δs| 超过全段 |Δs| 分布的 6×MAD
  与"布段"的交叉：本脚本不猜布段，只报台阶位置；
                 布段由 |g-1|（相对首帧）的平台段界定。

判决问题：**在同一条 s(t) 平台上，是否存在"布已合上且光度=I"的帧？**
即 s(t)≈1 与"布段"是否有交集。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

CROPPED = os.path.join(PROJ, 'data', 'My_Reflection_Videos_cropped')
APR = os.path.join(PROJ, 'data', 'test_sets', 'real_pair_apr_0917')


def fnum(name: str) -> int:
    m = re.search(r'(\d{6})', os.path.basename(name))
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PROJ, 'eval', 'apr_photometric_step_1012.json'))
    ap.add_argument('--ref-frames', type=int, default=24,
                    help='用于估计 s(t) 的像素采样数上限（取中心区域子采样，省内存）')
    args = ap.parse_args()

    scenes = sorted(x for x in os.listdir(APR) if x.startswith('VID_'))
    rep = {}

    for sc in scenes:
        cdir = os.path.join(CROPPED, sc)
        files = sorted((f for f in os.listdir(cdir) if f.endswith('.jpg')), key=fnum)
        imgs = []
        for f in files:
            im = cv2.imread(os.path.join(cdir, f), cv2.IMREAD_COLOR)
            if im is None:
                continue
            imgs.append((fnum(f), im.astype(np.float32)))
        if len(imgs) < 6:
            continue

        h = min(im.shape[0] for _, im in imgs)
        w = min(im.shape[1] for _, im in imgs)
        # 统一到同一尺寸，子采样加快
        step_y = max(1, h // 400)
        step_x = max(1, w // 400)
        arr = np.stack([im[:h, :w][::step_y, ::step_x] for _, im in imgs])  # (n,h',w',3)
        fno = np.array([n for n, _ in imgs], dtype=float)

        ref = arr[0]
        eps = 1e-3
        # s(t) = 逐通道 median(arr[t] / ref)，用中位数抗遮挡
        ratio = arr / np.maximum(ref, eps)
        s = np.median(ratio.reshape(len(arr), -1, 3), axis=1)      # (n,3)
        lum = np.array([cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).mean()
                        for _, im in imgs])

        ds = np.diff(s, axis=0)                                     # (n-1,3)
        ds_mag = np.abs(ds).max(axis=1)
        med = float(np.median(ds_mag))
        mad = float(np.median(np.abs(ds_mag - med))) + 1e-9
        z = (ds_mag - med) / (1.4826 * mad)
        steps = np.where(z > 6.0)[0]

        # s 偏离 1 超过 5% 的帧区间
        dev = np.abs(s - 1).max(axis=1)
        off = dev > 0.05

        # 布段（用 luma：布合上时被遮区域变暗/变灰，取 luma 相对首帧下降的连续尾部）
        # 不猜：报告 luma 尾部平台
        rep_sc = {
            'scene': sc,
            'n_frames': len(imgs),
            'frame_first': int(fno[0]),
            'frame_last': int(fno[-1]),
            's_first': [round(float(x), 4) for x in s[0]],
            's_last': [round(float(x), 4) for x in s[-1]],
            's_max_dev': round(float(dev.max()), 4),
            's_max_dev_frame': int(fno[int(np.argmax(dev))]),
            'n_steps': int(len(steps)),
            'step_frames': [int(fno[i]) for i in steps],
            'step_sizes': [round(float(ds_mag[i]), 4) for i in steps],
            'max_step_size': round(float(ds_mag.max()), 4),
            'max_step_frame': int(fno[int(np.argmax(ds_mag))]),
            'median_abs_ds': round(med, 5),
            'frac_frames_off_5pct': round(float(off.mean()), 3),
            'off_frame_range': [int(fno[off][0]), int(fno[off][-1])] if off.any() else None,
            'luma_first': round(float(lum[0]), 2),
            'luma_last': round(float(lum[-1]), 2),
            'luma_min': round(float(lum.min()), 2),
            'luma_max': round(float(lum.max()), 2),
        }
        rep[sc] = rep_sc

        print(f"\n=== {sc}  n={len(imgs)}  帧 {int(fno[0])}-{int(fno[-1])}")
        print(f"  s(首)={rep_sc['s_first']}  s(末)={rep_sc['s_last']}")
        print(f"  s 最大偏离 1: {dev.max():.4f} @帧 {int(fno[int(np.argmax(dev))])}")
        print(f"  台阶: {len(steps)} 处  {[int(fno[i]) for i in steps][:10]}"
              f"   最大单步 {ds_mag.max():.4f} @帧 {int(fno[int(np.argmax(ds_mag))])}")
        print(f"  相邻帧 |Δs| 中位 {med:.5f} (MAD {mad:.5f})")
        print(f"  |s-1|>5% 的帧占比 {off.mean()*100:.0f}%"
              f"  区间 {rep_sc['off_frame_range']}")
        print(f"  luma 首 {lum[0]:.1f} 末 {lum[-1]:.1f}  [{lum.min():.1f}, {lum.max():.1f}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out}')

    print('\n' + '=' * 104)
    print(f'{"场景":<24}{"最大单步":>9}{"@帧":>7}{"台阶数":>7}{"s末[B,G,R]":>26}'
          f'{"偏离>5%占比":>11}')
    print('-' * 104)
    for sc, r in rep.items():
        print(f'{sc:<24}{r["max_step_size"]:>9.4f}{r["max_step_frame"]:>7}'
              f'{r["n_steps"]:>7}{str(r["s_last"]):>26}{r["frac_frames_off_5pct"]*100:>10.0f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
