#!/usr/bin/env python3
"""apr 光度漂移的**无循环**诊断（2026-10-12，修正 scan_apr_gt_1012.py 的锚点自比）

上一版 scan_apr_gt_1012.py 把"最优帧"定义成 g 最接近 1 的帧，
而锚点本身就是 I 帧（frame 000000），于是 `g(anchor, anchor) ≡ 1` 必然夺魁——
**结论由构造决定，不是测量结果**。本脚本换成无循环的测量：

指标（全部只用**帧自身**，不引入 GT、不引入任何参考答案）
-------------------------------------------------------
1. 逐帧光度轨迹：luma(t) 与三通道均值(B,G,R)(t)
   场景静止 → 轨迹的变化 = 相机 AE/AWB 的变化。
2. 相邻候选帧的**帧间差分**：|luma(t+1)-luma(t)| 与逐通道差
   - 若某点出现孤立大跳 → **瞬时台阶**（对应六月"布合上一瞬即完整存在"）
   - 若全段平滑单调     → **连续斜坡**（AE 追赶）
3. 漂移幅度 vs 信号幅度：
   drift = max-min of luma over the clip
   signal = |I_mean - GT_mean|  （photometric 口径）
   若 drift 与 signal 同量级 → 选帧有机会救；若 drift << signal → 选帧没用。
4. 布段稳定性：布段内 luma 的 std 与极差，找出最长的"静止窗口"。
   交接页 §4 记"布段内自身不稳 g 抖动 0.04-0.21"，本脚本给出时间定位。

判读不看绝对值，只看**形状**：斜坡 vs 台阶。
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
APR_SETS = os.path.join(PROJ, 'data', 'test_sets', 'real_pair_apr_0917')


def frame_num(name: str) -> int:
    m = re.search(r'(\d{6})', os.path.basename(name))
    return int(m.group(1)) if m else -1


def stats(path: str):
    """返回 (luma, [B,G,R] 均值, h, w)。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # 统一缩到 320 宽做统计，省内存也去掉缩放无关的噪声
    h, w = img.shape[:2]
    if w > 320:
        s = 320.0 / w
        img = cv2.resize(img, (320, max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    luma = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    ch = [float(img[..., i].mean()) for i in range(3)]
    return luma, ch, h, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PROJ, 'eval', 'apr_photometric_drift_1012.json'))
    args = ap.parse_args()

    scenes = sorted(x for x in os.listdir(APR_SETS) if x.startswith('VID_'))
    allrep = {}

    for sc in scenes:
        cdir = os.path.join(CROPPED, sc)
        if not os.path.isdir(cdir):
            print(f'{sc}: 无裁剪帧目录，跳过')
            continue
        files = sorted((f for f in os.listdir(cdir) if f.endswith('.jpg')), key=frame_num)
        traj = []
        for f in files:
            s = stats(os.path.join(cdir, f))
            if s is None:
                continue
            traj.append((frame_num(f), s[0], s[1]))
        if len(traj) < 5:
            print(f'{sc}: 帧太少，跳过')
            continue

        fno = np.array([t[0] for t in traj], dtype=float)
        lum = np.array([t[1] for t in traj], dtype=float)
        chs = np.array([t[2] for t in traj], dtype=float)   # (n,3) BGR

        d_lum = np.diff(lum)
        d_fno = np.diff(fno)

        # 台阶检测：某相邻帧差远大于整体分布
        med_d = float(np.median(np.abs(d_lum)))
        mad_d = float(np.median(np.abs(np.abs(d_lum) - med_d))) + 1e-6
        z = (np.abs(d_lum) - med_d) / (1.4826 * mad_d)
        step_idx = np.where(z > 6.0)[0]

        drift = float(lum.max() - lum.min())
        ch_drift = [float(chs[:, i].max() - chs[:, i].min()) for i in range(3)]
        ch_spread = float(max(ch_drift) - min(ch_drift))

        # 现行 GT 的 luma（用 test set 里的 GT.png）
        gt_l = None
        gi = os.path.join(APR_SETS, sc, 'GT.png')
        if os.path.exists(gi):
            s = stats(gi)
            gt_l = None if s is None else s[0]

        # 各 I 帧 luma
        i_l = []
        idir = os.path.join(APR_SETS, sc)
        if os.path.isdir(idir):
            for f in sorted((x for x in os.listdir(idir) if x.startswith('I_')), key=frame_num):
                s = stats(os.path.join(idir, f))
                if s is not None:
                    i_l.append((frame_num(f), round(s[0], 2)))

        # 平滑性：|g-1| 轨迹的单调度 —— 用 luma 轨迹与帧号的秩相关
        from scipy.stats import spearmanr
        rho, _ = spearmanr(fno, lum)

        # 布段 vs 反射段：不做人工分段，用"轨迹是否单调"替代
        # 最长静止窗口（相邻帧 |dluma| <= 0.5 的连续段）
        best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
        for i, d in enumerate(d_lum):
            if abs(d) <= 0.5:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0

        rep = {
            'scene': sc,
            'n_frames': len(traj),
            'frame_span': [int(fno.min()), int(fno.max())],
            'luma_min': round(float(lum.min()), 2),
            'luma_max': round(float(lum.max()), 2),
            'luma_drift': round(drift, 2),
            'luma_std': round(float(lum.std()), 2),
            'ch_drift_BGR': [round(x, 2) for x in ch_drift],
            'ch_drift_spread': round(ch_spread, 3),
            'spearman_frame_luma': round(float(rho), 4),
            'median_abs_dluma': round(med_d, 3),
            'max_abs_dluma': round(float(np.abs(d_lum).max()), 3),
            'n_step_like_transitions': int(len(step_idx)),
            'step_frames': [int(fno[i]) for i in step_idx][:8],
            'longest_static_window': {
                'n_pairs': int(best_len),
                'from_frame': int(fno[best_start]) if best_len else None,
                'to_frame': int(fno[best_start + best_len]) if best_len else None,
            },
            'gt_luma': None if gt_l is None else round(gt_l, 2),
            'i_frames': i_l,
        }
        if gt_l is not None and i_l:
            rep['signal_vs_drift_ratio'] = round(
                abs(np.mean([x[1] for x in i_l]) - gt_l) / max(drift, 1e-6), 3)

        # 形状判读
        if len(step_idx) == 0 and abs(rho) > 0.5:
            shape = '连续斜坡(单调漂移)'
        elif len(step_idx) >= 1 and abs(rho) < 0.5:
            shape = f'瞬时台阶 x{len(step_idx)}'
        elif len(step_idx) >= 1:
            shape = f'斜坡+台阶(斜坡主, 台阶{len(step_idx)})'
        else:
            shape = '基本恒定'
        rep['drift_shape'] = shape

        print(f"\n=== {sc}  n={len(traj)}  帧 {int(fno.min())}-{int(fno.max())}")
        print(f"  luma [{lum.min():.1f}, {lum.max():.1f}] 漂移 {drift:.2f} (std {lum.std():.2f})"
              f"   逐通道漂移 BGR {[round(x,2) for x in ch_drift]} 极差 {ch_spread:.3f}")
        print(f"  Spearman(帧号, luma) = {rho:+.3f}   相邻帧|dluma| 中位 {med_d:.2f} max {np.abs(d_lum).max():.2f}")
        print(f"  台阶数(z>6) = {len(step_idx)}  位置 {[int(fno[i]) for i in step_idx][:8]}")
        print(f"  最长静止窗口: {best_len} 对  {int(fno[best_start]) if best_len else '-'}"
              f" -> {int(fno[best_start+best_len]) if best_len else '-'}")
        print(f"  I 帧 luma {i_l}   GT luma {gt_l}")
        if 'signal_vs_drift_ratio' in rep:
            print(f"  信号/漂移 比 = {rep['signal_vs_drift_ratio']}")
        print(f"  形状判读: {shape}")

        allrep[sc] = rep

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(allrep, f, ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out}')

    print('\n' + '=' * 110)
    print(f'{"场景":<24}{"帧数":>5}{"luma漂移":>10}{"通道极差":>10}{"Spearman":>11}'
          f'{"台阶":>6}{"静止窗":>8}{"信号/漂移":>11}  形状')
    print('-' * 110)
    for sc, r in allrep.items():
        print(f'{sc:<24}{r["n_frames"]:>5}{r["luma_drift"]:>10.2f}{r["ch_drift_spread"]:>10.3f}'
              f'{r["spearman_frame_luma"]:>11.3f}{r["n_step_like_transitions"]:>6}'
              f'{r["longest_static_window"]["n_pairs"]:>8}'
              f'{r.get("signal_vs_drift_ratio", float("nan")):>11.2f}  {r["drift_shape"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
