#!/usr/bin/env python3
"""AE 漂移诊断：现有三域数据的曝光差是"固定偏置"还是"逐帧漂移"？（2026-10-08）

目的：指导手机拍摄时该怎么锁色调。
  若 g 在同场景内恒定（只是跨档差一个常数）→ 只要保证"挡黑布前后曝光档一致"即可；
  若 g 在同场景内逐帧变          → 必须真锁 AE（手动曝光），否则帧间无法对齐。

指标（每个场景内）：
  g 序列的极差 / 标准差       —— 漂移幅度
  corr(g, 输入帧号)           —— 是否随时间单调漂（AE 追赶）
  corr(g, luma(I))            —— 是否 AE 在响应画面亮度
  |g-1| 与协议阈 0.03 的比较
用法: python3 scripts/probe_ae_drift.py
"""
import os, sys, collections
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_protocol import robust_gain          # noqa: E402

SETS = {
    'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png'),
    'may': ('data/test_sets/real_pair_may_0917', 'GT.png'),
    'jun': ('data/test_sets/real_pair_0917', 'T.png'),
}


def main():
    print(f"{'域/场景':<34}{'n':>3}{'g中位':>8}{'极差':>8}{'std':>7}"
          f"{'corr(g,帧号)':>13}{'corr(g,luma)':>13}   漂移型")
    allg = collections.defaultdict(list)
    for dom, (root, gtn) in SETS.items():
        root = os.path.join(PROJ, root)
        for sc in sorted(x for x in os.listdir(root) if x.startswith('VID_')):
            sd = os.path.join(root, sc)
            T = cv2.imread(os.path.join(sd, gtn))
            if T is None:
                continue
            frames = sorted(f for f in os.listdir(sd) if f.startswith('I_'))
            if not frames:
                frames = ['I.png']
            gs, lumas, nums = [], [], []
            for fn in frames:
                I = cv2.imread(os.path.join(sd, fn))
                if I is None or I.shape != T.shape:
                    continue
                g = float(np.atleast_1d(robust_gain(I, T)).mean())
                gs.append(g)
                lumas.append(float(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).mean()))
                try:
                    nums.append(float(fn.split('_')[-1].split('.')[0]))
                except Exception:
                    nums.append(len(nums))
            if len(gs) < 2:
                continue
            gs = np.array(gs); lumas = np.array(lumas); nums = np.array(nums)
            allg[dom] += list(gs)
            rng = gs.max() - gs.min()
            cg = np.corrcoef(nums, gs)[0, 1] if nums.std() > 0 else float('nan')
            cl = np.corrcoef(lumas, gs)[0, 1] if lumas.std() > 0 else float('nan')
            if abs(cg) > 0.7:
                kind = '随时间单调漂(AE追赶)'
            elif abs(cl) > 0.7:
                kind = '随画面亮度(亮度计)'
            elif rng > 0.02:
                kind = '抖动'
            else:
                kind = '基本恒定'
            print(f"{dom+'/'+sc:<34}{len(gs):>3}{np.median(gs):>8.3f}{rng:>8.3f}"
                  f"{gs.std():>7.3f}{cg:>13.2f}{cl:>13.2f}   {kind}")
        print()
    print('=== 分域汇总 ===')
    for dom, gs in allg.items():
        gs = np.array(gs)
        bad = int(np.sum(np.abs(gs - 1) > 0.03))
        print(f'  {dom}: n={len(gs)}  |g-1|中位 {np.median(np.abs(gs-1)):.4f}  '
              f'超 0.03 的 {bad}/{len(gs)}  g 范围 [{gs.min():.3f}, {gs.max():.3f}]')
    print()
    print('判读：')
    print('  若多数场景"基本恒定" → 只需保证挡黑布前后用同一曝光档，可用手机 AE/AF 锁定')
    print('  若多为"随时间漂/随亮度变" → 手机 AE 锁定不够，必须切手动曝光(M档)把 ISO/快门写死')


if __name__ == '__main__':
    main()
