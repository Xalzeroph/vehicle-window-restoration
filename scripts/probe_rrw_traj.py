#!/usr/bin/env python3
"""RRW：g 随帧号轨迹 + 跨黑（反射帧 ↔ GT）跳变（2026-10-08）。

与车窗数据同口径，回答："RRW 的 AE 跳了吗？"
用法: python3 scripts/probe_rrw_traj.py [case ...]
"""
import os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_protocol import robust_gain          # noqa: E402

ROOT = '/DataDisk/zmy_code/datasets/RRWDataset'


def rd(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    targets = sys.argv[1:] or ['reflection_camera1/camera_objects',
                               'reflection_camera1/camera_objects2',
                               'reflection_camera2/indoor_object1',
                               'reflection_hf0/indoor_chairs']
    print(f"{'case':<40}{'帧数':>6}{'帧号max':>9}{'g起点':>8}{'g末点':>8}"
          f"{'g极差':>8}{'corr(g,帧号)':>13}")
    for t in targets:
        sub, cs = t.split('/')
        cd = os.path.join(ROOT, sub, cs)
        gtd = None
        for d in sorted(os.listdir(os.path.join(ROOT, sub))):
            if d.startswith('GT') and os.path.isdir(os.path.join(ROOT, sub, d)):
                gtd = os.path.join(ROOT, sub, d)
                break
        gtp = os.path.join(gtd, f'{cs}_GT.png') if gtd else None
        if not gtp or not os.path.exists(gtp):
            print(f'  {t}: 无 GT'); continue
        T = rd(gtp)
        fs = sorted(f for f in os.listdir(cd) if f.lower().endswith('.jpg'))
        if len(fs) < 5:
            continue
        nums = np.array([int(f.split('_')[1].split('.')[0]) for f in fs], float)
        idx = np.linspace(0, len(fs) - 1, min(12, len(fs))).astype(int)
        gs = []
        for i in idx:
            I = rd(os.path.join(cd, fs[i]))
            if I is None:
                gs.append(np.nan); continue
            if I.shape != T.shape:
                I = cv2.resize(I, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_AREA)
            gs.append(float(np.atleast_1d(robust_gain(I, T)).mean()))
        gs = np.array(gs)
        ok = ~np.isnan(gs)
        cg = float(np.corrcoef(nums[idx][ok], gs[ok])[0, 1]) if ok.sum() > 2 else float('nan')
        print(f"{t:<40}{len(fs):>6}{int(nums.max()):>9}{gs[ok][0]:>8.3f}{gs[ok][-1]:>8.3f}"
              f"{np.nanmax(gs)-np.nanmin(gs):>8.3f}{cg:>13.2f}")
    print()
    print('注：g 是"该帧相对 GT"的增益。g 随帧号明显变化 = AE 在漂。')
    print('    RRW 的帧号跨 0~10650（约 7 分钟视频），GT 是其中一帧。')


if __name__ == '__main__':
    main()
