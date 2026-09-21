#!/usr/bin/env python3
"""M1 掩码训练数据：apr 原始视频 359 帧的 Ω 标签生成（2026-09-21 轮）。

对 6 场景 × 全部 frame_*.jpg：T = 场景 GT（测试集 GT.png），Ω = omega_of(I, T)。
产物: data/train_sets/train_mask_0921/{I,T,mask}/<scene>__<frame>.png + meta.json
验收（预登记）: Ω 成功率 ≥70%；成功帧中位占比 ∈ [14%,33%]。
用法: python3 scripts/build_mask_train_0921.py
"""
import json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_g2_baseline import omega_of                # noqa: E402

SRC = os.path.join(PROJ, 'data/My_Reflection_Videos_cropped')
GTD = os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917')
OUT = os.path.join(PROJ, 'data/train_sets/train_mask_0921')
SCENES = ['VID_20260426_172719', 'VID_20260426_172933', 'VID_20260426_173237',
          'VID_20260426_173349', 'VID_20260426_173604', 'VID_20260426_173827']


def main():
    for sub in ('I', 'T', 'mask'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    meta, ok, fail = {}, 0, 0
    for sc in SCENES:
        T = cv2.imread(os.path.join(GTD, sc, 'GT.png'))
        assert T is not None, sc
        frames = sorted(f for f in os.listdir(os.path.join(SRC, sc)) if f.startswith('frame_'))
        for fn in frames:
            I = cv2.imread(os.path.join(SRC, sc, fn))
            if I is None or I.shape != T.shape:
                fail += 1
                continue
            key = f'{sc}__{fn[:-4]}'
            o = omega_of(I, T)
            om = o['Om'] if o is not None else np.zeros(I.shape[:2], bool)
            cv2.imwrite(os.path.join(OUT, 'I', key + '.png'), I)
            cv2.imwrite(os.path.join(OUT, 'T', key + '.png'), T)
            cv2.imwrite(os.path.join(OUT, 'mask', key + '.png'), (om * 255).astype(np.uint8))
            frac = float(om.mean())
            meta[key] = dict(scene=sc, frame=fn[:-4], omega_frac=round(frac, 4),
                             ok=int(frac > 0.01))
            ok += int(frac > 0.01)
        print(f'{sc}: {len(frames)} 帧完成', flush=True)
    with open(os.path.join(OUT, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=1)
    fr = [v['omega_frac'] for v in meta.values() if v['ok']]
    print(f'\n共 {len(meta)} 帧 | Ω 有效 {ok}（{ok/max(len(meta),1)*100:.1f}%，线 ≥70%）| '
          f'读取失败 {fail}')
    if fr:
        print(f'Ω 占比：中位 {np.median(fr)*100:.1f}%（线 [14,33]）min {min(fr)*100:.1f}% '
              f'max {max(fr)*100:.1f}%')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
