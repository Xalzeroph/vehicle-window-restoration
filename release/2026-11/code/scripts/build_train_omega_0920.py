#!/usr/bin/env python3
"""构建 S2 真实配对训练集 train_pairs_omega_0920（P3-2 + P3-3a）。

源：data/test_sets/real_pair_apr_0917（6 视频 16 帧，S2 已定 apr 解除 eval-only 转训练）
产出（data/train_sets/train_pairs_omega_0920/）：
  I/<case>.png      输入帧（含反射）
  T/<case>.png      GT（无反射帧）
  R/<case>.png      有符号 R = I − g·T（+128 存储；g=robust_gain）
  omega/<case>.png  Ω mask（人影洞区，omega_of v2 定式）
  meta.json         per-case g / Ω 占比 / 尺寸

用法: python3 scripts/build_train_omega_0920.py
"""
import json, os, shutil, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_protocol import robust_gain               # noqa: E402
from eval_g2_baseline import omega_of               # noqa: E402

SRC = os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917')
OUT = os.path.join(PROJ, 'data/train_sets/train_pairs_omega_0920')


def main():
    for sub in ('I', 'T', 'R', 'omega'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    meta = {}
    for sc in sorted(x for x in os.listdir(SRC) if x.startswith('VID_')):
        sd = os.path.join(SRC, sc)
        T = cv2.imread(os.path.join(sd, 'GT.png'))
        for fn in sorted(f for f in os.listdir(sd) if f.startswith('I_')):
            case = f'{sc}__{fn[:-4]}'
            I = cv2.imread(os.path.join(sd, fn))
            assert I is not None and T is not None and I.shape == T.shape, case
            g = robust_gain(I, T)
            R = I.astype(np.float64) - T.astype(np.float64) * np.atleast_1d(g).reshape(1, 1, 3)
            o = omega_of(I, T)
            om = o['Om'] if o is not None else np.zeros(I.shape[:2], bool)
            # C_I = 环带 luma 均值 − Ω 内 luma 均值（hinge 阈值 δ=max(4,0.5|C_I|) 用）
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            ring = (cv2.dilate(om.astype(np.uint8), ker) > 0) & (~om)
            lI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
            C_I = float(lI[ring].mean() - lI[om].mean()) if om.any() and ring.any() else 0.0
            cv2.imwrite(os.path.join(OUT, 'I', case + '.png'), I)
            cv2.imwrite(os.path.join(OUT, 'T', case + '.png'), T)
            cv2.imwrite(os.path.join(OUT, 'R', case + '.png'),
                        np.clip(R + 128.0, 0, 255).astype(np.uint8))
            cv2.imwrite(os.path.join(OUT, 'omega', case + '.png'),
                        (om * 255).astype(np.uint8))
            meta[case] = dict(scene=sc, src=fn, g=round(float(np.atleast_1d(g).mean()), 4),
                              C_I=round(C_I, 2),
                              omega_frac=round(float(om.mean()), 4),
                              R_abs_mean=round(float(np.abs(R.mean(axis=2)).mean()), 2),
                              size=list(I.shape[:2]))
            print(f'{case}: g={meta[case]["g"]:.3f} Ω={meta[case]["omega_frac"]*100:5.1f}% '
                  f'|R|={meta[case]["R_abs_mean"]:.1f}')
    with open(os.path.join(OUT, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=1, ensure_ascii=False)
    om = [v['omega_frac'] for v in meta.values()]
    print(f'\n共 {len(meta)} 对 -> {OUT}')
    print(f'Ω 占比: min {min(om)*100:.1f}% 中位 {np.median(om)*100:.1f}% max {max(om)*100:.1f}%')


if __name__ == '__main__':
    main()
