#!/usr/bin/env python3
"""apr 选帧扫描 —— 直接判据版（2026-10-12）

前两版的教训：
  scan_apr_gt_1012.py  以 I 帧为锚，"最优帧"必然是第一张 I 帧 → 循环论证；
  verdict_apr_gt_1012.py 以 I 帧均值为锚，只能回答"与 I 光度一致"，
                        不能回答"哪一帧最像现行 GT"。

本脚本用**唯一直接可比**的判据：现行 GT.png 自己。
对每个 apr 场景，沿裁剪帧目录逐帧算
    g(f, GT)        —— f 与现行 GT 的逐通道稳健增益
    |g-1| 均值      —— 光度状态距离
    spread          —— 白平衡一致性
若某帧 f* 满足 |g-1| <= 0.05 且 spread <= 0.03，则 f* 与现行 GT
**光度等价**，用它替换 GT 不会改变评测口径，却可能改变结构内容。
反之若全目录无此帧，说明现行 GT 的光度状态在整个裁剪片段里是孤立的
——那么"选帧"根本无从谈起。

同时报告：与 GT 光度等价的帧所占比例、这些帧的帧号分布（前段/后段）。
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
sys.path.insert(0, HERE)
from eval_protocol import robust_gain  # noqa: E402

CROPPED = os.path.join(PROJ, 'data', 'My_Reflection_Videos_cropped')
APR = os.path.join(PROJ, 'data', 'test_sets', 'real_pair_apr_0917')
THR_DEV, THR_SPR = 0.05, 0.03


def fnum(name: str) -> int:
    m = re.search(r'(\d{6})', os.path.basename(name))
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PROJ, 'eval', 'apr_gt_direct_1012.json'))
    args = ap.parse_args()

    scenes = sorted(x for x in os.listdir(APR) if x.startswith('VID_'))
    res = {}

    for sc in scenes:
        gtp = os.path.join(APR, sc, 'GT.png')
        if not os.path.exists(gtp):
            continue
        GT = cv2.imread(gtp, cv2.IMREAD_COLOR).astype(np.float64)
        gh, gw = GT.shape[:2]

        cdir = os.path.join(CROPPED, sc)
        files = sorted((f for f in os.listdir(cdir) if f.endswith('.jpg')), key=fnum)

        rows = []
        for f in files:
            im = cv2.imread(os.path.join(cdir, f), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if im.shape[:2] != (gh, gw):
                im = cv2.resize(im, (gw, gh), interpolation=cv2.INTER_AREA)
            g = np.atleast_1d(robust_gain(im.astype(np.float64), GT)).astype(float)
            rows.append({'frame': fnum(f),
                         'dev': float(np.abs(g - 1).mean()),
                         'spread': float(g.max() - g.min()),
                         'g': [round(float(x), 4) for x in g]})
        if not rows:
            continue
        fno = np.array([r['frame'] for r in rows], float)
        dev = np.array([r['dev'] for r in rows])
        spr = np.array([r['spread'] for r in rows])

        equiv = (dev <= THR_DEV) & (spr <= THR_SPR)
        # 现行 GT 的帧号（段末）；等价帧是落在它前面还是后面
        gt_frame = fno[-1]
        prev = fno[equiv & (fno < gt_frame)]
        n_equiv = int(equiv.sum())

        best_i = int(np.lexsort((spr, dev))[0])
        res[sc] = {
            'gt_frame': int(gt_frame),
            'n_frames': len(rows),
            'n_photometric_equiv': n_equiv,
            'equiv_frac': round(float(equiv.mean()), 3),
            'equiv_frames_head': [int(x) for x in fno[equiv][:10]],
            'equiv_before_gt': int(len(prev)),
            'best_frame': int(fno[best_i]),
            'best_dev': round(float(dev[best_i]), 4),
            'best_spread': round(float(spr[best_i]), 4),
            'min_dev_overall': round(float(dev.min()), 4),
            'dev_at_gt_minus_1': round(float(dev[-1]), 4),
            'verdict': ('存在与现行 GT 光度等价的其它帧 → 换帧在口径上是干净的'
                        if n_equiv > 0 else
                        '整段无帧与现行 GT 光度等价 → 选帧无从谈起'),
        }

        print(f"\n=== {sc}  n={len(rows)}  现行 GT ≈ 帧 {int(gt_frame)}")
        print(f"  与现行 GT 光度等价的帧: {n_equiv}/{len(rows)}"
              f" ({equiv.mean()*100:.0f}%)   其中在 GT 之前: {len(prev)}")
        print(f"  等价帧帧号(前10): {[int(x) for x in fno[equiv][:10]]}")
        print(f"  最优帧 {int(fno[best_i])}: |g-1|={dev[best_i]:.4f} spread={spr[best_i]:.4f}")
        print(f"  判决: {res[sc]['verdict']}")

        # 端点诊断：首帧与末帧对 GT 的 g
        print(f"  首帧 {int(fno[0])} g={rows[0]['g']}  |g-1|={rows[0]['dev']:.4f}")
        print(f"  末帧 {int(fno[-1])} g={rows[-1]['g']}  |g-1|={rows[-1]['dev']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out}')

    print('\n' + '=' * 96)
    print(f'{"场景":<24}{"现行GT帧":>9}{"等价帧数":>9}{"等价比例":>9}'
          f'{"GT前等价帧":>11}{"最小|g-1|":>11}  判决')
    print('-' * 96)
    for sc, r in res.items():
        print(f'{sc:<24}{r["gt_frame"]:>9}{r["n_photometric_equiv"]:>9}'
              f'{r["equiv_frac"]*100:>8.0f}%{r["equiv_before_gt"]:>11}'
              f'{r["min_dev_overall"]:>11.4f}  {r["verdict"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
