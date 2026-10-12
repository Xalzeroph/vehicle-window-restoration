#!/usr/bin/env python3
"""核对两条 April 测量是否自洽（2026-10-12）

scan_apr_gt_1012.py 说 corr(帧号, |g-1|) 为正（漂移随时间增大），
diag_apr_drift_1012.py 说 Spearman(帧号, luma) 为负（luma 随时间下降）。
两者都在量同一条光度轨迹，符号必须一致。本脚本打印原始曲线定位矛盾。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

CURVE = os.path.join(PROJ, 'eval', 'apr_gt_rescan_1012.json')


def main() -> int:
    d = json.load(open(CURVE, encoding='utf-8'))
    for sc in ['VID_20260426_173827', 'VID_20260426_172719']:
        r = d.get(sc)
        if not r:
            continue
        cur = r['curve']
        print(f'\n=== {sc}  anchor_frame={r["anchor_frame"]}  I帧={r["i_frames"]}')
        print(f'  现行GT |g-1|={r["current_gt"]["absdev"]} luma={r["current_gt"]["luma"]}')
        print(f'{"frame":>8}{"absdev":>9}{"spread":>9}{"luma":>9}   g[B,G,R]')
        for row in cur:
            print(f'{row["frame"]:>8}{row["absdev"]:>9.4f}{row["spread"]:>9.4f}'
                  f'{row["luma"]:>9.2f}   {row["g"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
