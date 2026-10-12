#!/usr/bin/env python3
"""apr 选帧扫描（2026-10-12）

交接页 §8 待办 1：扫 359 帧找"作 GT 时 |g-1| 与通道极差同时最小"的帧。
这是唯一可能"不重拍就救回 apr"的实验。

问题背景
--------
四月 5 个场景的 `label.txt` 只有一行裸帧号，且该帧号总是该场景**最后一帧**
（875/400/475/625/725/1025）。而 I 帧全取自 0-28。两者相隔 400-1000 帧。
若相机的 AE/AWB 在这段区间内漂移，GT 与 I 就落在不同的光度状态上，
downstream 的对齐（g）会把"相机漂移"误当成"反射带来的增益"。

判据
----
场景静止（三脚架/手持静止、黑布覆盖）→ 帧间差异**只有光度**。
因此对任意两帧 i,j，逐通道稳健增益 g(f_i, f_j) 量的就是**相机光度状态之比**。
以**输入 I 帧的光度状态**为锚：
    g_j = g(I, f_j)            越接近 1 → f_j 与 I 的光度状态越一致
    spread_j = max(g_j) - min(g_j)   越小 → 白平衡越一致

对每个场景报告：
    - 现行 GT（末帧）的 g 与 spread
    - 全帧扫描的最优帧（先按 |g-1|，再按 spread）
    - 黑布段的起点/终点（用同场景帧间差异突变检测）

注意：本脚本只报告**光度一致性**，不改变物理结论。
交接页 §4 已确认「遮挡改透射照明」不可修；本扫描只回答
"在已有帧里能否挑到一个光度状态更接近 I 的 GT"。
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
APR_SETS = os.path.join(PROJ, 'data', 'test_sets', 'real_pair_apr_0917')


def frame_num(path: str) -> int:
    m = re.search(r'(\d{6})', os.path.basename(path))
    return int(m.group(1)) if m else -1


def read_rgb(path: str):
    """cv2 读入 BGR -> 返回 BGR uint8（与 robust_gain 口径一致，通道顺序不影响结论）。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def luma(img) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PROJ, 'eval', 'apr_gt_rescan_1012.json'))
    ap.add_argument('--max-frames', type=int, default=0,
                    help='每场景最多扫描多少帧（0=全部）')
    ap.add_argument('--stride', type=int, default=1, help='帧抽样步长')
    args = ap.parse_args()

    scenes = sorted(x for x in os.listdir(APR_SETS) if x.startswith('VID_'))
    print(f'apr 场景数: {len(scenes)}')
    results = {}

    for sc in scenes:
        sd = os.path.join(APR_SETS, sc)
        meta = {}
        mp = os.path.join(sd, 'meta.json')
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding='utf-8'))

        gt_cur = read_rgb(os.path.join(sd, 'GT.png'))
        if gt_cur is None:
            print(f'{sc}: GT.png 读不到，跳过')
            continue

        # I 帧（输入）：取该场景全部 I_*.png
        i_files = sorted((f for f in os.listdir(sd) if f.startswith('I_')),
                         key=frame_num)
        i_imgs = [(frame_num(f), read_rgb(os.path.join(sd, f))) for f in i_files]
        i_imgs = [(n, im) for n, im in i_imgs if im is not None]
        if not i_imgs:
            print(f'{sc}: 无 I 帧，跳过')
            continue

        # 候选池：裁剪帧目录里的全部帧
        cdir = os.path.join(CROPPED, sc)
        cand = []
        if os.path.isdir(cdir):
            cand = sorted((f for f in os.listdir(cdir) if f.endswith('.jpg')), key=frame_num)
        if args.max_frames:
            cand = cand[:args.max_frames]
        cand = cand[::args.stride]

        print(f'\n=== {sc}  I帧 {[n for n, _ in i_imgs]}  候选 {len(cand)} 帧'
              f'  meta_keys={list(meta.keys())}')

        # 用第一个 I 帧作为光度锚（与"输入"同一时刻）
        anchor_n, anchor = i_imgs[0]

        rows = []
        for fn in cand:
            img = read_rgb(os.path.join(cdir, fn))
            if img is None:
                continue
            if img.shape != gt_cur.shape:
                img = cv2.resize(img, (gt_cur.shape[1], gt_cur.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            g = np.atleast_1d(robust_gain(img, anchor)).astype(float)
            rows.append({
                'frame': frame_num(fn),
                'g': [round(float(x), 4) for x in g],
                'absdev': round(float(np.abs(g - 1).mean()), 4),
                'spread': round(float(g.max() - g.min()), 4),
                'luma': round(luma(img), 2),
            })

        if not rows:
            print('  候选帧全读失败')
            continue

        dev = np.array([r['absdev'] for r in rows])
        spr = np.array([r['spread'] for r in rows])
        lum = np.array([r['luma'] for r in rows])
        fno = np.array([r['frame'] for r in rows])

        # 现行 GT 的位置与指标（GT 是否就是最后一帧？）
        g_gt = np.atleast_1d(robust_gain(gt_cur, anchor)).astype(float)
        cur = {
            'g': [round(float(x), 4) for x in g_gt],
            'absdev': round(float(np.abs(g_gt - 1).mean()), 4),
            'spread': round(float(g_gt.max() - g_gt.min()), 4),
            'luma': round(luma(gt_cur), 2),
        }
        last = rows[-1]
        print(f'  现行 GT : |g-1|={cur["absdev"]:.4f}  spread={cur["spread"]:.4f} '
              f'luma={cur["luma"]:.1f}')
        print(f'  末候选帧: frame={last["frame"]:06d} |g-1|={last["absdev"]:.4f} '
              f'spread={last["spread"]:.4f} luma={last["luma"]:.1f}'
              f'   <-- {"与现行 GT 一致" if abs(last["luma"] - cur["luma"]) < 0.5 else "与现行 GT 不同！"}')

        # 最优：先 |g-1| 最小，再 spread 最小
        order = np.lexsort((spr, dev))
        best = rows[int(order[0])]
        print(f'  最优候选: frame={best["frame"]:06d} |g-1|={best["absdev"]:.4f} '
              f'spread={best["spread"]:.4f} luma={best["luma"]:.1f}   '
              f'（改善 |g-1| {cur["absdev"]:.4f} -> {best["absdev"]:.4f}）')

        # 光度轨迹：|g-1| 随帧号的单调性（是否"漂移到末端"）
        if fno.std() > 0 and dev.std() > 0:
            r_dev = float(np.corrcoef(fno, dev)[0, 1])
        else:
            r_dev = float('nan')
        # 帧间光度抖动：相邻候选帧 luma 差的绝对值中位
        dl = float(np.median(np.abs(np.diff(lum)))) if len(lum) > 1 else float('nan')
        print(f'  corr(帧号, |g-1|)={r_dev:+.3f}   相邻帧 luma 差中位={dl:.2f}   '
              f'luma 范围=[{lum.min():.1f}, {lum.max():.1f}]')

        results[sc] = {
            'i_frames': [n for n, _ in i_imgs],
            'anchor_frame': anchor_n,
            'current_gt': cur,
            'last_candidate': last,
            'best_candidate': best,
            'corr_frame_absdev': None if np.isnan(r_dev) else round(r_dev, 4),
            'median_abs_dluma': None if np.isnan(dl) else round(dl, 3),
            'luma_range': [round(float(lum.min()), 2), round(float(lum.max()), 2)],
            'n_candidates': len(rows),
            'curve': rows,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out}')

    # 汇总表
    print('\n' + '=' * 100)
    print(f'{"场景":<24}{"现行|g-1|":>10}{"现行spread":>11}{"最优帧":>9}'
          f'{"最优|g-1|":>10}{"最优spread":>11}{"corr(帧,|g-1|)":>16}')
    print('-' * 100)
    for sc, r in results.items():
        c, b = r['current_gt'], r['best_candidate']
        print(f'{sc:<24}{c["absdev"]:>10.4f}{c["spread"]:>11.4f}'
              f'{b["frame"]:>9d}{b["absdev"]:>10.4f}{b["spread"]:>11.4f}'
              f'{(r["corr_frame_absdev"] if r["corr_frame_absdev"] is not None else float("nan")):>16.3f}')
    n_imp = sum(1 for r in results.values()
                if r['best_candidate']['absdev'] < r['current_gt']['absdev'] - 0.01)
    print('-' * 100)
    print(f'有实质改善(>0.01)的场景: {n_imp}/{len(results)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
