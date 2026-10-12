#!/usr/bin/env python3
"""apr 选帧扫描 —— 判决版（2026-10-12）

上一版 scan_apr_gt_1012.py 的"最优帧"必然是第一张 I 帧（自己和自己比 g≡1），
是循环论证。本脚本用**无循环**的口径给出判决：

对每个 apr 场景，以 input.png 侧的光度（全部 I 帧均值）为锚，
沿裁剪帧目录逐帧算 g(f, anchor)，直接看：
  1. GT 候选窗口（现行 label.txt 帧号附近）内 |g-1| 能否被别帧压到阈值以下？
  2. GT 候选窗口的**宽度**——即是否存在一段"与 I 光度状态一致"的帧区间。
     若窗口很宽（>=100 帧）→ 选帧自由度高，但现行 GT 已在窗口内 → **无可救**。
     若窗口很窄且现行 GT 在窗外 → 选帧才有意义。
  3. 光度扰动的**位置**：|g-1| 在哪些帧区间超出阈值。

阈值沿用交接页 §5.3 建议：段内 |g-1| <= 0.05 视为 AE 一致。
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
THR = 0.05


def fnum(name: str) -> int:
    m = re.search(r'(\d{6})', os.path.basename(name))
    return int(m.group(1)) if m else -1


def load(path: str, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = img.astype(np.float64)
    if size is not None and img.shape[:2] != size:
        img = cv2.resize(img.astype(np.uint8), (size[1], size[0]),
                         interpolation=cv2.INTER_AREA).astype(np.float64)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PROJ, 'eval', 'apr_gt_verdict_1012.json'))
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    scenes = sorted(x for x in os.listdir(APR) if x.startswith('VID_'))
    out = {}

    for sc in scenes:
        # 锚：全部 I 帧的**逐像素均值**（比取单帧稳，且不偏向任何一帧）
        idir = os.path.join(APR, sc)
        ifiles = sorted((f for f in os.listdir(idir) if f.startswith('I_')), key=fnum)
        imgs = [load(os.path.join(idir, f)) for f in ifiles]
        imgs = [x for x in imgs if x is not None]
        if not imgs:
            continue
        ref_shape = imgs[0].shape[:2]
        anchor = np.mean(np.stack([i if i.shape[:2] == ref_shape else
                                   cv2.resize(i.astype(np.uint8), (ref_shape[1], ref_shape[0]),
                                              interpolation=cv2.INTER_AREA).astype(np.float64)
                                   for i in imgs]), axis=0)
        i_nums = [fnum(f) for f in ifiles]

        # 现行 GT 帧号（label.txt 里的裸帧号）
        gtn = None
        lf = os.path.join(CROPPED, sc, 'label.txt')
        if os.path.exists(lf):
            for line in open(lf, encoding='utf-8'):
                s = line.strip()
                if re.fullmatch(r'\d{6}', s):
                    gtn = int(s)
        # 也读 meta.json 的 gt_frame
        mp = os.path.join(idir, 'meta.json')
        meta_gt = None
        if os.path.exists(mp):
            meta_gt = json.load(open(mp, encoding='utf-8')).get('gt_frame')

        cdir = os.path.join(CROPPED, sc)
        files = sorted((f for f in os.listdir(cdir) if f.endswith('.jpg')), key=fnum)
        files = files[::args.stride]

        rows = []
        for f in files:
            im = load(os.path.join(cdir, f), ref_shape)
            if im is None:
                continue
            g = np.atleast_1d(robust_gain(im, anchor)).astype(float)
            rows.append({'frame': fnum(f), 'absdev': float(np.abs(g - 1).mean()),
                         'spread': float(g.max() - g.min())})
        if not rows:
            continue
        fno = np.array([r['frame'] for r in rows], dtype=float)
        dev = np.array([r['absdev'] for r in rows])

        ok = dev <= THR
        # 最长连续 OK 窗口
        best = (0, None, None)
        i = 0
        while i < len(ok):
            if ok[i]:
                j = i
                while j + 1 < len(ok) and ok[j + 1]:
                    j += 1
                # 用帧号跨度衡量
                span = fno[j] - fno[i]
                if span > best[0]:
                    best = (span, int(fno[i]), int(fno[j]))
                i = j + 1
            else:
                i += 1

        # 现行 GT 所在帧的 |g-1|
        gt_dev = None
        if gtn is not None:
            k = np.where(fno == gtn)[0]
            if len(k):
                gt_dev = float(dev[k[0]])
        # 最接近 GT 的采样点
        if gtn is not None and gt_dev is None:
            k = int(np.argmin(np.abs(fno - gtn)))
            gt_dev = float(dev[k])
            gt_at = int(fno[k])
        else:
            gt_at = gtn

        bad = ~ok
        out[sc] = {
            'i_frames': i_nums,
            'label_gt_frame': gtn,
            'meta_gt_frame': meta_gt,
            'gt_absdev': gt_dev,
            'gt_passed': None if gt_dev is None else bool(gt_dev <= THR),
            'n_frames': len(rows),
            'n_pass': int(ok.sum()),
            'pass_frac': round(float(ok.mean()), 3),
            'longest_consistent_window': {'span_frames': int(best[0]),
                                          'from': best[1], 'to': best[2]},
            'first_bad_frame': int(fno[bad][0]) if bad.any() else None,
            'last_bad_frame': int(fno[bad][-1]) if bad.any() else None,
            'max_absdev': round(float(dev.max()), 4),
            'max_absdev_frame': int(fno[int(np.argmax(dev))]),
            'verdict': ('GT 本身已在一致窗口内 → 换帧救不了'
                        if (gt_dev is not None and gt_dev <= THR)
                        else 'GT 不在一致窗口内'),
        }

        print(f"\n=== {sc}  I帧 {i_nums}  n={len(rows)}")
        print(f"  label GT 帧 {gtn}  meta gt_frame {meta_gt}  采样点 {gt_at}")
        print(f"  现行 GT |g-1| = {gt_dev:.4f}  {'PASS' if gt_dev <= THR else 'FAIL'} (阈 {THR})")
        print(f"  一致帧比例 {ok.mean()*100:.0f}%  最长一致窗口 跨度 {best[0]} 帧"
              f"  [{best[1]} .. {best[2]}]")
        print(f"  |g-1| 最大 {dev.max():.4f} @帧 {int(fno[int(np.argmax(dev))])}"
              f"   首次超阈 @{int(fno[bad][0]) if bad.any() else '-'}"
              f"   末次超阈 @{int(fno[bad][-1]) if bad.any() else '-'}")
        print(f"  判决: {out[sc]['verdict']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out}')

    print('\n' + '=' * 104)
    print(f'{"场景":<24}{"GT帧":>7}{"GT|g-1|":>9}{"判决":>6}{"一致比例":>9}'
          f'{"最长一致窗(帧)":>15}{"最大|g-1|":>10}{"@帧":>7}')
    print('-' * 104)
    for sc, r in out.items():
        w = r['longest_consistent_window']
        print(f'{sc:<24}{r["label_gt_frame"]:>7}{r["gt_absdev"]:>9.4f}'
              f'{"PASS" if r["gt_passed"] else "FAIL":>6}{r["pass_frac"]*100:>8.0f}%'
              f'{w["span_frames"]:>15}{r["max_absdev"]:>10.4f}{r["max_absdev_frame"]:>7}')
    n_pass = sum(1 for r in out.values() if r['gt_passed'])
    print('-' * 104)
    print(f'现行 GT 光度一致性 PASS: {n_pass}/{len(out)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
