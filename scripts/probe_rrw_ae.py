#!/usr/bin/env python3
"""RRW 数据的 AE 漂移诊断 —— 与车窗数据同口径对比（2026-10-08）。

问题：RRW 也是手机视频模式拍的，为什么它的色调混杂没被当作问题？
做法：对 RRW 的每个 case，抽若干视频帧 vs 该 case 的 GT.png，
      算逐帧 robust_gain，看：
        - 同一 case 内 g 的极差/std（帧间漂移）
        - corr(g, 帧号)（是否 AE 追赶）
        - corr(g, luma)（是否随画面亮度）
      再与车窗 apr/may 的数字并排。

用法: python3 scripts/probe_rrw_ae.py --root /DataDisk/zmy_code/datasets/RRWDataset --stride 25
"""
import argparse, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_protocol import robust_gain          # noqa: E402


def imread_any(p):
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/DataDisk/zmy_code/datasets/RRWDataset')
    ap.add_argument('--stride', type=int, default=25)
    ap.add_argument('--max-cases', type=int, default=12)
    a = ap.parse_args()

    rows = []
    subs = [d for d in sorted(os.listdir(a.root)) if d.startswith('reflection_')]
    print(f'发现子集: {subs}\n')
    for sub in subs:
        sroot = os.path.join(a.root, sub)
        cases = sorted(d for d in os.listdir(sroot)
                       if os.path.isdir(os.path.join(sroot, d)) and not d.startswith('GT'))
        for cs in cases[:a.max_cases]:
            cd = os.path.join(sroot, cs)
            # GT 在 <sub>/GT_<suffix>/<case>_GT.png，不在子集根目录
            gt_dir = None
            for d in sorted(os.listdir(sroot)):
                if d.startswith('GT') and os.path.isdir(os.path.join(sroot, d)):
                    gt_dir = os.path.join(sroot, d)
                    break
            gtp = None
            for cand in ([os.path.join(gt_dir, f'{cs}_GT.png')] if gt_dir else []) + \
                        [os.path.join(sroot, f'{cs}_GT.png')]:
                if os.path.exists(cand):
                    gtp = cand
                    break
            if gtp is None:
                continue
            T = imread_any(gtp)
            if T is None:
                continue
            fs = sorted(f for f in os.listdir(cd) if f.lower().endswith(('.jpg', '.png')))
            if len(fs) < 3:
                continue
            pick = fs[::a.stride]
            if pick[-1] != fs[-1]:
                pick.append(fs[-1])
            gs, lumas, idx = [], [], []
            for i, fn in enumerate(pick):
                I = imread_any(os.path.join(cd, fn))
                if I is None:
                    continue
                if I.shape != T.shape:
                    I = cv2.resize(I, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_AREA)
                g = float(np.atleast_1d(robust_gain(I, T)).mean())
                gs.append(g)
                lumas.append(float(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).mean()))
                idx.append(i)
            if len(gs) < 3:
                continue
            gs = np.array(gs); lumas = np.array(lumas); idx = np.array(idx, float)
            rng = float(gs.max() - gs.min())
            cg = float(np.corrcoef(idx, gs)[0, 1]) if gs.std() > 0 else float('nan')
            cl = float(np.corrcoef(lumas, gs)[0, 1]) if gs.std() > 0 and lumas.std() > 0 else float('nan')
            rows.append(dict(sub=sub, case=cs, n_frame=len(fs), n_sampled=len(gs),
                             g_med=round(float(np.median(gs)), 4), g_range=round(rng, 4),
                             g_std=round(float(gs.std()), 4),
                             corr_idx=round(cg, 3), corr_luma=round(cl, 3),
                             abs_g1=round(float(np.median(np.abs(gs - 1))), 4)))
            print(f'  {sub}/{cs:<22} 帧{len(fs):>4} 抽{len(gs):>3} '
                  f'g中位 {np.median(gs):.3f} 极差 {rng:.4f} std {gs.std():.4f} '
                  f'corr(帧号) {cg:+.2f} corr(luma) {cl:+.2f}')

    if not rows:
        print('没测到任何 case'); return
    print('\n=== RRW 汇总 ===')
    rg = np.array([r['g_range'] for r in rows])
    st = np.array([r['g_std'] for r in rows])
    a1 = np.array([r['abs_g1'] for r in rows])
    print(f'  case 数 {len(rows)}')
    print(f'  |g-1| 中位      : 中位 {np.median(a1):.4f}  超0.03的 {int((a1>0.03).sum())}/{len(a1)}')
    print(f'  同 case 内 g 极差: 中位 {np.median(rg):.4f}  max {rg.max():.4f}')
    print(f'  同 case 内 g std : 中位 {np.median(st):.4f}  max {st.max():.4f}')
    print()
    print('  对照 车窗数据: |g-1| 中位 apr 0.1749 / may 0.0792；')
    print('               同场景内 g 极差 最大 0.225（may/162901）')


if __name__ == '__main__':
    main()
