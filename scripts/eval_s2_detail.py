#!/usr/bin/env python3
"""S2 修正版第⑤条验收：细节保留率（Laplacian 能量比 P/T）+ 透射区保真 |P−I|。

用法: python3 scripts/eval_s2_detail.py <pred_root> [tag]
  pred_root 下为 <key>/ytmt_ucs_sirs_l.png（如 runs/pred/may_s2v2it1000）
输出: 逐场景明细 + 汇总（⑤ 通过线：全图 ≥0.90 且 ≥4/5 场景、点名三场景必达；透射区 |P−I| ≤5/255
（2026-09-22 重锚：地板实测 4.7/2.6/3.6））
"""
import os, sys
import cv2
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(PROJ, 'data/test_sets/real_pair_may_0917')
MUST = ('164653', '163859', '162203')


def lap_e(x):
    return np.abs(cv2.Laplacian(x, cv2.CV_32F)).mean()


def main():
    root = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ''
    rows = []
    for sc in sorted(x for x in os.listdir(D) if x.startswith('VID_')):
        T = cv2.imread(os.path.join(D, sc, 'GT.png'))
        Tg = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for fn in sorted(f for f in os.listdir(os.path.join(D, sc)) if f.startswith('I_')):
            key = f'{sc}__{fn[:-4]}'
            P = cv2.imread(os.path.join(root, key, 'ytmt_ucs_sirs_l.png'))
            if P is None:
                continue
            I = cv2.imread(os.path.join(D, sc, fn))
            Pg = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY).astype(np.float32)
            Dm = np.abs(I.astype(np.float64) - T.astype(np.float64)).mean(2)
            tran = Dm <= np.percentile(Dm, 20)
            rows.append((sc[-6:], fn[:-4], lap_e(Pg) / max(lap_e(Tg), 1e-6),
                         float(np.abs(P.astype(np.float64) - I.astype(np.float64)).mean(2)[tran].mean())))
    print(f'== {tag} 第⑤条（may）：细节保留率 / 透射区 |P−I| ==')
    for s, f, r, e in rows:
        mark = ' ★' if s in MUST else ''
        print(f'  {s}__{f:<8} 细节保留 {r:.3f}  透射区|P−I| {e:5.2f}{mark}')
    if not rows:
        return
    r = np.array([x[2] for x in rows]); e = np.array([x[3] for x in rows])
    n_ok = int((r >= 0.90).sum())
    must = {s: rv for s, f, rv, ev in rows if s in MUST}
    print(f'汇总: 细节保留中位 {np.median(r):.3f}、≥0.90 的 {n_ok}/{len(rows)} 行'
          f'（线：≥4/5 场景）| 点名场景 ' + ' '.join(f'{k}:{v:.2f}' for k, v in must.items()))
    print(f'      透射区 |P−I| 中位 {np.median(e):.2f}/255（线 ≤5；2026-09-22 重锚）')


if __name__ == '__main__':
    main()
