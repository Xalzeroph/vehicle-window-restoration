#!/usr/bin/env python3
"""反射是"镜面"还是"漫反射"？—— 用【用户标注 ROI】而非 omega_of 残差掩膜（2026-10-08）。

修正说明：前一版用 omega_of(I,T) 的 Ω 定义"洞"，但 Ω = |I−gT|>3σ 本身就是
"差异大"的区域，拿它测结构是循环论证（且与用户所指"洞" IoU 仅 0.23）。
本版改用 data/annot_hole_cells.json 的**用户格号**（5×5 格）。

三判据（对用户 ROI 内 vs ROI 外邻域）：
  A 结构比   rank_sobel(ROI内)/rank(ROI外)   镜面→≥1（倒影有结构）；漫反射→<1
  B 对比度比 lstd(ROI内)/lstd(ROI外)          漫反射→<1（被杂散光压平）
  C 雾模型   I_in ≈ α·I_out + β 的 R²          漫反射→R² 高（全局线性压暗）
用法: python3 scripts/probe_diffuse_roi.py
"""
import csv, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
N = 5
K = 7

SETS = {
    'apr': (os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917'), 'GT.png'),
    'may': (os.path.join(PROJ, 'data/test_sets/real_pair_may_0917'), 'GT.png'),
    'jun': (os.path.join(PROJ, 'data/test_sets/real_pair_0917'), 'T.png'),
}


def cellmask(shape, cells):
    H, W = shape
    m = np.zeros((H, W), bool)
    for c in cells:
        col, row = ord(c[0]) - 65, int(c[1]) - 1
        m[row * H // N:(row + 1) * H // N, col * W // N:(col + 1) * W // N] = True
    return m


def local_std(g):
    m = cv2.blur(g.astype(np.float32), (K, K))
    m2 = cv2.blur((g * g).astype(np.float32), (K, K))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def main():
    J = json.load(open(f'{PROJ}/data/annot_hole_cells.json', encoding='utf-8'))
    rows = []
    for k, v in J.items():
        cells = v.get('pos') or []
        if not cells:
            continue
        dom, rest = k.split('/', 1)
        if dom not in SETS:
            continue
        root, gtn = SETS[dom]
        sc = rest.split('__')[0]
        T = cv2.imread(os.path.join(root, sc, gtn))
        if T is None:
            continue
        # key 形如 apr/VID_20260426_172719__000000（注意：帧号无 I_ 前缀）
        # 磁盘文件名 = I_<帧号>.png；jun 为 I.png
        if dom == 'jun':
            fn = 'I.png'
        else:
            num = rest.split('__', 1)[1]
            fn = (num if num.startswith('I_') else 'I_' + num) + '.png'
        I = cv2.imread(os.path.join(root, sc, fn))
        if I is None:
            print(f'  [警告] 读不到 {os.path.join(root, sc, fn)}')
        if I is None or I.shape != T.shape:
            continue
        H, W = I.shape[:2]
        Om = cellmask((H, W), cells)
        # ROI 外邻域 = 膨胀 3 格再减去 ROI，避免边界效应
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * (W // N) + 1, 2 * (H // N) + 1))
        ring = (cv2.dilate(Om.astype(np.uint8), ker) > 0) & (~Om)

        gI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gI32 = gI.astype(np.float32)
        sx = cv2.Sobel(gI32, cv2.CV_32F, 1, 0)
        sy = cv2.Sobel(gI32, cv2.CV_32F, 0, 1)
        mag = np.sqrt(sx * sx + sy * sy)
        rank = np.argsort(np.argsort(mag, axis=None)).reshape(mag.shape) / mag.size
        r_in, r_out = float(rank[Om].mean()), float(rank[ring].mean())
        ls = local_std(gI32)
        ls_in, ls_out = float(np.median(ls[Om])), float(np.median(ls[ring]))
        x, y = gI[Om], gI[ring]
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        if n > 50:
            A = np.stack([y, np.ones_like(y)], 1)
            c, *_ = np.linalg.lstsq(A, x, rcond=None)
            ss = ((x - x.mean()) ** 2).sum()
            r2 = float(1 - ((x - c[0] * y - c[1]) ** 2).sum() / ss) if ss > 0 else float('nan')
        else:
            c = [float('nan')] * 2; r2 = float('nan')

        rows.append(dict(set=dom, case=rest, n_cells=len(cells),
                         struct_ratio=round(r_in / max(r_out, 1e-9), 3),
                         contrast_ratio=round(ls_in / max(ls_out, 1e-9), 3),
                         haze_alpha=round(float(c[0]), 3), haze_r2=round(r2, 3),
                         luma_in=round(float(gI[Om].mean()), 1),
                         luma_out=round(float(gI[ring].mean()), 1),
                         gt_luma_in=round(float(cv2.cvtColor(T, cv2.COLOR_BGR2GRAY)[Om].mean()), 1)))

    out = os.path.join(PROJ, 'eval/probe_diffuse_roi.csv')
    keys = sorted({k for r in rows for k in r})
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f'写出 eval/probe_diffuse_roi.csv   n={len(rows)} 帧（用户标注 ROI 口径）\n')

    for dom in SETS:
        sub = [r for r in rows if r['set'] == dom]
        if not sub:
            continue
        sr = np.array([r['struct_ratio'] for r in sub])
        cr = np.array([r['contrast_ratio'] for r in sub])
        h2 = np.array([r['haze_r2'] for r in sub])
        li = np.array([r['luma_in'] for r in sub])
        lo = np.array([r['luma_out'] for r in sub])
        print(f'=== {dom}  n={len(sub)} ===')
        print(f'  A 结构比   中位 {np.median(sr):.3f}  [{sr.min():.2f}, {sr.max():.2f}]')
        print(f'  B 对比度比 中位 {np.median(cr):.3f}  [{cr.min():.2f}, {cr.max():.2f}]')
        print(f'  C 雾模型R² 中位 {np.nanmedian(h2):.3f}')
        print(f'  ROI 内/外 luma: {np.median(li):.1f} / {np.median(lo):.1f}'
              f'   (Δ={np.median(li-lo):+.1f})')
        print(f'  逐帧 A: ' + ' '.join(f'{x:.2f}' for x in sr))
        print()
    print('判读：')
    print('  镜面为主 → A ≥ 1（ROI 内有倒影结构）')
    print('  漫反射为主 → A 明显<1 且 B 明显<1 且 C 的 R² 高')
    print('  若 A≈1 且 B≈1 → ROI 内既没少结构、也没被压平 → 不像漫反射')


if __name__ == '__main__':
    main()
