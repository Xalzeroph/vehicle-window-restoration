#!/usr/bin/env python3
"""反射是"镜面"还是"漫反射(veiling glare)"？三个独立判据（2026-10-08）。

镜面: Ω 内保留源场景的边缘结构（清晰倒影）→ 正确操作=去结构
漫反射: Ω 内只有对比度损失、几乎无结构 → 正确操作=压暗+恢复对比度（等价去雾）

判据：
  A. 结构对比 —— 用 Sobel 边缘密度的**秩统计**比较 Ω 内 vs Ω 外
     （Ω 通常偏暗，绝对梯度会被亮度压低，故用秩/分位而非绝对值）
  B. 对比度压缩 —— Ω 内局部标准差 / Ω 外局部标准差（雾会压低它）
  C. 退化模型检验 —— 若为漫反射，则 Ω 内 luma ≈ α·luma_out + β（线性雾模型）
     报告拟合 R² 与残差结构；镜面反射应呈现"局部结构 + 双向偏差"
用法: python3 scripts/probe_diffuse_vs_specular.py --sets apr,may,jun
"""
import argparse, csv, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_g2_baseline import omega_of          # noqa: E402

SETS = {
    'apr': (os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917'), 'GT.png', 'I_'),
    'may': (os.path.join(PROJ, 'data/test_sets/real_pair_may_0917'), 'GT.png', 'I_'),
    'jun': (os.path.join(PROJ, 'data/test_sets/real_pair_0917'), 'T.png', 'I.png'),
}
K = 7          # 局部窗口


def local_std(gray):
    m = cv2.blur(gray, (K, K))
    m2 = cv2.blur(gray * gray, (K, K))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='apr,may,jun')
    ap.add_argument('--out', default='eval/probe_diffuse_vs_specular.csv')
    a = ap.parse_args()

    rows = []
    for tag in a.sets.split(','):
        if tag not in SETS:
            continue
        root, gtn, ipre = SETS[tag]
        for sc in sorted(x for x in os.listdir(root) if x.startswith('VID_')):
            sd = os.path.join(root, sc)
            T = cv2.imread(os.path.join(sd, gtn))
            if T is None:
                continue
            iframes = (sorted(f for f in os.listdir(sd) if f.startswith(ipre))
                       if ipre != 'I.png' else ['I.png'])
            for fn in iframes:
                I = cv2.imread(os.path.join(sd, fn))
                if I is None or I.shape != T.shape:
                    continue
                key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
                o = omega_of(I, T)
                if o is None or not o['Om'].any():
                    continue
                Om = o['Om']
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
                ring = (cv2.dilate(Om.astype(np.uint8), ker) > 0) & (~Om)

                gI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
                gT = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
                # --- A. 结构：Sobel 幅值的秩统计 ---
                # 注意：OpenCV 5.0 的 Sobel 只接受 float32（float64 报 buffer format 错）
                gI32 = gI.astype(np.float32)
                sx = cv2.Sobel(gI32, cv2.CV_32F, 1, 0); sy = cv2.Sobel(gI32, cv2.CV_32F, 0, 1)
                mag = np.sqrt(sx * sx + sy * sy)
                rank = np.argsort(np.argsort(mag, axis=None)).reshape(mag.shape) / mag.size
                r_in, r_out = rank[Om].mean(), rank[ring].mean()
                # --- B. 对比度：局部 std 中位 ---
                ls = local_std(gI)
                ls_in, ls_out = np.median(ls[Om]), np.median(ls[ring])
                # --- C. 线性雾模型：I_om ≈ α·I_ring + β ---
                x, y = gI[Om], gI[ring]
                n = min(len(x), len(y))
                x, y = x[:n], y[:n]
                if n > 50:
                    A = np.stack([y, np.ones_like(y)], 1)
                    c, *_ = np.linalg.lstsq(A, x, rcond=None)
                    pred = c[0] * y + c[1]
                    ss = ((x - x.mean()) ** 2).sum()
                    r2 = 1 - ((x - pred) ** 2).sum() / ss if ss > 0 else float('nan')
                else:
                    r2 = float('nan')
                # GT 侧同样做一遍（真值里 Ω 应已"去反射"）
                rows.append(dict(set=tag, case=key, omega_pct=round(100 * float(Om.mean()), 1),
                                 rank_in=round(float(r_in), 4), rank_out=round(float(r_out), 4),
                                 struct_ratio=round(float(r_in / max(r_out, 1e-9)), 3),
                                 lstd_in=round(float(ls_in), 2), lstd_out=round(float(ls_out), 2),
                                 contrast_ratio=round(float(ls_in / max(ls_out, 1e-9)), 3),
                                 haze_alpha=round(float(c[0]), 3) if n > 50 else '',
                                 haze_beta=round(float(c[1]), 2) if n > 50 else '',
                                 haze_r2=round(float(r2), 3),
                                 luma_in=round(float(gI[Om].mean()), 1),
                                 luma_out=round(float(gI[ring].mean()), 1)))

    os.makedirs(os.path.dirname(os.path.join(PROJ, a.out)), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(PROJ, a.out), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    print(f'写出 {a.out}   n={len(rows)} 帧\n')
    for tag in a.sets.split(','):
        sub = [r for r in rows if r['set'] == tag]
        if not sub:
            continue
        sr = np.array([r['struct_ratio'] for r in sub], float)
        cr = np.array([r['contrast_ratio'] for r in sub], float)
        h2 = np.array([r['haze_r2'] for r in sub], float)
        ha = np.array([r['haze_alpha'] for r in sub], float)
        print(f'=== {tag}  n={len(sub)} ===')
        print(f'  A 结构比 rank(Ω内)/rank(Ω外)  中位 {np.median(sr):.3f}   '
              f'<1 表示 Ω 内结构更少  [{np.percentile(sr,10):.2f}, {np.percentile(sr,90):.2f}]')
        print(f'  B 对比度比 lstd(Ω内)/lstd(Ω外) 中位 {np.median(cr):.3f}   '
              f'<1 表示 Ω 内更平  [{np.percentile(cr,10):.2f}, {np.percentile(cr,90):.2f}]')
        print(f'  C 线性雾模型 R²               中位 {np.nanmedian(h2):.3f}   '
              f'α 中位 {np.nanmedian(ha):.3f}')
        print()
    print('判读（三判据独立）：')
    print('  镜面为主 → A 接近或>1（Ω 内有清晰倒影结构），B 接近 1')
    print('  漫反射为主 → A 明显<1、B 明显<1、C 的 R² 偏高且 α<1（对比度被压缩）')


if __name__ == '__main__':
    main()
