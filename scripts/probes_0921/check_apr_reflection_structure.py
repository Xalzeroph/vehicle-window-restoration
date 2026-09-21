#!/usr/bin/env python3
"""反射层结构含量核查（用户问：黑膜车窗的数据里"成结构的反射"真的存在吗？还是只有黑洞？）。

R = I − (s·T_lin)^(1/γ)·255（同 omega_of 口径）。量：
- R 亮度分布（p10/p50/p90、亮区%/暗区%）；R 的局部结构能量 stR（9×9 盒滤波 |Laplacian|）对比
  T 自身结构 stT（比值 = 反射的"纹理含量"）；成结构面积 = frac(stR>1)；成结构亮反射 = frac(R>6 & stR>1)。
- Ω 环带均值 R（洞周围反射亮度）vs Ω 内均值。
产物：控制台表 + fig/r_reflection_structure_0921.jpg（I | R×3 增强 | T）。
用法: python3 scripts/probes_0921/check_apr_reflection_structure.py
"""
import importlib.util, os
import cv2
import numpy as np

os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')
spec = importlib.util.spec_from_file_location('eg2', 'scripts/eval_g2_baseline.py')
eg2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg2)
G = eg2.G
SETS = {'apr': 'data/test_sets/real_pair_apr_0917', 'may': 'data/test_sets/real_pair_may_0917'}
PICK = {'apr': [('VID_20260426_173237', 'I_000000.png'), ('VID_20260426_173827', 'I_000000.png'),
                ('VID_20260426_172719', 'I_000000.png')],
        'may': [('VID_20260512_164653', 'I_000000.png'), ('VID_20260512_163859', 'I_000000.png')]}


def refl(I, T):
    Il = (np.clip(I, 0, 255) / 255.0) ** G
    Tl = (np.clip(T, 0, 255) / 255.0) ** G
    valid = Tl.mean(axis=2) > (10 / 255.) ** G
    ratio = np.where(valid[..., None], Il / np.where(Tl > 1e-8, Tl, np.nan), np.nan)
    s = float(np.nanquantile(ratio.mean(axis=2), 0.05))
    Tsc = (np.clip(s * Tl, 0, 1)) ** (1 / G) * 255.0
    return (I.astype(np.float64) - Tsc).mean(axis=2), s


def main():
    rows, panels = [], []
    for setname, d in SETS.items():
        print(f'== {setname} ==')
        print(f'{"帧":<26}{"R中位":>7}{"R_p90":>7}{"亮%":>6}{"暗%":>6}{"stR中位":>8}{"stT中位":>8}'
              f'{"stR/stT":>8}{"成结构%":>7}{"亮+结构%":>8}')
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            T = cv2.imread(f'{d}/{sc}/GT.png')
            for fn in sorted(f for f in os.listdir(f'{d}/{sc}') if f.startswith('I')):
                I = cv2.imread(f'{d}/{sc}/{fn}')
                R, s = refl(I, T)
                stR = cv2.boxFilter(np.abs(cv2.Laplacian(R.astype(np.float32), cv2.CV_32F)), -1, (9, 9))
                stT = cv2.boxFilter(np.abs(cv2.Laplacian(
                    cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)), -1, (9, 9))
                bright = float((R > 6).mean()) * 100
                dark = float((R < -6).mean()) * 100
                struct = float((stR > 1.0).mean()) * 100
                bs = float(((R > 6) & (stR > 1.0)).mean()) * 100
                mR, mT = float(np.median(stR)), float(np.median(stT))
                rows.append((setname, sc, fn, np.median(R), np.percentile(R, 90), bright, dark, mR, mT,
                             mR / max(mT, 1e-6), struct, bs))
                print(f'{sc[-6:]}/{fn[:-4]:<12} {np.median(R):>7.1f}{np.percentile(R,90):>7.1f}'
                      f'{bright:>6.1f}{dark:>6.1f}{mR:>8.2f}{mT:>8.2f}{mR/max(mT,1e-6):>8.2f}'
                      f'{struct:>7.1f}{bs:>8.1f}')
        for sc, fn in PICK[setname]:
            I = cv2.imread(f'{d}/{sc}/{fn}'); T = cv2.imread(f'{d}/{sc}/GT.png')
            R, _ = refl(I, T)
            rv = np.clip(128 + 3 * R, 0, 255).astype(np.uint8)
            rv = cv2.cvtColor(rv, cv2.COLOR_GRAY2BGR)
            def fit(x, h=340, w=460):
                y = cv2.resize(x, (max(1, int(x.shape[1] * h / x.shape[0])), h), interpolation=cv2.INTER_AREA)
                c = np.zeros((h, w, 3), np.uint8); c[:, :min(y.shape[1], w)] = y[:, :w]; return c
            row = np.hstack([fit(I), fit(rv), fit(T)])
            lab = np.full((36, row.shape[1], 3), 20, np.uint8)
            cv2.putText(lab, f'{setname} {sc[-6:]}/{fn[:-4]}   I | R(x3) | T', (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(np.vstack([lab, row]))
    gap = np.full((8, panels[0].shape[1], 3), 235, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.vstack([out, gap, p])
    cv2.imwrite('fig/r_reflection_structure_0921.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    a = lambda k: np.mean([r[k] for r in rows])
    apr = [r for r in rows if r[0] == 'apr']; may = [r for r in rows if r[0] == 'may']
    m = lambda rs, k: np.mean([r[k] for r in rs])
    print(f'\n汇总：apr 亮% {m(apr,5):.1f} 暗% {m(apr,6):.1f} stR/stT {m(apr,9):.2f} 成结构% {m(apr,10):.1f} 亮+结构% {m(apr,11):.1f}')
    print(f'      may 亮% {m(may,5):.1f} 暗% {m(may,6):.1f} stR/stT {m(may,9):.2f} 成结构% {m(may,10):.1f} 亮+结构% {m(may,11):.1f}')
    print('-> fig/r_reflection_structure_0921.jpg')


if __name__ == '__main__':
    main()
