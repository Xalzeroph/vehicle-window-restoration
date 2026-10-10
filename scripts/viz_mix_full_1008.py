#!/usr/bin/env python3
"""整帧对比图：混训 vs 单域 vs 对照（2026-10-08）。

不裁剪，整帧显示；各帧原始尺寸不同（最大 1078×1757），统一缩放到同一高度再横向排列。
每帧按其原始长宽比缩放，列内对齐；行 = 该域内 Δρ 最差/中位/最好。
Ω 轮廓（红）画在 INPUT 上（缩放后重画，保证位置正确）。

用法: python3 scripts/viz_mix_full_1008.py
"""
import csv, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_g2_baseline import omega_of          # noqa: E402

OUT = os.path.join(PROJ, 'fig/mix_1008')
ROW_H = 300                                    # 每行图像的显示高度
INFO_W = 200

DOMS = {
    'may': ('b1_apr_may', 'm3_aprjun_may', 'm6_aprjun_gm'),
    'apr': ('b2_may_apr', 'm4_mayjun_apr', None),
    'jun': (None, 'm5_aprmay_jun', None),
}
SETDIR = {'may': 'data/test_sets/real_pair_may_0917',
          'apr': 'data/test_sets/real_pair_apr_0917',
          'jun': 'data/test_sets/real_pair_0917'}
GTNAME = {'may': 'GT.png', 'apr': 'GT.png', 'jun': 'T.png'}


def load(f):
    return [r for r in csv.DictReader(open(f, encoding='utf-8-sig')) if r.get('case')]


def fnum(r, k):
    try:
        return float(r.get(k, ''))
    except (TypeError, ValueError):
        return float('nan')


def get_I_T(dom, case):
    root = os.path.join(PROJ, SETDIR[dom])
    if '__' in case:
        sc, fn = case.split('__', 1)
        I = cv2.imread(os.path.join(root, sc, fn + '.png'))
    else:
        sc = case
        I = cv2.imread(os.path.join(root, sc, 'I.png'))
    T = cv2.imread(os.path.join(root, sc, GTNAME[dom]))
    return I, T


def get_pred(dom, arm, case):
    p = os.path.join(PROJ, f'runs/pred/{dom}_{arm}/{case}/ytmt_ucs_sirs_l.png')
    return cv2.imread(p) if os.path.exists(p) else None


def fit_h(img, h):
    """等比缩放到高 h。"""
    if img is None:
        return np.full((h, int(h * 4 / 3), 3), 240, np.uint8)
    r = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * r)), h), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(OUT, exist_ok=True)
    blocks = []            # (dom, [ (row_tiles, labels, case_info), ... ], colnames)
    maxw = 0

    for dom, (base, mix, ctl) in DOMS.items():
        f = os.path.join(PROJ, f'runs/mix_1008/eval/{mix}_on_{dom}.csv')
        if not os.path.exists(f):
            continue
        base_rho = {}
        if base:
            bf = os.path.join(PROJ, f'runs/mix_1008/eval/{base}_on_{dom}.csv')
            if os.path.exists(bf):
                for r in load(bf):
                    base_rho[r['case']] = fnum(r, f'{base}_rho')

        rows = [r for r in load(f) if r.get('den_ok') == '1']
        arms = [a for a in (base, mix, ctl) if a]
        items = []
        for r in rows:
            case = r['case']
            I, T = get_I_T(dom, case)
            if I is None or T is None or I.shape != T.shape:
                continue
            mixv = fnum(r, f'{mix}_rho')
            bv = base_rho.get(case, float('nan'))
            items.append(dict(r=r, case=case, I=I, T=T, mix=mixv, base=bv,
                              d=(mixv - bv) if base else float('nan')))
        if not items:
            continue
        if base:
            items.sort(key=lambda c: (c['d'] if c['d'] == c['d'] else 0))
            sel = [items[0], items[len(items) // 2], items[-1]]
        else:
            sel = [items[0], items[len(items) // 2], items[-1]]

        tiles_rows = []
        for c in sel:
            I, T = c['I'], c['T']
            o = omega_of(I, T)
            iv = I.copy()
            if o is not None and o['Om'].any():
                cnts, _ = cv2.findContours((o['Om'] * 255).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(iv, cnts, -1, (0, 0, 255), 2)
                c['om_pct'] = round(100 * float(o['Om'].mean()), 1)
            else:
                c['om_pct'] = 0.0
            tiles = [fit_h(iv, ROW_H), fit_h(T, ROW_H)]
            for a in arms:
                tiles.append(fit_h(get_pred(dom, a, c['case']), ROW_H))
            tiles_rows.append((tiles, c))
        blocks.append((dom, arms, tiles_rows, base, mix, ctl))
        w = sum(t.shape[1] for t in tiles_rows[0][0]) + INFO_W
        maxw = max(maxw, w)

    # ---- 拼图 ----
    HDR, SEP = 34, 10
    total_h = 60 + sum(3 * ROW_H + HDR + 3 * SEP + 26 for _ in blocks) + 40
    sheet = np.full((total_h, maxw, 3), 255, np.uint8)

    y = 8
    cv2.putText(sheet, '整帧对比（未裁剪）：左=INPUT(红=Ω轮廓)  次=GT(无反射)  右侧=各臂输出',
                (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    y += 46

    for dom, arms, tiles_rows, base, mix, ctl in blocks:
        cv2.putText(sheet, f'域 {dom}:  单域={base or "—"}  混训={mix}  对照={ctl or "—"}',
                    (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 160), 2)
        y += HDR
        # 列名
        names = ['INPUT', 'GT (无反射)'] + arms
        x = INFO_W
        row0 = tiles_rows[0][0]
        for j, nm in enumerate(names):
            cv2.putText(sheet, nm[:32], (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (0, 0, 0), 1)
            x += row0[j].shape[1]
        for tiles, c in tiles_rows:
            info = np.zeros((ROW_H, INFO_W, 3), np.uint8)
            lines = [c['case'][:32], f"Ω={c['om_pct']}%",
                     f"Δρ={c['d']:+.3f}" if c['d'] == c['d'] else 'Δρ=n/a',
                     f"base ρ={c['base']:+.2f}" if c['base'] == c['base'] else '',
                     f"mix  ρ={c['mix']:+.2f}",
                     f"帧尺寸 {c['I'].shape[1]}×{c['I'].shape[0]}"]
            for k, ln in enumerate(lines):
                if ln:
                    cv2.putText(info, ln, (6, 26 + k * 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.46, (255, 255, 255), 1)
            row = np.hstack([info] + tiles)
            sheet[y:y + ROW_H, :row.shape[1]] = row
            y += ROW_H + SEP
        y += 10

    p = os.path.join(OUT, 'mix_full_all.png')
    cv2.imwrite(p, sheet)
    print(f'[写出] {p}  {sheet.shape}')
    for dom, arms, rows, base, mix, ctl in blocks:
        print(f'  {dom}: {[r[1]["case"][:32] for r in rows]}')


if __name__ == '__main__':
    main()
