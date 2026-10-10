#!/usr/bin/env python3
"""可视化混训 6 臂的实际修复效果（2026-10-08）。

每个域一张图；每行一个代表帧（按 Δρ = 混训 − 单域 选最好/中位/最差）。
列：裁剪区(红框=Ω) | GT | 各臂输出（每格标注 ρ 与 ref_gain）
外加一条 Ω 强度带（下沿），便于看洞在哪。

用法: python3 scripts/viz_mix_1008.py
"""
import csv, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, 'fig/mix_1008')
CROP = 320          # 裁剪边长（比 224 更能看清洞的上下文）

# 域 -> (单域臂, 混训臂, 对照臂)
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
    v = r.get(k, '')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def get_I_T(dom, case):
    """返回 (I, T)。case 形如 VID_...__I_000000 或 VID_...（jun）。"""
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


def crop_of(I, T, y0, x0, s=CROP):
    def c(x):
        return x[y0:y0 + s, x0:x0 + s]
    return c(I), c(T)


def main():
    os.makedirs(OUT, exist_ok=True)
    for dom, (base, mix, ctl) in DOMS.items():
        f = os.path.join(PROJ, f'runs/mix_1008/eval/{mix}_on_{dom}.csv')
        if not os.path.exists(f):
            print(f'[跳过] {dom}: 无 {f}'); continue
        rows = load(f)
        # 每个 CSV 只含自己那一臂的列 → 基线臂的 rho 必须从它自己的 CSV 取
        base_rho, ctl_rho = {}, {}
        for tag, store in ((base, base_rho), (ctl, ctl_rho)):
            if not tag:
                continue
            bf = os.path.join(PROJ, f'runs/mix_1008/eval/{tag}_on_{dom}.csv')
            if os.path.exists(bf):
                for r in load(bf):
                    store[r['case']] = fnum(r, f'{tag}_rho')

        # 每帧的 Ω 与 ρ
        cases = []
        for r in rows:
            if r.get('den_ok') != '1':
                continue
            case = r['case']
            I, T = get_I_T(dom, case)
            if I is None or T is None or I.shape != T.shape:
                continue
            mixv = fnum(r, f'{mix}_rho')
            basev = base_rho.get(case, float('nan')) if base else float('nan')
            cases.append(dict(case=case, r=r, I=I, T=T, mix=mixv, base=basev,
                              ctl=ctl_rho.get(case, float('nan')),
                              d=(mixv - basev) if base else float('nan')))
        if not cases:
            print(f'[跳过] {dom}: 无可画帧'); continue

        # 选最好/中位/最差
        cases.sort(key=lambda c: c['d'])
        sel = [cases[0], cases[len(cases)//2], cases[-1]]

        # 每帧裁剪锚点 = Ω 质心
        for c in sel:
            from_eval = None
            I, T = c['I'], c['T']
            # 用 Ω 掩膜（从 eval CSV 无掩膜，改由 omega_of 现算）
            sys.path.insert(0, HERE)
            from eval_g2_baseline import omega_of
            o = omega_of(I, T)
            if o is None or not o['Om'].any():
                c['y0'] = c['x0'] = 0; c['om'] = np.zeros(I.shape[:2], np.uint8); continue
            Om = o['Om']
            H, W = Om.shape
            ys, xs = np.nonzero(Om)
            y0 = int(np.clip(ys.mean() - CROP // 2, 0, max(0, H - CROP)))
            x0 = int(np.clip(xs.mean() - CROP // 2, 0, max(0, W - CROP)))
            c['y0'], c['x0'] = y0, x0
            c['om'] = Om

        arms = [a for a in (base, mix, ctl) if a]
        ncol = 2 + len(arms)          # I(带框) + GT + 各臂
        rows_img = []
        for c in sel:
            y0, x0 = c['y0'], c['x0']
            ic = c['I'][y0:y0 + CROP, x0:x0 + CROP].copy()
            tc = c['T'][y0:y0 + CROP, x0:x0 + CROP].copy()
            # Ω 轮廓画在 I 上
            omc = c['om'][y0:y0 + CROP, x0:x0 + CROP].astype(np.uint8) * 255
            cnts, _ = cv2.findContours(omc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ic, cnts, -1, (0, 0, 255), 1)
            tiles = [ic, tc]
            labels = [f"INPUT  Ω={c['r']['omega_pct']}%", "GT (无反射)"]
            for a in arms:
                P = get_pred(dom, a, c['case'])
                if P is None:
                    tiles.append(np.full_like(tc, 255)); labels.append(f'{a}\n[无预测]')
                else:
                    tiles.append(P[y0:y0 + CROP, x0:x0 + CROP])
                    if a == base:
                        rho, rg, od = c['base'], float('nan'), fnum(c['r'], f'{a}_out_dPSNR')
                    elif a == ctl:
                        rho, rg, od = c['ctl'], float('nan'), fnum(c['r'], f'{a}_out_dPSNR')
                    else:
                        rho = fnum(c['r'], f'{a}_rho')
                        rg = fnum(c['r'], f'{a}_ref_gain')
                        od = fnum(c['r'], f'{a}_out_dPSNR')
                    labels.append(f'{a}\nρ={rho:+.2f} rg={rg:+.3f} ③={od:+.2f}')
            row = np.hstack([t for t in tiles])
            rows_img.append((row, labels, c))

        gap = 6
        tlab_h = 30                      # 每行左侧的行标签高度
        sep = np.full((gap, rows_img[0][0].shape[1] + 210, 3), 255, np.uint8)
        body = []
        for row, labels, c in rows_img:
            # 左侧行信息条（黑底白字），避免每格重复整列名
            info = np.zeros((row.shape[0], 210, 3), np.uint8)
            for j, line in enumerate([
                    c['case'][:30],
                    f"Ω={c['r']['omega_pct']}%",
                    f"Δρ={c['d']:+.3f}" if c['d'] == c['d'] else 'Δρ=n/a',
                    f"base ρ={c['base']:+.2f}" if c['base'] == c['base'] else 'base ρ=n/a',
                    f"mix  ρ={c['mix']:+.2f}",
                    f"ctl  ρ={c['ctl']:+.2f}" if c['ctl'] == c['ctl'] else '']):
                cv2.putText(info, line, (6, 26 + j * 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, (255, 255, 255), 1)
            body.append(np.hstack([info, row]))
            body.append(sep)
        sheet = np.vstack(body[:-1])

        # 标题条 + 列名条（各列只写一次）
        bar = np.full((46, sheet.shape[1], 3), 255, np.uint8)
        title = (f'域 {dom}:  单域={base or "—"}  |  混训={mix}  |  对照={ctl or "—"}'
                 f'   (每帧切 {CROP}×{CROP}，锚点=Ω 质心，红线=Ω 轮廓)')
        cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
        cv2.putText(bar, '行 = Δρ(混训−单域) 最差 / 中位 / 最好；左条给出该帧信息',
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1)
        sheet = np.vstack([bar, sheet])

        lab = np.zeros((30, sheet.shape[1], 3), np.uint8)
        first_labels = rows_img[0][1]
        for j, t in enumerate(first_labels):
            col = t.split('\n')[0]
            cv2.putText(lab, col[:34], (210 + j * CROP + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        sheet = np.vstack([sheet, lab])

        p = os.path.join(OUT, f'mix_{dom}.png')
        cv2.imwrite(p, sheet)
        print(f'[写出] {p}  {sheet.shape}')
        for c in sel:
            print(f"    {c['case'][:36]:<38} Δρ={c['d']:+.3f}  baseρ={c['base']:+.2f} mixρ={c['mix']:+.2f}")


if __name__ == '__main__':
    main()
