#!/usr/bin/env python3
"""多域版 S2 训练集构建（2026-10-08 混训实验）。

泛化自 build_train_omega_0920.py —— **口径完全一致**（robust_gain / omega_of / R 存 +128），
只增加：① 多域拼接；② 可选 gm 逐帧色调对齐（对照臂用）。

用法:
  python3 scripts/build_train_omega_multi.py --domain may --out data/train_sets/pairs_may
  python3 scripts/build_train_omega_multi.py --domain apr,jun --out data/train_sets/pairs_aprjun
  python3 scripts/build_train_omega_multi.py --domain apr,jun --out data/train_sets/pairs_aprjun_gm --gm-align

产出: I/ T/ R/ omega/ + meta.json（每条含 domain 字段）
"""
import argparse, collections, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_protocol import robust_gain      # noqa: E402
from eval_g2_baseline import omega_of      # noqa: E402

DOMAINS = {
    'apr': dict(root='data/test_sets/real_pair_apr_0917',
                gt='GT.png', is_I=lambda f: f.startswith('I_'),
                note='apr 实拍，GT.png + I_*.png'),
    'may': dict(root='data/test_sets/real_pair_may_0917',
                gt='GT.png', is_I=lambda f: f.startswith('I_'),
                note='may 实拍，GT.png + I_*.png'),
    # jun 用 real_pair_0917：每场景 1 输入帧，T=无反射低光档（与 apr/may 同为「无反射档」语义）。
    # litgt 的 N=带加光档，语义不同，不用。
    'jun': dict(root='data/test_sets/real_pair_0917',
                gt='T.png', is_I=lambda f: f == 'I.png',
                note='jun 实拍，T.png(无反射低光) + I.png(反射低光)'),
}


def collect(dom):
    cfg = DOMAINS[dom]
    root = os.path.join(PROJ, cfg['root'])
    assert os.path.isdir(root), f'{dom}: 源目录不存在 {root}'
    out = []
    for sc in sorted(x for x in os.listdir(root) if x.startswith('VID_')):
        sd = os.path.join(root, sc)
        if not os.path.exists(os.path.join(sd, cfg['gt'])):
            continue
        for fn in sorted(f for f in os.listdir(sd) if cfg['is_I'](f)):
            out.append((dom, sc, sd, fn, os.path.join(sd, cfg['gt'])))
    print(f'[{dom}] {cfg["note"]}  -> {len(out)} 帧', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, help='逗号分隔，如 apr,jun')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gm-align', action='store_true',
                    help='把 I 乘 robust_gain 落到 T 的曝光域（色调对照臂）')
    a = ap.parse_args()

    doms = [d.strip() for d in a.domain.split(',') if d.strip()]
    for d in doms:
        assert d in DOMAINS, f'未知域 {d}，可选 {list(DOMAINS)}'
    OUT = a.out if os.path.isabs(a.out) else os.path.join(PROJ, a.out)

    items = []
    for d in doms:
        items += collect(d)

    for sub in ('I', 'T', 'R', 'omega'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    meta, gs, n_empty = {}, [], 0
    for dom, sc, sd, fn, gp in items:
        case = f'{dom}__{sc}__{fn[:-4]}'        # 域前缀，防跨域重名
        T = cv2.imread(gp)
        I0 = cv2.imread(os.path.join(sd, fn))
        assert I0 is not None and T is not None, case
        assert I0.shape == T.shape, f'{case}: {I0.shape} vs {T.shape}'

        g = np.atleast_1d(robust_gain(I0, T)).reshape(1, 1, 3)   # 增益恒在**原始** I 上估
        gs.append(float(g.mean()))

        I = I0
        if a.gm_align:
            I = np.clip(I0.astype(np.float64) * g, 0, 255).astype(np.uint8)

        # R 用对齐后的 I 减 g·T → 与评测口径 (I·g − T·g) 一致，色调项抵消
        R = I.astype(np.float64) - T.astype(np.float64) * g

        o = omega_of(I, T)
        if o is None:                            # 无 Ω 的帧：全零掩膜（与 0920 脚本同处理）
            om = np.zeros(I.shape[:2], bool); n_empty += 1
        else:
            om = o['Om']

        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ring = (cv2.dilate(om.astype(np.uint8), ker) > 0) & (~om)
        lI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        C_I = float(lI[ring].mean() - lI[om].mean()) if om.any() and ring.any() else 0.0

        cv2.imwrite(os.path.join(OUT, 'I', case + '.png'), I)
        cv2.imwrite(os.path.join(OUT, 'T', case + '.png'), T)
        cv2.imwrite(os.path.join(OUT, 'R', case + '.png'),
                    np.clip(R + 128.0, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUT, 'omega', case + '.png'), (om * 255).astype(np.uint8))
        meta[case] = dict(domain=dom, scene=sc, src=fn,
                          g=round(float(g.mean()), 4), C_I=round(C_I, 2),
                          omega_frac=round(float(om.mean()), 4),
                          R_abs_mean=round(float(np.abs(R.mean(axis=2)).mean()), 2),
                          size=list(I.shape[:2]))

    with open(os.path.join(OUT, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=1, ensure_ascii=False)

    cnt = collections.Counter(v['domain'] for v in meta.values())
    om = [v['omega_frac'] for v in meta.values()]
    print(f'\n共 {len(meta)} 对 -> {OUT}')
    print(f'  按域: {dict(cnt)}   Ω 为空的帧: {n_empty}')
    print(f'  gm_align={a.gm_align}   |g-1| 中位 {np.median(np.abs(np.array(gs)-1)):.4f}')
    print(f'  Ω 占比: min {min(om)*100:.1f}% 中位 {np.median(om)*100:.1f}% max {max(om)*100:.1f}%')


if __name__ == '__main__':
    main()
