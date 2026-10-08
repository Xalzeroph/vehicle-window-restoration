#!/usr/bin/env python3
"""第 1 步：光度归一化下的旧数字重算（2026-10-08，零训练）
预登记判决门（先写后跑）：
  Q1 干净档(r_obs<0.05) 在 affine 对齐后 ΔPSNR 中位 ≤ −1.0 dB ?  → 是:"干净区被伤"存活
  Q2 oracle 二值 headroom 在 affine 对齐后 ≥ +1.0 dB ?           → 是:E4 存活
  Q3 被检对象相对 "identity+affine" 平凡基线的净收益 > 0 ?        → 是:门控线仍有意义
口径：原生分辨率 + 原生官方预测；三种对齐 = raw / 标量增益 / per-image affine
"""
import glob, os, sys
import cv2, numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = 64
EDGES = [0, .05, .1, .2, .3, .5, .8, 1.2, 9]
SETS = {'apr': f'{PROJ}/data/test_sets/real_pair_apr_0917',
        'may': f'{PROJ}/data/test_sets/real_pair_may_0917'}
PRED = {'apr': 'runs/pred/apr_rdnet_native', 'may': 'runs/pred/may_rdnet_native'}


def imread_any(path):
    """无扩展名的预测文件：cv2.imread 会因扩展名缺失返回 None，必须用 imdecode"""
    if path is None:
        return None
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))


def find_pred(d, case, stem):
    """预测可能是：单文件 / 符号链接 / 每 case 一个目录（内含 *_l.png 或 remo.png）"""
    cands = [os.path.join(d, f'{case}__I_{stem}.png'), os.path.join(d, f'{case}__I_{stem}'),
             os.path.join(d, f'{case}__{stem}.png'), os.path.join(d, f'{case}__{stem}'),
             os.path.join(d, f'{stem}.png'), os.path.join(d, f'{case}_{stem}.png')]
    cands += sorted(glob.glob(os.path.join(d, f'*{stem}*')))
    for p in cands:
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            inner = sorted(glob.glob(os.path.join(p, '*.png')))
            if inner:
                pref = [x for x in inner if x.endswith('_l.png') or 'remo' in os.path.basename(x)]
                return (pref or inner)[0]
    return None


def gm(X, T):                      # 标量增益：X→T 最优乘子
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    return float((x*t).sum()/max((x*x).sum(), 1e-9))


def affine(X, T):                  # 每图一条全局色调曲线
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1)
    c, *_ = np.linalg.lstsq(A, t, rcond=None)
    Y = np.clip(c[0]*X.astype(np.float64)+c[1], 0, 255).astype(np.uint8)
    return Y, float(c[0]), float(c[1])


irows, brows = [], []
for sname, d in SETS.items():
    for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        T = cv2.imread(f'{d}/{sc}/GT.png')
        if T is None:
            continue
        for f in sorted(x for x in os.listdir(f'{d}/{sc}') if x.startswith('I_')):
            stem = f[2:-4]
            I = cv2.imread(f'{d}/{sc}/{f}')
            pp = find_pred(os.path.join(PROJ, PRED[sname]), sc, stem)
            if I is None or I.shape != T.shape or pp is None:
                continue
            P = imread_any(pp)
            if P is None:
                continue
            if P.shape != T.shape:
                P = cv2.resize(P, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_AREA)
            Iaf, aI, bI = affine(I, T); Paf, _, _ = affine(P, T)
            gI = gm(I, T); Ig = np.clip(I.astype(np.float64)*gI, 0, 255).astype(np.uint8)
            gP = gm(P, T); Pg = np.clip(P.astype(np.float64)*gP, 0, 255).astype(np.uint8)
            irows.append(dict(set=sname, case=f'{sc}__{stem}',
                              raw_I=round(psnr(I, T), 2), raw_P=round(psnr(P, T), 2),
                              gm_I=round(psnr(Ig, T), 2), gm_P=round(psnr(Pg, T), 2),
                              af_I=round(psnr(Iaf, T), 2), af_P=round(psnr(Paf, T), 2),
                              b_gm_I=gI, b_gm_P=gP, affine_a=aI, affine_b=bI))
            h, w = T.shape[:2]; nh, nw = h//BS, w//BS
            for by in range(nh):
                for bx in range(nw):
                    sl = np.s_[by*BS:(by+1)*BS, bx*BS:(bx+1)*BS]
                    tb = T[sl].astype(np.float64)
                    ibr = I[sl].astype(np.float64); pb = P[sl].astype(np.float64)
                    ib_g = Ig[sl].astype(np.float64); pb_g = Pg[sl].astype(np.float64)
                    ib = Iaf[sl].astype(np.float64); pba = Paf[sl].astype(np.float64)
                    r_obs = float(np.abs(ib_g - tb).max(axis=2).mean()/(tb.mean()+1e-9))
                    a1 = (ib_g - tb).ravel(); b1 = (pb_g - ib_g).ravel(); bb = float((b1*b1).sum())
                    gs = float(np.clip(-(a1*b1).sum()/bb, 0, 1)) if bb > 1e-6 else np.nan
                    a2 = (ib - tb).ravel(); b2 = (pba - ib).ravel(); bb2 = float((b2*b2).sum())
                    gs_af = float(np.clip(-(a2*b2).sum()/bb2, 0, 1)) if bb2 > 1e-6 else np.nan
                    d_raw = psnr(pb, tb) - psnr(ibr, tb)
                    d_af = psnr(pba, tb) - psnr(ib, tb)
                    brows.append(dict(set=sname, case=f'{sc}__{stem}', by=by, bx=bx,
                                      luma=round(float(tb.mean()), 1), r_obs=round(r_obs, 4),
                                      d_raw=round(d_raw, 2), d_af=round(d_af, 2),
                                      gstar=round(gs, 3) if not np.isnan(gs) else '',
                                      gstar_af=round(gs_af, 3) if not np.isnan(gs_af) else ''))
import csv
os.makedirs(f'{PROJ}/eval', exist_ok=True)
for nm, rw in (('img', irows), ('blocks', brows)):
    with open(f'{PROJ}/eval/reaudit_1008_{nm}.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rw[0].keys())); wr.writeheader(); wr.writerows(rw)

print('==== 预登记判决门 ====')
for sname in SETS:
    ir = [r for r in irows if r['set'] == sname]; br = [r for r in brows if r['set'] == sname]
    if not ir:
        continue
    print(f'\n########## {sname}（{len(ir)} 图 / {len(br)} 块）##########')
    rawi = np.mean([r['raw_I'] for r in ir]); rawp = np.mean([r['raw_P'] for r in ir])
    afi = np.mean([r['af_I'] for r in ir]); afp = np.mean([r['af_P'] for r in ir])
    gmi = np.mean([r['gm_I'] for r in ir]); gmp = np.mean([r['gm_P'] for r in ir])
    print(f'  图级 PSNR：raw  I {rawi:6.2f}  P {rawp:6.2f}   (P−I {rawp-rawi:+.2f})')
    print(f'             标量  I {gmi:6.2f}  P {gmp:6.2f}   (P−I {gmp-gmi:+.2f})')
    print(f'             affine I {afi:6.2f}  P {afp:6.2f}   (P−I {afp-afi:+.2f})   ← Q3 平凡基线')
    clean = [r for r in br if r['r_obs'] < 0.05]
    if clean:
        print(f'  Q1 干净档 r_obs<0.05（{len(clean)} 块）：d_raw 中位 {np.median([r["d_raw"] for r in clean]):+.2f} dB | '
              f'd_af 中位 {np.median([r["d_af"] for r in clean]):+.2f} dB')
    else:
        print('  Q1 干净档：无块')
    bi_raw = np.mean([max(0.0, r['raw_I'] - r['raw_P']) for r in ir])
    bi_af = np.mean([max(0.0, r['af_I'] - r['af_P']) for r in ir])
    print(f'  Q2 oracle 二值 headroom（图级）raw {bi_raw:+.2f} dB | affine 对齐后 {bi_af:+.2f} dB')
    print(f'  {"r_obs 档":<14}{"块数":>7}{"d_raw中位":>11}{"d_af中位":>10}{"g*中位":>8}{"g*_af中位":>10}')
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sub = [r for r in br if lo <= r['r_obs'] < hi and r['gstar'] != '']
        if len(sub) < 8:
            continue
        print(f'  [{lo:>5g},{hi:>5g}){len(sub):>7}{np.median([r["d_raw"] for r in sub]):>+11.2f}'
              f'{np.median([r["d_af"] for r in sub]):>+10.2f}{np.median([float(r["gstar"]) for r in sub]):>8.2f}'
              f'{np.median([float(r["gstar_af"]) for r in sub if r["gstar_af"] != ""] or [float("nan")]):>10.2f}')
print('\n[写出] eval/reaudit_1008_img.csv, eval/reaudit_1008_blocks.csv')
