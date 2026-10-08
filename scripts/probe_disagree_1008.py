#!/usr/bin/env python3
"""第 2 步：臂间分歧能否当"该收手"的信号（2026-10-08，零训练）
预登记判决门（先写后跑）：
  PASS: 高分歧四分位块的 ΔPSNR(affine 对齐) 中位 比低分歧档低 ≥0.5 dB 且 Spearman ρ(D, ΔPSNR) ≤ −0.2
  FAIL: ρ > −0.15 → 该信号不可用
附带：D 作为"域外度"的读数（may vs apr 的 D 均值比）
"""
import glob, os, sys
import cv2, numpy as np
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = 64
SETS = {'apr': (f'{PROJ}/data/test_sets/real_pair_apr_0917',
                ['apr_rdnet_native', 'apr_remo', 'apr_lovo_k2', 'apr_lovo_k4',
                 'apr_s101_apr10k', 'apr_s2v2_s101_apr', 'apr_v1at6k']),
        'may': (f'{PROJ}/data/test_sets/real_pair_may_0917',
                ['may_rdnet_native', 'may_remo', 'may_lovo_k2', 'may_lovo_k4',
                 'may_rdt_s2_main_s101_10000', 'may_rdt_s2v2_main_s101_6000', 'may_v1at6k'])}


def imread_any(path):
    if path is None:
        return None
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def find_pred(d, case, stem):
    cands = [os.path.join(d, f'{case}__I_{stem}.png'), os.path.join(d, f'{case}__I_{stem}'),
             os.path.join(d, f'{case}__{stem}.png'), os.path.join(d, f'{case}__{stem}'),
             os.path.join(d, f'{stem}.png')]
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


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))


def affine(X, T):
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1); c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.clip(c[0]*X.astype(np.float64)+c[1], 0, 255).astype(np.uint8)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra*rb).sum()/np.sqrt((ra*ra).sum()*(rb*rb).sum() + 1e-12))


rows = []
for sname, (d, arms) in SETS.items():
    used = []
    for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        T = cv2.imread(f'{d}/{sc}/GT.png')
        if T is None:
            continue
        for f in sorted(x for x in os.listdir(f'{d}/{sc}') if x.startswith('I_')):
            stem = f[2:-4]
            I = cv2.imread(f'{d}/{sc}/{f}')
            if I is None or I.shape != T.shape:
                continue
            stack = []
            for a in arms:
                pp = find_pred(os.path.join(PROJ, 'runs/pred', a), sc, stem)
                X = imread_any(pp)
                if X is None:
                    continue
                if X.shape != T.shape:
                    X = cv2.resize(X, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_AREA)
                stack.append(X.astype(np.float32))
            if len(stack) < 3:
                continue
            used.append(len(stack))
            S = np.stack(stack, 0)
            D = S.std(0).mean(2)                        # 逐像素臂间分歧
            Pall = affine(stack[0], T); Iall = affine(I, T)
            h, w = T.shape[:2]; nh, nw = h//BS, w//BS
            for by in range(nh):
                for bx in range(nw):
                    sl = np.s_[by*BS:(by+1)*BS, bx*BS:(bx+1)*BS]
                    tb = T[sl].astype(np.float64)
                    d_af = psnr(Pall[sl], tb) - psnr(Iall[sl], tb)
                    rows.append(dict(set=sname, case=f'{sc}__{stem}', D=float(D[sl].mean()),
                                     luma=float(tb.mean()), d_af=d_af,
                                     d_raw=psnr(stack[0][sl], tb) - psnr(I[sl], tb)))
    print(f'[{sname}] 用到的臂数（多数 case）: {np.bincount(used).argmax() if used else 0}')

import csv
with open(f'{PROJ}/eval/probe_disagree_1008.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

print('\n==== 预登记判决门 ====')
for sname in SETS:
    sub = [r for r in rows if r['set'] == sname]
    if len(sub) < 20:
        continue
    D = np.array([r['D'] for r in sub]); Y = np.array([r['d_af'] for r in sub])
    Yr = np.array([r['d_raw'] for r in sub])
    q1, q3 = np.quantile(D, [0.25, 0.75])
    lo = [y for dd, y in zip(D, Y) if dd <= q1]; hi = [y for dd, y in zip(D, Y) if dd >= q3]
    rho_af = spearman(D, Y); rho_raw = spearman(D, Yr)
    print(f'\n--- {sname}（{len(sub)} 块）---')
    print(f'  分歧 D：均值 {D.mean():6.2f} | 低四分位 D≤{q1:.2f} 的 d_af 中位 {np.median(lo):+6.2f} dB')
    print(f'                                高四分位 D≥{q3:.2f} 的 d_af 中位 {np.median(hi):+6.2f} dB')
    print(f'  高−低 = {np.median(hi)-np.median(lo):+.2f} dB | Spearman ρ(D,d_af) = {rho_af:+.3f} | ρ(D,d_raw) = {rho_raw:+.3f}')
    print(f'  判定：{"PASS" if (np.median(hi)-np.median(lo) <= -0.5 and rho_af <= -0.2) else ("FAIL" if rho_af > -0.15 else "灰区")}')
print('\n[写出] eval/probe_disagree_1008.csv')
