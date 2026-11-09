#!/usr/bin/env python3
"""S1/S1.5：现后端在 Ω 上的 G2 判据基线（口径 v2，Opus 判决 2026-09-20）。

对每案：Ω（v2 定式）→ 三判据：
  ① 消退（T 锚定修复进度；T=黑布档去反射参考）：ρ=(C_I−C_A)/(C_I−C_T) ≥0.5，且不得过冲
     （Ω 内 luma ≥ T 的 Ω 内 luma − max(4, 0.5|C_I|)）；den=C_I−C_T 不足则①剔除
  ② 参考改善：Ω 外扩带裁剪 LPIPS(out vs T) 相对 (I vs T) 改善 ≥0.02
  ③ 不劣：Ω 外 ΔPSNR ≥ −0.2
  诊断列：dC（旧口径，存档）、med_ratio=med|A−T|[Ω]/med|I−T|[Ω]、外/ctrl luma 位移
臂：identity / perfect(=T，上限检验) / rdnet / remo。
产物：eval/eval_g2_baseline.csv
"""
import argparse, csv, os, sys
import cv2
import numpy as np
import torch
import pyiqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import lpips_pair, psnr  # noqa: E402

G = 2.2
PAD = 10
SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet', 'ytmt_ucs_sirs_l.png', 'runs/pred/real_pair_0917_remo', 'remo.png'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet', 'ytmt_ucs_sirs_l.png', 'runs/pred/apr_remo', 'remo.png'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet', 'ytmt_ucs_sirs_l.png', 'runs/pred/may_remo', 'remo.png')}


def omega_of(I, T):
    Il = (np.clip(I, 0, 255) / 255.0) ** G
    Tl = (np.clip(T, 0, 255) / 255.0) ** G
    valid = Tl.mean(axis=2) > (10 / 255.) ** G
    ratio = np.where(valid[..., None], Il / np.where(Tl > 1e-8, Tl, np.nan), np.nan)
    s = float(np.nanquantile(ratio.mean(axis=2), 0.05))
    Tsc = (np.clip(s * Tl, 0, 1)) ** (1 / G) * 255.0
    Rg = (I.astype(np.float64) - Tsc).mean(axis=2)
    R8 = np.clip(Rg + 128, 0, 255).astype(np.uint8)
    sig = 1.4826 * float(np.median(np.abs(Rg - (cv2.medianBlur(R8, 3).astype(np.float64) - 128))))
    tau = max(6.0, 5 * sig)
    h, w = Rg.shape
    blk = 48
    hh, ww = h // blk, w // blk
    t = Rg[:hh * blk, :ww * blk].reshape(hh, blk, ww, blk).transpose(0, 2, 1, 3).reshape(hh, ww, -1)
    bg = cv2.resize(np.percentile(t, 90, axis=2).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR).astype(np.float64)
    if float(np.percentile(Rg, 90)) < max(12.0, 6 * sig):
        return None
    M = ((S := bg - Rg) > tau) & (2 * Rg < bg)
    n, lab, st, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), 8)
    Om = np.zeros_like(M, bool)
    for i in range(1, n):
        if st[i][4] / M.size >= 0.005:
            Om |= (lab == i)
    inv = (~Om).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n2):
        x, y, w2, h2, area = st2[i]
        if not (x == 0 or y == 0 or x + w2 >= w or y + h2 >= h) and area / M.size <= 0.25:
            Om |= (lab2 == i)
    return dict(Om=Om, S=S, bg=bg)


def luma(x):
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='jun,apr,may', help='逗号分隔，如 may')
    ap.add_argument('--extra-arm', action='append', default=[], metavar='NAME=ROOT',
                    help='外部预测臂（目录下 <key>/ytmt_ucs_sirs_l.png），可多次')
    ap.add_argument('--skip-official', action='store_true', help='跳过 rdnet/remo 官方臂')
    ap.add_argument('--out', default='eval/eval_g2_baseline.csv')
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    lp = pyiqa.create_metric('lpips', device=dev)
    rows = []
    todo = [t for t in SETS if t in args.sets.split(',')]
    with torch.no_grad():
        for tag, (d, gtn, pr1, pf1, pr2, pf2) in [(t, SETS[t]) for t in todo]:
            for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
                sd = os.path.join(d, sc)
                T = cv2.imread(os.path.join(sd, gtn))
                for fn in (sorted(f for f in os.listdir(sd) if f.startswith('I')) or ['I.png']):
                    I = cv2.imread(os.path.join(sd, fn))
                    key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
                    o = omega_of(I, T) if (I is not None and T is not None and I.shape == T.shape) else None
                    if o is None:
                        rows.append(dict(set=tag, case=key, omega_pct=0.0, note='Omega 空/不适用')); continue
                    Om = o['Om']
                    if not Om.any():
                        rows.append(dict(set=tag, case=key, omega_pct=0.0, note='Omega 空/不适用')); continue
                    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PAD + 1, 2 * PAD + 1))
                    ring = (cv2.dilate(Om.astype(np.uint8), ker) > 0) & (~Om)
                    arms = {'identity': I, 'perfect': T}
                    if not args.skip_official:
                        P = cv2.imread(os.path.join(pr1, key, pf1))
                        if P is not None:
                            arms['rdnet'] = cv2.resize(P, (T.shape[1], T.shape[0])) if P.shape != T.shape else P
                        P2 = cv2.imread(os.path.join(pr2, key, pf2))
                        if P2 is not None:
                            arms['remo'] = cv2.resize(P2, (T.shape[1], T.shape[0])) if P2.shape != T.shape else P2
                    for spec in args.extra_arm:
                        an, _, root = spec.partition('=')
                        root = root if os.path.isabs(root) else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', root)
                        Px = cv2.imread(os.path.join(root, key, 'ytmt_ucs_sirs_l.png'))
                        if Px is not None:
                            arms[an] = cv2.resize(Px, (T.shape[1], T.shape[0])) if Px.shape != T.shape else Px
                    lI = luma(I); lT = luma(T)
                    C_I = lI[ring].mean() - lI[Om].mean()
                    C_T = lT[ring].mean() - lT[Om].mean()
                    den = C_I - C_T
                    den_ok = bool(den >= max(4.0, 0.3 * abs(C_I)))
                    delta = max(4.0, 0.5 * abs(C_I))
                    mean_T_om = float(lT[Om].mean())
                    out_m = ~cv2.dilate(Om.astype(np.uint8), ker).astype(bool)
                    ctrl = (lI < np.percentile(lI, 10)) & out_m
                    dIT_med = max(float(np.median(np.abs(I[Om].astype(np.float64) - T[Om]))), 1e-6)
                    rec = dict(set=tag, case=key, omega_pct=round(100 * float(Om.mean()), 1),
                               n_ring=int(ring.sum()), C_I=round(float(C_I), 2), C_T=round(float(C_T), 2),
                               den_ok=int(den_ok), note='' if den_ok else '参考对比不足(①剔除)')
                    for an, A in arms.items():
                        lA = luma(A)
                        C_A = lA[ring].mean() - lA[Om].mean()
                        rec[f'{an}_dC'] = (round(float((C_I - C_A) / C_I), 3) if C_I > 2 else None)
                        rec[f'{an}_rho'] = round(float((C_I - C_A) / den), 3) if den_ok else None
                        rec[f'{an}_no_overshoot'] = int(lA[Om].mean() >= mean_T_om - delta)
                        rec[f'{an}_pass1'] = (int(rec[f'{an}_rho'] >= 0.5 and rec[f'{an}_no_overshoot'])
                                              if den_ok else None)
                        rec[f'{an}_out_luma_shift'] = round(float(np.abs(lA[out_m] - lI[out_m]).mean()), 2)
                        rec[f'{an}_ctrl_luma_shift'] = round(float((lA[ctrl] - lI[ctrl]).mean()), 2) if ctrl.sum() > 100 else None
                        rec[f'{an}_med_ratio'] = round(float(np.median(np.abs(A[Om].astype(np.float64) - T[Om])) / dIT_med), 3)
                        rec[f'{an}_pass4'] = int(rec[f'{an}_med_ratio'] <= 1.0)
                        # ② Ω 外扩带裁剪 LPIPS 改善
                        ys, xs = np.where(Om)
                        y0, y1 = max(0, ys.min() - 16), min(T.shape[0], ys.max() + 16)
                        x0, x1 = max(0, xs.min() - 16), min(T.shape[1], xs.max() + 16)
                        l_i = lpips_pair(lp, T[y0:y1, x0:x1], I[y0:y1, x0:x1])
                        l_a = lpips_pair(lp, T[y0:y1, x0:x1], A[y0:y1, x0:x1])
                        rec[f'{an}_ref_gain'] = round(l_i - l_a, 4)
                        # ③ Ω 外不劣
                        rec[f'{an}_out_dPSNR'] = round(psnr(A[out_m].reshape(-1, 3), T[out_m].reshape(-1, 3))
                                                       - psnr(I[out_m].reshape(-1, 3), T[out_m].reshape(-1, 3)), 2)
                        rec[f'{an}_pass_all'] = (int(rec[f'{an}_pass1'] == 1 and rec[f'{an}_pass4'] == 1
                                                     and rec[f'{an}_ref_gain'] >= 0.02 and rec[f'{an}_out_dPSNR'] >= -0.2)
                                                 if den_ok else None)
                    rows.append(rec)

                    def fm(v, fmt):
                        return 'NA' if v is None else format(v, fmt)

                    print(f"{tag}/{key}: Ω {rec['omega_pct']}% C_I={rec['C_I']} C_T={rec['C_T']} "
                          + (' ' if den_ok else '[den不足] ') + ' '.join(
                        f"{an}:ρ{fm(rec[f'{an}_rho'], '+.2f')}/①{rec[f'{an}_pass1']}/②{fm(rec[f'{an}_ref_gain'], '+.3f')}"
                        f"/③{fm(rec[f'{an}_out_dPSNR'], '+.2f')}" for an in arms))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, restval='')
        wr.writeheader(); wr.writerows(rows)
    print(f'\n写出 {args.out}')


if __name__ == '__main__':
    main()
