#!/usr/bin/env python3
"""补测②：手搓信号的块级 AUC + 逐月 R²(T≈a·I+b)（2026-09-19 方法线 P2 前置）。

- 块级（128²，与补测①同网格）：信号 s0..s7（推断时可得）预测两类标签：
    harm  = 纯 RDNet 伤害该块（dP_rd<0）
    act   = 该动该块（w_star>0）
  标签取自 eval/eval_probe_oracle_blocks.csv（先跑补测①）。
- 逐月 R²：luma(T) ≈ a·luma(I) + b 的最小二乘拟合决定系数 → 检验"去反射≈全局调色"捷径。
产物：eval/eval_probe_signals.csv、eval/eval_probe_monthr2.csv
"""
import csv, os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

BS = 128
SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet')}


def cases(cfg):
    d, gtn, _ = cfg
    out = []
    for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        sd = os.path.join(d, sc)
        for fn in (sorted(f for f in os.listdir(sd) if f.startswith('I')) or ['I.png']):
            key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
            out.append((key, os.path.join(sd, fn), os.path.join(sd, gtn)))
    return out


def signals_of(I, P, Rr, T):
    g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ge = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mu = cv2.GaussianBlur(g, (0, 0), 8)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 8) - mu ** 2, 0))
    sat = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    cross = np.abs(cv2.GaussianBlur(g, (0, 0), 2) - cv2.GaussianBlur(g, (0, 0), 8))
    r_abs = np.abs(Rr.astype(np.float32)).mean(axis=2)
    dpi = np.abs(P.astype(np.float32) - I.astype(np.float32)).mean(axis=2)
    return {'s0_|R|': r_abs, 's1_grad': ge, 's2_bright': mu, 's3_std': sd,
            's4_grad+std': ge / (ge.mean() + 1e-6) + sd / (sd.mean() + 1e-6),
            's5_|P-I|': dpi, 's6_sat': sat, 's7_crossscale': cross}


def main():
    blocks = list(csv.DictReader(open('eval/eval_probe_oracle_blocks.csv')))
    by_key = {}
    for b in blocks:
        by_key.setdefault((b['set'], b['case']), []).append(b)
    rows, r2_rows = [], []
    for tag, cfg in SETS.items():
        d, gtn, pred_root = cfg
        for key, ip, tp in cases(cfg):
            I = cv2.imread(ip); T = cv2.imread(tp)
            pp = os.path.join(pred_root, key, 'ytmt_ucs_sirs_l.png')
            rp = os.path.join(pred_root, key, 'ytmt_ucs_sirs_r.png')
            if I is None or T is None or not os.path.exists(pp) or not os.path.exists(rp):
                print(f'{key}: 缺文件 -> 跳过'); continue
            P = cv2.imread(pp); Rr = cv2.imread(rp)
            if P.shape != T.shape:
                P = cv2.resize(P, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
                Rr = cv2.resize(Rr, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
            sig = signals_of(I, P, Rr, T)
            for b in by_key.get((tag, key), []):
                y, x = int(b['y']), int(b['x'])
                rec = dict(set=tag, case=key, y=y, x=x, dP_rd=float(b['dP_rd']),
                           w_star=float(b['w_star']),
                           harm=int(float(b['dP_rd']) < 0), act=int(float(b['w_star']) > 0))
                for name, m in sig.items():
                    rec[name] = float(m[y:y + BS, x:x + BS].mean())
                rows.append(rec)
            # 逐月 R²：luma(T) ≈ a·luma(I) + b
            li = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
            lt = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
            A = np.vstack([li, np.ones_like(li)]).T
            coef, res, *_ = np.linalg.lstsq(A, lt, rcond=None)
            pred = A @ coef
            ss_res = float(((lt - pred) ** 2).sum()); ss_tot = float(((lt - lt.mean()) ** 2).sum())
            r2_rows.append(dict(set=tag, case=key, a=round(float(coef[0]), 3), b=round(float(coef[1]), 2),
                                r2=round(1 - ss_res / max(ss_tot, 1e-9), 4)))
    with open('eval/eval_probe_signals.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open('eval/eval_probe_monthr2.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(r2_rows[0].keys())); wr.writeheader(); wr.writerows(r2_rows)

    sig_names = [k for k in rows[0] if k.startswith('s') and k != 'set']
    print(f'块 {len(rows)} | 逐月 R²（luma(T)≈a·luma(I)+b）：')
    for tag in SETS:
        v = [r['r2'] for r in r2_rows if r['set'] == tag]
        a = [r['a'] for r in r2_rows if r['set'] == tag]
        if v:
            print(f'  {tag}: 中位 R²={np.median(v):.3f}（{min(v):.3f}~{max(v):.3f}）  a 中位 {np.median(a):.2f}')
    print(f'\n块级 AUC（信号 → 标签）：')
    print(f'{"signal":<14}{"harm(jun/apr/may/all)":>30}{"act(all)":>10}')
    summ = {}
    for s in sig_names:
        vals = []
        for lab in ('harm', 'act'):
            row = []
            for tag in ['jun', 'apr', 'may', 'ALL']:
                sub = rows if tag == 'ALL' else [r for r in rows if r['set'] == tag]
                y = np.array([r[lab] for r in sub])
                if len(set(y)) < 2:
                    row.append(float('nan')); continue
                auc = roc_auc_score(y, np.array([r[s] for r in sub]))
                row.append(auc)
            vals.append(row)
        summ[s] = dict(harm={t: round(v, 3) for t, v in zip(['jun', 'apr', 'may', 'ALL'], vals[0])},
                       act={t: round(v, 3) for t, v in zip(['jun', 'apr', 'may', 'ALL'], vals[1])})
        print(f'{s:<14}' + ' '.join(f'{v:.3f}' for v in vals[0]) + '   ' + ' '.join(f'{v:.3f}' for v in vals[1]))
    print('\n（AUC<0.5 表示信号方向相反，实际判别力 = 1−AUC）')
    import json
    json.dump(summ, open('eval/eval_probe_signals_auc.json', 'w'), indent=1, ensure_ascii=False)
    print('写出 eval/eval_probe_signals.csv、eval_probe_monthr2.csv、eval_probe_signals_auc.json')


if __name__ == '__main__':
    main()
