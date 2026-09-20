#!/usr/bin/env python3
"""oracle-v3（LPIPS 裁决后，2026-09-20）：结构收益为目标的正确天花板。

逐组件（掩膜形状动作、新门槛 0.3%/32px）：
  w* = argmax_w ΔNGR_m(c,w)   s.t. ΔLPIPS_crop(w) ≤ 0 且 组件 ΔPSNR(w) ≥ 0
  无可行 w → w=0 并记 infeasible。
v2（min LPIPS s.t. ΔPSNR≥0）降为 ablation（目标错位代价）。
产物：runs/pred/{set}_oraclev3/、eval/eval_probe_oracle_v3_choices.csv
"""
import csv, os, sys
import cv2
import numpy as np
import torch
import pyiqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import (compute_mask, crop_components, ls_scale, apply_gm,  # noqa: E402
                           lpips_pair, psnr, ngr_ratio_map, N0)

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet', 'runs/pred/real_pair_0917_oraclev3'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet', 'runs/pred/apr_oraclev3'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet', 'runs/pred/may_oraclev3')}
WS = [0.0, 0.25, 0.5, 0.75, 1.0]
PAD = 16


def cases(cfg):
    d, gtn, _, _ = cfg
    out = []
    for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        sd = os.path.join(d, sc)
        for fn in (sorted(f for f in os.listdir(sd) if f.startswith('I')) or ['I.png']):
            key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
            out.append((key, os.path.join(sd, fn), os.path.join(sd, gtn)))
    return out


def comp_ngr(ratio, cm):
    return float(ratio.mean(axis=2)[cm].mean())


def main():
    dev = torch.device('cuda:0')
    lp = pyiqa.create_metric('lpips', device=dev)
    choices = []
    with torch.no_grad():
        for tag, cfg in SETS.items():
            d, gtn, pred_root, out_root = cfg
            for key, ip, tp in cases(cfg):
                I = cv2.imread(ip); T = cv2.imread(tp)
                pp = os.path.join(pred_root, key, 'ytmt_ucs_sirs_l.png')
                if I is None or T is None or not os.path.exists(pp):
                    print(f'{key}: 缺文件 -> 跳过'); continue
                P = cv2.imread(pp)
                if P.shape != T.shape:
                    P = cv2.resize(P, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
                h, w = T.shape[:2]
                M, _, _ = compute_mask(I, T)
                _, lab, _, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), 8)
                GI = apply_gm(I, ls_scale(I, T)).astype(np.uint8)
                Tf = T.astype(np.float64)
                ref_ratio, _ = ngr_ratio_map(Tf, GI.astype(np.float64), N0)
                I64 = I.astype(np.float64); P64 = P.astype(np.float64)
                wmap = np.zeros((h, w), np.float64)
                comps, _, _ = crop_components(M, min_area_frac=0.003, min_side=32)
                for (cx, cy, cw, ch) in comps:
                    sub = lab[cy:cy + ch, cx:cx + cw]
                    vals, counts = np.unique(sub[sub > 0], return_counts=True)
                    if len(vals) == 0:
                        continue
                    lbl = vals[np.argmax(counts)]
                    cm = np.zeros((h, w), bool)
                    cm[cy:cy + ch, cx:cx + cw] = (sub == lbl)
                    x0, y0 = max(0, cx - PAD), max(0, cy - PAD)
                    x1, y1 = min(w, cx + cw + PAD), min(h, cy + ch + PAD)
                    ngr0 = comp_ngr(ref_ratio, cm)
                    p0 = psnr(GI[cm], T[cm])
                    best = None
                    for wv in WS:
                        O = I64.copy()
                        O[cm] = I64[cm] * (1 - wv) + P64[cm] * wv
                        O = np.clip(O, 0, 255).astype(np.uint8)
                        Og = apply_gm(O, ls_scale(O, T)).astype(np.uint8)
                        dP = psnr(Og[cm], T[cm]) - p0
                        dL = lpips_pair(lp, T[y0:y1, x0:x1], Og[y0:y1, x0:x1]) - \
                            lpips_pair(lp, T[y0:y1, x0:x1], GI[y0:y1, x0:x1])
                        ratio_w, _ = ngr_ratio_map(Tf, Og.astype(np.float64), N0)
                        gain = ngr0 - comp_ngr(ratio_w, cm)      # 正 = 结构变好
                        feas = (dP >= -1e-9) and (dL <= 0)
                        if best is None or (feas, gain) > (best[0], best[1]):
                            best = (feas, gain, wv, dP, dL)
                    feas, gain, wv, dP, dL = best
                    if not feas:
                        wv = 0.0
                    wmap[cm] = np.maximum(wmap[cm], wv)
                    choices.append(dict(set=tag, case=key, x=cx, y=cy, w=cw, h=ch,
                                        w_star=wv, feasible=int(feas), dNGR=round(gain, 4),
                                        dP=round(dP, 2), dL=round(dL, 4)))
                wsm = cv2.GaussianBlur(wmap.astype(np.float32), (0, 0), 3.0)
                O = np.clip(I64 + wsm[..., None] * (P64 - I64), 0, 255).astype(np.uint8)
                od = os.path.join(out_root, key)
                os.makedirs(od, exist_ok=True)
                cv2.imwrite(os.path.join(od, 'oraclev3.png'), O)
                n_act = sum(1 for c in choices if c['case'] == key and c['w_star'] > 0)
                print(f'{tag}/{key}: 组件 {len(comps)} 动手 {n_act}')
    with open('eval/eval_probe_oracle_v3_choices.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(choices[0].keys()))
        wr.writeheader(); wr.writerows(choices)
    for tag in SETS:
        sub = [c for c in choices if c['set'] == tag]
        if sub:
            act = [c for c in sub if c['w_star'] > 0]
            print(f"{tag}: 组件 {len(sub)} 动手 {len(act)} ({100*len(act)/len(sub):.0f}%) "
                  f"不可行 {sum(1 for c in sub if not c['feasible'])} | 动手时 ΔNGR 中位 {np.median([c['dNGR'] for c in act]) if act else 'NA'}")
    print('写出 eval/eval_probe_oracle_v3_choices.csv')


if __name__ == '__main__':
    main()
