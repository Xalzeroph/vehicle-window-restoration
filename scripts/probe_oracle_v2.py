#!/usr/bin/env python3
"""补测②（v2）：协议口径对齐的 oracle 天花板（2026-09-19 方法线 P2 前置）。

与 v1（块级 LPIPS 约束）不同：逐**掩膜连通域**（面积≥1%）在 {0,.25,.5,.75,1} 内选
使该组件裁剪 LPIPS 最小、且组件 ΔPSNR≥0 的 w；σ8 羽化后整臂交给 eval_protocol（gm 口径）评测。
产物：runs/pred/{set}_oraclev2/、eval/eval_probe_oracle_v2_choices.csv
"""
import csv, os, sys
import cv2
import numpy as np
import torch
import pyiqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import compute_mask, ls_scale, apply_gm, crop_components, lpips_pair, psnr, to_tensor  # noqa: E402

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet', 'runs/pred/real_pair_0917_oraclev2'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet', 'runs/pred/apr_oraclev2'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet', 'runs/pred/may_oraclev2')}
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
                I64 = I.astype(np.float64); P64 = P.astype(np.float64)
                wmap = np.zeros((h, w), np.float64)
                comps, _, _ = crop_components(M, min_area_frac=0.003, min_side=32)
                for (cx, cy, cw, ch) in comps:
                    sub = lab[cy:cy + ch, cx:cx + cw]
                    vals, counts = np.unique(sub[sub > 0], return_counts=True)
                    if len(vals) == 0:
                        continue
                    lbl = vals[np.argmax(counts)]
                    cmask = np.zeros((h, w), bool)
                    cmask[cy:cy + ch, cx:cx + cw] = (sub == lbl)
                    x0, y0 = max(0, cx - PAD), max(0, cy - PAD)
                    x1, y1 = min(w, cx + cw + PAD), min(h, cy + ch + PAD)
                    p0 = psnr(GI[cmask], T[cmask])
                    best = None
                    for wv in WS:
                        O = I64.copy()
                        O[cmask] = I64[cmask] * (1 - wv) + P64[cmask] * wv
                        O = np.clip(O, 0, 255).astype(np.uint8)
                        Og = apply_gm(O, ls_scale(O, T)).astype(np.uint8)
                        dP_comp = psnr(Og[cmask], T[cmask]) - p0
                        l = lpips_pair(lp, T[y0:y1, x0:x1], Og[y0:y1, x0:x1])
                        feas = (dP_comp >= -1e-9)
                        key_ = (l if feas else 1e9, wv)
                        if best is None or key_ < best[0]:
                            best = (key_, wv, l, dP_comp)
                    wv = best[1]
                    wmap[cmask] = np.maximum(wmap[cmask], wv)
                    choices.append(dict(set=tag, case=key, x=cx, y=cy, w=cw, h=ch,
                                        w_star=wv, crop_lpips=round(best[2], 4), dP_comp=round(best[3], 2)))
                wsm = cv2.GaussianBlur(wmap.astype(np.float32), (0, 0), 3.0)
                O = np.clip(I64 + wsm[..., None] * (P64 - I64), 0, 255).astype(np.uint8)
                od = os.path.join(out_root, key)
                os.makedirs(od, exist_ok=True)
                cv2.imwrite(os.path.join(od, 'oraclev2.png'), O)
                print(f'{tag}/{key}: 组件 {len(comps)}，w*={[c["w_star"] for c in choices if c["case"] == key]}')
    with open('eval/eval_probe_oracle_v2_choices.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(choices[0].keys()))
        wr.writeheader(); wr.writerows(choices)
    print(f'\n组件记录 {len(choices)} -> eval/eval_probe_oracle_v2_choices.csv')


if __name__ == '__main__':
    main()
