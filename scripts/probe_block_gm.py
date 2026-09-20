#!/usr/bin/env python3
"""补测①b：块级 gm 口径复算（2026-09-19 方法线）。

对照 E7 的 raw 块表（raw 复现已核：三套均差 0.000 dB），算 gm 口径（I、P 各自 LS 对齐 T 后混合）的
块级 ΔPSNR / 约束 oracle，量化口径切换对块级结论的影响。
产物：eval/eval_probe_block_gm.csv
"""
import csv, os, sys
import cv2
import numpy as np
import torch
import pyiqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import ls_scale, apply_gm  # noqa: E402

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet')}
BS = 128
WS = [0.0, 0.25, 0.5, 0.75, 1.0]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def to_t(b):
    return torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)


def cases(cfg):
    d, gtn, _ = cfg
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
    rows = []
    with torch.no_grad():
        for tag, cfg in SETS.items():
            d, gtn, pred_root = cfg
            for key, ip, tp in cases(cfg):
                I = cv2.imread(ip); T = cv2.imread(tp)
                pp = os.path.join(pred_root, key, 'ytmt_ucs_sirs_l.png')
                if I is None or T is None or not os.path.exists(pp):
                    continue
                P = cv2.imread(pp)
                if P.shape != T.shape:
                    P = cv2.resize(P, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
                GI = apply_gm(I, ls_scale(I, T)).astype(np.uint8)
                GP = apply_gm(P, ls_scale(P, T)).astype(np.uint8)
                h, w = T.shape[:2]
                for y in range(0, h - BS + 1, BS):
                    for x in range(0, w - BS + 1, BS):
                        tb = T[y:y + BS, x:x + BS]
                        if tb.mean() < 15:
                            continue
                        ib, pb = GI[y:y + BS, x:x + BS], GP[y:y + BS, x:x + BS]
                        tg = to_t(tb).to(dev)
                        li = float(lp(to_t(ib).to(dev), tg))
                        p0 = psnr(ib, tb)
                        dPs, dLs = [], []
                        for wv in WS:
                            o = np.clip(ib.astype(np.float64) * (1 - wv) + pb.astype(np.float64) * wv, 0, 255).astype(np.uint8)
                            dPs.append(psnr(o, tb) - p0)
                            dLs.append(float(lp(to_t(o).to(dev), tg)) - li)
                        feas = [k for k in range(len(WS)) if dLs[k] <= 0]
                        kstar = max(feas, key=lambda k: dPs[k]) if feas else 0
                        rows.append(dict(set=tag, case=key, y=y, x=x, w_star=WS[kstar],
                                         dP_rd=round(dPs[-1], 3), dL_rd=round(dLs[-1], 4),
                                         dP_con=round(dPs[kstar], 3)))
    with open('eval/eval_probe_block_gm.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    raw = list(csv.DictReader(open('eval/eval_probe_oracle_blocks.csv')))
    print(f'块 {len(rows)}（gm 口径）')
    for tag in SETS:
        g = [r for r in rows if r['set'] == tag]
        r_ = [r for r in raw if r['set'] == tag]
        if not g:
            continue
        print(f"{tag}: gm 纯RDNet {np.mean([b['dP_rd'] for b in g]):+.2f} (raw {np.mean([float(b['dP_rd']) for b in r_]):+.2f}) "
              f"| gm oracle {np.mean([b['dP_con'] for b in g]):+.2f} (raw {np.mean([float(b['dP_con']) for b in r_]):+.2f}) "
              f"| gm 选w=0 {100*np.mean([b['w_star']==0 for b in g]):.0f}% (raw {100*np.mean([float(b['w_star'])==0 for b in r_]):.0f}%)")
    print('写出 eval/eval_probe_block_gm.csv')


if __name__ == '__main__':
    main()
