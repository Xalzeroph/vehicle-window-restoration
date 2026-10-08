#!/usr/bin/env python3
"""选项B 筛选 v2：差分暗度判据
ΔI = mean(I_unmarked) − mean(I_cell)；ΔT = mean(T_unmarked) − mean(T_cell)（gain 对齐域）
保留条件：ΔI − ΔT ≥ 10 灰度  → 该格"比周围暗"无法由透射层解释，即反射被压
产物：data/annot_hole_cells_filtered.json（pos=保留, neg=剔除）
"""
import json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from eval_protocol import robust_gain
N, MARGIN = 5, 10.0
J = json.load(open('data/annot_hole_cells.json', encoding='utf-8'))
out = {}
tot_k = tot_d = 0
print(f'{"帧":<42}{"原":>4}{"留":>4}{"剔":>4}   剔除格 (ΔI/ΔT)')
for k, v in J.items():
    sname, case = k.split('/')
    f = f'data/gate_v2/lovo_k2/{sname}/{case}.npz'
    if not os.path.exists(f) or not v.get('pos'):
        continue
    z = np.load(f); I, T = z['I'], z['T']
    H, W = I.shape[:2]
    g = robust_gain(I, T)
    Ig = np.clip(I.astype(np.float64)*g.reshape(1,1,3), 0, 255).mean(2)
    Tg = T.astype(np.float64).mean(2)
    allpos = np.zeros((H, W), bool)
    for c in v['pos']:
        col = ord(c[0])-65; row = int(c[1:])-1
        allpos[int(row*H/N):int((row+1)*H/N), int(col*W/N):int((col+1)*W/N)] = True
    muI, muT = Ig[~allpos].mean(), Tg[~allpos].mean()
    keep, drop, info = [], [], []
    for c in v['pos']:
        col = ord(c[0])-65; row = int(c[1:])-1
        sl = np.s_[int(row*H/N):int((row+1)*H/N), int(col*W/N):int((col+1)*W/N)]
        dI = muI - Ig[sl].mean(); dT = muT - Tg[sl].mean()
        (keep if (dI - dT) >= MARGIN else drop).append(c)
        if (dI - dT) < MARGIN:
            info.append(f'{c}({dI:+.0f}/{dT:+.0f})')
    out[k] = {'pos': keep, 'neg': (v.get('neg') or []) + drop, 'w': W, 'h': H}
    tot_k += len(keep); tot_d += len(drop)
    if drop:
        print(f'{k:<42}{len(v["pos"]):>4}{len(keep):>4}{len(drop):>4}   ' + ' '.join(info[:6]) + (' …' if len(info) > 6 else ''))
json.dump(out, open('data/annot_hole_cells_filtered.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n合计：保留 {tot_k} 格 / 剔除 {tot_d} 格；仍有正标记的帧 {sum(1 for v in out.values() if v["pos"])}/{len(out)}')
