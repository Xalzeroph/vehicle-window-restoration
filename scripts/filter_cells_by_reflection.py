#!/usr/bin/env python3
"""按"反射是否被压"筛选用户格子（选项 B）
判据：R = I(gain对齐) − T；若格子内 mean(R) < 0.8 × 全图未标区中位 → 判为"反射被压"保留
      否则（R 正常但 T 更暗）判为"透射层暗物"→ 剔除，并作为 DINOv2 的负提示
产物：data/annot_hole_cells_filtered.json + 统计
"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, 'scripts')
from eval_protocol import robust_gain
N = 5
J = json.load(open('data/annot_hole_cells.json', encoding='utf-8'))
out = {}
print(f'{"帧":<42}{"原标":>5}{"保留":>5}{"剔除":>5}   剔除的格子（R比 / T比）')
tot_k = tot_d = 0
for k, v in J.items():
    sname, case = k.split('/')
    f = f'data/gate_v2/lovo_k2/{sname}/{case}.npz'
    if not os.path.exists(f) or not v.get('pos'):
        continue
    z = np.load(f); I, T = z['I'], z['T']
    H, W = I.shape[:2]
    g = robust_gain(I, T)
    R = np.clip(I.astype(np.float64)*g.reshape(1, 1, 3), 0, 255).mean(2) - T.astype(np.float64).mean(2)
    Tg = T.astype(np.float64).mean(2)
    def cm(cells):
        m = np.zeros((H, W), bool)
        for c in cells:
            col = ord(c[0])-65; row = int(c[1:])-1
            m[int(row*H/N):int((row+1)*H/N), int(col*W/N):int((col+1)*W/N)] = True
        return m
    Mpos = cm(v['pos']); baseR = np.median(R[~Mpos]); baseT = np.median(Tg[~Mpos])
    keep, drop, info = [], [], []
    for c in v['pos']:
        col = ord(c[0])-65; row = int(c[1:])-1
        sl = np.s_[int(row*H/N):int((row+1)*H/N), int(col*W/N):int((col+1)*W/N)]
        rR = R[sl].mean()/max(baseR, 1e-6); rT = Tg[sl].mean()/max(baseT, 1e-6)
        if rR < 0.8:
            keep.append(c)
        else:
            drop.append(c); info.append(f'{c}(R{rR:.2f}/T{rT:.2f})')
    out[k] = {'pos': keep, 'neg': (v.get('neg') or []) + drop, 'w': W, 'h': H}
    tot_k += len(keep); tot_d += len(drop)
    if drop:
        print(f'{k:<42}{len(v["pos"]):>5}{len(keep):>5}{len(drop):>5}   ' + ' '.join(info[:8]) + (' …' if len(info) > 8 else ''))
json.dump(out, open('data/annot_hole_cells_filtered.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n合计：保留 {tot_k} 格，剔除 {tot_d} 格（剔除的用作负提示）')
nf = sum(1 for v in out.values() if v['pos'])
print(f'仍有正标记的帧数：{nf}/{len(out)}')
