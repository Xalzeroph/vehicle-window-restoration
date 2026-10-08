#!/usr/bin/env python3
"""P2: ROI 内细节损失归因（低通 / 色调 / 噪声）
分解：总缺失 = GE(GT) − GE(arm)；色调可找回 = GE(affine(arm)) − GE(arm)；结构缺失 = GE(GT) − GE(affine(arm))
噪声：GT 平坦区分位内 arm 的 Laplacian MAD 与 GT 之比
预登记：结构缺失占比 ≥50% → 与 9.21「域外全图低通」一致；<50% → 修正机制表述
"""
import glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, 'scripts')
N = 5

def gemap(im):
    """逐像素梯度幅值（Sobel），用于分区统计"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx*gx + gy*gy)


def affine(X, T):
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1); c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.clip(c[0]*X.astype(np.float64)+c[1], 0, 255).astype(np.uint8)

def lap_mad(im, m):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    l = cv2.Laplacian(g, cv2.CV_32F)
    v = l[m]
    return float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0

rows = []
for f in sorted(glob.glob('data/hole_masks_v2/*.png')):
    base = os.path.basename(f)[:-4]
    sname, sc, fr = base.split('__'); case = f'{sc}__{fr}'
    zp = f'data/gate_v2/lovo_k2/{sname}/{case}.npz'
    if not os.path.exists(zp):
        continue
    z = np.load(zp); I, T = z['I'], z['T']; H, W = I.shape[:2]
    Poff = np.load(f'data/gate_v1/{sname}/{case}.npz')['P']
    if Poff.shape[:2] != (H, W):
        Poff = cv2.resize(Poff, (W, H), interpolation=cv2.INTER_AREA)
    arms = {'identity': I, 'official': Poff, 'lovo_k2': z['P'],
            's2main10k': np.load(f'data/gate_v2/s2main10k/{sname}/{case}.npz')['P']}
    M = cv2.imread(f, 0) > 0
    M = cv2.erode(M.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0      # 去边界伪影
    if M.sum() < 500:
        continue
    geT = gemap(T)[M].mean()
    def fine(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(((g - cv2.GaussianBlur(g, (0, 0), 1.0))**2)[M].mean())
    nT = fine(T)
    for an, P in arms.items():
        Pa = affine(P, T)
        geP, gePa = gemap(P)[M].mean(), gemap(Pa)[M].mean()
        tot = geT - geP
        tone = gePa - geP
        struct = geT - gePa
        rows.append(dict(set=sname, case=case, arm=an, roi=round(float(M.mean()), 3),
                         ge_T=round(geT, 1), ge_arm=round(geP, 1), ge_tone=round(gePa, 1),
                         deficit=round(tot, 1), tone_gain=round(tone, 1), struct_loss=round(struct, 1),
                         struct_share=round(abs(struct)/max(abs(struct)+abs(tone), 1e-6), 3),
                         noise_ratio=round(fine(P)/max(nT, 1e-6), 3)))
import csv
os.makedirs('eval', exist_ok=True)
with open('eval/roi_detail_decomp_1008.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'{"臂":<12}{"n":>4}{"GE(GT)":>9}{"GE(臂)":>9}{"总缺失":>9}{"色调可找回":>11}{"结构缺失":>10}{"结构占比":>9}{"噪声比":>8}')
for an in ('identity', 'official', 'lovo_k2', 's2main10k'):
    sub = [r for r in rows if r['arm'] == an]
    if not sub: continue
    mu = lambda k: np.mean([r[k] for r in sub])
    print(f'{an:<12}{len(sub):>4}{mu("ge_T"):>9.1f}{mu("ge_arm"):>9.1f}{mu("deficit"):>9.1f}'
          f'{mu("tone_gain"):>11.1f}{mu("struct_loss"):>10.1f}{mu("struct_share"):>9.2f}{mu("noise_ratio"):>8.2f}')
for s in ('apr', 'may'):
    sub = [r for r in rows if r['set'] == s and r['arm'] == 's2main10k']
    if sub:
        print(f'  [{s}] s2main10k 结构占比 {np.mean([r["struct_share"] for r in sub]):.2f} | 噪声比 {np.mean([r["noise_ratio"] for r in sub]):.2f}')
print('[写出] eval/roi_detail_decomp_1008.csv')
