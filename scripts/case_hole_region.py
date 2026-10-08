#!/usr/bin/env python3
"""个案研究：在"用户指认的洞区"里，各模型到底干了什么？
洞区（用户指认，apr 000000）：x∈[W/4,W], y∈[3H/4,H]（B4–C4 手拿手机 + C4–D4 人头肩）
指标：区内 luma 变化、区内 ΔPSNR（对齐口径）、细节保留
"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, 'scripts')
from eval_protocol import robust_gain

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))

def affine(X, T):
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1); c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.clip(c[0]*X.astype(np.float64)+c[1], 0, 255).astype(np.uint8)

def lap(im):
    return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())

CASES = [('apr','VID_20260426_172719__000000'), ('may','VID_20260512_163859__000000')]
for sname, c in CASES:
    z1 = np.load(f'data/gate_v1/{sname}/{c}.npz')          # P = 官方
    z2 = np.load(f'data/gate_v2/lovo_k2/{sname}/{c}.npz')  # P = lovo_k2 臂
    z3 = np.load(f'data/gate_v2/s2main10k/{sname}/{c}.npz')# P = s2main10k 臂
    I, T = z2['I'], z2['T']; H, W = I.shape[:2]
    hh = lambda X: cv2.resize(X, (W, H), interpolation=cv2.INTER_AREA) if X.shape[:2] != (H, W) else X
    arms = {'官方RDNet': hh(z1['P']), 'lovo_k2 臂': z2['P'], 's2main10k 臂': z3['P']}
    y0, x0 = 3*H//4, W//4
    sl = np.s_[y0:H, x0:W]
    print(f'\n===== {sname}/{c}  洞区(B4–C4+C4–D4) 尺寸 {W-x0}×{H-y0} =====')
    print(f'  真值 T 区内 luma {T[sl].mean():6.1f} | 细节(Lap) {lap(T[sl]):7.1f}')
    print(f'  输入 I 区内 luma {I[sl].mean():6.1f}（比真值 {I[sl].mean()-T[sl].mean():+.1f}）| 细节 {lap(I[sl]):7.1f} | 对齐后区内 PSNR {psnr(affine(I,T)[sl], T[sl]):5.2f}')
    for name, P in arms.items():
        Pa = affine(P, T)
        print(f'  {name:<12} 区内 luma {P[sl].mean():6.1f}（比真值 {P[sl].mean()-T[sl].mean():+.1f}）| '
              f'细节 {lap(P[sl]):7.1f} | 对齐后区内 PSNR {psnr(Pa[sl], T[sl]):5.2f} | 区内 ΔPSNR(vs输入) {psnr(Pa[sl], T[sl])-psnr(affine(I,T)[sl], T[sl]):+5.2f}')
    # 拼图：I / T / 三臂（全图 + 洞区放大）
    def fit(im, h=260):
        return cv2.resize(im, (int(im.shape[1]*h/im.shape[0]), h), interpolation=cv2.INTER_AREA)
    row = [fit(I), fit(T)] + [fit(P) for P in arms.values()]
    labs = ['INPUT', 'GT', '官方RDNet', 'lovo_k2', 's2main10k']
    row = [cv2.copyMakeBorder(r, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)) for r in row]
    for r, l in zip(row, labs):
        cv2.putText(r, l, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    top = np.hstack(row)
    def crop_fit(im, w=380):
        c = im[sl]
        return cv2.resize(c, (w, int(c.shape[0]*w/c.shape[1])), interpolation=cv2.INTER_NEAREST)
    row2 = [crop_fit(I), crop_fit(T)] + [crop_fit(P) for P in arms.values()]
    row2 = [cv2.copyMakeBorder(r, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)) for r in row2]
    for r, l in zip(row2, labs):
        cv2.putText(r, l+' (洞区1:1)', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    hmax = max(r.shape[0] for r in row2)
    row2 = [cv2.copyMakeBorder(r, 0, hmax-r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)) for r in row2]
    bot = np.hstack(row2)
    Wm = max(top.shape[1], bot.shape[1])
    top = cv2.copyMakeBorder(top, 0, 0, 0, Wm-top.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
    bot = cv2.copyMakeBorder(bot, 0, 0, 0, Wm-bot.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
    sheet = np.vstack([top, np.full((8, Wm, 3), 255, np.uint8), bot])
    os.makedirs('fig', exist_ok=True)
    cv2.imwrite(f'fig/hole_region_{sname}_{c[-6:]}.png', sheet)
    print(f'  [写出] fig/hole_region_{sname}_{c[-6:]}.png {sheet.shape}')
