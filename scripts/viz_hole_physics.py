#!/usr/bin/env python3
"""看清"人影洞"到底是什么：输入 / 真值 / Ω(协议定义) / 带符号差分 / 残差 / 低残差大块"""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, 'scripts')
from eval_protocol import compute_mask, robust_gain
PROJ = os.getcwd()
H, W = 300, 380


def fit(im):
    im = cv2.resize(im, (max(1, int(im.shape[1]*H/im.shape[0])), H), interpolation=cv2.INTER_AREA)
    if im.shape[1] > W:
        x0 = (im.shape[1]-W)//2; im = im[:, x0:x0+W]
    c = np.zeros((H, W, 3), np.uint8); x = (W-im.shape[1])//2; c[:, x:x+im.shape[1]] = im
    return c


def tag(im, t):
    cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def hs(imgs, gap=5):
    out = imgs[0]
    for i in imgs[1:]:
        out = np.hstack([out, np.full((out.shape[0], gap, 3), 240, np.uint8), i])
    return out


CASES = [('apr', 'VID_20260426_172719__000000'), ('apr', 'VID_20260426_173604__000014'),
         ('may', 'VID_20260512_162203__000000'), ('may', 'VID_20260512_163859__000000')]
rows = []
for sname, case in CASES:
    f = f'data/gate_v2/lovo_k2/{sname}/{case}.npz'
    if not os.path.exists(f):
        continue
    z = np.load(f); I, T = z['I'], z['T']
    g = robust_gain(I, T)
    Ta = np.clip(T.astype(np.float64)*g.reshape(1, 1, 3), 0, 255)
    signed = I.astype(np.float64) - Ta                     # >0 输入比真值亮（反射叠加）；<0 输入更暗
    D = np.abs(signed).max(2)
    M, info, _ = compute_mask(I, T)
    # 带符号热图（蓝=负、红=正）
    s_norm = np.clip(signed.mean(2)/60.0, -1, 1)
    heat = np.zeros_like(I)
    heat[..., 2] = np.clip(-s_norm, 0, 1)*255     # 蓝：比真值暗
    heat[..., 1] = (1-np.abs(s_norm))*90
    heat[..., 1] = np.clip(heat[...,1], 0, 255)
    heat[..., 2] = np.clip(heat[..., 2], 0, 255)
    heat[..., 1] = np.clip((1-np.abs(s_norm))*80, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip((1-np.abs(s_norm))*80, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip((-s_norm).clip(0, 1)*255 + (s_norm.clip(0, 1)*0), 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip((-s_norm).clip(0, 1)*255, 0, 255).astype(np.uint8)
    heat[..., 0] = np.clip((s_norm).clip(0, 1)*255, 0, 255).astype(np.uint8)   # 红：比真值亮
    # 低残差大连通域（疑似"反射被挡"区）
    low = ((D < np.percentile(D, 40))*255).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(low)
    cand = np.zeros_like(low)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, 4]))
        if stats[k, 4] > 0.03*low.size:
            cand = (lab == k).astype(np.uint8)*255
    ov = I.copy(); ov[M] = (0.45*ov[M] + 0.55*np.array([0, 0, 255])).astype(np.uint8)
    ov2 = I.copy(); m2 = cand > 0
    ov2[m2] = (0.45*ov2[m2] + 0.55*np.array([0, 255, 0])).astype(np.uint8)
    d8 = np.clip(D*3, 0, 255).astype(np.uint8)
    row = hs([tag(fit(I), f'{sname} INPUT'), tag(fit(T), 'GT(黑布档)'),
              tag(fit(ov), f'Ω 协议掩膜 {100*M.mean():.0f}%'),
              tag(fit(heat), '带符号差分 红=亮/蓝=暗'),
              tag(fit(cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)), '残差 |I-gT|'),
              tag(fit(ov2), f'低残差大块 {100*m2.mean():.0f}%')])
    rows.append(row)
    print(f'  {case}: Ω={100*M.mean():.0f}% | 低残差大块={100*m2.mean():.0f}% | 洞内I-T均值={signed[M].mean():+.1f} 洞外={signed[~M].mean():+.1f}', flush=True)

head = np.zeros((54, rows[0].shape[1], 3), np.uint8)
cv2.putText(head, 'r_obs 口径的 Omega 到底是什么？  （注意：Omega 由 |I-gT|>3sigma 定义，故"洞内残差更大"是同义反复）',
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
cv2.putText(head, '关键看第 4 格：红=输入比真值亮（反射叠加）；蓝=输入比真值暗（反射被挡）',
            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
sheet = np.vstack([head] + [np.vstack([r, np.full((5, r.shape[1], 3), 240, np.uint8)]) for r in rows])
os.makedirs('fig', exist_ok=True)
cv2.imwrite('fig/hole_physics.png', sheet)
print('saved fig/hole_physics.png', sheet.shape)
