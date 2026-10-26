#!/usr/bin/env python3
"""反射洞掩膜 v2：闭运算估"未遮挡反射水平"，洞 = 显著低于该水平的大连通区
另：与用户指认区（apr 000000 的 B4–C4 手拿手机 + C4–D4 人头肩）做重叠核对
"""
import sys, os
import cv2, numpy as np
sys.path.insert(0, 'scripts')
from eval_protocol import robust_gain


def mask_v2(I, T, ksize=101, k=1.5, min_frac=0.002, max_frac=0.40, min_solid=0.30):
    g = robust_gain(I, T)
    Ial = np.clip(I.astype(np.float64)*g.reshape(1, 1, 3), 0, 255).mean(2)
    R = Ial - T.astype(np.float64).mean(2)
    q = np.clip(R + 128, 0, 255).astype(np.uint8)          # 有符号平移进 uint8
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    base = cv2.morphologyEx(q, cv2.MORPH_CLOSE, ker)        # 闭运算 ≈ 未遮挡反射水平
    d = q.astype(np.float64) - base.astype(np.float64)      # 负 = 比周围反射暗
    lo = d[d < np.percentile(d, 50)]
    sig = 1.4826*np.median(np.abs(lo - np.median(lo))) if lo.size else 2.0
    m = ((d < -k*max(sig, 1.5))*255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m); kept = []
    for i in range(1, n):
        a = st[i, 4]
        if a < min_frac*m.size or a > max_frac*m.size:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull > 0 and cv2.contourArea(c)/hull < min_solid:
            continue
        out |= comp; kept.append((st[i], cent[i]))
    return out*255, R, d, sig, kept


CASE = ('apr', 'VID_20260426_172719__000000')
z = np.load(f'data/gate_v2/lovo_k2/{CASE[0]}/{CASE[1]}.npz')
I, T = z['I'], z['T']
H, W = I.shape[:2]
m, R, d, sig, kept = mask_v2(I, T)
mm = m > 0
# 用户指认区：B4–C4（x∈[W/4,W/2]）与 C4–D4（x∈[W/2,W]），y∈[3H/4,H]
user = np.zeros((H, W), bool)
user[3*H//4:, W//4:] = True
ov = I.copy()
ov[mm] = (0.35*ov[mm] + 0.65*np.array([0, 0, 255])).astype(np.uint8)
cv2.rectangle(ov, (W//4, 3*H//4), (W-1, H-1), (0, 255, 255), 2)     # 黄框 = 用户指认
for i in range(1, 4):
    cv2.line(ov, (i*W//4, 0), (i*W//4, H), (255, 255, 255), 1)
    cv2.line(ov, (0, i*H//4), (W, i*H//4), (255, 255, 255), 1)
for i in range(4):
    for j in range(4):
        cv2.putText(ov, f'{chr(65+j)}{i+1}', (j*W//4+6, i*H//4+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(ov, f'{chr(65+j)}{i+1}', (j*W//4+6, i*H//4+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
inter = (mm & user).sum(); uni = (mm | user).sum()
print(f'掩膜占比 {100*mm.mean():.1f}% | σ={sig:.1f} | 保留连通域 {len(kept)} 个')
print(f'与用户指认区（B4–C4 + C4–D4 黄框）：掩膜落在其中的比例 {100*inter/max(mm.sum(),1):.1f}% | IoU {inter/max(uni,1):.3f}')
for st, c in kept:
    x, y, w, h, a = st
    print(f'  连通域 框=({x},{y},{w},{h}) 面积={a} 中心=({c[0]:.0f},{c[1]:.0f}) → 列{chr(65+min(3,int(c[0]/(W/4))))} 行{min(3,int(c[1]/(H/4)))+1}')
dn = np.clip((d + 40)/80*255, 0, 255).astype(np.uint8)
dn = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
sheet = np.hstack([I, np.full((H, 6, 3), 255, np.uint8), ov, np.full((H, 6, 3), 255, np.uint8), dn])
cv2.imwrite('fig/hole_mask_check2.png', sheet)
print('saved fig/hole_mask_check2.png', sheet.shape)
