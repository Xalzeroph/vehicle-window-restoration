#!/usr/bin/env python3
"""按"物理定义"造反射洞掩膜：该处反射被局部压掉、且成连贯物体形状
R̂ = I(光度对齐) − GT        （= 反射层估计）
base = 大窗中值(R̂)          （= 未被挡时的局部反射水平）
hole = 连通域{ R̂ − base < −τ }，面积/实心度过滤
"""
import sys
import cv2, numpy as np
sys.path.insert(0, 'scripts')
from eval_protocol import robust_gain


def hole_mask(I, T, win=61, k=1.2, min_frac=0.004, max_frac=0.35):
    g = robust_gain(I, T)
    Ial = np.clip(I.astype(np.float64)*g.reshape(1, 1, 3), 0, 255).mean(2)
    R = Ial - T.astype(np.float64).mean(2)
    # OpenCV 5 的 medianBlur 只吃 CV_8U：量化到 uint8 再还原（0.5 灰阶精度足够估"局部反射水平"）
    rmin, rmax = float(R.min()), float(R.max())
    q = ((R - rmin) / max(rmax - rmin, 1e-6) * 255).astype(np.uint8)
    base = cv2.medianBlur(q, win).astype(np.float64) / 255.0 * (rmax - rmin) + rmin
    d = R - base
    lo = d[d < np.percentile(d, 50)]
    sig = 1.4826*np.median(np.abs(lo - np.median(lo))) if lo.size else 1.0
    m = ((d < -k*max(sig, 1.0))*255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        area = st[i, 4]
        if area < min_frac*m.size or area > max_frac*m.size:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea); a = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull > 0 and a/hull >= 0.35:          # 实心度：物体轮廓应较实
            out |= comp
    return out*255, R, base, d, sig


if __name__ == '__main__':
    import os
    CASE = ('apr', 'VID_20260426_172719__000000')
    z = np.load(f'data/gate_v2/lovo_k2/{CASE[0]}/{CASE[1]}.npz')
    I, T = z['I'], z['T']
    m, R, base, d, sig = hole_mask(I, T)
    H, W = I.shape[:2]
    ov = I.copy(); mm = m > 0
    ov[mm] = (0.35*ov[mm] + 0.65*np.array([0, 0, 255])).astype(np.uint8)
    for im, name in ((I, 'INPUT'), (ov, f'物理洞掩膜 {100*mm.mean():.1f}%')):
        for i in range(1, 4):
            cv2.line(im, (i*W//4, 0), (i*W//4, H), (0, 255, 255), 1)
            cv2.line(im, (0, i*H//4), (W, i*H//4), (0, 255, 255), 1)
        for i in range(4):
            for j in range(4):
                cv2.putText(im, f'{chr(65+j)}{i+1}', (j*W//4+6, i*H//4+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(im, f'{chr(65+j)}{i+1}', (j*W//4+6, i*H//4+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(im, name, (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(im, name, (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    dn = np.clip((d-d.min())/(np.ptp(d)+1e-6)*255, 0, 255).astype(np.uint8)
    dn = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
    cv2.putText(dn, 'd = R - 局部反射水平（暗=反射被压）', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
    cv2.putText(dn, 'd = R - 局部反射水平（暗=反射被压）', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    sheet = np.hstack([I, np.full((H, 6, 3), 255, np.uint8), ov, np.full((H, 6, 3), 255, np.uint8), dn])
    os.makedirs('fig', exist_ok=True)
    cv2.imwrite('fig/hole_mask_check.png', sheet)
    print(f'saved fig/hole_mask_check.png {sheet.shape} | 洞占比 {100*mm.mean():.1f}% | sigma={sig:.1f}')
    n, lab, st, cent = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        cx, cy = cent[i]
        print(f'  连通域{i}: 框=({x},{y},{w},{h}) 面积={a} 中心=({cx:.0f},{cy:.0f}) '
              f'→ 格子列={chr(65+min(3,int(cx/(W/4))))} 行={min(3,int(cy/(H/4)))+1}')
