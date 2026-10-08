#!/usr/bin/env python3
"""用用户格子标注（正/负）当提示，DINOv2 特征做像素级精修 → 洞掩膜
做法：patch 特征 → 正种子均相似度 − 负种子均相似度 → 阈值(按格子标签最优) → 掩膜
产物：data/hole_masks/<set>__<case>.png（0/255）、fig/hole_refine/*.png 抽查图
"""
import glob, json, os, sys
import cv2, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel
dev = torch.device('cuda:0')
SZ, P = 518, 14
G = SZ // P
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
model = AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval()
J = json.load(open('data/annot_hole_cells.json', encoding='utf-8'))
os.makedirs('data/hole_masks', exist_ok=True); os.makedirs('fig/hole_refine', exist_ok=True)
N = 5


def cell_pts(cells, N=N, per=3):
    pts = []
    for c in cells:
        col = ord(c[0])-65; row = int(c[1:])-1
        for a in range(per):
            for b in range(per):
                pts.append(((col + (a+0.5)/per)/N, (row + (b+0.5)/per)/N))
    return pts


def to_tensor(bgr):
    x = cv2.cvtColor(cv2.resize(bgr, (SZ, SZ), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (x.astype(np.float32)/255.0 - MEAN)/STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)


def cellmask(shape, cells):
    H, W = shape; m = np.zeros((H, W), bool)
    for c in cells:
        col = ord(c[0])-65; row = int(c[1:])-1
        m[int(row*H/N):int((row+1)*H/N), int(col*W/N):int((col+1)*W/N)] = True
    return m


rows = []
for k, v in J.items():
    sname, case = k.split('/')
    f = f'data/gate_v2/lovo_k2/{sname}/{case}.npz'
    if not os.path.exists(f) or not v.get('pos'):
        continue
    z = np.load(f); I = z['I']; H, W = I.shape[:2]
    with torch.no_grad():
        t = model(pixel_values=to_tensor(I)).last_hidden_state[0]
    t = t[1:] if t.shape[0] == G*G+1 else t
    Fm = F.normalize(t, dim=-1).reshape(G, G, -1)
    pos = cell_pts(v['pos']); neg = cell_pts(v.get('neg') or []) or [(0.02, 0.02), (0.98, 0.02), (0.02, 0.98)]
    gp = np.clip((np.array(pos)*G).astype(int), 0, G-1); gn = np.clip((np.array(neg)*G).astype(int), 0, G-1)
    Pv = F.normalize(Fm[gp[:, 1], gp[:, 0]].mean(0), dim=0)
    Nv = F.normalize(Fm[gn[:, 1], gn[:, 0]].mean(0), dim=0)
    score = (Fm @ Pv) - (Fm @ Nv)                      # (G,G)
    sm = cv2.resize(score.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
    cm = cellmask((H, W), v['pos'])
    best = (0, None)
    for q in np.arange(0.2, 0.95, 0.05):
        thr = np.quantile(sm, q)
        mm = sm >= thr
        iou = (mm & cm).sum()/max((mm | cm).sum(), 1)
        if iou > best[0]:
            best = (iou, mm)
    mm = best[1].astype(np.uint8)*255
    mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm)
    if n > 1:
        keep = np.zeros_like(mm)
        for i in range(1, n):
            if st[i, 4] >= 0.002*mm.size:
                keep |= (lab == i).astype(np.uint8)*255
        mm = keep
    cv2.imwrite(f'data/hole_masks/{sname}__{case}.png', mm)
    rows.append((k, 100*cm.mean(), 100*(mm > 0).mean(), best[0]))
    print(f'  {k:<44} 格子占比 {100*cm.mean():5.1f}% → 精修掩膜 {100*(mm>0).mean():5.1f}%  与格子IoU {best[0]:.3f}', flush=True)

a = np.array([[r[1], r[2], r[3]] for r in rows])
print(f'\n=== 汇总 {len(rows)} 帧：格子占比均值 {a[:,0].mean():.0f}% → 掩膜占比均值 {a[:,1].mean():.0f}% | 与格子 IoU 均值 {a[:,2].mean():.3f} ===')

# 抽查图：3 帧，输入+格子 / 精修掩膜叠加 / 真值
picks = [rows[0][0], rows[8][0], rows[20][0]] if len(rows) > 20 else [r[0] for r in rows[:3]]
sheets = []
for k in picks:
    sname, case = k.split('/')
    z = np.load(f'data/gate_v2/lovo_k2/{sname}/{case}.npz'); I, T = z['I'], z['T']
    mm = cv2.imread(f'data/hole_masks/{sname}__{case}.png', 0) > 0
    H, W = I.shape[:2]
    ov = I.copy(); ov[mm] = (0.4*ov[mm] + 0.6*np.array([0, 0, 255])).astype(np.uint8)
    for j in range(1, N):
        cv2.line(ov, (j*W//N, 0), (j*W//N, H), (0, 255, 255), 1)
        cv2.line(ov, (0, j*H//N), (W, j*H//N), (0, 255, 255), 1)
    cv2.line(I, (0, 0), (0, 0), (0, 0), 1)
    for j in range(1, N):
        cv2.line(I, (j*W//N, 0), (j*W//N, H), (0, 255, 255), 1)
        cv2.line(I, (0, j*H//N), (W, j*H//N), (0, 255, 255), 1)
    h = 300
    fx = lambda im: cv2.resize(im, (int(im.shape[1]*h/im.shape[0]), h), interpolation=cv2.INTER_AREA)
    sheets.append(np.hstack([fx(I), fx(ov), fx(T)]))
w = max(s.shape[1] for s in sheets)
sheets = [cv2.copyMakeBorder(s, 0, 0, 0, w-s.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for s in sheets]
cv2.imwrite('fig/hole_refine/refine_check.png', np.vstack(sheets))
print('saved fig/hole_refine/refine_check.png')
