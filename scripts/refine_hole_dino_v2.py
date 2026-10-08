#!/usr/bin/env python3
"""DINOv2 精修 v2：加反例（全部未标格）+ 空间约束（标注格外扩 1 格内）
score = sim(正种子) − sim(负种子)；掩膜 = score ≥ 阈值 ∧ 允许区
"""
import glob, json, os, sys
import cv2, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel
dev = torch.device('cuda:0')
SZ = 518; G = SZ // 14; N = 5
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
model = AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval()
J = json.load(open('data/annot_hole_cells.json', encoding='utf-8'))
os.makedirs('data/hole_masks_v2', exist_ok=True)

def pts_of(cells, per=3):
    out = []
    for c in cells:
        col, row = ord(c[0])-65, int(c[1:])-1
        for a in range(per):
            for b in range(per):
                out.append(((col+(a+0.5)/per)/N, (row+(b+0.5)/per)/N))
    return out

def cellmask(shape, cells, dilate=0):
    H, W = shape; m = np.zeros((H, W), np.uint8)
    for c in cells:
        col, row = ord(c[0])-65, int(c[1:])-1
        m[int(row*H/N):int((row+1)*H/N), int(col*W/N):int((col+1)*W/N)] = 1
    if dilate:
        k = np.ones((int(H/N)*2*dilate+1, int(W/N)*2*dilate+1), np.uint8)
        m = cv2.dilate(m, k)
    return m.astype(bool)

def to_tensor(bgr):
    x = cv2.cvtColor(cv2.resize(bgr, (SZ, SZ), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (x.astype(np.float32)/255.0 - MEAN)/STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)

rows = []
for k, v in J.items():
    if not v.get('pos'):
        continue
    sname, case = k.split('/')
    f = f'data/gate_v2/lovo_k2/{sname}/{case}.npz'
    if not os.path.exists(f):
        continue
    I, T = (lambda z: (z['I'], z['T']))(np.load(f)); H, W = I.shape[:2]
    allc = [f'{chr(65+c)}{r+1}' for r in range(N) for c in range(N)]
    negc = [c for c in allc if c not in v['pos']]
    with torch.no_grad():
        t = model(pixel_values=to_tensor(I)).last_hidden_state[0]
    t = t[1:] if t.shape[0] == G*G+1 else t
    Fm = F.normalize(t, dim=-1).reshape(G, G, -1)
    gp = np.clip((np.array(pts_of(v['pos']))*G).astype(int), 0, G-1)
    gn = np.clip((np.array(pts_of(negc, per=2))*G).astype(int), 0, G-1)
    Pv = F.normalize(Fm[gp[:, 1], gp[:, 0]].mean(0), dim=0)
    Nv = F.normalize(Fm[gn[:, 1], gn[:, 0]].mean(0), dim=0)
    score = (Fm @ Pv) - (Fm @ Nv)
    sm = cv2.resize(score.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
    allow = cellmask((H, W), v['pos'], dilate=1)          # 允许区 = 标注格 + 1 格
    cm = cellmask((H, W), v['pos'])
    best = (0, None)
    for q in np.arange(0.55, 0.98, 0.02):
        thr = np.quantile(sm, q)
        mm = (sm >= thr) & allow
        iou = (mm & cm).sum()/max((mm | cm).sum(), 1)
        if iou > best[0]:
            best = (iou, mm)
    mm = (best[1].astype(np.uint8))*255
    mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm)
    if n > 1:
        keep = np.zeros_like(mm)
        for i in range(1, n):
            if st[i, 4] >= 0.001*mm.size:
                keep |= (lab == i).astype(np.uint8)*255
        mm = keep
    cv2.imwrite(f'data/hole_masks_v2/{sname}__{case}.png', mm)
    rows.append((k, cm.mean(), (mm > 0).mean(), best[0]))
a = np.array([[r[1], r[2], r[3]] for r in rows])
print(f'=== v2 汇总 {len(rows)} 帧：格子 {100*a[:,0].mean():.0f}% → 掩膜 {100*a[:,1].mean():.0f}%（v1 是 49%）| IoU {a[:,2].mean():.3f}（v1 是 0.593）===')
# 对比图
COLS, WW, HH = 4, 430, 310
def panel(sname, case, idx):
    z = np.load(f'data/gate_v2/lovo_k2/{sname}/{case}.npz'); I = z['I']
    im = cv2.resize(I, (WW, HH), interpolation=cv2.INTER_AREA)
    for tag, d, col in (('v1', 'data/hole_masks', (0, 0, 255)), ('v2', 'data/hole_masks_v2', (255, 0, 255))):
        p = f'{d}/{sname}__{case}.png'
        if os.path.exists(p):
            m = cv2.resize((cv2.imread(p, 0) > 0).astype(np.uint8), (WW, HH), interpolation=cv2.INTER_NEAREST) > 0
            if tag == 'v1':
                im[m] = (0.6*im[m] + 0.4*np.array(col)).astype(np.uint8)
    p2 = f'data/hole_masks_v2/{sname}__{case}.png'
    if os.path.exists(p2):
        m2 = cv2.resize((cv2.imread(p2, 0) > 0).astype(np.uint8), (WW, HH), interpolation=cv2.INTER_NEAREST) > 0
        edge = cv2.morphologyEx(m2.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        im[edge] = (255, 0, 255)
    for c in (J[f'{sname}/{case}'].get('pos') or []):
        x0, y0 = int((ord(c[0])-65)*WW/N), int((int(c[1:])-1)*HH/N)
        cv2.rectangle(im, (x0+1, y0+1), (x0+WW//N-1, y0+HH//N-1), (0, 255, 255), 1)
    bar = np.zeros((20, WW, 3), np.uint8)
    cv2.putText(bar, f'{idx:02d} {sname} {case[-6:]}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
    return np.vstack([bar, im])
for sname in ('apr', 'may'):
    fs = sorted(glob.glob(f'data/gate_v2/lovo_k2/{sname}/*.npz'))
    ps = [panel(sname, os.path.basename(f)[:-4], i+1) for i, f in enumerate(fs)]
    while len(ps) % COLS:
        ps.append(np.zeros_like(ps[0]))
    rows2 = [np.hstack(ps[i:i+COLS]) for i in range(0, len(ps), COLS)]
    sheet = np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in rows2])
    cv2.imwrite(f'fig/dino_seg_v2_{sname}.png', sheet)
    print(f'fig/dino_seg_v2_{sname}.png {sheet.shape}')
