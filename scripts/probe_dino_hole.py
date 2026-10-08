#!/usr/bin/env python3
"""自监督模型能否"察觉"反射洞？
DINOv2-base @518 的 patch 特征 →
  ① k-means 聚类（谁与用户指认的洞区吻合）
  ② 种子相似度扫描（取洞内一点，看相似区能否点亮整块洞）
  ③ 非语义基线：按暗度阈值
核对区（用户指认，apr 000000）：宽松 x≥0.25,y≥0.75；紧 x≥0.5,y≥0.75
"""
import os, sys
import cv2, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel
dev = torch.device('cuda:0')
SZ = 518
model = AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(bgr):
    """手写 DINOv2 预处理（缺 preprocessor_config.json，按官方 ImageNet 归一化）"""
    x = cv2.cvtColor(cv2.resize(bgr, (SZ, SZ), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (x.astype(np.float32)/255.0 - MEAN)/STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)
P = 14
G = SZ // P


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def kmeans(X, k=4, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - c[None])**2).sum(-1)
        a = d.argmin(1)
        for i in range(k):
            if (a == i).any():
                c[i] = X[a == i].mean(0)
    return a


def analyze(tag, I, boxes):
    im = cv2.resize(I, (SZ, SZ), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        t = model(pixel_values=to_tensor(im)).last_hidden_state[0]
    t = t[1:] if t.shape[0] == G*G+1 else t                     # 去掉 CLS
    Fm = F.normalize(t, dim=-1).cpu().numpy().reshape(G, G, -1)
    # ① 聚类
    lab = kmeans(Fm.reshape(-1, Fm.shape[-1]), k=4).reshape(G, G)
    # ② 种子相似度扫描
    base = cv2.cvtColor(cv2.resize(I, (SZ, SZ), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    print(f'\n=== {tag} ===')
    for bname, bx in boxes.items():
        m = np.zeros((G, G), bool)
        y0, x0 = int(bx[1]*G), int(bx[0]*G)
        m[y0:, x0:] = True
        best = (0, None, None)
        for gy in range(0, G, 2):
            for gx in range(0, G, 2):
                if not m[gy, gx]:
                    continue
                sim = (Fm*Fm[gy, gx]).sum(-1)
                for q in (0.6, 0.7, 0.8):
                    thr = np.quantile(sim, q)
                    mm = sim >= thr
                    v = iou(mm, m)
                    if v > best[0]:
                        best = (v, (gy, gx), mm)
        # ③ 非语义基线：暗度阈值
        dark_q = np.quantile(base, 0.30)
        dark = cv2.resize((base <= dark_q).astype(np.uint8), (G, G), interpolation=cv2.INTER_NEAREST) > 0
        print(f'  [{bname}] 聚类最大吻合 IoU={max(iou(lab==c, m) for c in range(4)):.3f} | '
              f'种子相似度最佳 IoU={best[0]:.3f}（种子格 {best[1]}）| 暗度基线 IoU={iou(dark, m):.3f}')
    # 可视化
    vis = cv2.applyColorMap((lab*255//max(lab.max(),1)).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.resize(vis, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
    with torch.no_grad():
        sim0 = (Fm*Fm[G*3//4, G*3//4]).sum(-1)
    hm = cv2.applyColorMap(np.clip((sim0-sim0.min())/(np.ptp(sim0)+1e-6)*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    hm = cv2.resize(hm, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.hstack([I, vis, hm])


CASES = [('apr', 'VID_20260426_172719__000000'), ('may', 'VID_20260512_163859__000000')]
BOXES = {'宽松 x≥.25,y≥.75': (0.25, 0.75), '紧 x≥.50,y≥.75': (0.50, 0.75)}
panels = []
for sname, c in CASES:
    z = np.load(f'data/gate_v2/lovo_k2/{sname}/{c}.npz')
    panels.append(analyze(f'{sname}/{c}', z['I'].copy(), BOXES))
sheet = np.vstack([np.vstack([p, np.full((6, p.shape[1], 3), 255, np.uint8)]) for p in panels])
cv2.putText(sheet, '左: INPUT | 中: DINOv2 patch 聚类(4类) | 右: 以洞内点为种子的相似度', (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
cv2.putText(sheet, '左: INPUT | 中: DINOv2 patch 聚类(4类) | 右: 以洞内点为种子的相似度', (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
os.makedirs('fig', exist_ok=True)
cv2.imwrite('fig/dino_hole.png', sheet)
print('\nsaved fig/dino_hole.png', sheet.shape)
