#!/usr/bin/env python3
"""A: FoCo 式"强-弱模型差距"能否当"该不该信"信号（2026-10-08，零训练）
机制（照 FoCo GaTA）：强模型 CLIP 零样本 + T 次增广投票 → 伪标签；弱模型 SegFormer-B0 出预测；
                    两者在【同一标签空间 ADE-150】上的一致性 = gap。
预登记判决门（先写后跑）：
  G1  sanity：gap(GT) − gap(I) ≥ +0.05（干净图更一致）——不达则本探针失去意义
  G2  域移可测：|gap(I)_apr − gap(I)_may| ≥ 0.05 且 may 更低
  G3  money：Spearman ρ(gap(I), α*) ≤ −0.4（差距越大→越不该信）
  PASS = G1 ∧ G2 ∧ G3
产物：eval/trustgap_1008.csv
"""
import glob, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from transformers import (CLIPModel, CLIPProcessor, SegformerForSemanticSegmentation,
                          AutoImageProcessor)
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = torch.device('cuda:0')
ARMS = {'lovo_k2': 'lovo_k2', 's2main10k': 's2main10k'}
T_AUG, SEED = 8, 20261008
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 3)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))


def affine(X, T):
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1); c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.clip(c[0]*X.astype(np.float64)+c[1], 0, 255).astype(np.uint8)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra*rb).sum()/np.sqrt((ra*ra).sum()*(rb*rb).sum() + 1e-12))


print('[load] CLIP + SegFormer(ADE-150)', flush=True)
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval()
clip_pr = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
seg = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512').to(dev).eval()
seg_pr = AutoImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
id2label = seg.config.id2label
classes = [id2label[i] for i in range(len(id2label))]
prompts = [f'a photo of a {c}.' for c in classes]
with torch.no_grad():
    txt = clip_pr(text=prompts, return_tensors='pt', padding=True).to(dev)
    tfeat = clip.get_text_features(**txt)
    tfeat = tfeat if torch.is_tensor(tfeat) else tfeat.pooler_output
    tfeat = F.normalize(tfeat, dim=-1)
print(f'[load] 文本标签 {len(prompts)} 个（ADE-150）', flush=True)


def clip_probs(img, rng):
    """强模型：T 次随机增广 → 概率平均 → 伪标签分布（FoCo GaTA 的投票做法）"""
    h, w = img.shape[:2]
    ps = []
    with torch.no_grad():
        for _ in range(T_AUG):
            s = rng.uniform(0.7, 1.0)
            ch, cw = int(h*s), int(w*s)
            y0 = rng.integers(0, h-ch+1); x0 = rng.integers(0, w-cw+1)
            crop = img[y0:y0+ch, x0:x0+cw]
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            inp = clip_pr(images=np.ascontiguousarray(crop), return_tensors='pt').to(dev)
            f = clip.get_image_features(**inp)
            f = f if torch.is_tensor(f) else f.pooler_output
            f = F.normalize(f, dim=-1)
            ps.append((100.0 * f @ tfeat.T).softmax(-1)[0])
    return torch.stack(ps).mean(0)


def seg_probs(img):
    """弱模型：SegFormer-B0 逐像素 logits → 全图平均分布"""
    with torch.no_grad():
        inp = seg_pr(images=img, return_tensors='pt').to(dev)
        logits = seg(logits=None, **inp).logits if False else seg(**inp).logits
        p = logits.softmax(1)[0]                     # (150,h,w)
        return p.mean((1, 2))


rng = np.random.default_rng(SEED)
rows = []
for arm in ARMS:
    for sname in ('apr', 'may'):
        for f in sorted(glob.glob(f'{PROJ}/data/gate_v2/{arm}/{sname}/*.npz')):
            z = np.load(f); I, P, T = z['I'], z['P'], z['T']
            pc = {'I': clip_probs(I, rng), 'T': clip_probs(T, rng), 'P': clip_probs(P, rng)}
            sw = {'I': seg_probs(I), 'T': seg_probs(T), 'P': seg_probs(P)}
            gap = {k: float((pc[k].argmax() == sw[k].argmax()).item()) for k in ('I', 'T', 'P')}
            kl = {k: float(F.kl_div((sw[k]+1e-9).log(), pc[k], reduction='sum').item()) for k in ('I', 'T', 'P')}
            # 每图 oracle α（对齐口径）
            al_ps = [psnr(affine(((I.astype(np.float32)+a*(P.astype(np.float32)-I.astype(np.float32))).clip(0, 255).astype(np.uint8)), T), T)
                     for a in ALPHAS]
            astar = float(ALPHAS[int(np.argmax(al_ps))])
            rows.append(dict(arm=arm, set=sname, case=os.path.basename(f)[:-4],
                             gap_I=gap['I'], gap_T=gap['T'], gap_P=gap['P'],
                             kl_I=round(kl['I'], 3), kl_T=round(kl['T'], 3),
                             astar=astar, d0=round(al_ps[0], 2), d1=round(al_ps[-1], 2),
                             gain=round(al_ps[-1]-al_ps[0], 2)))
            print(f'  {arm}/{sname}/{os.path.basename(f)[:-4]}  gap I/T/P={gap["I"]:.0f}/{gap["T"]:.0f}/{gap["P"]:.0f}  α*={astar:.2f}', flush=True)

import csv
with open(f'{PROJ}/eval/trustgap_1008.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

print('\n######## 预登记判决门 ########')
for arm in ARMS:
    sub = [r for r in rows if r['arm'] == arm]
    if not sub:
        continue
    A = [r for r in sub if r['set'] == 'apr']; M = [r for r in sub if r['set'] == 'may']
    g1 = np.mean([r['gap_T']-r['gap_I'] for r in sub])
    g2 = np.mean([r['gap_I'] for r in A]) - np.mean([r['gap_I'] for r in M])
    rho = spearman([r['gap_I'] for r in sub], [r['astar'] for r in sub])
    rho_kl = spearman([r['kl_I'] for r in sub], [r['astar'] for r in sub])
    print(f'\n--- {arm}（n={len(sub)}）---')
    print(f'  gap 均值：apr I={np.mean([r["gap_I"] for r in A]):.2f} | may I={np.mean([r["gap_I"] for r in M]):.2f}'
          f' | GT={np.mean([r["gap_T"] for r in sub]):.2f} | P={np.mean([r["gap_P"] for r in sub]):.2f}')
    print(f'  G1 gap(GT)−gap(I) = {g1:+.2f}（门 ≥+0.05）→ {"PASS" if g1 >= 0.05 else "FAIL"}')
    print(f'  G2 gap(I)_apr − gap(I)_may = {g2:+.2f}（门 ≥+0.05）→ {"PASS" if g2 >= 0.05 else "FAIL"}')
    print(f'  G3 ρ(gap_I, α*) = {rho:+.3f}（门 ≤−0.4）→ {"PASS" if rho <= -0.4 else "FAIL"}'
          f'   [参考 ρ(KL_I, α*) = {rho_kl:+.3f}]')
    print(f'  α* 均值：apr {np.mean([r["astar"] for r in A]):.2f} | may {np.mean([r["astar"] for r in M]):.2f}')
