#!/usr/bin/env python3
"""A-2: 忠实复现 FoCo GaTA 的"强-弱差距"（ImageNet 标签空间，零训练）
强=C LIP ViT-B/32 零样本(1000 类, T 次增广投票)  弱=ResNet18(a1_in1k)
信号1 gap      = 两者 top-1 是否一致
信号2 stab_*   = 各自在 T 次增广下的投票稳定率（连续量，补充梯度信息）
预登记门：G1 gap(T)−gap(I) ≥ +0.05；G2 gap(I)_apr − gap(I)_may ≥ +0.05；
          G3 域内 ρ(signal, α*) 显著（|ρ| ≥ 0.4 且两域同号）
产物：eval/trustgap_imagenet_1008.csv
"""
import glob, os, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import timm
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = torch.device('cuda:0')
T_AUG, SEED = 8, 20261008
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 3)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))


def affine(X, T):
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1); c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.clip(c[0]*X.astype(np.float64)+c[1], 0, 255).astype(np.uint8)


print('[load] CLIP + ResNet18 + ImageNet-1000 可读类名', flush=True)
from timm.data import ImageNetInfo
names = ImageNetInfo().label_descriptions()
names = [n.split(',')[0].strip().lower() for n in names]
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval()
cpr = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
res = timm.create_model('resnet18.a1_in1k', pretrained=True).to(dev).eval()
with torch.no_grad():
    t = cpr(text=[f'a photo of a {n}.' for n in names], return_tensors='pt', padding=True).to(dev)
    tfeat = clip.get_text_features(**t)
    tfeat = F.normalize(tfeat if torch.is_tensor(tfeat) else tfeat.pooler_output, dim=-1)


def views(img, rng, n=T_AUG, size=224):
    h, w = img.shape[:2]; out = []
    for _ in range(n):
        s = rng.uniform(0.5, 1.0); ch, cw = int(h*s), int(w*s)
        y0 = rng.integers(0, h-ch+1); x0 = rng.integers(0, w-cw+1)
        crop = img[y0:y0+ch, x0:x0+cw]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float().unsqueeze(0)/255.
        out.append(F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False))
    return torch.cat(out, 0)


def predict(img, rng):
    v = views(img, rng)
    with torch.no_grad():
        inp = cpr(images=[np.ascontiguousarray(v[i].permute(1, 2, 0).numpy()*255).astype(np.uint8) for i in range(v.shape[0])], return_tensors='pt').to(dev) if False else None
        # CLIP: 用 processor 归一化后的张量（与文本特征同一空间）
        pv = (v - 0.48145466) / 0.4578275
        f = clip.get_image_features(pixel_values=pv.to(dev))
        f = F.normalize(f if torch.is_tensor(f) else f.pooler_output, dim=-1)
        pl = (100.0 * f @ tfeat.T).softmax(-1)                     # (T,1000)
        rl = res(((v - MEAN) / STD).to(dev)).softmax(-1)           # (T,1000)
    cvote = pl.mean(0); rvote = rl.mean(0)
    ctop = int(cvote.argmax()); rtop = int(rvote.argmax())
    stab_c = float((pl.argmax(-1) == ctop).float().mean())
    stab_r = float((rl.argmax(-1) == rtop).float().mean())
    return ctop, rtop, stab_c, stab_r, float(cvote.max()), float(rvote.max())


def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra*rb).sum()/np.sqrt((ra*ra).sum()*(rb*rb).sum()+1e-12))


rng = np.random.default_rng(SEED)
rows = []
for arm in ('lovo_k2', 's2main10k'):
    for sname in ('apr', 'may'):
        for f in sorted(glob.glob(f'{PROJ}/data/gate_v2/{arm}/{sname}/*.npz')):
            z = np.load(f); I, P, T = z['I'], z['P'], z['T']
            r = {}
            for k, X in (('I', I), ('T', T), ('P', P)):
                ct, rt, sc, sr, cc, rc = predict(X, rng)
                r[f'gap_{k}'] = int(ct == rt); r[f'stabc_{k}'] = round(sc, 3)
                r[f'stabr_{k}'] = round(sr, 3); r[f'confc_{k}'] = round(cc, 3); r[f'confr_{k}'] = round(rc, 3)
            al = [psnr(affine(((I.astype(np.float32)+a*(P.astype(np.float32)-I.astype(np.float32))).clip(0, 255).astype(np.uint8)), T), T) for a in ALPHAS]
            r.update(arm=arm, set=sname, case=os.path.basename(f)[:-4], astar=float(ALPHAS[int(np.argmax(al))]),
                     gain=round(al[-1]-al[0], 2))
            rows.append(r)
            print(f'  {arm}/{sname}/{os.path.basename(f)[:-4]} gap I/T/P={r["gap_I"]}/{r["gap_T"]}/{r["gap_P"]} '
                  f'stabR I={r["stabr_I"]:.2f} T={r["stabr_T"]:.2f} α*={r["astar"]:.2f}', flush=True)
import csv
with open(f'{PROJ}/eval/trustgap_imagenet_1008.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

print('\n######## 预登记判决门（ImageNet 空间）########')
for arm in ('lovo_k2', 's2main10k'):
    sub = [r for r in rows if r['arm'] == arm]
    A = [r for r in sub if r['set'] == 'apr']; M = [r for r in sub if r['set'] == 'may']
    mu = lambda rs, k: float(np.mean([r[k] for r in rs]))
    print(f'\n--- {arm}（n={len(sub)}）---')
    print(f'  gap 均值 I={mu(sub,"gap_I"):.2f}（apr {mu(A,"gap_I"):.2f} / may {mu(M,"gap_I"):.2f}）GT={mu(sub,"gap_T"):.2f} P={mu(sub,"gap_P"):.2f}')
    print(f'  ResNet 稳定率 I={mu(sub,"stabr_I"):.2f}（apr {mu(A,"stabr_I"):.2f} / may {mu(M,"stabr_I"):.2f}）GT={mu(sub,"stabr_T"):.2f}')
    print(f'  CLIP   稳定率 I={mu(sub,"stabc_I"):.2f}（apr {mu(A,"stabc_I"):.2f} / may {mu(M,"stabc_I"):.2f}）GT={mu(sub,"stabc_T"):.2f}')
    print(f'  CLIP  置信 I={mu(sub,"confc_I"):.3f} GT={mu(sub,"confc_T"):.3f} | ResNet 置信 I={mu(sub,"confr_I"):.3f} GT={mu(sub,"confr_T"):.3f}')
    g1 = mu(sub, 'gap_T') - mu(sub, 'gap_I'); g2 = mu(A, 'gap_I') - mu(M, 'gap_I')
    print(f'  G1 gap(GT)−gap(I) = {g1:+.2f} → {"PASS" if g1 >= 0.05 else "FAIL"}')
    print(f'  G2 gap(I)_apr − gap(I)_may = {g2:+.2f} → {"PASS" if g2 >= 0.05 else "FAIL"}')
    for k in ('gap_I', 'stabr_I', 'stabr_T', 'confc_I', 'confr_I'):
        rho_all = sp([r[k] for r in sub], [r['astar'] for r in sub])
        rho_a = sp([r[k] for r in A], [r['astar'] for r in A])
        rho_m = sp([r[k] for r in M], [r['astar'] for r in M])
        print(f'    ρ({k:9s}, α*): 全 {rho_all:+.3f} | apr {rho_a:+.3f} | may {rho_m:+.3f}')
