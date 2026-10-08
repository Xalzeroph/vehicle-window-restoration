#!/usr/bin/env python3
"""门控 v1 人眼审阅图（2026-10-08）
5 列：INPUT | 官方RDNet | 门控O | affine(I)（仅全局色调） | GT
4 行：全图 / Ω区放大 / 1:1 像素 / [门控热力图 | 亮度均值条形图 | 三张误差图]
产物：fig/gate_review/{set}_{case}.png
"""
import glob, os, sys
import cv2, numpy as np, torch
sys.path.insert(0, 'scripts')
from gate_v1_train import Gate
from eval_protocol import compute_mask

H, WP = 300, 380
CROP = 224
dev = torch.device('cuda:0')


def fit(im, h=H, wp=WP):
    im = cv2.resize(im, (max(1, int(im.shape[1]*h/im.shape[0])), h), interpolation=cv2.INTER_AREA)
    if im.shape[1] > wp:
        x0 = (im.shape[1]-wp)//2; im = im[:, x0:x0+wp]
    c = np.zeros((h, wp, 3), np.uint8); x = (wp-im.shape[1])//2; c[:, x:x+im.shape[1]] = im
    return c


def tag(img, txt, org=(8, 24), sc=0.58):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 4)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 1)
    return img


def hs(imgs, gap=6):
    hmax = max(i.shape[0] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, hmax-i.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for i in imgs]
    out = imgs[0]
    for i in imgs[1:]:
        out = np.hstack([out, np.full((out.shape[0], gap, 3), 245, np.uint8), i])
    return out


def vs(imgs, gap=6):
    w = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, w-i.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)) for i in imgs]
    out = imgs[0]
    for i in imgs[1:]:
        out = np.vstack([out, np.full((gap, w, 3), 245, np.uint8), i])
    return out


def cpad(im, y0, x0, y1, x1):
    y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(im.shape[0], y1), min(im.shape[1], x1)
    return im[y0:y1, x0:x1]


def affine_best(I, T):
    x = I.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1); coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.clip(coef[0]*I.astype(np.float64)+coef[1], 0, 255).astype(np.uint8), coef


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))


net = Gate().to(dev); sd = torch.load('runs/ckpt/gate_v1_apr/gate.pth', map_location='cpu')
net.load_state_dict(sd['sd']); net.eval()
os.makedirs('fig/gate_review', exist_ok=True)
CASES = [('apr', 'VID_20260426_172719__000000'), ('apr', 'VID_20260426_173604__000014'),
         ('apr', 'VID_20260426_173827__000000'),
         ('may', 'VID_20260512_162623__000700'), ('may', 'VID_20260512_162203__000000'),
         ('may', 'VID_20260512_163859__000000')]

for sname, case in CASES:
    fp = f'data/gate_v1/{sname}/{case}.npz'
    if not os.path.exists(fp):
        print('skip', fp); continue
    z = np.load(fp); I, P, T = z['I'], z['P'], z['T']
    xI = torch.from_numpy(I.transpose(2, 0, 1)[None].astype(np.float32)/255.).to(dev)
    xP = torch.from_numpy(P.transpose(2, 0, 1)[None].astype(np.float32)/255.).to(dev)
    with torch.no_grad():
        _, g = net(xI, xP); O = (xI + g*(xP-xI)).clamp(0, 1)
    gnp = g[0, 0].cpu().numpy()
    O = (O[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
    A, coef = affine_best(I, T)
    cols = [('INPUT', I), ('official', P), ('gate O', O), ('affine(I)', A), ('GT', T)]
    M, info, _ = compute_mask(I, T)
    ys, xs = np.nonzero(M)
    if len(ys) == 0:
        by0, bx0, by1, bx1 = 0, 0, T.shape[0], T.shape[1]
    else:
        pad = 70
        by0, bx0, by1, bx1 = int(ys.min())-pad, int(xs.min())-pad, int(ys.max())+pad, int(xs.max())+pad
    cy, cx = (by0+by1)//2, (bx0+bx1)//2
    r1 = [tag(fit(im), n) for n, im in cols]
    r2 = [tag(fit(cpad(im, by0, bx0, by1, bx1)), n) for n, im in cols]
    r3 = [tag(cpad(im, cy-CROP//2, cx-CROP//2, cy+CROP//2, cx+CROP//2), n + ' 1:1') for n, im in cols]
    # 第 4 行：门控热力图 + 亮度条形图 + 三张误差图
    gh = cv2.applyColorMap(((1-gnp)*255).astype(np.uint8), cv2.COLORMAP_JET)
    gh = fit(gh); tag(gh, f'gate g  (mean {gnp.mean():.2f}, std {gnp.std():.2f})')
    bar = np.full((H, WP, 3), 255, np.uint8)
    lums = [im.mean() for _, im in cols]
    for i, (n, _) in enumerate(cols):
        x0 = 30 + i*66
        hh = int(lums[i]/255*(H-80))
        cv2.rectangle(bar, (x0, H-40-hh), (x0+40, H-40), (200, 180, 120) if i != 4 else (120, 160, 120), -1)
        cv2.putText(bar, f'{lums[i]:.0f}', (x0-2, H-46-hh), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.line(bar, (0, H-40-int(lums[4]/255*(H-80))), (WP, H-40-int(lums[4]/255*(H-80))), (60, 120, 60), 2)
    tag(bar, 'mean luma (green line = GT)')
    errs = []
    for n, im in [('|O-T|', O), ('|affine-T|', A), ('|P-T|', P)]:
        e = np.abs(im.astype(np.float32)-T.astype(np.float32)).mean(2)
        e = np.clip(e*4, 0, 255).astype(np.uint8)
        errs.append(tag(fit(cv2.applyColorMap(e, cv2.COLORMAP_INFERNO)), n))
    r4 = [gh, bar] + errs
    sheet = vs([hs(r1), hs(r2), hs(r3), hs(r4)])
    head = np.zeros((66, sheet.shape[1], 3), np.uint8)
    ps = '  '.join(f'{n}:{psnr(im, T):.1f}' for n, im in cols)
    cv2.putText(head, f'{sname}  {case}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(head, f'PSNR vs GT ->  {ps}      affine: a={coef[0]:.3f} b={coef[1]:+.1f}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
    sheet = np.vstack([head, sheet])
    out = f'fig/gate_review/{sname}_{case}.png'
    cv2.imwrite(out, sheet)
    print(f'[ok] {out}  {sheet.shape[1]}x{sheet.shape[0]}')
