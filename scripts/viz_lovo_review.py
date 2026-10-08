#!/usr/bin/env python3
"""LOVO 结论人眼审阅图（2026-10-08）

每场景一张 contact sheet：3 行 × 6 列
  列: INPUT | 官方RDNet(native) | k=2 | k=4 | k=6(S2 main s101@10k) | GT
  行: ① 全图  ② Ω 区放大  ③ Ω 中心 1:1 原始像素（判"糊"用）
非 GT 列统一做 per-image 增益对齐（对齐到 GT 尺度），消除曝光差对肉眼的干扰。
图上刻意不打印任何指标数字。
输出: fig/lovo_review/{set}_{scene}.png
"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, 'scripts')
from eval_protocol import ls_scale, apply_gm, compute_mask  # noqa: E402

H, WP = 340, 450          # 全图面板
HZ = 340                  # Ω 放大面板
CROPSZ = 256              # 1:1 像素面板边长

SETCFG = {
    'may': dict(test='data/test_sets/real_pair_may_0917',
                official='runs/pred/may_rdnet_native',
                k2='runs/pred/may_lovo_k2', k4='runs/pred/may_lovo_k4',
                k6='runs/pred/may_rdt_s2_main_s101_10000'),
    'apr': dict(test='data/test_sets/real_pair_apr_0917',
                official='runs/pred/apr_rdnet_native',
                k2='runs/pred/apr_lovo_k2', k4='runs/pred/apr_lovo_k4',
                k6='runs/pred/apr_s101_apr10k'),
}


def fit(im, h=H, wp=WP):
    im = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
    if im.shape[1] > wp:
        x0 = (im.shape[1] - wp) // 2
        im = im[:, x0:x0 + wp]
    canvas = np.zeros((h, wp, 3), np.uint8)
    x = (wp - im.shape[1]) // 2
    canvas[:, x:x + im.shape[1]] = im
    return canvas


def tag(img, txt, org=(8, 26), scale=0.62):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return img


def pad_to(im, h, w, bg=250):
    canvas = np.full((h, w, 3), bg, np.uint8)
    hh, ww = min(h, im.shape[0]), min(w, im.shape[1])
    canvas[:hh, :ww] = im[:hh, :ww]
    return canvas


def hstack(imgs, gap=6, bg=240):
    h = max(im.shape[0] for im in imgs)
    imgs = [pad_to(im, h, im.shape[1], bg) for im in imgs]
    out = imgs[0]
    for im in imgs[1:]:
        out = np.hstack([out, np.full((out.shape[0], gap, 3), bg, np.uint8), im])
    return out


def vstack(imgs, gap=6, bg=240):
    w = max(im.shape[1] for im in imgs)
    imgs = [pad_to(im, im.shape[0], w, bg) for im in imgs]
    out = imgs[0]
    for im in imgs[1:]:
        out = np.vstack([out, np.full((gap, out.shape[1], 3), bg, np.uint8), im])
    return out


def crop_pad(im, y0, x0, y1, x1):
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(im.shape[0], y1), min(im.shape[1], x1)
    return im[y0:y1, x0:x1]


def panel_zoom(im, bbox, h=HZ, wp=WP):
    y0, x0, y1, x1 = bbox
    c = crop_pad(im, y0, x0, y1, x1)
    if c.size == 0:
        return np.zeros((h, wp, 3), np.uint8)
    return fit(c, h, wp)


def panel_pixel(im, cy, cx, sz=CROPSZ):
    y0, x0 = int(cy - sz // 2), int(cx - sz // 2)
    c = crop_pad(im, y0, x0, y0 + sz, x0 + sz)
    canvas = np.zeros((sz, sz, 3), np.uint8)
    canvas[:c.shape[0], :c.shape[1]] = c
    return canvas


def load_pred(root, scene, stem):
    # 预测目录命名 = {scene}__I_{stem}（注意 I_ 前缀）
    for pat in (f'{root}/{scene}__I_{stem}/ytmt_ucs_sirs_l.png', f'{root}/{scene}__{stem}/ytmt_ucs_sirs_l.png'):
        if os.path.exists(pat):
            return cv2.imread(pat)
    return None


def build(setname, scene, stem, outdir):
    cfg = SETCFG[setname]
    d = cfg['test']
    I = cv2.imread(f'{d}/{scene}/I_{stem}.png')
    T = cv2.imread(f'{d}/{scene}/GT.png')
    if I is None or T is None:
        print(f'[skip] {setname} {scene} {stem}: 缺 I/GT'); return None
    arms = [('INPUT', I), ('official', load_pred(cfg['official'], scene, stem)),
            ('k=2', load_pred(cfg['k2'], scene, stem)), ('k=4', load_pred(cfg['k4'], scene, stem)),
            ('k=6', load_pred(cfg['k6'], scene, stem)), ('GT', T)]
    # 统一到 GT 尺寸 + 增益对齐
    cols, gains = [], []
    for name, im in arms:
        if im is None:
            cols.append((name, None, None)); gains.append((name, float('nan'))); continue
        if im.shape[:2] != T.shape[:2]:
            im = cv2.resize(im, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
        if name == 'GT':
            g = 1.0
        else:
            g = float(ls_scale(im, T))
            im = apply_gm(im, g)
        cols.append((name, im, g)); gains.append((name, g))
    # Ω 掩膜与 bbox
    M, info, D = compute_mask(I, T)
    ys, xs = np.nonzero(M)
    if len(ys) == 0:
        bbox = (0, 0, T.shape[0], T.shape[1]); cy, cx = T.shape[0] // 2, T.shape[1] // 2
    else:
        pad = 90
        bbox = (int(ys.min()) - pad, int(xs.min()) - pad, int(ys.max()) + pad, int(xs.max()) + pad)
        cy, cx = int(ys.mean()), int(xs.mean())
    # 三行
    r1, r2, r3 = [], [], []
    for name, im, g in cols:
        if im is None:
            print(f'  [missing arm] {setname} {scene} {stem} -> {name}')
            ph1 = np.zeros((H, WP, 3), np.uint8)
            ph3 = np.zeros((CROPSZ, CROPSZ, 3), np.uint8)
            r1.append(tag(ph1, f'{name}: MISSING')); r2.append(ph1.copy()); r3.append(ph3); continue
        full = fit(im)
        if name == 'INPUT':
            cont = full.copy()
            xs2 = (xs * WP / T.shape[1]).astype(int); ys2 = (ys * H / T.shape[0]).astype(int)
            m2 = np.zeros((H, WP), np.uint8)
            ok = (xs2 < WP) & (ys2 < H) & (xs2 >= 0) & (ys2 >= 0)
            m2[ys2[ok], xs2[ok]] = 255
            m2 = cv2.dilate(m2, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cont, cnts, -1, (0, 255, 0), 2)
            full = cont
        r1.append(tag(full, name))
        r2.append(tag(panel_zoom(im, bbox), name))
        r3.append(tag(panel_pixel(im, cy, cx), name))
    sheet = vstack([hstack(r1), hstack(r2), hstack(r3)])
    # 顶部说明条
    bar = np.zeros((46, sheet.shape[1], 3), np.uint8)
    gs = '  '.join(f'{n}:{g:.2f}' for n, g in gains if n != 'GT' and not np.isnan(g))
    cv2.putText(bar, f'{setname}  {scene}  {stem}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, f'rows: full / omega-zoom / omega 1:1      gain(aligned to GT): {gs}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    sheet = np.vstack([bar, sheet])
    os.makedirs(outdir, exist_ok=True)
    out = f'{outdir}/{setname}_{scene}_{stem}.png'
    cv2.imwrite(out, sheet)
    print(f'[ok] {out}  ({sheet.shape[1]}x{sheet.shape[0]})')
    return out


def main():
    outdir = 'fig/lovo_review'
    cases = {
        'may': [('VID_20260512_162203', '000000'), ('VID_20260512_162623', '000700'),
                ('VID_20260512_162901', '000675'), ('VID_20260512_163859', '000000'),
                ('VID_20260512_164653', '000000')],
        'apr': [('VID_20260426_172719', '000000'), ('VID_20260426_172933', '000250'),
                ('VID_20260426_173237', '000000'), ('VID_20260426_173349', '000000'),
                ('VID_20260426_173604', '000007'), ('VID_20260426_173827', '000000')],
    }
    made = []
    for setname, lst in cases.items():
        for scene, stem in lst:
            r = build(setname, scene, stem, outdir)
            if r:
                made.append(r)
    print(f'\n共 {len(made)} 张 -> {outdir}/')


if __name__ == '__main__':
    main()
