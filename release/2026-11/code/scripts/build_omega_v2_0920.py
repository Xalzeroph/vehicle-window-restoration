#!/usr/bin/env python3
"""Ω 重建 v2（Opus 定案，2026-09-20）：物理曝光 s + 结构凹陷检测。

口径：
  γ=2.2 线性域估 s = q05(I_lin/T_lin)；T_scale = (s·T_lin)^(1/γ)·255（回到 0-255 显示域）
  Rg = I − T_scale（物理校正后的反射，非负为主）
  S  = med_big(Rg) − Rg（大核中值背景 − 反射 = 凹陷深度）
  Ω  = CC{S > τ}，τ = max(8, 0.4·p90(Rg))，面积 ≥0.5%
统一配色：照片上红 = Ω；热图用 JET（红=反射强；蓝/暗=反射缺失=洞）。
产物：data/omega_0920/{name}_omega.png、eval/eval_omega_v2.csv、fig/omega_v2_0920.jpg
"""
import csv, os, sys
import cv2
import numpy as np

G = 2.2
RRW = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/datasets/RRWDataset"
PAIRS = [
    ('jun', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_170941', 'I.png'),
    ('jun', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_170534', 'I.png'),
    ('jun', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_172143', 'I.png'),
    ('apr', 'data/test_sets/real_pair_apr_0917', 'GT.png', 'VID_20260426_173827', 'I_000009.png'),
    ('apr', 'data/test_sets/real_pair_apr_0917', 'GT.png', 'VID_20260426_172719', 'I_000000.png'),
    ('may', 'data/test_sets/real_pair_may_0917', 'GT.png', 'VID_20260512_162203', 'I_000000.png'),
    ('may', 'data/test_sets/real_pair_may_0917', 'GT.png', 'VID_20260512_164653', 'I_000023.png'),
    ('rrw', RRW, None, 'reflection_camera1/camera_objects', 'frame_000180.jpg'),
]


def omega_of(I, T):
    Il = (np.clip(I, 0, 255) / 255.0) ** G
    Tl = (np.clip(T, 0, 255) / 255.0) ** G
    valid = Tl.mean(axis=2) > (10 / 255.) ** G
    ratio = np.where(valid[..., None], Il / np.where(Tl > 1e-8, Tl, np.nan), np.nan)
    s = float(np.nanquantile(ratio.mean(axis=2), 0.05))
    Tsc = (np.clip(s * Tl, 0, 1)) ** (1 / G) * 255.0
    Rg = (I.astype(np.float64) - Tsc).mean(axis=2)
    R8 = np.clip(Rg + 128, 0, 255).astype(np.uint8)
    # 帧级稳健噪声（3×3 中值残差 MAD）
    med3 = cv2.medianBlur(R8, 3).astype(np.float64) - 128.0
    sig = 1.4826 * float(np.median(np.abs(Rg - med3)))
    tau = max(6.0, 5 * sig)
    h, w = Rg.shape
    blk = 48
    hh, ww = h // blk, w // blk
    t = Rg[:hh * blk, :ww * blk].reshape(hh, blk, ww, blk).transpose(0, 2, 1, 3).reshape(hh, ww, -1)
    bg = cv2.resize(np.percentile(t, 90, axis=2).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR).astype(np.float64)
    S = bg - Rg
    # 帧级闸：该帧几乎无反射 → Ω=∅
    if float(np.percentile(Rg, 90)) < max(12.0, 6 * sig):
        return dict(s=s, tau=tau, Rg=Rg, S=S, Om=np.zeros_like(Rg, bool), gate=True)
    M = ((S > tau) & (2 * Rg < bg)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(M, 8)
    Om = np.zeros_like(M, bool)
    for i in range(1, n):
        if st[i][4] / M.size >= 0.005:
            Om |= (lab == i)
    # 填内洞（≤25% 面积）
    inv = (~Om).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n2):
        x, y, ww, hh, area = st2[i]
        touches = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches and area / M.size <= 0.25:
            Om |= (lab2 == i)
    return dict(s=s, tau=tau, Rg=Rg, S=S, Om=Om, gate=False)


def main():
    os.makedirs('data/omega_0920', exist_ok=True)
    rows, panels = [], []
    H, WP = 300, 400
    for tag, d, gtn, sc, fn in PAIRS:
        if tag == 'rrw':
            I = cv2.imread(f'{d}/{sc}/{fn}')
            T = cv2.imread(f'{d}/reflection_camera1/GT_camera1/{sc.split("/")[-1]}_GT.png')
        else:
            I = cv2.imread(f'{d}/{sc}/{fn}')
            T = cv2.imread(f'{d}/{sc}/{gtn}')
        if I is None or T is None:
            print(f'{tag}/{sc}: 缺文件'); continue
        if I.shape != T.shape:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))
        o = omega_of(I, T)
        name = f'{tag}__{sc.split("/")[-1]}__{fn[:-4]}'.replace('/', '_')
        cv2.imwrite(f'data/omega_0920/{name}_omega.png', (o['Om'] * 255).astype(np.uint8))
        # 170941 人影 bbox 核验
        hit = ''
        if sc.endswith('170941'):
            for (x0, y0, x1, y1) in [(0, 86, 83, 302), (0, 302, 83, 423)]:
                ov = float(o['Om'][y0:y1, x0:x1].mean())
                hit += f' 人影bbox({x0},{y0})覆盖{ov*100:.0f}%'
        rows.append(dict(name=name, s=round(o['s'], 3), tau=round(o['tau'], 1),
                         omega_pct=round(float(o['Om'].mean()) * 100, 2), bbox=hit.strip()))
        print(f"{tag:<4} {sc.split('/')[-1][:18]:<20} s={o['s']:.2f} τ={o['tau']:.1f} Ω={o['Om'].mean()*100:5.2f}%{hit}")
        # 拼图
        def fit(im):
            x = cv2.resize(im, (max(1, int(im.shape[1] * H / im.shape[0])), H), interpolation=cv2.INTER_AREA)
            c = np.zeros((H, WP, 3), np.uint8); x = x[:, :WP] if x.shape[1] > WP else x
            c[:, :x.shape[1]] = x
            return c
        heat = cv2.applyColorMap(np.clip((o['Rg'] + 20) / 80 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        dip = cv2.applyColorMap(np.clip(o['S'] / 60 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        ov = I.copy(); ov[o['Om']] = (0, 0, 255)
        ov = cv2.addWeighted(ov, 0.45, I, 0.55, 0)
        lab = np.zeros((H, 270, 3), np.uint8)
        for i, t in enumerate([f'{tag}/{sc.split("/")[-1][:18]}', f's={o["s"]:.2f} tau={o["tau"]:.1f}',
                               f'Omega {o["Om"].mean()*100:.2f}%', 'I+Omega | Rg(JET 红=反射强) | Dip S | T+Omega']):
            cv2.putText(lab, t, (6, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ovt = T.copy(); ovt[o['Om']] = (0, 0, 255)
        ovt = cv2.addWeighted(ovt, 0.45, T, 0.55, 0)
        panels.append(np.hstack([lab, fit(ov), fit(heat), fit(dip), fit(ovt)]))
    with open('eval/eval_omega_v2.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    gap = np.full((8, panels[0].shape[1], 3), 235, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.vstack([out, gap, p])
    cv2.imwrite('fig/omega_v2_0920.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('\n写出 eval/eval_omega_v2.csv、data/omega_0920/、fig/omega_v2_0920.jpg')


if __name__ == '__main__':
    main()
