#!/usr/bin/env python3
"""定案实验（Opus 设计，2026-09-20）：Ω(R<−20) 到底是"暗反射（洞）"还是 AE/增益伪影。

8 对：apr×3 / jun×3（含 170941 人影）/ may×2 + RRW×2（无人影对照）。
每对：
 (a) 复现 Ω0 = CC(R<−20)，R = I − g·T（g=robust_gain）
 (b) s_q = q05(I_lin/T_lin)（γ=2.2 线性化），R̂ = I_lin − s_q·T_lin；Ω0 内按 R̂ 分三档面积：
     <−10(以 255 尺度换算) / |≤10| / >10
 (c) g×(0.92,1.08) 重算 Ω → 面积摆动 + IoU
 (d) Ω0 与 T 暗区(luma<p20)重叠率
 (e) 拼图 fig/omega_semantics_0920.jpg（I | R̂热图 | Ω0 | T）
"""
import csv, glob, os, re, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import robust_gain  # noqa: E402

GAMMA = 2.2
RRW = '/DataDisk/zmy_code/datasets/RRWDataset'


def lin(x):
    return (np.clip(x, 0, 255) / 255.0) ** GAMMA


def omega_mask(R, thr=-20.0, min_frac=0.005):
    m = (R < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m, bool)
    for i in range(1, n):
        if st[i][4] / R.size >= min_frac:
            out |= (lab == i)
    return out


def ioU(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


PAIRS = [('apr', 'data/test_sets/real_pair_apr_0917', 'GT.png', 'VID_20260426_173827', 'I_000009.png'),
         ('apr', 'data/test_sets/real_pair_apr_0917', 'GT.png', 'VID_20260426_172719', 'I_000000.png'),
         ('apr', 'data/test_sets/real_pair_apr_0917', 'GT.png', 'VID_20260426_173237', 'I_000013.png'),
         ('jun', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_170941', 'I.png'),
         ('jun', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_170534', 'I.png'),
         ('jun', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_172143', 'I.png'),
         ('may', 'data/test_sets/real_pair_may_0917', 'GT.png', 'VID_20260512_162203', 'I_000000.png'),
         ('may', 'data/test_sets/real_pair_may_0917', 'GT.png', 'VID_20260512_164653', 'I_000023.png'),
         ('rrw', RRW, None, 'reflection_camera1/camera_objects', 'frame_000180.jpg'),
         ('rrw', RRW, None, 'reflection_camera1/camera_objects1', 'frame_000150.jpg')]


def load_pair(tag, d, gtn, sc, fn):
    if tag == 'rrw':
        I = cv2.imread(f'{d}/{sc}/{fn}')
        T = cv2.imread(f'{d}/reflection_camera1/GT_camera1/{sc.split("/")[-1]}_GT.png')
    else:
        I = cv2.imread(f'{d}/{sc}/{fn}')
        T = cv2.imread(f'{d}/{sc}/{gtn}')
    if I is None or T is None:
        return None, None
    if I.shape != T.shape:
        T = cv2.resize(T, (I.shape[1], I.shape[0]))
    return I, T


def main():
    rows, panels = [], []
    H, WP = 300, 400
    for tag, d, gtn, sc, fn in PAIRS:
        I, T = load_pair(tag, d, gtn, sc, fn)
        if I is None:
            print(f'{tag}/{sc}/{fn}: 缺文件'); continue
        g = robust_gain(I, T)
        R = (I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3)).mean(axis=2)
        O0 = omega_mask(R)
        # (b) 物理 s
        Il, Tl = lin(I.astype(np.float64)), lin(T.astype(np.float64))
        Tl_safe = np.where(Tl > (10 / 255.) ** GAMMA, Tl, np.nan)
        ratio = Il / Tl_safe
        s_q = float(np.nanquantile((ratio.mean(axis=2)), 0.05))
        Rh = (Il - s_q * Tl).mean(axis=2) * 255.0      # 回到 0-255 尺度便于分档
        if O0.sum() > 0:
            v = Rh[O0]
            f_neg = float((v < -10).mean()); f_mid = float((np.abs(v) <= 10).mean()); f_pos = float((v > 10).mean())
        else:
            f_neg = f_mid = f_pos = float('nan')
        # (c) 增益摆动
        outs = [omega_mask((I.astype(np.float64) - T.astype(np.float64) * (g * k).reshape(1, 1, 3)).mean(axis=2))
                for k in (0.92, 1.08)]
        swing = max(float(outs[0].mean()), float(outs[1].mean())) / max(float(O0.mean()), 1e-9)
        iou_min = min(ioU(O0, outs[0]), ioU(O0, outs[1]))
        # (d) 与 T 暗区重叠
        Tl_gray = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
        darkT = Tl_gray < np.percentile(Tl_gray, 20)
        ov_darkT = float((O0 & darkT).sum() / max(O0.sum(), 1))
        rows.append(dict(tag=tag, case=f'{sc}/{fn}', g=round(float(g.mean()), 3), s_q=round(s_q, 3),
                         om0_pct=round(float(O0.mean()) * 100, 1), f_neg=round(f_neg, 2), f_mid=round(f_mid, 2),
                         f_pos=round(f_pos, 2), swing=round(swing, 2), iou_min=round(iou_min, 2),
                         omega_in_darkT=round(ov_darkT, 2)))
        print(f"{tag:<4} {sc.split('/')[-1]:<18} g={g.mean():.2f} s_q={s_q:.2f} | Ω0 {O0.mean()*100:5.1f}% | Ω0内R̂档 负{f_neg:.2f}/洞{f_mid:.2f}/正{f_pos:.2f} | 摆动x{swing:.1f} IoU≥{iou_min:.2f} | Ω∩T暗区 {ov_darkT:.2f}")
        # (e) 拼图
        heat = cv2.applyColorMap(np.clip((Rh + 40) / 80 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        ov = I.copy(); ov[O0] = (0, 0, 255)
        ov = cv2.addWeighted(ov, 0.45, I, 0.55, 0)
        def fit(im):
            x = cv2.resize(im, (max(1, int(im.shape[1] * H / im.shape[0])), H), interpolation=cv2.INTER_AREA)
            c = np.zeros((H, WP, 3), np.uint8); x = x[:, :WP] if x.shape[1] > WP else x
            c[:, :x.shape[1]] = x
            return c
        lab = np.zeros((H, 260, 3), np.uint8)
        for i, t in enumerate([f'{tag}/{sc.split("/")[-1][:16]}', f'Ω0 {O0.mean()*100:.1f}% 档 负{f_neg:.2f}/洞{f_mid:.2f}',
                               f'摆动x{swing:.1f} IoU{iou_min:.2f}', 'I | R̂热图 | Ω0 | T']):
            cv2.putText(lab, t, (6, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(np.hstack([lab, fit(I), fit(heat), fit(ov), fit(T)]))
    with open('eval/eval_omega_semantics.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    gap = np.full((8, panels[0].shape[1], 3), 235, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.vstack([out, gap, p])
    cv2.imwrite('fig/omega_semantics_0920.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('\n写出 eval/eval_omega_semantics.csv 与 fig/omega_semantics_0920.jpg')


if __name__ == '__main__':
    main()
