#!/usr/bin/env python3
"""LOVO 人眼观察的数字验证 v2（2026-10-08，零训练）

用户目检三条判断：
  A. 跨域时"反射洞"与"透射层固有阴影"分不清
  B. 在亮反射上犹豫 → 亮反射收益不如官方
  C. 对阴影/暗区趋向于模糊

v2 修正（v1 的问题）：**所有臂（含 INPUT）逐图增益对齐到 GT 尺度**（ls_scale+apply_gm），
再做分块统计，否则全局曝光差会污染 dluma 与 Laplacian 比值。

分层维度：
  (1) r_obs = mean|g·I−T| / mean(T)（E3 口径，考 B）
  (2) GT 亮度 × Ω 归属（考 A/C：Ω 内是"洞"，Ω 外暗块才是"真阴影"）
输出 eval/probe_lovo_strata_{set}.csv
"""
import csv, os, sys
import cv2
import numpy as np

sys.path.insert(0, 'scripts')
from eval_protocol import ls_scale, apply_gm, compute_mask  # noqa: E402

BS = 64
SETS = {
 'may': dict(test='data/test_sets/real_pair_may_0917', off='runs/pred/may_rdnet_native',
             k2='runs/pred/may_lovo_k2', k4='runs/pred/may_lovo_k4',
             k6='runs/pred/may_rdt_s2_main_s101_10000'),
 'apr': dict(test='data/test_sets/real_pair_apr_0917', off='runs/pred/apr_rdnet_native',
             k2='runs/pred/apr_lovo_k2', k4='runs/pred/apr_lovo_k4',
             k6='runs/pred/apr_s101_apr10k'),
}
ARMS = ['off', 'k2', 'k4', 'k6']


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def lap_e(x):
    g = cv2.cvtColor(x.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var() ** 0.5)


def load(root, scene, stem, T):
    p = f'{root}/{scene}__I_{stem}/ytmt_ucs_sirs_l.png'
    im = cv2.imread(p) if os.path.exists(p) else None
    if im is not None and im.shape[:2] != T.shape[:2]:
        im = cv2.resize(im, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
    return im


rows = []
for setname, cfg in SETS.items():
    d = cfg['test']
    for scene in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        T = cv2.imread(f'{d}/{scene}/GT.png')
        if T is None:
            continue
        for fn in sorted(f for f in os.listdir(f'{d}/{scene}') if f.startswith('I_')):
            stem = fn[2:-4]
            I = cv2.imread(f'{d}/{scene}/{fn}')
            if I is None:
                continue
            arms = {a: load(cfg[a], scene, stem, T) for a in ARMS}
            if any(v is None for v in arms.values()):
                print(f'  [skip] {setname} {scene} {stem}: 缺臂'); continue
            M, info, D = compute_mask(I, T)          # Ω 掩膜
            # 逐图增益对齐到 GT
            t_ = T.astype(np.float64)
            gI = float(ls_scale(I, T)); Ig = apply_gm(I, gI)
            Ag = {a: apply_gm(arms[a], float(ls_scale(arms[a], T))) for a in ARMS}
            i_ = Ig.astype(np.float64)
            Dw = np.abs(i_ - t_).max(axis=2)          # 用对齐后口径算 r_obs（与视觉一致）
            h, w = T.shape[:2]
            for y in range(0, h - BS + 1, BS):
                for x in range(0, w - BS + 1, BS):
                    tb, ib = T[y:y+BS, x:x+BS], Ig[y:y+BS, x:x+BS]
                    frac_om = float(M[y:y+BS, x:x+BS].mean())
                    lg = lap_e(tb)
                    rec = dict(set=setname, scene=scene, stem=stem, by=y, bx=x,
                               r_obs=round(float(Dw[y:y+BS, x:x+BS].mean() / (tb.astype(np.float64).mean() + 1e-9)), 4),
                               luma_gt=round(float(tb.mean()), 1),
                               luma_in=round(float(ib.astype(np.float64).mean()), 1),
                               omega=round(frac_om, 3),
                               in_omega=int(frac_om >= 0.5),
                               lap_in=round(lap_e(ib) / (lg + 1e-9), 3))
                    for a in ARMS:
                        ab = Ag[a][y:y+BS, x:x+BS]
                        rec[f'dpsnr_{a}'] = round(psnr(ab, tb) - psnr(ib, tb), 3)
                        rec[f'lap_{a}'] = round(lap_e(ab) / (lg + 1e-9), 3)
                        rec[f'dluma_{a}'] = round(float(ab.astype(np.float64).mean() - tb.astype(np.float64).mean()), 2)
                    rows.append(rec)

os.makedirs('eval', exist_ok=True)


def med(sub, k):
    return np.median([r[k] for r in sub]) if sub else float('nan')


for setname in SETS:
    rs = [r for r in rows if r['set'] == setname]
    if not rs:
        continue
    with open(f'eval/probe_lovo_strata_{setname}.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rs[0].keys())); wr.writeheader(); wr.writerows(rs)

    def tbl(title, key, edges, extra=None):
        print(f'\n===== {setname} | {title} =====')
        print(f'{"区间":<16}{"块数":>7}' + ''.join(f'{a:>9}' for a in ARMS)
              + f'{"in_lap":>8}' + ''.join(f'{a+"_lap":>9}' for a in ARMS))
        for a, b in zip(edges[:-1], edges[1:]):
            sub = [r for r in rs if a <= r[key] < b and (extra is None or extra(r))]
            if len(sub) < 8:
                continue
            line = f'[{a:>5.1f},{b:>5.1f}){len(sub):>7}'
            line += ''.join(f'{med(sub, "dpsnr_"+x):>+9.2f}' for x in ARMS)
            line += f'{med(sub, "lap_in"):>8.2f}'
            line += ''.join(f'{med(sub, "lap_"+x):>9.2f}' for x in ARMS)
            print(line)

    tbl('按 r_obs 分箱（考 B：亮反射收益）', 'r_obs', [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 9])
    tbl('Ω 外 · 按 GT 亮度（真阴影/暗区，考 C）', 'luma_gt', [0, 25, 40, 60, 90, 140, 255],
        extra=lambda r: r['in_omega'] == 0)
    tbl('Ω 内 · 按 GT 亮度（洞，考 A/C）', 'luma_gt', [0, 25, 40, 60, 90, 140, 255],
        extra=lambda r: r['in_omega'] == 1)
    print(f'\n[写出] eval/probe_lovo_strata_{setname}.csv ({len(rs)} 块)')
