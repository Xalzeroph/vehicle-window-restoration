#!/usr/bin/env python3
"""S2 机制核验（用户茶盒假设）：透射层结构在 I 里可见性 vs S2 保留率，may 全 14 帧。

用户假设（2026-09-21）：S2 "分不清透射层阴影是反射还是透射已有阴影 → 整体糊一遍"；
茶盒糊的成因 = "输入很暗没细节 / 被漫反射遮住了一部分"。
测量：结构强度 s = 9×9 盒滤波 |Laplacian|（灰度）；has_T = sT > p70(sT)（真值里纹理清楚的 ~30%，口径约定）；
vis = has_T ∧ sI > thr·sT（结构在输入里仍可见，thr 为口径阈值，默认 0.5）；cov = 其余（被盖）。
指标：sI/sT（输入保留真值多少对比度，跨帧量/近似）、sP/sI（输出把输入里这批结构毁了多少，同帧量）、
σ_eq（高斯模糊 I 的等效半径，在 vis 区匹配 sP/sI，量化"低通"程度）。
用法:
  python3 scripts/probes_0921/check_structure_visibility.py                        # 主表（s101@10k）
  python3 scripts/probes_0921/check_structure_visibility.py --pred runs/pred/may_rdnet_native \
      --out eval/probe_s2_structure_visibility_official_0921.csv                   # 官方原生对照
  python3 scripts/probes_0921/check_structure_visibility.py --robust               # 3 阈值 × 3 seed 汇总
"""
import argparse, csv, os
import cv2
import numpy as np

os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')
SETS = {'may': dict(d='data/test_sets/real_pair_may_0917',
                    main='runs/pred/may_rdt_s2_main_s{}_10000',
                    out='eval/probe_s2_structure_visibility_0921.csv',
                    robust='eval/probe_s2_structure_visibility_robust_0921.csv'),
        'apr': dict(d='data/test_sets/real_pair_apr_0917',
                    main='runs/pred/apr_s{}_apr10k',
                    out='eval/probe_s2_structure_visibility_apr_0921.csv',
                    robust='eval/probe_s2_structure_visibility_apr_robust_0921.csv')}


def struct(x):
    return cv2.boxFilter(np.abs(cv2.Laplacian(x, cv2.CV_32F)), -1, (9, 9))


def ratios(sT, sI, sP, mask):
    """(均值比 sP/sI, 比值均值 sP/sI, sI/sT, sP/sT)，mask 太小返回 nan。"""
    if mask.sum() < 200:
        return (float('nan'),) * 4
    n, d = sP[mask], sI[mask]
    mor = float((n / np.maximum(d, 1e-3)).mean())
    rom = float(n.sum() / max(d.sum(), 1e-3))
    return mor, rom, float((sI[mask] / np.maximum(sT[mask], 1e-3)).mean()), \
        float((sP[mask] / np.maximum(sT[mask], 1e-3)).mean())


def measure(d, pred_dir, thr, sigma_grid=None):
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = []
    for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        gT = g(cv2.imread(f'{d}/{sc}/GT.png'))
        for fn in sorted(f for f in os.listdir(f'{d}/{sc}') if f.startswith('I')):
            key = f'{sc}__{fn[:-4]}' if fn != 'I.png' else sc
            pp = f'{pred_dir}/{key}/ytmt_ucs_sirs_l.png'
            if not os.path.exists(pp):
                print(f'  [缺预测] {key}')
                continue
            gI, gP = g(cv2.imread(f'{d}/{sc}/{fn}')), g(cv2.imread(pp))
            sT, sI, sP = struct(gT), struct(gI), struct(gP)
            has_T = sT > np.percentile(sT, 70)
            vis = has_T & (sI > thr * sT)
            cov = has_T & (sI <= thr * sT)
            pv, pv_rom, iv, tv = ratios(sT, sI, sP, vis)
            pc, pc_rom, _, _ = ratios(sT, sI, sP, cov)
            sig = float('nan')
            if sigma_grid is not None and not np.isnan(pv):
                best = (1e9, float('nan'))
                for s in sigma_grid:
                    r = ratios(sT, sI, struct(cv2.GaussianBlur(gI, (0, 0), float(s))), vis)[0]
                    if abs(r - pv) < best[0]:
                        best = (abs(r - pv), float(s))
                sig = best[1]
            rows.append(dict(scene=sc, frame=fn[:-4], hasT_pct=round(100 * has_T.mean(), 1),
                             vis_pct=round(100 * vis.sum() / max(has_T.sum(), 1), 1),
                             cov_pct=round(100 * cov.sum() / max(has_T.sum(), 1), 1),
                             sI_sT_vis=round(iv, 3), sP_sT_vis=round(tv, 3),
                             sP_sI_vis=round(pv, 3), sP_sI_vis_rom=round(pv_rom, 3),
                             sP_sI_cov=round(pc, 3), sP_sI_cov_rom=round(pc_rom, 3),
                             sigma_eq=round(sig, 2)))
    return rows


def write_csv(rows, out):
    with open(out, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['may', 'apr'], default='may')
    ap.add_argument('--pred', default=None, help='默认按 set 的 S2 s101@10k 目录')
    ap.add_argument('--out', default=None)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--robust', action='store_true', help='3 阈值 × 3 seed 汇总（跳过 σ 搜索）')
    args = ap.parse_args()
    cfg = SETS[args.set]
    if args.robust:
        out = cfg['robust']
        agg = []
        for seed in (101, 202, 303):
            for thr in (0.3, 0.5, 0.7):
                rows = measure(cfg['d'], cfg['main'].format(seed), thr)
                a = lambda k: float(np.nanmean([r[k] for r in rows]))
                agg.append(dict(seed=seed, thr=thr, n=len(rows),
                                vis_pct=round(a('vis_pct'), 1), sI_sT_vis=round(a('sI_sT_vis'), 3),
                                sP_sI_vis=round(a('sP_sI_vis'), 3), sP_sI_cov=round(a('sP_sI_cov'), 3),
                                sP_sI_cov_rom=round(a('sP_sI_cov_rom'), 3)))
                print(f'  seed {seed} thr {thr}: vis {agg[-1]["vis_pct"]}%  '
                      f'sP/sI(vis) {agg[-1]["sP_sI_vis"]}  cov(均值比 {agg[-1]["sP_sI_cov"]} '
                      f'/ 比值均值 {agg[-1]["sP_sI_cov_rom"]})', flush=True)
        write_csv(agg, out)
        print(f'-> {out}')
        return
    rows = measure(cfg['d'], args.pred or cfg['main'].format(101), args.thr,
                   sigma_grid=np.arange(0.2, 3.05, 0.1))
    out = args.out or cfg['out']
    write_csv(rows, out)
    a = lambda k: float(np.nanmean([r[k] for r in rows]))
    print(f'{"场景/帧":<24} {"T结构区":>6} {"I可见%":>7} {"被盖%":>6} {"vis sI/sT":>9} '
          f'{"vis sP/sT":>9} {"vis sP/sI":>9} {"cov sP/sI":>9} {"σ_eq":>5}')
    for r in rows:
        print(f'{r["scene"][-6:]}/{r["frame"]:<10} {r["hasT_pct"]:>5.1f}% {r["vis_pct"]:>6.1f}% '
              f'{r["cov_pct"]:>5.1f}% {r["sI_sT_vis"]:>9.2f} {r["sP_sT_vis"]:>9.2f} '
              f'{r["sP_sI_vis"]:>9.2f} {r["sP_sI_cov"]:>9.2f} {r["sigma_eq"]:>5.2f}')
    print(f'\n=== {len(rows)} 帧汇总: 可见占比 {a("vis_pct"):.1f}% | vis sI/sT {a("sI_sT_vis"):.2f} | '
          f'vis sP/sT {a("sP_sT_vis"):.2f} | vis S2相对I {a("sP_sI_vis"):.2f} '
          f'(rom {a("sP_sI_vis_rom"):.2f}) | cov S2相对I {a("sP_sI_cov"):.2f} '
          f'(rom {a("sP_sI_cov_rom"):.2f}) | σ_eq 中位 {np.nanmedian([r["sigma_eq"] for r in rows]):.2f} ===\n'
          f'-> {out}')


if __name__ == '__main__':
    main()
