#!/usr/bin/env python3
"""指标排序秩相关表（2026-09-19 轮 §2d）：同一组臂上，不同指标的排序一致性。

口径：5 臂 = identity_gm / const / blur_identity / rdnet / remo；逐案对各指标给臂排序，
算指定指标对的 Spearman；跨案取中位 + 场景聚类 bootstrap CI。
用途：证明"PSNR 排序 ≠ 感知排序"（既有评测与感知判据脱节）。
用法：python scripts/eval_rank_corr.py
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import scene_bootstrap  # noqa: E402
from scipy.stats import spearmanr

ARMS = ['identity_gm', 'const', 'blur_identity', 'rdnet', 'remo']
PAIRS = [('psnr', 'lpips_crop'), ('psnr', 'ngr_m'), ('psnr', 'ngr_bg'),
         ('ssim_m', 'lpips_crop'), ('ngr_m', 'lpips_crop')]
SETS = {'jun': 'eval/eval_proto_jun.csv', 'apr': 'eval/eval_proto_apr.csv', 'may': 'eval/eval_proto_may.csv'}


def value(row, arm, metric):
    if metric == 'lpips_crop':
        v = row.get(f'{arm}__lpips_crop')
        if v in (None, '', 'None'):
            v = row.get(f'{arm}__lpips')
    else:
        v = row.get(f'{arm}__{metric}')
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    out_rows, summary = [], {}
    for tag, path in SETS.items():
        rows = list(csv.DictReader(open(path)))
        per_pair = {}
        for pa, pb in PAIRS:
            by_scene = {}
            for r in rows:
                va = [value(r, a, pa) for a in ARMS]
                vb = [value(r, a, pb) for a in ARMS]
                if any(v is None for v in va + vb):
                    continue
                rho = float(spearmanr(va, vb).statistic)
                out_rows.append(dict(set=tag, case=r['case'], pair=f'{pa}~{pb}', spearman=round(rho, 4)))
                by_scene.setdefault(r['scene'], []).append(rho)
                per_pair.setdefault(f'{pa}~{pb}', {}).update({r['case']: rho})
            bs = scene_bootstrap(by_scene)
            summary[f'{tag}:{pa}~{pb}'] = bs
    with open('eval/eval_rank_corr.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    json.dump(summary, open('eval/eval_rank_corr.json', 'w'), indent=1, ensure_ascii=False)
    for k, v in summary.items():
        ci = f"[{v['ci_lo']:+.2f},{v['ci_hi']:+.2f}]" if v['median'] is not None else 'NA'
        print(f"{k:<28} 秩相关中位 {v['median']:+.3f} {ci} (n场景 {v['n_scenes']})")
    print('写出 eval/eval_rank_corr.csv 与 .json')


if __name__ == '__main__':
    main()
