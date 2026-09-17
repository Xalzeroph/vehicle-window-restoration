#!/usr/bin/env python3
"""E3 补：量"残留"而不是 MSE（2026-09-17）。

动机：块级 PSNR 显示强反射区 RDNet +2~4.8 dB，但人眼说"人影没去掉"——
怀疑是**残留剪影感知显著、但占 MSE 比例小**。本脚本直接量残留幅度：

  residual_ratio  = mean|RDNet − GT| / mean|输入 − GT|      （还剩百分之几）
  并在同一批块上算 1-近邻梯度相关（无 pyiqa 时的感知代理：结构残留）

口径：只在 r_obs > 0.5 的块上统计（强反射区）。
输出 eval/eval_e3_residual.csv + 控制台摘要。
"""
import csv, os
import cv2
import numpy as np

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet')}
BS = 64


def pred_path(tag, sc, stem):
    base = SETS[tag][2]
    return (f'{base}/{sc}/ytmt_ucs_sirs_l.png' if tag == 'jun'
            else f'{base}/{sc}__{stem}/ytmt_ucs_sirs_l.png')


def main():
    rows = []
    for r in csv.DictReader(open('eval/eval_e3_robs_bins.csv')):
        if float(r['r_obs']) < 0.5:
            continue
        tag, sc, stem = r['set'], r['scene'], r['stem']
        y, x = int(r['by']), int(r['bx'])
        d, gtn, _ = SETS[tag]
        GT = cv2.imread(f'{d}/{sc}/{gtn}')
        I = cv2.imread(f'{d}/{sc}/{stem}.png')
        pp = pred_path(tag, sc, stem)
        if GT is None or I is None or not os.path.exists(pp):
            continue
        R = cv2.resize(cv2.imread(pp), (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_LINEAR)
        gtb = GT[y:y + BS, x:x + BS].astype(np.float64)
        ib = I[y:y + BS, x:x + BS].astype(np.float64)
        rb = R[y:y + BS, x:x + BS].astype(np.float64)
        e_in = np.abs(ib - gtb).mean()
        e_rd = np.abs(rb - gtb).mean()
        # 感知代理：GT 的结构（高频）在输出里还剩多少
        def hf(a):
            gr = cv2.cvtColor(a.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return cv2.magnitude(cv2.Sobel(gr, cv2.CV_32F, 1, 0), cv2.Sobel(gr, cv2.CV_32F, 0, 1)).mean()
        h_gt, h_in, h_rd = hf(gtb), hf(ib), hf(rb)
        rows.append(dict(set=tag, scene=sc, stem=stem, by=y, bx=x, r_obs=r['r_obs'],
                         err_in=round(float(e_in), 2), err_rd=round(float(e_rd), 2),
                         residual_ratio=round(float(e_rd / (e_in + 1e-9)), 3),
                         hf_in_gt=round(float(h_in / (h_gt + 1e-9)), 3),
                         hf_rd_gt=round(float(h_rd / (h_gt + 1e-9)), 3)))
    os.makedirs('eval', exist_ok=True)
    with open('eval/eval_e3_residual.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    rr = np.array([x['residual_ratio'] for x in rows])
    er = np.array([x['err_in'] for x in rows])
    print(f'强反射块 (r_obs>0.5): n={len(rows)}')
    print(f'  误差: 输入中位 {np.median(er):.1f} → RDNet 中位 {np.median([x["err_rd"] for x in rows]):.1f}')
    print(f'  **残留比 residual_ratio 中位 {np.median(rr):.2f}**  '
          f'(p10 {np.percentile(rr,10):.2f}, p90 {np.percentile(rr,90):.2f})')
    print(f'  残留 >0.3 的块占比: {100*(rr>0.3).mean():.0f}%   残留 >0.5 的块占比: {100*(rr>0.5).mean():.0f}%')
    print(f'  写出 eval/eval_e3_residual.csv')


if __name__ == '__main__':
    main()
