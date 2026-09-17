#!/usr/bin/env python3
"""E4 step1：oracle 选择器的上限检查（2026-09-17 免伤算子轮）。

问题：若每个块/每张图都能**完美地**在 {输入, RDNet 输出} 中选优，能拿多少？
判据（预登记）：块级 oracle 相对纯 RDNet 提升 >= +1.0 dB（PSNR）且 LPIPS 同向改善 → 门控有空间。
零训练，复用现成预测。输出 eval/eval_e4_oracle.csv + summary。
"""
import csv, os
import cv2
import numpy as np
import torch
import pyiqa

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet')}
BS = 64


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def pred_path(tag, sc, stem):
    base = SETS[tag][2]
    return (f'{base}/{sc}/ytmt_ucs_sirs_l.png' if tag == 'jun'
            else f'{base}/{sc}__{stem}/ytmt_ucs_sirs_l.png')


def to_tensor(bgr):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def main():
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    try:
        lpips = pyiqa.create_metric('lpips', device=dev)
    except Exception as e:
        print(f'[warn] LPIPS 不可用（{type(e).__name__}）-> 本轮只报 PSNR')
        lpips = None
    rows = []
    for tag, (d, gtn, _) in SETS.items():
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            GT = cv2.imread(f'{d}/{sc}/{gtn}')
            if GT is None:
                continue
            ins = sorted(f for f in os.listdir(f'{d}/{sc}') if f.startswith('I')) or ['I.png']
            for fn in ins:
                I = cv2.imread(f'{d}/{sc}/{fn}')
                pp = pred_path(tag, sc, fn[:-4])
                if I is None or not os.path.exists(pp):
                    continue
                R = cv2.resize(cv2.imread(pp), (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_LINEAR)
                # 图像级
                p_in, p_rd = psnr(I, GT), psnr(R, GT)
                # 块级
                h, w = GT.shape[:2]
                bi, br = [], []
                for y in range(0, h - BS + 1, BS):
                    for x in range(0, w - BS + 1, BS):
                        g = GT[y:y + BS, x:x + BS]
                        if g.mean() < 10:
                            continue
                        bi.append(psnr(I[y:y + BS, x:x + BS], g))
                        br.append(psnr(R[y:y + BS, x:x + BS], g))
                bi, br = np.array(bi), np.array(br)
                bo = np.maximum(bi, br)
                pick_rd = float((br > bi).mean())
                # 感知（图像级）
                l_in = l_rd = float('nan')
                if lpips is not None:
                    with torch.no_grad():
                        l_in = float(lpips(to_tensor(I).to(dev), to_tensor(GT).to(dev)))
                        l_rd = float(lpips(to_tensor(R).to(dev), to_tensor(GT).to(dev)))
                rows.append(dict(set=tag, scene=sc, stem=fn[:-4],
                                 img_psnr_in=round(p_in, 2), img_psnr_rd=round(p_rd, 2),
                                 img_oracle=round(max(p_in, p_rd), 2),
                                 blk_oracle=round(float(bo.mean()), 2),
                                 blk_rdnet=round(float(br.mean()), 2),
                                 blk_input=round(float(bi.mean()), 2),
                                 blk_pick_rd_frac=round(pick_rd, 3),
                                 lpips_in=round(l_in, 4), lpips_rd=round(l_rd, 4)))
    os.makedirs('eval', exist_ok=True)
    with open('eval/eval_e4_oracle.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ('set', 'scene', 'stem')}
    print(f'n={len(rows)} 张图（三套合并）')
    print(f'  图像级 PSNR 中位:  输入 {np.median(a["img_psnr_in"]):6.2f} | RDNet {np.median(a["img_psnr_rd"]):6.2f} | '
          f'oracle {np.median(a["img_oracle"]):6.2f}')
    print(f'  块级   PSNR 均值:  输入 {a["blk_input"].mean():6.2f} | RDNet {a["blk_rdnet"].mean():6.2f} | '
          f'oracle {a["blk_oracle"].mean():6.2f}')
    print(f'  块级 oracle − RDNet = {(a["blk_oracle"] - a["blk_rdnet"]).mean():+.2f} dB '
          f'（选 RDNet 的块占比中位 {np.median(a["blk_pick_rd_frac"]) * 100:.0f}%）')
    print(f'  图像级 oracle − RDNet = {(a["img_oracle"] - a["img_psnr_rd"]).mean():+.2f} dB')
    if not np.isnan(a["lpips_in"]).all():
        print(f'  LPIPS 中位（越低越好）: 输入 {np.median(a["lpips_in"]):.4f} | RDNet {np.median(a["lpips_rd"]):.4f} '
              f'| 差值 {np.median(a["lpips_rd"] - a["lpips_in"]):+.4f}')
    else:
        print('  LPIPS: 未获取（权重下载失败）')
    print('  写出 eval/eval_e4_oracle.csv')


if __name__ == '__main__':
    main()
