#!/usr/bin/env python3
"""开源 benchmark 回归检查（nature20 + real20，40 对，与训练池 md5 零命中）。

用途：**标准去反射能力的防遗忘检查**（不含"洞"，不测人影修复）。
对照：官方 RDNet(zero-shot) / v1@10k / v2@6k / identity。
指标：PSNR、crop-LPIPS（全图）。产物 eval/eval_bench_open_0921.csv
"""
import os, sys, csv
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ORIG_CWD = os.getcwd()
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, 'RDNet'))
os.chdir(os.path.join(PROJ, 'RDNet'))
from finetune_rdnet_peft import build                      # noqa: E402
import pyiqa                                               # noqa: E402

B = '/home/zmy/mycode/datasets'
BENCH = [('nature20', f'{B}/nature/nature20/blended', f'{B}/nature/nature20/transmission'),
         ('real20', f'{B}/real/real20/blended', f'{B}/real/real20/transmission_layer'),
         ('objects200', f'{B}/SIR2/SolidObjectDataset/solid200/blended', f'{B}/SIR2/SolidObjectDataset/solid200/transmission_layer'),
         ('postcard199', f'{B}/SIR2/postcard/postcard199/blended', f'{B}/SIR2/postcard/postcard199/transmission_layer'),
         ('wild55', f'{B}/SIR2/Wildscene/wild55/blended', f'{B}/SIR2/Wildscene/wild55/transmission_layer')]
MODELS = [('official', None), ('v1_10k', os.path.join(PROJ, 'runs/ckpt/rdt_s2_main_s101/ckpt_final.pth')),
          ('v2_6k', os.path.join(PROJ, 'runs/ckpt/rdt_s2v2_main_s101/ckpt_final.pth'))]


def pad32(x):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REFLECT), h, w


def infer(net, net_c, dev, I):
    Ip, h, w = pad32(I)
    x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        code = net_c(x)
        _, outs = net(x, code, prompt=True)
        T_hat = outs[-1][:, 3:]
    out = T_hat[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1][:h, :w]
    return (out * 255).round().astype(np.uint8)


def main():
    dev = torch.device('cuda:0')
    lp = pyiqa.create_metric('lpips', device=dev)
    rows = []
    for mname, ckpt in MODELS:
        net, net_c = build(save_memory=True, mona='none' if ckpt is None else 'both', grad_ckpt=False)
        if ckpt is not None:
            ck = torch.load(ckpt, map_location='cpu')
            miss, _ = net.load_state_dict(ck['icnn'], strict=False)
            assert not [k for k in miss if 'mona' not in k], miss[:5]
        net.to(dev).eval(); net_c.to(dev)
        for bname, idir, tdir in BENCH:
            for fn in sorted(os.listdir(idir)):
                stem = os.path.splitext(fn)[0]
                cand = [f'{stem}{e}' for e in ('.png', '.jpg', '.jpeg')]
                tf = next((c for c in cand if os.path.exists(os.path.join(tdir, c))), None)
                if tf is None:
                    continue
                I = cv2.imread(os.path.join(idir, fn))
                T = cv2.imread(os.path.join(tdir, tf))
                if I is None or T is None or I.shape != T.shape:
                    continue
                if max(I.shape[:2]) > 1024:      # 大图等比缩到长边 1024（全部臂同处理）
                    sc = 1024 / max(I.shape[:2])
                    sz = (int(I.shape[1] * sc), int(I.shape[0] * sc))
                    I, T = cv2.resize(I, sz, interpolation=cv2.INTER_AREA), cv2.resize(T, sz, interpolation=cv2.INTER_AREA)
                for an, A in (('identity', I), (mname, infer(net, net_c, dev, I))):
                    mse = np.mean((A.astype(np.float64) - T.astype(np.float64)) ** 2)
                    ps = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
                    with torch.no_grad():
                        a = torch.from_numpy(A[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
                        b = torch.from_numpy(T[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
                        lpv = float(lp(a, b))
                    rows.append(dict(bench=bname, case=stem, arm=an, psnr=round(ps, 2),
                                     lpips=round(lpv, 4)))
        print(f'[{mname}] done', flush=True)
        del net, net_c
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(PROJ, 'eval'), exist_ok=True)
    with open(os.path.join(PROJ, 'eval/eval_bench_open_0921.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['bench', 'case', 'arm', 'psnr', 'lpips'])
        wr.writeheader(); wr.writerows(rows)
    print('=== 汇总（PSNR 均值↑ / LPIPS 均值↓）===')
    for bname, _, _ in BENCH:
        for an in ['identity'] + [m for m, _ in MODELS]:
            sub = [r for r in rows if r['bench'] == bname and r['arm'] == an]
            if sub:
                print(f'{bname:<9} {an:<9} n={len(sub)} PSNR {np.mean([r["psnr"] for r in sub]):6.2f} | '
                      f'LPIPS {np.mean([r["lpips"] for r in sub]):.4f}')


if __name__ == '__main__':
    main()
