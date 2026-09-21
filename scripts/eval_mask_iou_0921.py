#!/usr/bin/env python3
"""M1 主判：跨场景掩码 IoU 评测（预登记门，docs/2026.9.21_方法线战略_待做清单.md §3.2）。

对 may（跨域，11 帧/5 场景）与 apr 留出场景（域内）：预测 m̂ = head(官方特征)，
与真值 Ω（omega_of(I,T) 配对算出）比 IoU。
判定：场景级 IoU 中位 ≥0.50 且 ≥3/5 场景 ≥0.50 → 可迁移；[0.30,0.50) 部分；<0.30 判死。
用法: CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_mask_iou_0921.py --ckpt runs/ckpt/mask_w96/ckpt_final.pth
"""
import argparse, os, sys
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ORIG_CWD = os.getcwd()
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "RDNet"))
os.chdir(os.path.join(PROJ, "RDNet"))
from finetune_rdnet_peft import build                       # noqa: E402
sys.path.insert(0, HERE)
from finetune_rdnet_mask import MaskHead                    # noqa: E402
import importlib.util                                       # noqa: E402
spec = importlib.util.spec_from_file_location("eg2", os.path.join(HERE, "eval_g2_baseline.py"))
eg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(eg2)

SETS = {'may': (os.path.join(PROJ, 'data/test_sets/real_pair_may_0917'), 'GT.png'),
        'apr_hold': (os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917'), 'GT.png')}
HOLD = ('VID_20260426_173604', 'VID_20260426_173827')


def pad32(x):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REFLECT), h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sets', default='may,apr_hold')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.gpu}')
    net, net_c = build(save_memory=True, mona='none', grad_ckpt=False)
    net.eval().requires_grad_(False)
    net.to(dev); net_c.to(dev)
    feats = {}
    net.baseball.register_forward_hook(lambda m, i, o: feats.update(base=o[0]))
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ORIG_CWD, args.ckpt)
    ck = torch.load(ckpt, map_location='cpu')
    head = MaskHead(ck['in_chs'], width=ck['width']).to(dev)
    head.load_state_dict(ck['head']); head.eval()
    print(f'[iou] ckpt {os.path.basename(os.path.dirname(args.ckpt))} width={ck["width"]} iter={ck["iter"]}')
    for sname in args.sets.split(','):
        d, gtn = SETS[sname]
        per_scene = {}
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            if sname == 'apr_hold' and sc not in HOLD:
                continue
            T = cv2.imread(os.path.join(d, sc, gtn))
            for fn in sorted(f for f in os.listdir(os.path.join(d, sc)) if f.startswith('I')):
                key = f'{sc}__{fn[:-4]}' if fn != 'I.png' else sc
                I = cv2.imread(os.path.join(d, sc, fn))
                o = eg2.omega_of(I, T)
                if o is None:
                    continue
                gt = o['Om']
                Ip, h, w = pad32(I)
                x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
                with torch.no_grad():
                    net(x, net_c(x), prompt=True)
                    lg = head(feats['base'], (h, w))
                    pr = (torch.sigmoid(lg)[0, 0, :h, :w] > 0.5).cpu().numpy()
                inter = float((pr & gt).sum()); union = float((pr | gt).sum())
                iou = inter / max(union, 1)
                per_scene.setdefault(sc, []).append(iou)
                print(f'  {sname}/{key:<38} IoU {iou:.3f} (Ω {gt.mean()*100:.0f}%, pred {pr.mean()*100:.0f}%)')
        meds = {sc: float(np.mean(v)) for sc, v in per_scene.items()}
        if meds:
            vals = sorted(meds.values())
            med = vals[len(vals) // 2]
            n_ok = sum(1 for v in vals if v >= 0.50)
            verdict = '可迁移' if (med >= 0.50 and n_ok >= max(1, int(np.ceil(0.6*len(vals))))) \
                else ('部分迁移' if med >= 0.30 else '判死')
            print(f'== {sname}: 场景级 IoU 中位 {med:.3f} | ≥0.50 场景 {n_ok}/{len(vals)} -> **{verdict}** ==')


if __name__ == '__main__':
    main()
