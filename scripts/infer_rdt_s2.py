#!/usr/bin/env python3
"""S2 推理：加载 Mona-PEFT ckpt，对测试集产出 T̂ 预测（兼容现有评测的 ytmt_ucs_sirs_l.png 命名）。

产物: runs/pred/<set>_<tag>/<key>/ytmt_ucs_sirs_l.png
用法: CUDA_VISIBLE_DEVICES=1 python3 scripts/infer_rdt_s2.py \
        --ckpt runs/ckpt/rdt_s2_main_s101/ckpt_latest.pth --sets may --tag s2iter1k --gpu 0
"""
import argparse, os, sys
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ORIG_CWD = os.getcwd()                               # chdir 前记录（相对 --ckpt 路径基准）
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "RDNet"))
os.chdir(os.path.join(PROJ, "RDNet"))
from finetune_rdnet_peft import build                # noqa: E402

SETS = {'jun': (os.path.join(PROJ, 'data/test_sets/real_pair_0917'), 'T.png'),
        'apr': (os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917'), 'GT.png'),
        'may': (os.path.join(PROJ, 'data/test_sets/real_pair_may_0917'), 'GT.png')}


def pad32(x):
    h, w = x.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    return cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT), ph, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sets', default='apr,may,jun')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.gpu}')
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ORIG_CWD, args.ckpt)
    net, net_c = build(save_memory=True, mona='both', grad_ckpt=False)
    ck = torch.load(ckpt, map_location='cpu')
    miss, unex = net.load_state_dict(ck['icnn'], strict=False)
    assert not [k for k in miss if 'mona' not in k], f'非 mona 权重缺失: {miss[:5]}'
    print(f'[infer] ckpt iter={ck.get("iter")} arm={ck.get("arm")} missing={len(miss)}', flush=True)
    net.to(dev).eval(); net_c.to(dev)
    for sname in args.sets.split(','):
        d, _ = SETS[sname]
        out_root = os.path.join(PROJ, f'runs/pred/{sname}_{args.tag}')
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            sd = os.path.join(d, sc)
            for fn in (sorted(f for f in os.listdir(sd) if f.startswith('I')) or ['I.png']):
                key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
                I = cv2.imread(os.path.join(sd, fn))
                if I is None:
                    continue
                Ip, ph, pw = pad32(I)
                x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
                with torch.no_grad():
                    code = net_c(x)
                    _, outs = net(x, code, prompt=True)
                    T_hat = outs[-1][:, 3:]                       # 后 3 通道 = 干净图（官方口径）
                Tn = T_hat[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]
                h, w = I.shape[:2]
                Tn = (Tn[:h, :w] * 255).round().astype(np.uint8)
                od = os.path.join(out_root, key)
                os.makedirs(od, exist_ok=True)
                cv2.imwrite(os.path.join(od, 'ytmt_ucs_sirs_l.png'), Tn)
        print(f'[infer] {sname} -> {out_root}', flush=True)


if __name__ == '__main__':
    main()
