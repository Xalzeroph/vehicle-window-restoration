#!/usr/bin/env python3
"""M1 预登记自检：oracle 掩码臂 —— m = 真值 Ω（配对算出），D = 官方残差。

T̂ = I − m_Ω ⊙ D。若该臂在 dev ① 也 ≤1/11 → M1 性能侧解读作废（归因接法/残差，
非定位）。产物: runs/pred/<set>_oraclemask/<key>/ytmt_ucs_sirs_l.png
用法: CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_oracle_mask_0921.py --sets may,apr --gpu 0
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
from finetune_rdnet_peft import build               # noqa: E402

sys.path.insert(0, HERE)
import importlib.util                               # noqa: E402
spec = importlib.util.spec_from_file_location("eg2", os.path.join(HERE, "eval_g2_baseline.py"))
eg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(eg2)   # noqa: E402

SETS = {'may': (os.path.join(PROJ, 'data/test_sets/real_pair_may_0917'), 'GT.png'),
        'apr': (os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917'), 'GT.png')}


def pad32(x):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REFLECT), h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='may,apr')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.gpu}')
    net, net_c = build(save_memory=True, mona='none', grad_ckpt=False)
    net.to(dev).eval(); net_c.to(dev)
    for sname in args.sets.split(','):
        d, gtn = SETS[sname]
        root = os.path.join(PROJ, f'runs/pred/{sname}_oraclemask')
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            T = cv2.imread(os.path.join(d, sc, gtn))
            for fn in sorted(f for f in os.listdir(os.path.join(d, sc)) if f.startswith('I_')):
                key = f'{sc}__{fn[:-4]}'
                I = cv2.imread(os.path.join(d, sc, fn))
                o = eg2.omega_of(I, T)
                m = o['Om'].astype(np.float32) if o is not None else np.zeros(I.shape[:2], np.float32)
                Ip, h, w = pad32(I)
                mp = cv2.copyMakeBorder(m, 0, (-I.shape[0]) % 32, 0, (-I.shape[1]) % 32,
                                        cv2.BORDER_CONSTANT, value=0)
                x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
                mp_t = torch.from_numpy(mp)[None, None].to(dev)
                with torch.no_grad():
                    code = net_c(x)
                    _, outs = net(x, code, prompt=True)
                    D = x - outs[-1][:, 3:]
                    Tout = x - mp_t * D                      # T̂ = I − m_Ω ⊙ D
                Tn = Tout[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1][:h, :w]
                od = os.path.join(root, key); os.makedirs(od, exist_ok=True)
                cv2.imwrite(os.path.join(od, 'ytmt_ucs_sirs_l.png'),
                            (Tn * 255).round().astype(np.uint8))
        print(f'[oracle-mask] {sname} -> {root}', flush=True)


if __name__ == '__main__':
    main()
