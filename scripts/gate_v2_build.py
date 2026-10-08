#!/usr/bin/env python3
"""门控 v2 建集：P 换成自有微调臂（预测直接读缓存，零 GPU 推理）
用法: python3 scripts/gate_v2_build.py --arm lovo_k2 --res 640
产物: data/gate_v2/<arm>/{apr,may}/*.npz  (I,P,T 原生缩放到长边 res)
"""
import argparse, glob, os, sys
import cv2, numpy as np
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {'apr': f'{PROJ}/data/test_sets/real_pair_apr_0917',
        'may': f'{PROJ}/data/test_sets/real_pair_may_0917'}
ARMS = {'lovo_k2':   {'apr': 'apr_lovo_k2', 'may': 'may_lovo_k2'},
        's2main10k': {'apr': 'apr_s101_apr10k', 'may': 'may_rdt_s2_main_s101_10000'}}


def imread_any(path):
    if path is None:
        return None
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def find_pred(d, case, stem):
    cands = [os.path.join(d, f'{case}__I_{stem}.png'), os.path.join(d, f'{case}__I_{stem}'),
             os.path.join(d, f'{case}__{stem}.png'), os.path.join(d, f'{case}__{stem}')]
    cands += sorted(glob.glob(os.path.join(d, f'*{stem}*')))
    for p in cands:
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            inner = sorted(glob.glob(os.path.join(p, '*.png')))
            if inner:
                pref = [x for x in inner if x.endswith('_l.png') or 'remo' in os.path.basename(x)]
                return (pref or inner)[0]
    return None


def fit_cap(im, cap):
    if max(im.shape[:2]) <= cap:
        return im
    sc = cap/max(im.shape[:2]); sz = (int(im.shape[1]*sc), int(im.shape[0]*sc))
    return cv2.resize(im, sz, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True); ap.add_argument('--res', type=int, default=640)
    a = ap.parse_args()
    root = os.path.join(PROJ, 'data/gate_v2', a.arm)
    for sname, d in SETS.items():
        outd = os.path.join(root, sname); os.makedirs(outd, exist_ok=True)
        pd = os.path.join(PROJ, 'runs/pred', ARMS[a.arm][sname])
        n = 0
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            T = cv2.imread(f'{d}/{sc}/GT.png')
            if T is None:
                continue
            for f in sorted(x for x in os.listdir(f'{d}/{sc}') if x.startswith('I_')):
                stem = f[2:-4]
                I = cv2.imread(f'{d}/{sc}/{f}')
                pp = find_pred(pd, sc, stem)
                P = imread_any(pp)
                if I is None or P is None:
                    print(f'   SKIP {sc}__{stem}: 读入失败', flush=True)
                    continue
                # 以 GT 尺寸为准（注意：GT 在测试集里已是长边 640；不可重绑定外层 T！）
                Tc = T if max(T.shape[:2]) <= a.res else fit_cap(T, a.res)
                th, tw = Tc.shape[:2]
                Ic = I if I.shape[:2] == (th, tw) else cv2.resize(I, (tw, th), interpolation=cv2.INTER_AREA)
                Pc = P if P.shape[:2] == (th, tw) else cv2.resize(P, (tw, th), interpolation=cv2.INTER_AREA)
                h, w = th//32*32, tw//32*32
                np.savez_compressed(os.path.join(outd, f'{sc}__{stem}.npz'),
                                    I=Ic[:h, :w], P=Pc[:h, :w], T=Tc[:h, :w])
                n += 1
        print(f'[{a.arm}/{sname}] {n} 例 -> {outd}', flush=True)


if __name__ == '__main__':
    main()
