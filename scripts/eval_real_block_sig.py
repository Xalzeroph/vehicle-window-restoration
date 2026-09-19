#!/usr/bin/env python3
"""三套实拍块级签名（2026-09-19 轮 §4c，RemoVal"通病"判据）。

按 e3_robs_binning 口径（块 64²，r_obs=mean(D_块)/mean(T_块)，T 块均值<25 跳过）：
  A 低反射分位区（r_obs<0.05 块）ΔPSNR(arm − identity_gm)：场景聚类 bootstrap 中位 <0 且 CI 上界 ≤0
  B 干净区梯度能量损失：1 − ge(arm)/ge(identity_gm)（低反射块），逐案中位 >20%
单套"支持" = A 或 B；≥2/3 套支持 → 主张"既有评测通病"成立。
用法：python scripts/eval_real_block_sig.py
"""
import csv, json, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import psnr, grad_energy, scene_bootstrap  # noqa: E402

BS = 64
SETS = {
    'jun': dict(d='data/test_sets/real_pair_0917', gtn='T.png', layout='jun',
                arms={'rdnet': ('runs/pred/real_pair_0917_rdnet', 'ytmt_ucs_sirs_l.png'),
                      'remo': ('runs/pred/real_pair_0917_remo', 'remo.png')}),
    'apr': dict(d='data/test_sets/real_pair_apr_0917', gtn='GT.png', layout='multi',
                arms={'rdnet': ('runs/pred/apr_rdnet', 'ytmt_ucs_sirs_l.png'),
                      'remo': ('runs/pred/apr_remo', 'remo.png')}),
    'may': dict(d='data/test_sets/real_pair_may_0917', gtn='GT.png', layout='multi',
                arms={'rdnet': ('runs/pred/may_rdnet', 'ytmt_ucs_sirs_l.png'),
                      'remo': ('runs/pred/may_remo', 'remo.png')}),
}


def cases_of(cfg):
    out = []
    for sc in sorted(x for x in os.listdir(cfg['d']) if x.startswith('VID_')):
        sd = os.path.join(cfg['d'], sc)
        if cfg['layout'] == 'jun':
            out.append((sc, sc, os.path.join(sd, 'I.png')))
        else:
            for fn in sorted(f for f in os.listdir(sd) if f.startswith('I_')):
                out.append((sc, f'{sc}__{fn[:-4]}', os.path.join(sd, fn)))
    return out


def blocks_case(I, T, arm_img):
    """主口径 gm：臂输出先 LS 对齐 T（GM）再算 PSNR/梯度，与 eval_protocol 一致。"""
    i_, t_ = I.astype(np.float64), T.astype(np.float64)
    g = float((i_ * t_).sum() / ((i_ * i_).sum() + 1e-9))
    D = np.abs(g * i_ - t_).max(axis=2)
    IG = np.clip(g * i_, 0, 255).astype(np.uint8)
    a_ = arm_img.astype(np.float64)
    ga = float((a_ * t_).sum() / ((a_ * a_).sum() + 1e-9))
    GM = np.clip(ga * a_, 0, 255).astype(np.uint8)
    h, w = T.shape[:2]
    recs = []
    for y in range(0, h - BS + 1, BS):
        for x in range(0, w - BS + 1, BS):
            tb = t_[y:y + BS, x:x + BS]
            if tb.mean() < 25:
                continue
            r_obs = float(D[y:y + BS, x:x + BS].mean() / (tb.mean() + 1e-9))
            recs.append(dict(r_obs=r_obs,
                             d_psnr=psnr(GM[y:y + BS, x:x + BS], T[y:y + BS, x:x + BS])
                             - psnr(IG[y:y + BS, x:x + BS], T[y:y + BS, x:x + BS]),
                             ge_ratio=grad_energy(GM[y:y + BS, x:x + BS])
                             / max(grad_energy(IG[y:y + BS, x:x + BS]), 1e-6)))
    return recs


def main():
    rows, verdict = [], {}
    for tag, cfg in SETS.items():
        for arm, (root, pf) in cfg['arms'].items():
            by_scene_low, loss_by_case = {}, {}
            for sc, key, ip in cases_of(cfg):
                I = cv2.imread(ip); T = cv2.imread(os.path.join(cfg['d'], sc, cfg['gtn']))
                pp = os.path.join(root, key, pf)
                if I is None or T is None or not os.path.exists(pp):
                    print(f'{key}: 缺文件 -> 跳过'); continue
                A = cv2.imread(pp)
                if A.shape != T.shape:
                    A = cv2.resize(A, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
                recs = blocks_case(I, T, A)
                for b in recs:
                    b.update(set=tag, arm=arm, scene=sc, case=key)
                rows += recs
                low = [b for b in recs if b['r_obs'] < 0.05]
                for b in low:
                    by_scene_low.setdefault(sc, []).append(b['d_psnr'])
                if low:
                    loss_by_case[key] = 1.0 - float(np.mean([b['ge_ratio'] for b in low]))
                print(f'{tag}/{arm}/{key}: {len(recs)} blocks (low {len(low)})')
            sA = scene_bootstrap(by_scene_low) if by_scene_low else dict(median=None, ci_lo=None, ci_hi=None)
            sigA = (sA['median'] is not None) and sA['median'] < 0 and sA['ci_hi'] <= 0
            loss_med = float(np.median(list(loss_by_case.values()))) if loss_by_case else None
            sigB = (loss_med is not None) and loss_med > 0.20
            verdict[f'{tag}/{arm}'] = dict(A=dict(**sA, pass_=bool(sigA)), B=dict(loss_median=loss_med, pass_=bool(sigB)),
                                           support=bool(sigA or sigB))
            print(f'  → {tag}/{arm}: A(低反射区ΔPSNR) 中位 {sA["median"]} CI [{sA["ci_lo"]},{sA["ci_hi"]}] {"支持" if sigA else "不支持"} | '
                  f'B(干净区梯度损失) {loss_med} {"支持" if sigB else "不支持"}')

    os.makedirs('eval', exist_ok=True)
    with open('eval/eval_real_block_sig.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for arm in ('rdnet', 'remo'):
        sup = sum(verdict[f'{t}/{arm}']['support'] for t in SETS)
        verdict[f'{arm}_verdict'] = f'{sup}/3 套支持' + ('（≥2/3 → 通病成立）' if sup >= 2 else '（<2/3 → 不支持通病）')
        print(f'\n★ {arm}: {verdict[arm + "_verdict"]}')
    json.dump(verdict, open('eval/eval_real_block_sig_verdict.json', 'w'), indent=1, ensure_ascii=False)
    print('写出 eval/eval_real_block_sig.csv 与 verdict.json')


if __name__ == '__main__':
    main()
