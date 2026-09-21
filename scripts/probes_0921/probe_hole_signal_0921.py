#!/usr/bin/env python3
"""洞信号判决探针（Opus 设计）：区分"记住 apr 帧的映射" vs "按洞深信号驱动修复"。

实验 A（记忆测试，apr 抹洞）：I' = I 但 Ω 填真值 T（洞消失）。若 S2 仍大改 Ω（|P'−I'| ≈ |P−I|）
  → 按帧改（记忆）；若 ≈ 全图低通底噪 floor → 无洞不修（信号驱动）。
实验 B（覆盖测试，注入 apr 级暗块）：向帧注入 depth≈30 的暗块（形状 = apr Ω 最大连通域，软边）。
  corr = 1 − mean|P_inj − P|[mask] / δ：1 = 注入的暗块被"修掉"，0 = 原样透过（不理）。
  may 帧（164653/163859/162203，ρ 0.01–0.47）+ apr 阳性对照（注入到非 Ω 区，若同样被修=操作有效）。
用法: CUDA_VISIBLE_DEVICES=1 python3 scripts/probes_0921/probe_hole_signal_0921.py
"""
import csv, importlib.util, os, sys
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
ORIG_CWD = os.getcwd()
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
sys.path.insert(0, os.path.join(PROJ, 'RDNet'))
os.chdir(PROJ)
spec = importlib.util.spec_from_file_location('eg2', os.path.join(PROJ, 'scripts/eval_g2_baseline.py'))
eg2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg2)
os.chdir(os.path.join(PROJ, 'RDNet'))
from finetune_rdnet_peft import build                       # noqa: E402

CKPT = os.path.join(PROJ, 'runs/ckpt/rdt_s2_main_s101/ckpt_10000.pth')
SETS = {'may': ('data/test_sets/real_pair_may_0917', [('VID_20260512_164653', 'I_000000.png'),
                                                      ('VID_20260512_163859', 'I_000000.png'),
                                                      ('VID_20260512_162203', 'I_000000.png')]),
        'apr': ('data/test_sets/real_pair_apr_0917', [('VID_20260426_173237', 'I_000000.png'),
                                                      ('VID_20260426_173604', 'I_000007.png')])}
EXP_A = ('apr', [('VID_20260426_173237', 'I_000000.png'), ('VID_20260426_173604', 'I_000007.png'),
                 ('VID_20260426_173349', 'I_000000.png')])


def pad32(x, p=32):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % p, 0, (-w) % p, cv2.BORDER_REFLECT)


def infer(net, net_c, I, dev):
    Ip = pad32(I)
    x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        code = net_c(x)
        _, outs = net(x, code, prompt=True)
        T_hat = outs[-1][:, 3:]
    Tn = T_hat[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]
    h, w = I.shape[:2]
    return (Tn[:h, :w] * 255).round().astype(np.uint8)


def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def largest_cc(mask):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    return (lab == i), st[i]


def pick_site(g, mask_h, w, h, avoid):
    """网格扫落点：环形邻域最平 + 尽量避开已有 Ω（评分 = ring std + 200×重叠率）。"""
    best = None
    mh, mw = mask_h.shape
    for fy in np.linspace(0.16, 0.84, 7):
        for fx in np.linspace(0.14, 0.86, 7):
            cx, cy = int(fx * w), int(fy * h)
            y0, y1 = cy - mh // 2 - 20, cy + mh // 2 + 20
            x0, x1 = cx - mw // 2 - 20, cx + mw // 2 + 20
            if x0 < 2 or y0 < 2 or x1 > w - 2 or y1 > h - 2:
                continue
            ov = float(avoid[cy - mh // 2:cy + mh // 2, cx - mw // 2:cx + mw // 2].mean())
            score = float(g[y0:y1, x0:x1].std()) + 200.0 * ov
            if best is None or score < best[0]:
                best = (score, cx, cy)
    return (best[1], best[2]) if best else (w // 2, h // 2)


def main():
    dev = torch.device('cuda:0')
    net, net_c = build(save_memory=True, mona='both', grad_ckpt=False)
    ck = torch.load(CKPT, map_location='cpu')
    miss, _ = net.load_state_dict(ck['icnn'], strict=False)
    assert not [k for k in miss if 'mona' not in k], miss[:5]
    net.to(dev).eval(); net_c.to(dev)
    print(f'[probe] ckpt iter={ck.get("iter")} 加载完成', flush=True)

    def load(setname, sc, fn):
        d = SETS[setname][0]
        I = cv2.imread(os.path.join(PROJ, d, sc, fn))
        T = cv2.imread(os.path.join(PROJ, d, sc, 'GT.png'))
        return I, T

    rows = []
    # ---- 实验 A：apr 抹洞（Ω 填真值 T）
    print('\n== 实验 A（apr 抹洞=Ω 填真值 T；比值≈1=记忆，≈floor/fix=信号驱动）==')
    print(f'{"frame":<22}{"|Ω|%":>6}{"fix_in":>8}{"erase_in":>9}{"floor_out":>10}{"比值":>7}')
    for sc, fn in EXP_A[1]:
        I, T = load('apr', sc, fn)
        o = eg2.omega_of(I, T)
        if not o:
            print(f'{sc[-6:]}/{fn[:-4]:<10}  无 Ω'); continue
        Om = o['Om']
        P = infer(net, net_c, I, dev)
        sfile = os.path.join(PROJ, f'runs/pred/apr_s101_apr10k/{sc}__{fn[:-4]}/ytmt_ucs_sirs_l.png')
        if os.path.exists(sfile):
            Ps = cv2.imread(sfile)
            if Ps is not None and Ps.shape == P.shape:
                print(f'  [复现核验] {sc[-6:]}/{fn[:-4]} max|P−P_stored| = '
                      f'{int(np.abs(P.astype(int) - Ps.astype(int)).max())}')
        I2 = I.copy(); I2[Om] = T[Om]
        P2 = infer(net, net_c, I2, dev)
        fix_in = float(np.abs(gray(P) - gray(I))[Om].mean())
        er_in = float(np.abs(gray(P2) - gray(I2))[Om].mean())
        floor = float(np.abs(gray(P) - gray(I))[~Om].mean())
        mIT_in = float(np.abs(gray(I) - gray(T))[Om].mean())
        mIT_out = float(np.abs(gray(I) - gray(T))[~Om].mean())
        pT_in = float(np.abs(gray(P) - gray(T))[Om].mean())
        pp = float(np.abs(gray(P2) - gray(P))[Om].mean())
        print(f'{sc[-6:]}/{fn[:-4]:<10} {100*Om.mean():>6.1f} {fix_in:>8.1f} {er_in:>9.1f} '
              f'{floor:>10.1f} {er_in/max(fix_in,1e-6):>7.2f} | |I-T|in {mIT_in:>5.1f} out {mIT_out:>5.1f} '
              f'|P-T|in {pT_in:>5.1f} |P\'-P|in {pp:>5.1f}')
        rows.append(dict(exp='A', frame=f'{sc}__{fn[:-4]}', omega_pct=round(100*Om.mean(), 1),
                         fix_in=round(fix_in, 2), erase_in=round(er_in, 2), floor_out=round(floor, 2),
                         ratio=round(er_in/max(fix_in, 1e-6), 3), mIT_in=round(mIT_in, 1),
                         mIT_out=round(mIT_out, 1), pT_in=round(pT_in, 1), dPP_in=round(pp, 1)))

    # ---- 实验 B：注入 apr 级暗块
    print('\n== 实验 B（注入 depth≈30；corr_dark=暗块被修掉的比例；corr_light=亮块被"修掉"比例）==')
    print(f'{"frame":<22}{"注入后depth":>10}{"δ":>6}{"corr_dark":>10}{"corr_light":>11}{"落点":>12}')
    shape = None
    Iap, Tap = load('apr', *EXP_A[1][0])
    oap = eg2.omega_of(Iap, Tap)
    cc = largest_cc(oap['Om']) if oap else None
    if cc:
        m0, st0 = cc
        x, y, w0, h0 = st0[0], st0[1], st0[2], st0[3]
        shape = m0[y:y+h0, x:x+w0].astype(np.uint8) * 255
    assert shape is not None, '未取到 apr Ω 形状'
    for setname in ('may', 'apr'):
        for sc, fn in SETS[setname][1]:
            I, T = load(setname, sc, fn)
            h, w = I.shape[:2]
            tgt_h = int(0.25 * h)
            tgt_w = max(8, int(shape.shape[1] * tgt_h / shape.shape[0]))
            m = cv2.resize(shape, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
            m = cv2.GaussianBlur(m.astype(np.float32) / 255., (0, 0), 4.0)
            g = gray(I)
            o = eg2.omega_of(I, T)
            avoid = o['Om'] if o else np.zeros_like(I[..., 0], bool)
            cx, cy = pick_site(g, m, w, h, avoid)
            y0, yo = cy - tgt_h // 2, cy + tgt_h // 2
            x0, xo = cx - tgt_w // 2, cx + tgt_w // 2
            mm = m[max(0, -y0):tgt_h - max(0, yo - h), max(0, -x0):tgt_w - max(0, xo - w)]
            y0, x0 = max(0, y0), max(0, x0)
            hard = mm > 0.5
            ring = g[y0:y0+mm.shape[0], x0:x0+mm.shape[1]][~hard]
            d0 = float(ring.mean() - g[y0:y0+mm.shape[0], x0:x0+mm.shape[1]][hard].mean())
            delta = max(10.0, 30.0 - d0)
            If = I.astype(np.float32).copy()
            sub = If[y0:y0+mm.shape[0], x0:x0+mm.shape[1]]
            sub -= delta * mm[..., None]
            Iinj = np.clip(If, 0, 255).astype(np.uint8)
            P = infer(net, net_c, I, dev)
            Pinj = infer(net, net_c, Iinj, dev)
            d1 = float(gray(Iinj)[y0:y0+mm.shape[0], x0:x0+mm.shape[1]][hard].mean())
            depth1 = float(ring.mean() - d1)
            resid = float(np.abs(gray(Pinj) - gray(P))[y0:y0+mm.shape[0], x0:x0+mm.shape[1]][hard].mean())
            corr = 1 - resid / delta
            # 亮块对照（+δ）：洞修复器应保留亮块（corr_light≈0），"整帧映射到记忆干净版"会连亮块一起抹
            Ibr = I.astype(np.float32).copy()
            Ibr[y0:y0+mm.shape[0], x0:x0+mm.shape[1]] += delta * mm[..., None]
            Ibr = np.clip(Ibr, 0, 255).astype(np.uint8)
            Pbr = infer(net, net_c, Ibr, dev)
            resid_b = float(np.abs(gray(Pbr) - gray(P))[y0:y0+mm.shape[0], x0:x0+mm.shape[1]][hard].mean())
            corr_b = 1 - resid_b / delta
            tag = f'{setname} {sc[-6:]}/{fn[:-4]}'
            print(f'{tag:<22}{depth1:>10.1f}{delta:>6.0f}{corr:>7.2f}{corr_b:>9.2f}   ({cx},{cy})')
            rows.append(dict(exp='B', frame=f'{setname} {sc}__{fn[:-4]}', depth=round(depth1, 1),
                             delta=round(delta, 1), corr_dark=round(corr, 3), corr_light=round(corr_b, 3),
                             site=f'{cx},{cy}'))

    out = os.path.join(PROJ, 'eval/probe_hole_signal_0921.csv')
    with open(out, 'w', newline='') as fp:
        wtr = csv.DictWriter(fp, fieldnames=sorted({k for r in rows for k in r}))
        wtr.writeheader(); wtr.writerows(rows)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
