#!/usr/bin/env python3
"""门控 v2 验收（预登记门 G1-G4，2026-10-08）
G1 apr 保收益：(af(O)-af(I))/(af(P)-af(I)) >= 0.70
G2 apr 不劣于输入：no-harm 帧 >= 14/16
G3 may 收手：af(O) >= af(P) + 0.5 dB
G4 may 胜过平凡基线：af(O) > af(affine(I))
口径：每张图/每个输出都给一条免费的全局色调曲线（affine 对齐）
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gate_v1_train import Gate          # noqa: E402
import pyiqa                            # noqa: E402


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if mse <= 1e-12 else float(10*np.log10(255.0**2/mse))


def affine(X, T):
    x = X.astype(np.float64).ravel(); t = T.astype(np.float64).ravel()
    A = np.stack([x, np.ones_like(x)], 1); c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return np.clip(c[0]*X.astype(np.float64)+c[1], 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--arm', required=True)
    ap.add_argument('--tag', required=True); ap.add_argument('--sets', default='apr,may')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}')
    net = Gate().to(dev); sd = torch.load(os.path.join(PROJ, a.ckpt), map_location='cpu')
    net.load_state_dict(sd['sd']); net.eval()
    lp = pyiqa.create_metric('lpips', device=dev)
    print(f'[eval] arm={a.arm} ckpt={a.ckpt} train={sd.get("train")}', flush=True)
    res = {}
    for sname in a.sets.split(','):
        rows = []
        for f in sorted(glob.glob(os.path.join(PROJ, 'data/gate_v2', a.arm, sname, '*.npz'))):
            z = np.load(f); I, P, T = z['I'], z['P'], z['T']
            xI = torch.from_numpy(I.transpose(2, 0, 1)[None].astype(np.float32)/255.).to(dev)
            xP = torch.from_numpy(P.transpose(2, 0, 1)[None].astype(np.float32)/255.).to(dev)
            with torch.no_grad():
                _, g = net(xI, xP); O = (xI + g*(xP-xI)).clamp(0, 1)
            Onp = (O[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
            raw = {k: psnr(X, T) for k, X in (('I', I), ('P', P), ('O', Onp))}
            af = {k: psnr(affine(X, T), T) for k, X in (('I', I), ('P', P), ('O', Onp))}
            af['trivial'] = af['I']                    # identity + 全局色调曲线 = 平凡基线
            lps = {}
            for k, X in (('I', I), ('P', P), ('O', Onp)):
                with torch.no_grad():
                    t1 = torch.from_numpy(X[:, :, ::-1].astype(np.float32)/255.).permute(2, 0, 1)[None].to(dev)
                    t2 = torch.from_numpy(T[:, :, ::-1].astype(np.float32)/255.).permute(2, 0, 1)[None].to(dev)
                    lps[k] = float(lp(t1, t2))
            rows.append(dict(case=os.path.basename(f)[:-4],
                             raw_I=round(raw['I'], 2), raw_P=round(raw['P'], 2), raw_O=round(raw['O'], 2),
                             af_I=round(af['I'], 2), af_P=round(af['P'], 2), af_O=round(af['O'], 2),
                             lpips_I=round(lps['I'], 4), lpips_P=round(lps['P'], 4), lpips_O=round(lps['O'], 4),
                             g_mean=round(float(g.mean()), 3),
                             noharm=int(af['O'] >= af['I'] - 1e-9 and lps['O'] <= lps['I'] + 1e-9)))
        res[sname] = rows
        import csv
        with open(os.path.join(PROJ, f'eval/gate_v2_{sname}_{a.tag}.csv'), 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        m = lambda k: float(np.mean([r[k] for r in rows]))
        print(f'\n===== {sname}（n={len(rows)}）=====')
        print(f'  raw    I {m("raw_I"):6.2f} | P {m("raw_P"):6.2f} | O {m("raw_O"):6.2f}')
        print(f'  对齐   I {m("af_I"):6.2f} | P {m("af_P"):6.2f} | O {m("af_O"):6.2f}   （I 即平凡基线）')
        print(f'  LPIPS  I {m("lpips_I"):.4f} | P {m("lpips_P"):.4f} | O {m("lpips_O"):.4f}')
        print(f'  门控均值 {m("g_mean"):.3f} | no-harm 帧 {sum(r["noharm"] for r in rows)}/{len(rows)}')
    print('\n######## 预登记判决门 ########')
    if 'apr' in res and 'may' in res:
        A, M = res['apr'], res['may']
        gm = lambda rs, k: float(np.mean([r[k] for r in rs]))
        denom = gm(A, 'af_P') - gm(A, 'af_I')
        keep = (gm(A, 'af_O') - gm(A, 'af_I'))/denom if abs(denom) > 1e-6 else float('nan')
        nh = sum(r['noharm'] for r in A)
        print(f'  G1 apr 保收益 = {keep:.2f}（门 ≥0.70）  → {"PASS" if keep >= 0.70 else "FAIL"}')
        print(f'  G2 apr no-harm = {nh}/{len(A)}（门 ≥14/16）→ {"PASS" if nh >= 14 else "FAIL"}')
        d3 = gm(M, 'af_O') - gm(M, 'af_P')
        print(f'  G3 may 收手 = {d3:+.2f} dB（门 ≥+0.50）→ {"PASS" if d3 >= 0.5 else "FAIL"}')
        d4 = gm(M, 'af_O') - gm(M, 'af_I')
        print(f'  G4 may 胜平凡基线 = {d4:+.2f} dB（门 >0）  → {"PASS" if d4 > 0 else "FAIL"}')


if __name__ == '__main__':
    main()
