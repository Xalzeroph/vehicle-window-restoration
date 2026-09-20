#!/usr/bin/env python3
"""补测①：协议口径的 oracle-w 上限（2026-09-19 方法线 P2 前置）。

复用 E7 块级机制（128² 块、WS 五档、GT 块均≥15），逐块取"ΔLPIPS≤0 中最大 ΔPSNR"的 w*，
落成全图 w-map（nearest 上采样 + σ=8 羽化）→ oracle 预测图；另出 random-w 对照臂。
产物：runs/pred/{set}_oraclew/、runs/pred/{set}_randw/、eval/eval_probe_oracle_blocks.csv
随后由 eval_protocol.py 以 --arm oraclew=... --arm randw=... 出协议口径判决。
"""
import csv, os, sys
import cv2
import numpy as np
import torch
import pyiqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet', 'runs/pred/real_pair_0917_oraclew', 'runs/pred/real_pair_0917_randw'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet', 'runs/pred/apr_oraclew', 'runs/pred/apr_randw'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet', 'runs/pred/may_oraclew', 'runs/pred/may_randw')}
BS = 128
WS = [0.0, 0.25, 0.5, 0.75, 1.0]
GT_MIN_MEAN = 15.0


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def to_t(b):
    return torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)


def cases(cfg):
    d, gtn, _, _, _ = cfg
    out = []
    for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        sd = os.path.join(d, sc)
        for fn in (sorted(f for f in os.listdir(sd) if f.startswith('I')) or ['I.png']):
            key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
            out.append((sc, key, os.path.join(sd, fn), os.path.join(sd, gtn)))
    return out


def main():
    dev = torch.device('cuda:0')
    lp = pyiqa.create_metric('lpips', device=dev)
    rng = np.random.default_rng(20260919)
    blocks = []
    with torch.no_grad():
        for tag, cfg in SETS.items():
            d, gtn, pred_root, ow_root, rw_root = cfg
            for sc, key, ip, tp in cases(cfg):
                I = cv2.imread(ip); T = cv2.imread(tp)
                pp = os.path.join(pred_root, key, 'ytmt_ucs_sirs_l.png')
                if I is None or T is None or not os.path.exists(pp):
                    print(f'{key}: 缺文件 -> 跳过'); continue
                P = cv2.imread(pp)
                if P.shape != T.shape:
                    P = cv2.resize(P, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
                h, w = T.shape[:2]
                nb_y, nb_x = (h - BS) // BS + 1, (w - BS) // BS + 1
                wm = np.zeros((nb_y, nb_x), np.float64)
                rm = np.zeros((nb_y, nb_x), np.float64)
                for by in range(nb_y):
                    for bx in range(nb_x):
                        y, x = by * BS, bx * BS
                        tb = T[y:y + BS, x:x + BS]
                        if tb.mean() < GT_MIN_MEAN:
                            continue
                        ib, pb = I[y:y + BS, x:x + BS], P[y:y + BS, x:x + BS]
                        tg, ti = to_t(tb).to(dev), to_t(ib).to(dev)
                        li = float(lp(ti, tg))
                        p0 = psnr(ib, tb)
                        dPs, dLs = [], []
                        for wv in WS:
                            o = np.clip(ib * (1 - wv) + pb * wv, 0, 255).astype(np.uint8)
                            dPs.append(psnr(o, tb) - p0)
                            dLs.append(float(lp(to_t(o).to(dev), tg)) - li)
                        feas = [k for k in range(len(WS)) if dLs[k] <= 0]
                        kstar = max(feas, key=lambda k: dPs[k]) if feas else 0
                        wm[by, bx] = WS[kstar]; rm[by, bx] = float(rng.choice(WS))
                        blocks.append(dict(set=tag, case=key, y=y, x=x, w_star=WS[kstar],
                                           dP_rd=round(dPs[-1], 3), dL_rd=round(dLs[-1], 4),
                                           dP_con=round(dPs[kstar], 3), dL_con=round(dLs[kstar], 4),
                                           n_feas=len(feas)))
                for name, grid, root in (('oraclew', wm, ow_root), ('randw', rm, rw_root)):
                    raw = np.kron(grid, np.ones((BS, BS), np.float64))
                    full = np.zeros((h, w), np.float64)      # 未覆盖的边缘余量保持 w=0（恒等）
                    full[:raw.shape[0], :raw.shape[1]] = raw
                    up = cv2.GaussianBlur(full, (0, 0), 8.0)
                    O = np.clip(I.astype(np.float64) + up[..., None] * (P.astype(np.float64) - I.astype(np.float64)), 0, 255).astype(np.uint8)
                    od = os.path.join(root, key)
                    os.makedirs(od, exist_ok=True)
                    cv2.imwrite(os.path.join(od, f'{name}.png'), O)
                print(f'{tag}/{key}: {nb_y*nb_x} 块，w*=0 占比 {float((wm==0).mean()):.2f}')

    with open('eval/eval_probe_oracle_blocks.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(blocks[0].keys()))
        wr.writeheader(); wr.writerows(blocks)
    print(f'\n块记录 {len(blocks)} 条 -> eval/eval_probe_oracle_blocks.csv')
    for tag in SETS:
        sub = [b for b in blocks if b['set'] == tag]
        if not sub:
            continue
        w0 = 100 * np.mean([b['w_star'] == 0 for b in sub])
        print(f"{tag}: 块 {len(sub)} | 纯RDNet ΔPSNR 均值 {np.mean([b['dP_rd'] for b in sub]):+.2f} "
              f"| oracle(约束) ΔPSNR 均值 {np.mean([b['dP_con'] for b in sub]):+.2f} "
              f"| 选 w=0 的块 {w0:.0f}% | 全档不可行(恒等)块 {100*np.mean([b['n_feas']==0 for b in sub]):.0f}%")


if __name__ == '__main__':
    main()
