#!/usr/bin/env python3
"""OpenRR-5k 可移植性验证（2026-09-19 轮 §4a）。

流程：按 |R| 分层抽 200 对（seed=0919）→ RDNet zero-shot 推理 → 块级(64²)四签名判决：
  ① 低反射分位区（r_obs<0.05 块）ΔPSNR(rdnet−identity_gm) 中位 <0 且 CI 上界 ≤0
  ② 干净区梯度能量损失 >20%（identity_gm → rdnet）
  ③ 强反射区（各对 r_obs 上四分位块）rdnet 残留 ≤ identity_gm 残留
  ④ identity_gm ≥ identity（PSNR 中位）
PASS=四向全对；幅度只记区间（OpenRR |R| 仅为目标域 1/3~1/4）。

口径与 e3_robs_binning.py 一致：g=<I,T>/<I,I>，D=max_c|g·I−T|，
r_obs(块)=mean(D_块)/mean(T_块)，T 块均值<25 跳过。

用法：
  python scripts/eval_openrr_port.py --stage sample
  python scripts/eval_openrr_port.py --stage infer      # 需 CUDA_VISIBLE_DEVICES 已设
  python scripts/eval_openrr_port.py --stage metrics
"""
import argparse, csv, json, os, subprocess, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import psnr, grad_energy, scene_bootstrap  # noqa: E402

SRC = 'data/external/OpenRR-5k/unpacked/trian_5k'
STATS = 'data/external/OpenRR-5k/analysis/r_stats.csv'
SAMPLE = 'data/external/OpenRR-5k/analysis/port200_sample.csv'
IN_DIR = 'data/external/OpenRR-5k/analysis/port200_input'
PRED = 'runs/pred/openrr200_rdnet'
OUT_CSV = 'eval/eval_openrr_port.csv'
BS = 64


def stage_sample(args):
    rows = list(csv.DictReader(open(STATS)))
    vals = np.array([float(r['abs_mean']) for r in rows])
    qs = np.quantile(vals, [0.25, 0.5, 0.75])
    rng = np.random.default_rng(args.seed)
    strata = [np.where(vals <= qs[0])[0], np.where((vals > qs[0]) & (vals <= qs[1]))[0],
              np.where((vals > qs[1]) & (vals <= qs[2]))[0], np.where(vals > qs[2])[0]]
    picked = []
    per = args.n // 4
    for si, idx in enumerate(strata):
        sel = rng.choice(idx, size=min(per, len(idx)), replace=False)
        for i in sel:
            picked.append(dict(name=rows[i]['name'], abs_mean=rows[i]['abs_mean'], stratum=si))
    with open(SAMPLE, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['name', 'abs_mean', 'stratum'])
        w.writeheader(); w.writerows(picked)
    amb = np.array([float(p['abs_mean']) for p in picked])
    print(f'抽样 {len(picked)} 对（每层 {per}）| |R| 范围 {amb.min():.2f}~{amb.max():.2f} 中位 {np.median(amb):.2f}')
    print(f'写出 {SAMPLE}')


def stage_infer(args):
    rows = list(csv.DictReader(open(SAMPLE)))
    os.makedirs(IN_DIR, exist_ok=True)
    for r in rows:
        src = os.path.join(SRC, 'blended', r['name'])
        dst = os.path.join(IN_DIR, r['name'])
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    os.makedirs(PRED, exist_ok=True)
    cmd = [sys.executable, 'inference_simple.py', os.path.abspath(IN_DIR), os.path.abspath(PRED),
           'pretrained/rdnet.pth', '0']
    print('运行:', ' '.join(cmd))
    subprocess.run(cmd, cwd='RDNet', check=True)


def blocks_of(I, T, R):
    """返回块级记录列表。"""
    i_, t_ = I.astype(np.float64), T.astype(np.float64)
    g = float((i_ * t_).sum() / ((i_ * i_).sum() + 1e-9))
    D = np.abs(g * i_ - t_).max(axis=2)
    IG = np.clip(g * i_, 0, 255).astype(np.uint8)
    r_ = R.astype(np.float64)
    go = float((r_ * t_).sum() / ((r_ * r_).sum() + 1e-9))
    RM = np.clip(go * r_, 0, 255).astype(np.uint8)          # gm 口径臂输出
    Do = np.abs(go * r_ - t_).max(axis=2)
    h, w = T.shape[:2]
    recs = []
    for y in range(0, h - BS + 1, BS):
        for x in range(0, w - BS + 1, BS):
            tb = t_[y:y + BS, x:x + BS]
            if tb.mean() < 25:
                continue
            rb = D[y:y + BS, x:x + BS].mean() / (tb.mean() + 1e-9)
            rob = Do[y:y + BS, x:x + BS].mean() / (tb.mean() + 1e-9)
            ge_in = grad_energy(IG[y:y + BS, x:x + BS])
            ge_rd = grad_energy(RM[y:y + BS, x:x + BS])
            recs.append(dict(r_obs=float(rb), r_obs_out=float(rob),
                             psnr_in=psnr(I[y:y + BS, x:x + BS], T[y:y + BS, x:x + BS]),
                             psnr_igm=psnr(IG[y:y + BS, x:x + BS], T[y:y + BS, x:x + BS]),
                             psnr_rd=psnr(RM[y:y + BS, x:x + BS], T[y:y + BS, x:x + BS]),
                             ge_in=ge_in, ge_rd=ge_rd))
    return recs


def stage_metrics(args):
    rows = list(csv.DictReader(open(SAMPLE)))
    out_rows = []
    for r in rows:
        name = r['name']
        I = cv2.imread(os.path.join(SRC, 'blended', name))
        T = cv2.imread(os.path.join(SRC, 'transmission_layer', name))
        pp = os.path.join(PRED, os.path.splitext(name)[0], 'ytmt_ucs_sirs_l.png')
        if I is None or T is None or not os.path.exists(pp):
            print(f'{name}: 缺文件 -> 跳过'); continue
        R = cv2.imread(pp)
        if R.shape != T.shape:
            R = cv2.resize(R, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
        if I.shape != T.shape:
            I = cv2.resize(I, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
        recs = blocks_of(I, T, R)
        for b in recs:
            b.update(pair=name, abs_mean=float(r['abs_mean']), stratum=int(r['stratum']))
        out_rows += recs
        print(f'{name}: {len(recs)} blocks')
    if not out_rows:
        print('无块，退出'); sys.exit(1)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # ---- 四签名 ----
    low = [b for b in out_rows if b['r_obs'] < 0.05]
    by_pair_low = {}
    for b in low:
        by_pair_low.setdefault(b['pair'], []).append(b['psnr_rd'] - b['psnr_igm'])
    s1 = scene_bootstrap(by_pair_low) if by_pair_low else dict(median=None, ci_lo=None, ci_hi=None)
    sig1 = (s1['median'] is not None) and s1['median'] < 0 and s1['ci_hi'] <= 0

    loss_by_pair = {}
    for b in low:
        if b['ge_in'] > 1e-6:
            loss_by_pair.setdefault(b['pair'], []).append(1.0 - b['ge_rd'] / b['ge_in'])
    losses = np.array([np.median(v) for v in loss_by_pair.values()]) if loss_by_pair else np.array([])
    loss_med = float(np.median(losses)) if losses.size else None
    sig2 = (loss_med is not None) and loss_med > 0.20

    strong_by_pair = {}
    for p in {b['pair'] for b in out_rows}:
        pb = [b for b in out_rows if b['pair'] == p]
        thr = np.percentile([b['r_obs'] for b in pb], 75)
        sel = [b for b in pb if b['r_obs'] >= thr]
        strong_by_pair[p] = (float(np.median([b['r_obs'] for b in sel])),
                             float(np.median([b['r_obs_out'] for b in sel])))
    m_in = float(np.median([v[0] for v in strong_by_pair.values()]))
    m_out = float(np.median([v[1] for v in strong_by_pair.values()]))
    sig3 = m_out <= m_in

    psi = float(np.median([b['psnr_igm'] for b in out_rows]))
    psi_raw = float(np.median([b['psnr_in'] for b in out_rows]))
    sig4 = psi >= psi_raw

    verdict = dict(s1_lowref_psnr=s1, s1_pass=bool(sig1),
                   s2_clean_ge_loss=loss_med, s2_pass=bool(sig2),
                   s3_strong_resid=dict(identity_gm=m_in, rdnet=m_out), s3_pass=bool(sig3),
                   s4_gm_vs_raw=dict(identity_gm=psi, identity=psi_raw), s4_pass=bool(sig4),
                   n_blocks=len(out_rows), n_pairs=len(strong_by_pair),
                   n_low_blocks=len(low))
    verdict['PASS_ALL'] = all(verdict[k] for k in ['s1_pass', 's2_pass', 's3_pass', 's4_pass'])
    json.dump(verdict, open(os.path.splitext(OUT_CSV)[0] + '_verdict.json', 'w'), indent=1, ensure_ascii=False)

    print('\n=== OpenRR-5k 四签名 ===')
    print(f"① 低反射区 ΔPSNR(rdnet−igm) 中位 {s1['median']} [{s1['ci_lo']},{s1['ci_hi']}]  "
          f"{'PASS' if sig1 else 'FAIL'}")
    print(f"② 干净区梯度损失 中位 {loss_med}  {'PASS' if sig2 else 'FAIL'}")
    print(f"③ 强反射区残留 identity_gm {m_in:.4f} -> rdnet {m_out:.4f}  {'PASS' if sig3 else 'FAIL'}")
    print(f"④ identity_gm {psi:.2f} vs identity {psi_raw:.2f}  {'PASS' if sig4 else 'FAIL'}")
    print(f"\n★ 总判决: {'PASS（同签名复现）' if verdict['PASS_ALL'] else 'FAIL'}")
    print(f'写出 {OUT_CSV} 与 verdict.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['sample', 'infer', 'metrics'], required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()
    {'sample': stage_sample, 'infer': stage_infer, 'metrics': stage_metrics}[args.stage](args)


if __name__ == '__main__':
    main()
