#!/usr/bin/env python3
"""最小充分评测协议（2026-09-19 重定位轮，docs/2026.9.19_评测模式重定位_方案.md §3）。

口径（实现要点，供 P4 复核）：
- mask：D = max_c|I − g·T|（g = robust LS 增益 T→I），thr = max(k·σ̂_D, 8)，
  σ̂_D = 1.4826·MAD(D | D ≤ p70(D))（低分位子集 MAD，q=0.70 经三套实拍标定）；
  mask 只用 I/T，禁含被评输出。
- NGR：E = G₂*|T−Ŷ|；N = sqrt(block16(G₄*(T−G₄T)²))（σ4 低通 + 16² 块级），下限 n0；
  逐通道比值后跨通道均值；域 = M（反射区）。
- NGR_bg：同式，域 = M̄ ∩ {G₈*luma(T) ≥ 20}（非反射且足够亮）。
- 实拍主口径 gm：各臂输出先做 LS 增益对齐 T（k=<Ŷ,T>/<Ŷ,Ŷ>），d_* 均对 identity_gm 计。
- Exist（可选报告）：G = G₄*(I−g·T)，E = G₄*(Ŷ−T)，ρ = Σ_M EG/√(ΣE²ΣG²)，A = √(Σ_M E²/Σ_M G²)，
  逐通道后平均；增益口径待复核。
- 裁剪 LPIPS：M 的 8-连通域，面积 ≥2% 且 bbox 短边 ≥64px；bbox 外扩 16px；
  若有效组件 <1 个或最大 bbox 面积 >50% 全图 → 降级整图并标 degraded。
- 统计：场景聚类 bootstrap（B=1e4）+ 场景级符号检验；不劣约束
  Δper 中位 ≤0 且 CI 上界 ≤ thr（LPIPS 0.005 / NGR 0.02，可调，待复核）。

用法：
  python scripts/eval_protocol.py --set data/test_sets/real_pair_0917 --layout jun \
      --arm rdnet=runs/pred/real_pair_0917_rdnet --out eval/eval_proto_jun.csv
  python scripts/eval_protocol.py --selftest
"""
import argparse, csv, json, os, sys
import cv2
import numpy as np

N0 = 3.0
LPIPS_THR = 0.005
NGR_THR = 0.02


def luma(x):
    return cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def robust_gain(I, T, keep_frac=0.7, iters=3):
    """g 使 I ≈ g·T（逐通道稳健 LS，与 build_real_pair_testset.py 同口径）。"""
    I = I.astype(np.float64); T = T.astype(np.float64)
    g = np.ones(3)
    for c in range(3):
        i = I[..., c].ravel(); t = T[..., c].ravel()
        gc = 1.0
        for _ in range(iters):
            den = float((t * t).sum())
            gc = float((i * t).sum() / den) if den > 1e-9 else 1.0
            res = np.abs(i - gc * t)
            keep = res <= np.percentile(res, keep_frac * 100)
            i, t = i[keep], t[keep]
            if len(i) < 100:
                break
        g[c] = gc
    return g


def ls_scale(y, ref):
    """k 使 y·k ≈ ref（LS）。"""
    y = y.astype(np.float64).ravel(); r = ref.astype(np.float64).ravel()
    den = float((y * y).sum())
    return float((y * r).sum() / den) if den > 1e-9 else 1.0


def apply_gm(y, k):
    return np.clip(y.astype(np.float64) * k, 0, 255)


def compute_mask(I, T, k_thr=3.0, thr_min=8.0, sig_q=0.70):
    """σ̂_D = 1.4826·MAD(D | D ≤ p_{sig_q}(D))。
    q=0.70 经三套实拍标定（2026-09-19）：σ̂ 中位 jun 7.5 / apr 9.3 / may 6.7，
    thr=3σ̂ → 22 / 28 / 20，落在构建口径（max(3·floor,8)=11~92）同区间。"""
    g = robust_gain(I, T)
    D = np.abs(I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3)).max(axis=2)
    cut = float(np.percentile(D, sig_q * 100))
    low = D[D <= cut]
    sig = float(1.4826 * np.median(np.abs(low - np.median(low)))) if low.size else 0.0
    thr = max(k_thr * sig, thr_min)
    M = D > thr
    return M, dict(g=[float(x) for x in g], sigma_d=sig, thr=float(thr), frac_M=float(M.mean())), D


def _low(x, s):
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)


def ngr_ratio_map(T, Y, n0=N0):
    """返回 (逐像素×通道比值图, 钳死掩膜)。T/Y 为 float 0..255 的 HxWx3。"""
    E = _low(np.abs(T - Y), 2.0)
    Hp = T - _low(T, 4.0)
    N = np.sqrt(np.maximum(cv2.blur(_low(Hp * Hp, 4.0), (16, 16)), 0))
    ratio = E / np.maximum(N, n0)
    return ratio, (N < n0)


def domain_mean(ratio_map, domain, clamp_map=None):
    """逐通道比值 → 跨通道均值 → 域内均值；域空返回 None。"""
    if domain.sum() < 1:
        return None, None
    ch_mean = ratio_map.mean(axis=2)
    val = float(ch_mean[domain].mean())
    clamp = float(clamp_map[domain].any(axis=1).mean()) if clamp_map is not None else None
    return val, clamp


def bg_domain(T, M):
    Tg = luma(np.clip(T, 0, 255))
    bright = _low(Tg, 8.0) >= 20
    return (~M) & bright


def exist_scores(I, T, Y, g, M):
    """Exist ρ/A（逐通道后平均）；域空返回 (None, None)。"""
    if M.sum() < 1:
        return None, None
    rhos, As = [], []
    for c in range(3):
        Gc = _low(I[..., c] - g[c] * T[..., c], 4.0)
        Ec = _low(Y[..., c] - T[..., c], 4.0)
        gm_, em = Gc[M], Ec[M]
        den = float(np.sqrt((em ** 2).sum() * (gm_ ** 2).sum()))
        rhos.append(float((em * gm_).sum() / den) if den > 1e-12 else None)
        denA = float((gm_ ** 2).sum())
        As.append(float(np.sqrt((em ** 2).sum() / denA)) if denA > 1e-12 else None)
    r = [x for x in rhos if x is not None]; a = [x for x in As if x is not None]
    return (sum(r) / len(r) if r else None), (sum(a) / len(a) if a else None)


def grad_energy(x):
    g = luma(np.clip(x, 0, 255)).astype(np.float32)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0); sy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(cv2.magnitude(sx, sy).mean())


def matched_blur(img, target_ge):
    img = np.clip(img, 0, 255).astype(np.uint8)
    lo, hi = 0.0, 8.0
    for _ in range(16):
        mid = (lo + hi) / 2
        if grad_energy(cv2.GaussianBlur(img, (0, 0), mid)) > target_ge:
            lo = mid
        else:
            hi = mid
    return cv2.GaussianBlur(img, (0, 0), (lo + hi) / 2)


def ssim_map_mean(T, Y, domain):
    from skimage.metrics import structural_similarity as ssim_fn
    if domain.sum() < 1:
        return None
    t8 = np.clip(T, 0, 255).astype(np.uint8); y8 = np.clip(Y, 0, 255).astype(np.uint8)
    _, S = ssim_fn(t8, y8, channel_axis=2, data_range=255, full=True)
    return float(S[domain].mean())


def crop_components(M, min_area_frac=0.02, min_side=64, degrade_bbox_frac=0.5):
    """返回 (bbox 列表, 是否降级, 原因)。"""
    H, W = M.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area_frac * H * W and min(w, h) >= min_side:
            comps.append((int(x), int(y), int(w), int(h)))
    if not comps:
        return [], True, 'no_valid_comp'
    if max(w * h for _, _, w, h in comps) > degrade_bbox_frac * H * W:
        return comps, True, 'bbox_too_large'
    return comps, False, ''


def to_tensor(img_bgr):
    import torch
    rgb = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def lpips_pair(lp, T, Y):
    import torch
    with torch.no_grad():
        return float(lp(to_tensor(T).to(lp.device), to_tensor(Y).to(lp.device)))


def crop_lpips(lp, T, Y, M, pad=16):
    """组件不满足门时按预登记降级为整图 LPIPS（并标注 degraded/reason），不剔除该案。"""
    H, W = M.shape
    comps, degraded, reason = crop_components(M)
    if degraded:
        return lpips_pair(lp, T, Y), len(comps), True, reason
    vals = []
    for x, y, w, h in comps:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        vals.append(lpips_pair(lp, T[y0:y1, x0:x1], Y[y0:y1, x0:x1]))
    return (float(np.median(vals)) if vals else None), len(comps), False, ''


def scene_bootstrap(deltas_by_scene, B=10000, seed=20260919):
    """deltas_by_scene: {scene: [d1, d2, ...]}；场景聚类重采样 → 中位差 CI。"""
    rng = np.random.default_rng(seed)
    scenes = list(deltas_by_scene)
    if not scenes:
        return dict(median=None, ci_lo=None, ci_hi=None, n_scenes=0)
    pooled = np.array([d for sc in scenes for d in deltas_by_scene[sc]], dtype=np.float64)
    med = float(np.median(pooled))
    boots = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(scenes), size=len(scenes), replace=True)
        pool = np.concatenate([deltas_by_scene[scenes[i]] for i in pick])
        boots[b] = np.median(pool)
    return dict(median=med, ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), n_scenes=len(scenes))


def sign_test(deltas_by_scene):
    from scipy.stats import binomtest
    meds = np.array([np.median(v) for v in deltas_by_scene.values()])
    n = len(meds); k = int((meds > 0).sum())
    if n == 0:
        return dict(k=0, n=0, p=1.0)
    p = float(binomtest(k, n, 0.5, alternative='two-sided').pvalue)
    return dict(k=k, n=n, p=p)


def eval_case(case, arms_raw, lp, k_thr, thr_min, n0):
    T = cv2.imread(case['tp'])
    I = cv2.imread(case['ip'])
    if T is None or I is None:
        print(f"{case['case']}: 缺 I/T -> 跳过"); return None
    T = T.astype(np.float64); I = I.astype(np.float64)
    I8 = np.clip(I, 0, 255).astype(np.uint8)

    M, minfo, _ = compute_mask(I, T, k_thr, thr_min)
    rec = dict(scene=case['scene'], case=case['case'],
               sigma_d=round(minfo['sigma_d'], 3), thr=round(minfo['thr'], 2),
               frac_M=round(minfo['frac_M'], 4),
               g_i=[round(x, 3) for x in minfo['g']])
    for kk in (2.0, 4.0):
        Mk, _, _ = compute_mask(I, T, kk, thr_min)
        rec[f'frac_M_k{int(kk)}'] = round(float(Mk.mean()), 4)

    # 基线臂（与 eval_real_pair.py 同规则；主口径 gm → identity 即 identity_gm）
    gi = ls_scale(I, T)
    base = {'identity_gm': apply_gm(I, gi),
            'const': np.full_like(I, float(int(round(I.mean()))))}
    ref_img = None
    for name, (root, pf) in arms_raw.items():
        p = os.path.join(root, case['key'], pf)
        if os.path.exists(p):
            R = cv2.imread(p)
            if R is not None:
                if R.shape != T.shape:
                    R = cv2.resize(R, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
                base[name] = R.astype(np.float64)
                if ref_img is None:
                    ref_img = R
        else:
            print(f"{case['case']}: 臂 {name} 缺预测 -> 跳过该臂")
    if ref_img is None:
        ref_img = I8
    ge_target = grad_energy(I) * (grad_energy(ref_img) / max(grad_energy(I), 1e-6))
    base['blur_identity'] = matched_blur(I8, ge_target).astype(np.float64)

    domain_bg = bg_domain(T, M)
    Y_dom = M
    rec['frac_bg_domain'] = round(float(domain_bg.mean()), 4)

    per_arm = {}
    for name, Y in base.items():
        k = ls_scale(Y, T)
        Yg = apply_gm(Y, k)
        ratio_m, clamp_m = ngr_ratio_map(T, Yg, n0)
        ngr_m, clampf_m = domain_mean(ratio_m, Y_dom, clamp_m)
        ngr_bg, clampf_bg = domain_mean(ratio_m, domain_bg, clamp_m)
        ssim_m = ssim_map_mean(T, Yg, Y_dom)
        rho, A = exist_scores(I, T, Yg, minfo['g'], Y_dom)
        vals = dict(k=round(k, 4), psnr=psnr(Yg, T), ngr_m=ngr_m, ngr_bg=ngr_bg,
                    clamp_m=clampf_m, clamp_bg=clampf_bg, ssim_m=ssim_m,
                    exist_rho=rho, exist_A=A)
        if lp is not None:
            vals['lpips'] = lpips_pair(lp, T, Yg)
            cl, ncomp, deg, reason = crop_lpips(lp, T, Yg, M)
            vals['lpips_crop'] = cl
            vals['ncomp'] = ncomp
            vals['degraded'] = int(deg)
            vals['deg_reason'] = reason
        per_arm[name] = vals

    ref = per_arm['identity_gm']
    for name, vals in per_arm.items():
        for key_in, key_out in [('ngr_m', 'd_struct'), ('ngr_bg', 'd_bg'), ('lpips_crop', 'd_perc')]:
            a, b = vals.get(key_in), ref.get(key_in)
            rec[f'{name}__{key_out}'] = (None if a is None or b is None else round(a - b, 6))
        for kk, vv in vals.items():
            if isinstance(vv, float):
                vals[kk] = round(vv, 6)
            rec[f'{name}__{kk}'] = vv
    rec['_per_arm'] = per_arm
    return rec


def stats_report(rows, lpips_thr=LPIPS_THR, ngr_thr=NGR_THR):
    out = {}
    model_arms = sorted({k.split('__')[0] for r in rows for k in r
                         if k.endswith('__d_struct') and not k.startswith(('identity', 'const', 'blur'))})
    for arm in model_arms:
        entry = {}
        for dkey, thr in [('d_perc', lpips_thr), ('d_bg', ngr_thr)]:
            by_scene = {}
            for r in rows:
                v = r.get(f'{arm}__{dkey}')
                if v is not None:
                    by_scene.setdefault(r['scene'], []).append(v)
            bs = scene_bootstrap(by_scene)
            bs['noninferior'] = bool(bs['median'] is not None and bs['median'] <= 0 and bs['ci_hi'] <= thr)
            entry[dkey] = bs
        by_scene = {}
        for r in rows:
            v = r.get(f'{arm}__d_struct')
            if v is not None:
                by_scene.setdefault(r['scene'], []).append(v)
        entry['d_struct'] = scene_bootstrap(by_scene)
        entry['d_struct']['sign_test'] = sign_test(by_scene)
        entry['pass_noninferior'] = bool(entry['d_perc']['noninferior'] and entry['d_bg']['noninferior'])
        out[arm] = entry
    return out


# ---------------- 自检（§4d 6 项解析用例） ----------------

def _synth(h=256, w=256, seed=0):
    """带纹理的合成图（含细尺度正弦 → 高频分量可支撑 N 分母）。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = (90 + 20 * np.sin(2 * np.pi * xx / 37) * np.cos(2 * np.pi * yy / 43)
            + 12 * np.sin(2 * np.pi * (xx / 7 + yy / 9)) + 8 * np.sin(2 * np.pi * xx / 5))
    tex = np.stack([base, base * 0.9 + 10, base * 0.8 + 20], axis=2)
    tex += rng.normal(0, 3, (h, w, 3))
    return np.clip(tex, 0, 255)


def selftest():
    ok = True
    T = _synth(seed=1)
    # 1. 全掩膜分支：M 覆盖全图 → NGR_bg 定义域空 → 返回 None（不崩）
    I = np.clip(T + 60, 0, 255)
    Mall = np.ones(T.shape[:2], bool)
    dom1 = bg_domain(T, Mall)
    rm1, cm1 = ngr_ratio_map(T, I)
    ngr_bg1, _ = domain_mean(rm1, dom1, cm1)
    c1 = (dom1.sum() == 0) and (ngr_bg1 is None)
    print(f"[1] 全掩膜下 NGR_bg 域空->None {c1}  {'PASS' if c1 else 'FAIL'}")
    ok &= c1

    # 2. 无反射图：thr 落到 8 下限，mask ≈ 空
    I2 = np.clip(T + np.random.default_rng(2).normal(0, 0.5, T.shape), 0, 255)
    M2, info2, _ = compute_mask(I2, T)
    c2 = abs(info2['thr'] - 8.0) < 1e-9 and info2['frac_M'] < 0.01
    print(f"[2] 无反射 thr={info2['thr']:.2f} (==8) frac_M={info2['frac_M']:.4f} (<0.01)  {'PASS' if c2 else 'FAIL'}")
    ok &= c2

    # 3. 常数增益：Ŷ=1.5·I 经 gm 对齐后 == identity_gm → ΔNGR≈0
    I3 = np.clip(T * 1.2 + 25, 0, 255)
    M3, info3, _ = compute_mask(I3, T)
    dom3 = bg_domain(T, M3)
    r_ref, _ = ngr_ratio_map(T, apply_gm(I3, ls_scale(I3, T)))
    r_gain, _ = ngr_ratio_map(T, apply_gm(I3 * 1.5, ls_scale(I3 * 1.5, T)))
    a, _ = domain_mean(r_ref, dom3); b, _ = domain_mean(r_gain, dom3)
    c3 = (a is not None) and (b is not None) and abs(a - b) < 0.01
    print(f"[3] 常数增益 ΔNGR_bg={abs(a - b):.5f} (<0.01)  {'PASS' if c3 else 'FAIL'}")
    ok &= c3

    # 4. JPEG 重压：钳死比例可报且 ≤30%
    enc = cv2.imencode('.jpg', np.clip(I3, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    Yj = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float64)
    rim, cl = ngr_ratio_map(T, Yj)
    _, clf = domain_mean(rim, dom3, cl)
    c4 = (clf is not None) and clf <= 0.30
    print(f"[4] JPEG 钳死比例 {None if clf is None else round(clf, 4)} (≤0.30)  {'PASS' if c4 else 'FAIL'}")
    ok &= c4

    # 5. 对角宽带 30% 面积（bbox 撑满全图）→ 裁剪 LPIPS 降级触发
    H5, W5 = T.shape[:2]
    M5 = np.zeros((H5, W5), bool)
    band = int(0.30 * W5)
    for i in range(H5):
        x0 = (i * W5 // H5)
        M5[i, x0:min(W5, x0 + band)] = True
        M5[i, :max(0, x0 + band - W5)] = True
    comps, deg, reason = crop_components(M5)
    c5 = deg and reason == 'bbox_too_large'
    print(f"[5] 斜带 30% 面积 -> degraded={deg} ({reason})  {'PASS' if c5 else 'FAIL'}")
    ok &= c5

    # 6. 单色块：σ̂_D 稳定（=0），thr=8，无 NaN
    Tc = np.full((64, 64, 3), 100.0)
    _, info6, _ = compute_mask(Tc, Tc)
    c6 = info6['sigma_d'] == 0.0 and info6['thr'] == 8.0
    print(f"[6] 单色块 σ̂_D={info6['sigma_d']} thr={info6['thr']}  {'PASS' if c6 else 'FAIL'}")
    ok &= c6

    print('\n=== 自检 ' + ('全部通过' if ok else '存在失败') + ' ===')
    return 0 if ok else 1


def discover(case_set, layout):
    cases = []
    for sc in sorted(d for d in os.listdir(case_set) if d.startswith('VID_')):
        sd = os.path.join(case_set, sc)
        tp = os.path.join(sd, 'T.png' if layout == 'jun' else 'GT.png')
        if layout == 'jun':
            cases.append(dict(scene=sc, case=sc, key=sc, ip=os.path.join(sd, 'I.png'), tp=tp))
        else:
            for fn in sorted(f for f in os.listdir(sd) if f.startswith('I_')):
                cases.append(dict(scene=sc, case=f'{sc}__{fn[:-4]}', key=f'{sc}__{fn[:-4]}',
                                  ip=os.path.join(sd, fn), tp=tp))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='data/test_sets/real_pair_0917')
    ap.add_argument('--layout', choices=['jun', 'multi'], default='jun')
    ap.add_argument('--arm', action='append', default=[], help='NAME=PRED_ROOT[|pred_file]，可重复')
    ap.add_argument('--pred-file', default='ytmt_ucs_sirs_l.png')
    ap.add_argument('--out', default='eval/eval_proto.csv')
    ap.add_argument('--thr-k', type=float, default=3.0)
    ap.add_argument('--thr-min', type=float, default=8.0)
    ap.add_argument('--n0', type=float, default=N0)
    ap.add_argument('--no-lpips', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    arms = {}
    for a in args.arm:
        name, val = a.split('=', 1)
        root, _, pf = val.partition('|')
        arms[name] = (root, pf or args.pred_file)

    lp = None
    if not args.no_lpips:
        import torch, pyiqa
        dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        lp = pyiqa.create_metric('lpips', device=dev)
        print(f'LPIPS device: {dev}')

    cases = discover(args.set, args.layout)
    if args.limit:
        cases = cases[:args.limit]
    rows = []
    for c in cases:
        r = eval_case(c, arms, lp, args.thr_k, args.thr_min, args.n0)
        if r is not None:
            rows.append(r)
        print(f"{c['case']} done")
    if not rows:
        print('无有效用例，退出'); sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    flat = [{k: v for k, v in r.items() if k != '_per_arm'} for r in rows]
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)

    stats = stats_report(rows)
    sj = os.path.splitext(args.out)[0] + '_stats.json'
    json.dump(dict(stats=stats, n_cases=len(rows), args=vars(args)), open(sj, 'w'), indent=1, ensure_ascii=False)

    print(f'\n{"臂":<14}{"Δper 中位 [CI]":>34}{"Δbg 中位 [CI]":>32}{"Δstruct 中位":>14}  不劣')
    for arm, e in stats.items():
        dp, db, ds = e['d_perc'], e['d_bg'], e['d_struct']
        f = lambda x: 'NA' if x['median'] is None else f"{x['median']:+.4f}[{x['ci_lo']:+.4f},{x['ci_hi']:+.4f}]"
        print(f"{arm:<14}{f(dp):>34}{f(db):>32}"
              f"{('NA' if ds['median'] is None else format(ds['median'], '+.4f')):>14}  "
              f"{'PASS' if e['pass_noninferior'] else 'FAIL'}")
    print(f'\n写出 {args.out} 与 {sj}')


if __name__ == '__main__':
    main()
