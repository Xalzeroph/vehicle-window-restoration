#!/usr/bin/env python3
"""拍完即验：一键核验采集质量（2026-10-11）。

用法:
    python qa_capture.py <视频.mp4|帧目录> [--fps 5] [--out report.md]

自动完成：
    1) （视频）抽帧到 <输入>_frames/
    2) 亮度剖面分段：反射段 / 布段
    3) 门 1  段内漂移      —— AE/AWB 是否真的锁死（v2 增设门 1b：逐通道）
    4) 门 2  逐通道一致性  —— 是否需逐通道对齐
    5) 门 3  跨段偏移      —— **仅记录**（v2：反射强度是信号，不判决）
    6) 门 4  结构可分性    —— 有没有东西可学，还是只剩色调差
    7) 门 5  反射强度      —— 反射够不够强
    8) 门 6  几何稳定      —— EIS/抖动是否破坏配对
输出 Markdown 报告 + 终端摘要。

判据来源：docs/2026.10.10_实拍数据诊断与VLM探针_结论.md 与
          旧数据实测（apr |g−1| 0.1749 / 逐通道极差 0.308 / 跨段变异大）。
"""
import argparse, glob, os, subprocess, sys
import numpy as np
import cv2


# ---------------- 基础工具 ----------------
def imread_any(p, max_side=768):
    """读图并限制长边（控内存；所有度量都是尺度无关的）。"""
    im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    m = max(h, w)
    if m > max_side:
        r = max_side / m
        im = cv2.resize(im, (max(1, int(w * r)), max(1, int(h * r))),
                        interpolation=cv2.INTER_AREA)
    return im


def center_crop(im, frac=0.5):
    h, w = im.shape[:2]
    dh, dw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return im[dh:h - dh, dw:w - dw]


def robust_gain(I, T, keep_frac=0.7, iters=3):
    """g 使 I ≈ g·T（逐通道稳健 LS）。"""
    I = I.astype(np.float64); T = T.astype(np.float64)
    g = np.ones(3)
    for c in range(3):
        i = I[..., c].ravel(); t = T[..., c].ravel(); gc = 1.0
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


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else float('nan')


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64)


def sobel_mean(g):
    g32 = g.astype(np.float32)
    return float(np.sqrt(cv2.Sobel(g32, cv2.CV_32F, 1, 0) ** 2 +
                         cv2.Sobel(g32, cv2.CV_32F, 0, 1) ** 2).mean())


def local_std_median(g):
    g32 = g.astype(np.float32)
    v = cv2.blur(g32 * g32, (9, 9)) - cv2.blur(g32, (9, 9)) ** 2
    return float(np.median(np.sqrt(np.maximum(v, 0))))


def phase_shift(a, b):
    ga = gray(a).astype(np.float32); gb = gray(b).astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return dx, dy, resp


# ---------------- 抽帧 ----------------
def extract_frames(video, fps, outdir):
    os.makedirs(outdir, exist_ok=True)
    exe = None
    for cand in ('ffmpeg',):
        try:
            subprocess.run([cand, '-version'], capture_output=True, check=True)
            exe = cand; break
        except Exception:
            pass
    if exe is None:
        print(f'[错] 找不到 ffmpeg，请手动抽帧到目录后用 --frames 指定')
        sys.exit(2)
    cmd = [exe, '-hide_banner', '-loglevel', 'error', '-y', '-i', video,
           '-vf', f'fps={fps}', '-q:v', '2', os.path.join(outdir, 'f_%05d.jpg')]
    print(f'[抽帧] {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
    return outdir


# ---------------- 分段 ----------------
def segment(lums):
    base = float(np.percentile(lums, 75))
    dark = lums < base * 0.92
    n = len(lums); best = (0, 0, 0); cur = None
    for i, dk in enumerate(dark):
        if dk and cur is None:
            cur = i
        elif not dk and cur is not None:
            if i - cur > best[0]:
                best = (i - cur, cur, i)
            cur = None
    if cur is not None and n - cur > best[0]:
        best = (n - cur, cur, n)
    return base, best


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='视频文件 或 帧目录')
    ap.add_argument('--fps', type=float, default=5.0, help='抽帧帧率（仅视频输入时用）')
    ap.add_argument('--out', default=None, help='报告输出路径（默认 <src>_QA.md）')
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    if os.path.isdir(src):
        fdir = src
    else:
        fdir = extract_frames(src, a.fps, src + '_frames')

    # ============ 流式处理：不把全部帧留在内存 ============
    CROP_FRAC = 0.5
    fs = sorted(glob.glob(os.path.join(fdir, '*.jpg')) +
                glob.glob(os.path.join(fdir, '*.png')))
    if not fs:
        print(f'[错] {fdir} 下没有 jpg/png 帧')
        sys.exit(2)
    lums, crop0 = [], None
    for f in fs:
        im = imread_any(f)
        if im is None:
            continue
        if crop0 is None:
            crop0 = center_crop(im, CROP_FRAC)
        lums.append(float(gray(im).mean()))
    lums = np.array(lums)
    n = len(lums)
    if n < 20:
        print(f'[错] 有效帧太少（{n}）')
        sys.exit(2)
    print(f'[载入] {n} 帧  中央裁剪处理尺寸 {crop0.shape[1]}×{crop0.shape[0]}'
          f'（原图已限长边 768 后取中央 {int(CROP_FRAC*100)}%）')
    del crop0

    base, (L, s, e) = segment(lums)
    ok_seg = L >= max(10, n * 0.05)
    print(f'[分段] 反射段 0–{s-1}（{s} 帧）   布段 {s}–{e}（{L} 帧）')
    if not ok_seg:
        print(f'[警告] 布段太短或不存在（{L} 帧）——分段不可靠，门限仅供参考')

    refl = list(range(0, max(1, s - 3)))
    cloth = list(range(min(n - 1, s + 5), max(s + 6, e - 5))) if ok_seg else \
        list(range(max(0, n // 2), n))

    def load_crops(idx):
        """只载入指定帧的中央裁剪，用完即弃。"""
        out = []
        for i in idx:
            im = imread_any(fs[i])
            if im is not None:
                out.append(center_crop(im, CROP_FRAC))
        return out

    res = {}

    # ---- 门 1/2：段内漂移 + 逐通道一致性 ----
    def in_seg_drift(seg, max_pts=20):
        if len(seg) < 3:
            return float('nan'), float('nan')
        pick = seg[::max(1, len(seg) // max_pts)]
        cs = load_crops(pick)
        if len(cs) < 3:
            return float('nan'), float('nan')
        ref = cs[0]
        gs = np.array([robust_gain(x, ref) for x in cs[1:]])
        return float(np.median(np.abs(gs.mean(1) - 1))), float(np.median(gs.max(1) - gs.min(1)))

    d_refl, c_refl = in_seg_drift(refl)
    d_cloth, c_cloth = in_seg_drift(cloth)
    res['drift_refl'] = d_refl; res['drift_cloth'] = d_cloth
    res['chan_refl'] = c_refl; res['chan_cloth'] = c_cloth

    # v2 新增：段内**逐通道** |g−1| 最大值（抓单通道白平衡漂移）
    def in_seg_drift_ch(seg, max_pts=20):
        if len(seg) < 3:
            return float('nan')
        pick = seg[::max(1, len(seg) // max_pts)]
        cs = load_crops(pick)
        if len(cs) < 3:
            return float('nan')
        ref = cs[0]
        gs = np.array([robust_gain(x, ref) for x in cs[1:]])
        return float(np.median(np.abs(gs - 1).max(axis=1)))

    res['drift_refl_ch'] = in_seg_drift_ch(refl)
    res['drift_cloth_ch'] = in_seg_drift_ch(cloth)

    # ---- 门 3：跨段一致性 ----
    if refl and cloth:
        pick = refl[::max(1, len(refl) // 20)]
        crefs = load_crops([cloth[len(cloth) // 2]])
        m_ref = crefs[0]
        gs_l, robs = [], []
        for x in load_crops(pick):
            g = robust_gain(x, m_ref)
            gs_l.append(g)
            xf = x.astype(np.float64); yf = m_ref.astype(np.float64)
            gs_ = float((xf * yf).sum() / (yf * yf).sum())
            robs.append(float(np.abs(xf - gs_ * yf).mean() / yf.mean()))
        gs_c = np.array(gs_l); robs = np.array(robs)
        res['cross_g_mean'] = float(np.median(gs_c.mean(1)))
        res['cross_g_chan'] = gs_c.mean(0)
        res['cross_cv'] = float(robs.std() / max(np.median(robs), 1e-9))
        res['r_obs'] = float(np.median(robs))
        del crefs, m_ref
    else:
        res['cross_g_mean'] = res['cross_cv'] = res['r_obs'] = float('nan')
        res['cross_g_chan'] = np.array([float('nan')] * 3)

    # ---- 门 4：结构可分性 ----
    if refl and cloth:
        i_repr = refl[int(np.argmax(lums[refl]))]
        c_repr = cloth[int(np.argmin(lums[cloth]))]
        two = load_crops([i_repr, c_repr])
        if len(two) == 2:
            I, M = two
            gI, gM = gray(I), gray(M)
            res['corr_IM'] = corr(gI, gM)
            res['grad_ratio'] = sobel_mean(gI) / max(sobel_mean(gM), 1e-9)
            resid = np.zeros(gI.shape)
            per = []
            for ch in range(3):
                X = I[:, :, ch].astype(np.float64); Tc = M[:, :, ch].astype(np.float64)
                x = X.ravel(); t = Tc.ravel()
                mx, mt = x.mean(), t.mean()
                xc, tc = x - mx, t - mt
                den = float((xc * xc).sum())
                aa = float((xc * tc).sum() / den) if den > 1e-9 else 1.0
                bb = mt - aa * mx
                Af = np.clip(aa * X + bb, 0, 255)
                resid += (Af - Tc); per.append(float(np.abs(Af - Tc).mean()))
            resid /= 3.0
            res['resid_raw'] = float(np.abs(I.astype(np.float64) - M.astype(np.float64)).mean())
            res['resid_affine'] = float(np.mean(per))
            res['resid_struct'] = local_std_median(resid) / max(local_std_median(gM), 1e-9)
            res['i_repr'], res['c_repr'] = i_repr, c_repr
        del two
    if 'corr_IM' not in res:
        for k in ('corr_IM', 'grad_ratio', 'resid_raw', 'resid_affine', 'resid_struct'):
            res[k] = float('nan')

    # ---- 门 5：几何（相邻对，用相邻的两个索引） ----
    ds = []
    step = max(1, n // 30)
    for i in range(step, n - 1, step):
        pair = load_crops([i - step, i])
        if len(pair) == 2:
            dx, dy, _ = phase_shift(pair[0], pair[1])
            ds.append(float(np.hypot(dx, dy)))
    res['shift_med'] = float(np.median(ds)) if ds else float('nan')
    res['shift_max'] = float(np.max(ds)) if ds else float('nan')

    # ---- 判决 ----
    G = []
    def gate(name, val, lo=None, hi=None, ok=None, fmt='{:.4f}', note='',
             record_only=False):
        v = ok if ok is not None else ((lo is None or val >= lo) and (hi is None or val <= hi))
        mark = '📝' if record_only else ('✅' if v else '❌')
        G.append((name, fmt.format(val) if val == val else 'nan', mark, note))
        return v

    gate('门1 反射段内漂移 |g−1| 中位', res['drift_refl'], hi=0.02,
         note='AE/AWB 是否锁死（旧数据 0.1749）')
    # v2 新增：单通道偏色。三通道均值正常但某一通道偏，同样说明 AWB 没锁死。
    gate('门1b 反射段逐通道 |g−1|', res.get('drift_refl_ch', float('nan')), hi=0.05,
         note='v2 新增：抓单通道白平衡漂移（旧数据 0.1749）')
    gate('门1 布段内漂移 |g−1| 中位', res['drift_cloth'], hi=0.02,
         note='布段自身稳定性（旧数据布段抖 0.04–0.21）')
    gate('门2 反射段逐通道极差', res['chan_refl'], hi=0.03,
         note='需否逐通道对齐（旧数据 max 0.308）')
    # v2 改：跨段 g 现在是**反射强度信号**，不是采集缺陷，因此只记录不判决。
    # （旧数据里 g 量的是相机光度漂移，那时设上限才对。）
    gate('门3 跨段偏移 cross_g_mean（信号，仅记录）', res['cross_g_mean'],
         record_only=True, note='v2：反射强度是信号，不设上限；逐通道值见明细')
    gate('门4 corr(I, M)  <0.95', res['corr_IM'], hi=0.95,
         note='越低=结构差异越大=越有东西可学；>0.95 才是坏事（只剩色调差）')
    gate('门4 残差结构比', res['resid_struct'], lo=0.20,
         note='≥0.20=残差是信号（旧数据 1.61=混了光度干扰）')
    gate('门5 帧间位移中位 (px)', res['shift_med'], hi=1.0, fmt='{:.3f}',
         note='EIS/抖动是否破坏配对')
    gate('门5 反射强度 r_obs', res['r_obs'], lo=0.05,
         note='太弱学不到（旧数据 0.155，合成 0.155–0.634）')

    npass = sum(1 for g in G if g[2] == '✅')
    nrecord = sum(1 for g in G if g[2] == '📝')
    ntotal = len(G) - nrecord   # 分母只算真正判决的门

    # ---- 输出 ----
    lines = []
    lines.append(f'# 采集质量核验报告\n')
    lines.append(f'- 输入: `{a.src}`')
    lines.append(f'- 帧数: {n}')
    lines.append(f'- 分段: 反射段 0–{s-1}（{s} 帧） / 布段 {s}–{e}（{L} 帧）'
                 + ('' if ok_seg else '  ⚠️ 布段过短，分段不可靠'))
    if 'i_repr' in res:
        lines.append(f'- 代表帧: I={res["i_repr"]}  M={res["c_repr"]}')
    lines.append('')
    lines.append(f'## 判决: {npass}/{ntotal} 通过'
                 + (f'（另有 {nrecord} 项仅记录）' if nrecord else '') + '\n')
    lines.append('| 检查 | 实测 | 结果 | 说明 |')
    lines.append('|---|---|---|---|')
    for nm, val, mk, note in G:
        lines.append(f'| {nm} | {val} | {mk} | {note} |')
    lines.append('')
    lines.append('## 数值明细\n')
    lines.append('> v2 口径：`drift_*` = 采集纪律（越小越好，有门限）；'
                 '`cross_g_*` / `r_obs` = **反射信号**（越大越有东西可学，不设上限）。\n')
    lines.append('| 指标 | 值 |')
    lines.append('|---|---|')
    for k in ('drift_refl', 'drift_refl_ch', 'drift_cloth', 'drift_cloth_ch',
              'chan_refl', 'chan_cloth',
              'cross_g_mean', 'cross_g_chan', 'cross_cv', 'r_obs',
              'corr_IM', 'grad_ratio', 'resid_raw', 'resid_affine', 'resid_struct',
              'shift_med', 'shift_max'):
        v = res.get(k, float('nan'))
        s_ = np.array2string(np.atleast_1d(v), precision=4) if isinstance(v, np.ndarray) \
            else (f'{v:.4f}' if v == v else 'nan')
        lines.append(f'| {k} | {s_} |')
    lines.append('')
    lines.append('## 判据出处\n')
    lines.append('- 旧数据实测（对照基线）: apr `|g−1|` 0.1749 / 逐通道极差 max 0.308 / '
                 '8-11 场景 AE 逐帧追赶')
    lines.append('- 详见 `docs/2026.10.10_实拍数据诊断与VLM探针_结论.md` 与 '
                 '`docs/采集方法决策卡_MR相减vs黑布_2026.10.10.md`')
    rep = a.out or (os.path.splitext(src)[0] + '_QA.md')
    with open(rep, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    print()
    print('=' * 64)
    print(f'  判决: {npass}/{ntotal} 通过'
          + (f'（另有 {nrecord} 项仅记录）' if nrecord else ''))
    print('=' * 64)
    for nm, val, mk, note in G:
        print(f'  {mk} {nm:<28} {val:>10}   {note}')
    print()
    print(f'[报告] {rep}')


if __name__ == '__main__':
    main()
