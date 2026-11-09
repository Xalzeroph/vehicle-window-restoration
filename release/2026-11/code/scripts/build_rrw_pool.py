#!/usr/bin/env python3
"""构建 RRW 真实反射池（基准 v2）：R = 逐帧 (frame − GT) 的时域中值。

背景：E1 的 R 池（build_real_r_pool.py，clip(B−T,0) 标准集残差）强度≈RRW 中位，
缺"深色剪影"结构，2024 方法即可去除 → 基准区分度不足。本脚本从 RRW 提取真实
反射层，并落实 2026-09-12 确认的三条 v2 修正：
  ① 深色剪影：RRW 含人影/暗物体反射（有效帧时域中值保留静态结构，压制 JPEG/对齐噪声）；
  ② 强度上调：native 过弱(R̄<4，夜暗退化)→剔除；其余按
       target_i = clip(base·c_i·ρ_i, lo, hi)，c=(native_i/中位)^0.5∈[0.45,1.8]，ρ~logU[1,2.6]
     与"强度×结构"综合分位正相关（中位≈1.6），scale 上限 8×。
     中位场景落在 base=0.06·255≈15（旧池水平），强场景最高 ~4-5 倍（对齐并略超实拍最强
     4.26 的 R̂/T̂=0.230，留基准区分度）；弱场景保持弱，不被全局归一化放大噪声；
  ③ 非均匀叠加：反射层本身取自真实场景，空间分布天然非均匀。

v3 修正（--tone-correct，见 2026.9.13_R池失配诊断与能量比先验实验.md §8）：保留帧里的
反射光会触发手机逐帧 AE 压曝光 + AWB 色增益（frame ≈ s_c·GT + R_optical），frame−GT 里
混入 (s−1)·GT 的场景负残影（corr(R,GT)≈−0.4~−0.8 的主导项）。默认 none=旧行为；
slope=低分位包络斜率估计 s_c；**tv=在"时域低方差像素"（反射从不出现处）上 LS 拟合
s_c，最贴物理、泄漏最低**；env=全包络对齐（过削直流，弃用，仅对照）。
活跃帧选择恒按原始 (frame−GT) 均值口径，与 v2 池一致。

产物：
  <out>/raw/{subset}__{scene}.png    时域中值 R（未缩放，0-255 灰阶）
  <out>/pool/{subset}__{scene}.png   按 ρ 缩放后的成品池（合成器可直接叠加，勿再归一化）
  <out>/pool/{...}_alt.png           剪影显著场景的"单帧强结构"备选层（可选）
  <out>/meta.csv                     逐场景元数据
  <out>/overview.png                 GT|帧|R_raw|R_scaled 拼图
  <out>/strength_compare.png         实拍|RRW|合成 强度对照图（需 --figure）

用法：
  python3 scripts/build_rrw_pool.py \
      --src /DataDisk/zmy_code/datasets/RRWDataset \
      --out /DataDisk/zmy_code/TintedGlass_Enhance/data/pools/r_rrw_strong
"""
import argparse, csv, glob, os, re, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.stats import rankdata

AUG_T = "/home/zmy/vehicle_ssd/datasets/augmented/a00000_label.jpg"


def imread_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def fit_size(im, w, h):
    if im.shape[1] == w and im.shape[0] == h:
        return im
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


def downscale(im, max_side):
    h, w = im.shape[:2]
    if max(h, w) <= max_side:
        return im
    s = max_side / max(h, w)
    return cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def save_u8(im, path):
    Image.fromarray(np.clip(im, 0, 255).round().astype(np.uint8)).save(path)


def frame_key(p):
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 0


def holes_score(Rg):
    """暗洞分：局部背景明亮处的反射缺失（深色剪影 = 亮反射里的暗结构）。"""
    sigma = max(6.0, min(Rg.shape[:2]) / 120.0)
    Rb = cv2.GaussianBlur(Rg, (0, 0), sigma)
    gate = Rb > np.percentile(Rb, 60)
    dip = np.clip(Rb - Rg, 0, None) * gate
    return float(dip.mean() / (Rg.mean() + 1e-6))


def edge_score(Rg):
    lap = cv2.Laplacian(Rg, cv2.CV_32F)
    return float(np.abs(lap).mean() / (Rg.mean() + 1e-6))


def tone_curve(gt_c, fr_c, bins, q):
    """低分位包络 (centers, vals)：按 GT 等频分箱取 frame 的 q% 低分位并单调化。

    取下沿而非中值是为贴住"无反射贡献"的基线，不被亮反射抬走。
    """
    edges = np.unique(np.quantile(gt_c, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return None
    b = np.clip(np.searchsorted(edges, gt_c, side="right") - 1, 0, edges.size - 2)
    centers, vals = [], []
    for i in range(edges.size - 1):
        m = b == i
        if int(m.sum()) < 16:
            continue
        centers.append(float(np.median(gt_c[m])))
        vals.append(float(np.percentile(fr_c[m], q)))
    if len(centers) < 3:
        return None
    return np.asarray(centers), np.maximum.accumulate(np.asarray(vals))


def tone_gain(gt_c, fr_c, args):
    """稳健估计逐通道乘性响应 s（AE 压曝光 / AWB 色增益），供 R = frame − s·GT。

    反射光出现后手机 AE/AWB 自适应：frame ≈ s_c·GT + R_optical（s<1 且逐通道
    不同），frame−GT 里混入 (s−1)·GT 的场景负残影。用低分位包络在 GT∈[lo,hi]
    段拟合斜率（包络对亮反射稳健），只扣乘性分量、保住反射直流。
    """
    tc = tone_curve(gt_c, fr_c, args.tone_bins, args.tone_q)
    if tc is None:
        return 1.0
    centers, vals = tc
    sel = (centers >= args.tone_fit_lo) & (centers <= args.tone_fit_hi)
    if int(sel.sum()) < 4:
        sel = np.ones(centers.shape, bool)
    A = np.stack([centers[sel], np.ones(int(sel.sum()))], 1)
    s = float(np.linalg.lstsq(A, vals[sel], rcond=None)[0][0])
    return float(np.clip(s, 0.5, 1.5))


def temporal_gains(imgs, gt, args):
    """逐帧逐通道 s：在时域低方差像素(反射从不出现)上 LS 拟合 frame ≈ s·GT + c。

    比低分位包络更贴物理——那些像素上 frame 与 GT 只差相机响应，不含反射内容。
    """
    st = np.stack(imgs)
    tstd = st.std(axis=0).mean(axis=2)
    m = tstd <= np.percentile(tstd, args.tv_low_pct)
    if int(m.sum()) < 500:
        m = tstd <= np.percentile(tstd, 50)
    s_all = []
    for f in st:
        s = []
        for c in range(f.shape[2]):
            x = gt[..., c][m]
            y = f[..., c][m]
            A = np.stack([x, np.ones_like(x)], 1)
            s.append(float(np.clip(np.linalg.lstsq(A, y, rcond=None)[0][0], 0.4, 1.6)))
        s_all.append(s)
    return np.asarray(s_all)


def residual(fr, gt, args, s=None):
    if args.tone_correct == "none":
        return np.clip(fr - gt, 0, None)
    if args.tone_correct == "env":
        R = np.empty_like(fr)
        for c in range(fr.shape[2]):
            tc = tone_curve(gt[..., c], fr[..., c], args.tone_bins, args.tone_q)
            if tc is None:
                R[..., c] = fr[..., c] - gt[..., c]
            else:
                R[..., c] = fr[..., c] - np.interp(gt[..., c], tc[0], tc[1])
        return np.clip(R, 0, None)
    if s is None:  # slope
        s = [tone_gain(gt[..., c], fr[..., c], args) for c in range(fr.shape[2])]
    return np.clip(fr - np.asarray(s, np.float32) * gt, 0, None)


def discover(src):
    scenes = []
    for sub in sorted(os.listdir(src)):
        sd = os.path.join(src, sub)
        if not os.path.isdir(sd):
            continue
        gtd = glob.glob(os.path.join(sd, "GT_*"))
        if not gtd:
            continue
        gtd = gtd[0]
        for scene in sorted(os.listdir(sd)):
            if scene.startswith("GT_"):
                continue
            sdir = os.path.join(sd, scene)
            if not os.path.isdir(sdir):
                continue
            frames = sorted(glob.glob(os.path.join(sdir, "frame_*.jpg")), key=frame_key)
            gt = os.path.join(gtd, f"{scene}_GT.png")
            if not frames or not os.path.exists(gt):
                continue
            scenes.append((sub, scene, frames, gt))
    return scenes


def extract_one(sub, scene, frames, gt_path, args):
    gt = downscale(imread_rgb(gt_path), args.max_side)
    h, w = gt.shape[:2]
    n = len(frames)
    k = min(args.max_frames, n)
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    imgs = [fit_size(downscale(imread_rgb(frames[i]), args.max_side), w, h) for i in idx]
    ts = temporal_gains(imgs, gt, args) if args.tone_correct == "tv" else None
    raws = [float(np.clip(f - gt, 0, None).mean()) for f in imgs]
    layers = [residual(f, gt, args, None if ts is None else ts[j]) for j, f in enumerate(imgs)]
    stack = np.stack(layers)
    # 活跃帧选择恒按原始 (frame-GT) 均值口径，保持与 v2 池同一选择规则
    fm = np.asarray(raws)
    thr = max(float(fm.max()) * 0.35, float(np.percentile(fm, 25)))
    act = fm >= thr
    if act.sum() < 6:
        act = fm >= np.percentile(fm, 60)
    R = np.median(stack[act], axis=0)
    if args.denoise:
        R = cv2.bilateralFilter(R, 9, args.sigma_color, 7)
    alt_hs, alt_j = -1.0, -1
    act_ids = np.where(act)[0]
    for j in act_ids:
        hs_j = holes_score(stack[j].mean(axis=2))
        if hs_j > alt_hs:
            alt_hs, alt_j = hs_j, int(j)
    Rg = R.mean(axis=2)
    return dict(sub=sub, scene=scene, R=R, n=n, k=len(idx), n_active=int(act.sum()), idx=idx,
                w=w, h=h, gt_mean=float(gt.mean()), r_mean=float(R.mean()),
                ratio=float(R.mean()) / (float(gt.mean()) + 1e-6),
                bg_p10=float(np.percentile(Rg, 10)),
                hole=holes_score(Rg), edge=edge_score(Rg),
                alt_j=alt_j, alt_hs=alt_hs, frames=frames, gt_path=gt_path,
                alt_s=(None if ts is None else [float(x) for x in ts[alt_j]]))


def make_overview(recs, out_dir, args):
    picks = []
    by_ratio = sorted(recs, key=lambda r: -r["ratio"])
    by_hole = sorted(recs, key=lambda r: -r["hole"])
    seen = set()

    def take(lst, n, key):
        got = []
        for r in lst:
            if r["scene"] in seen:
                continue
            seen.add(r["scene"])
            got.append(r)
            if len(got) >= n:
                break
        return got

    picks += [(r, "top-strength") for r in take(by_ratio, 3, "ratio")]
    picks += [(r, "top-silhouette") for r in take(by_hole, 3, "hole")]
    picks += [(r, "median") for r in take(sorted(recs, key=lambda r: abs(r["ratio"] - np.median([x["ratio"] for x in recs]))), 2, "med")]
    picks = picks[: args.montage_rows]

    cw, ch, pad, lab = 200, 112, 6, 18
    cols = ["GT", "frame", "R_raw", "R_scaled"]
    W = cw * len(cols) + pad * (len(cols) + 1)
    H = lab * 2 + pad + len(picks) * (ch + lab + pad)
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for ci, name in enumerate(cols):
        x0 = pad + ci * (cw + pad)
        d.rectangle([x0, pad, x0 + cw, pad + lab], fill=(60, 60, 70))
        d.text((x0 + 6, pad + 4), name, fill=(255, 255, 255))
    for ri, (r, tag) in enumerate(picks):
        y0 = lab + pad + ri * (ch + lab + pad)
        fr = fit_size(downscale(imread_rgb(r["frames"][r["idx"][len(r["idx"]) // 2]]),
                                args.max_side), r["w"], r["h"])
        gt = downscale(imread_rgb(r["gt_path"]), args.max_side)
        raw = np.asarray(Image.open(os.path.join(out_dir, "raw", r["png_raw"])).convert("RGB"), np.float32)
        psc = np.asarray(Image.open(os.path.join(out_dir, "pool", r["png_pool"])).convert("RGB"), np.float32)
        cells = [gt, fr, raw, psc]
        for ci, im in enumerate(cells):
            x0 = pad + ci * (cw + pad)
            cell = Image.fromarray(np.clip(im, 0, 255).round().astype(np.uint8)).resize((cw, ch))
            sheet.paste(cell, (x0, y0))
        lbl = (f"{r['sub']}/{r['scene']} [{tag}] ratio={r['ratio']:.2f} rho={r['rho']:.2f} "
               f"hole={r['hole']:.2f} Rm={r['r_mean_scaled']:.0f} clip={r['clip_frac']*100:.1f}%")
        d.rectangle([pad, y0 + ch, W - pad, y0 + ch + lab], fill=(45, 45, 55))
        d.text((pad + 6, y0 + ch + 4), lbl, fill=(230, 230, 230))
    p = os.path.join(out_dir, "overview.png")
    sheet.save(p)
    print(f"[fig] {p}")


def _find_real(stem_glob):
    for d in sorted(glob.glob(stem_glob)):
        mi = os.path.join(d, "m_input.png")
        rr = os.path.join(d, "ytmt_ucs_sirs_r.png")
        if os.path.exists(mi) and os.path.exists(rr):
            return mi, rr
    return None, None


def make_strength_figure(recs, out_dir, args):
    real_rows = []
    for tag, pat in [("4.26", "/home/zmy/vehicle_ssd/rdnet_real426_0909/*"),
                     ("162901", "/home/zmy/vehicle_ssd/rdnet_real901_0909/*")]:
        mi, rr = _find_real(pat)
        if mi:
            real_rows.append((f"real {tag} input", imread_rgb(mi)))
            real_rows.append((f"real {tag} Rhat", imread_rgb(rr)))

    rrw_cells = []
    for scene_name in ["camera_objects1", "outdoor_night_park3"]:
        r = next((x for x in recs if x["scene"] == scene_name), None)
        if not r:
            continue
        gt = downscale(imread_rgb(r["gt_path"]), args.max_side)
        fr = fit_size(downscale(imread_rgb(r["frames"][r["idx"][len(r["idx"]) // 2]]),
                                args.max_side), r["w"], r["h"])
        raw = np.asarray(Image.open(os.path.join(out_dir, "raw", r["png_raw"])).convert("RGB"), np.float32)
        psc = np.asarray(Image.open(os.path.join(out_dir, "pool", r["png_pool"])).convert("RGB"), np.float32)
        rrw_cells += [(f"RRW {scene_name} frame", fr), (f"RRW {scene_name} GT", gt),
                      (f"RRW R raw (m={raw.mean():.0f})", raw),
                      (f"RRW R x{r['rho']:.1f} (m={psc.mean():.0f})", psc)]

    synth_cells = []
    r = next((x for x in recs if x["scene"] == "camera_objects1"), None)
    if r and os.path.exists(AUG_T):
        T = imread_rgb(AUG_T)
        h, w = T.shape[:2]
        Rsc = np.asarray(Image.open(os.path.join(out_dir, "pool", r["png_pool"])).convert("RGB"),
                         np.float32)
        Rsc = fit_size(Rsc, w, h)
        rng = np.random.default_rng(args.seed)
        for a in [0.6, 0.3, 0.15]:
            I = a * T + Rsc
            I += rng.normal(0, args.noise / 255.0, I.shape)
            I = np.clip(I, 0, 255)
            synth_cells.append((f"synth a={a} (m={I.mean():.0f})", I))

    rows = [("ACTUAL FOOTAGE (input | RDNet R-hat)", real_rows),
            ("RRW REAL REFLECTION LAYERS", rrw_cells),
            ("SYNTHETIC: a*T + R_rrw (T=augmented)", synth_cells)]
    chh, lab, pad, title = 150, 20, 6, 24
    W = 1560
    H = pad + sum(title + chh + lab + pad for _, cells in rows if cells)
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    d = ImageDraw.Draw(sheet)
    y = pad
    for tname, cells in rows:
        if not cells:
            continue
        d.rectangle([pad, y, W - pad, y + title - 6], fill=(70, 70, 84))
        d.text((pad + 8, y + 5), tname, fill=(255, 255, 255))
        y += title
        x = pad
        for lbl, im in cells:
            h_, w_ = im.shape[:2]
            cw_ = max(60, round(chh * w_ / h_))
            cell = Image.fromarray(np.clip(im, 0, 255).round().astype(np.uint8)).resize((cw_, chh))
            if x + cw_ > W - pad:
                break
            sheet.paste(cell, (x, y))
            d.rectangle([x, y + chh, x + cw_, y + chh + lab], fill=(40, 40, 48))
            d.text((x + 5, y + chh + 5), lbl, fill=(230, 230, 230))
            x += cw_ + pad
        y += chh + lab + pad
    p = os.path.join(out_dir, "strength_compare.png")
    sheet.save(p)
    print(f"[fig] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/datasets/RRWDataset")
    ap.add_argument("--out", default=os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance") + "/data/pools/r_rrw_strong")
    ap.add_argument("--max-frames", type=int, default=24, help="每场景时域采样帧数(中值)")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--r-mean", type=float, default=0.06, help="中位场景目标 R 均值(base)")
    ap.add_argument("--rho-min", type=float, default=1.0)
    ap.add_argument("--rho-max", type=float, default=2.6)
    ap.add_argument("--comp", type=float, default=0.5, help="强度相对压缩指数(native/中位)^p")
    ap.add_argument("--comp-lo", type=float, default=0.45)
    ap.add_argument("--comp-hi", type=float, default=1.8)
    ap.add_argument("--clip-lo", type=float, default=6.0, help="目标 R̄ 下限(灰阶)")
    ap.add_argument("--clip-hi", type=float, default=75.0, help="目标 R̄ 上限(灰阶)")
    ap.add_argument("--scale-cap", type=float, default=8.0, help="单场景最大放大倍数")
    ap.add_argument("--min-r-mean", type=float, default=4.0, help="native R̄ 低于此值剔除(噪声退化)")
    ap.add_argument("--alt-ratio", type=float, default=1.35, help="单帧剪影强于中值多少倍才导出 alt")
    ap.add_argument("--sigma-color", type=float, default=15.0)
    ap.add_argument("--tone-correct", choices=["none", "slope", "env", "tv"], default="none",
                    help="none=frame-GT(旧行为); slope=低分位包络斜率扣 AE/AWB; "
                         "tv=时域低方差像素拟合响应(最贴物理); env=全包络对齐(对照)")
    ap.add_argument("--tv-low-pct", type=float, default=20.0,
                    help="tv 估计时取时域 std 最低的百分位像素")
    ap.add_argument("--tone-bins", type=int, default=24, help="色调包络 GT 分箱数(等频)")
    ap.add_argument("--tone-q", type=float, default=15.0, help="包络分位(q%% 低分位)")
    ap.add_argument("--tone-fit-lo", type=float, default=30.0, help="斜率拟合 GT 下限(灰阶)")
    ap.add_argument("--tone-fit-hi", type=float, default=230.0, help="斜率拟合 GT 上限(灰阶)")
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--denoise", action="store_true", default=True)
    ap.add_argument("--no-denoise", dest="denoise", action="store_false")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--only", default=None, help="逗号分隔 subset/scene 过滤(调试)")
    ap.add_argument("--figure", action="store_true", default=True)
    ap.add_argument("--no-figure", dest="figure", action="store_false")
    ap.add_argument("--montage-rows", type=int, default=8)
    args = ap.parse_args()

    scenes = discover(args.src)
    if args.only:
        want = set(args.only.split(","))
        scenes = [s for s in scenes if f"{s[0]}/{s[1]}" in want]
    print(f"发现 {len(scenes)} 个可用场景")
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "pool"), exist_ok=True)

    recs = []
    for i, (sub, scene, frames, gt) in enumerate(scenes):
        r = extract_one(sub, scene, frames, gt, args)
        recs.append(r)
        print(f"[{i+1}/{len(scenes)}] {sub}/{scene} n={r['n']} k={r['k']} "
              f"ratio={r['ratio']:.3f} hole={r['hole']:.3f} edge={r['edge']:.3f} "
              f"bgP10={r['bg_p10']:.1f} alt={r['alt_hs']:.3f}", flush=True)

    for r in recs:
        r["kept"] = bool(r["r_mean"] >= args.min_r_mean)
    kept = [r for r in recs if r["kept"]]
    dropped = [r for r in recs if not r["kept"]]
    m_nat = float(np.median([r["r_mean"] for r in kept]))
    pr = (rankdata([r["ratio"] for r in kept]) + rankdata([r["hole"] for r in kept])
          + rankdata([r["edge"] for r in kept])) / (3.0 * len(kept))
    rng = np.random.default_rng(args.seed)
    lo, hi = np.log(args.rho_min), np.log(args.rho_max)
    base = args.r_mean * 255.0
    for r, p in zip(kept, pr):
        u = float(np.clip(0.5 * rng.random() + 0.5 * p, 0, 1))
        r["rho"] = float(np.exp(lo + u * (hi - lo)))
        c = float(np.clip((r["r_mean"] / m_nat) ** args.comp, args.comp_lo, args.comp_hi))
        target = float(np.clip(base * c * r["rho"], args.clip_lo, args.clip_hi))
        scale = min(target / max(r["r_mean"], 1e-6), args.scale_cap)
        r["target"] = round(r["r_mean"] * scale, 1)
        r["r_mean_scaled"] = r["target"]
        Rs = np.clip(r["R"] * scale, 0, 255)
        r["clip_frac"] = float((Rs.max(axis=2) >= 254).mean())
        r["png_raw"] = f"{r['sub']}__{r['scene']}.png"
        r["png_pool"] = f"{r['sub']}__{r['scene']}.png"
        save_u8(r["R"], os.path.join(args.out, "raw", r["png_raw"]))
        save_u8(Rs, os.path.join(args.out, "pool", r["png_pool"]))
        r["alt_saved"] = False
        if r["alt_hs"] > args.alt_ratio * r["hole"] and r["alt_hs"] > 0.03:
            fr = fit_size(downscale(imread_rgb(r["frames"][r["idx"][r["alt_j"]]]),
                                    args.max_side), r["w"], r["h"])
            gt = downscale(imread_rgb(r["gt_path"]), args.max_side)
            Ra = residual(fr, gt, args, r["alt_s"])
            if args.denoise:
                Ra = cv2.bilateralFilter(Ra, 9, args.sigma_color, 7)
            Ras = np.clip(Ra * (r["target"] / max(Ra.mean(), 1e-6)), 0, 255)
            save_u8(Ras, os.path.join(args.out, "pool", f"{r['sub']}__{r['scene']}_alt.png"))
            r["alt_saved"] = True

    hole_p75 = np.percentile([r["hole"] for r in kept], 75)
    for r in kept:
        r["silhouette"] = bool(r["hole"] >= hole_p75)
    for r in dropped:
        r["rho"] = r["target"] = r["r_mean_scaled"] = r["clip_frac"] = 0.0
        r["silhouette"] = r["alt_saved"] = False
        r["png_raw"] = r["png_pool"] = ""
    cols = ["sub", "scene", "n", "k", "n_active", "w", "h", "gt_mean", "r_mean", "ratio",
            "kept", "rho", "target", "r_mean_scaled", "clip_frac", "bg_p10", "hole", "edge",
            "silhouette", "alt_hs", "alt_saved", "png_raw", "png_pool"]
    with open(os.path.join(args.out, "meta.csv"), "w", newline="") as fp:
        cw = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        cw.writeheader()
        for r in sorted(recs, key=lambda x: -x["ratio"]):
            cw.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})

    ratios = np.array([r["ratio"] for r in kept])
    natives = np.array([r["r_mean"] for r in kept])
    targets = np.array([r["target"] for r in kept])
    rhos = np.array([r["rho"] for r in kept])
    print(f"\n=== 汇总 (扫描 {len(recs)} 场景, 入池 {len(kept)}, 剔除 {len(dropped)}) ===")
    if dropped:
        print("剔除(过弱/噪声退化):", [(r["sub"] + "/" + r["scene"], round(r["r_mean"], 1)) for r in dropped])
    print(f"原生 R̄:    min={natives.min():.1f} 中位={np.median(natives):.1f} max={natives.max():.1f} (灰阶)")
    print(f"原生 R/GT:  min={ratios.min():.3f} 中位={np.median(ratios):.3f} max={ratios.max():.3f}")
    print(f"ρ:          min={rhos.min():.2f} 中位={np.median(rhos):.2f} max={rhos.max():.2f}")
    print(f"入池 R̄:     min={targets.min():.1f} 中位={np.median(targets):.1f} max={targets.max():.1f} (灰阶)")
    print(f"剪影标记: {sum(r['silhouette'] for r in kept)} 个 (hole≥P75); alt 导出 "
          f"{sum(r['alt_saved'] for r in kept)} 个")
    print("Top5 强度:", [(r["sub"] + "/" + r["scene"], round(r["ratio"], 3))
                         for r in sorted(kept, key=lambda x: -x["ratio"])[:5]])
    print("Top5 剪影:", [(r["sub"] + "/" + r["scene"], round(r["hole"], 3))
                         for r in sorted(kept, key=lambda x: -x["hole"])[:5]])
    print(f"meta -> {os.path.join(args.out, 'meta.csv')}")

    make_overview(kept, args.out, args)
    if args.figure:
        make_strength_figure(kept, args.out, args)


if __name__ == "__main__":
    main()
