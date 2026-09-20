#!/usr/bin/env python3
"""RRW 扩池（2026-09-18）：把 RRWDataset 里**尚未开采的帧**做成新的 R 候选层。

背景（盘点结论，见 docs/2026.9.17_RRW扩池_0918.md）：
  本地 RRWDataset 共 42 个可配对场景（refD_vivo1 17 / reflection_camera1 9 /
  reflection_camera2 12 / reflection_hf0 4），`r_rrw_strong/meta.csv` 已**全部扫过**
  （38 入池 + 4 因 native R̄<4 剔除）→ 没有"未用场景"，未开采的是**同一场景内未参与
  中值的帧**：旧配方每场景只等距抽 24 帧（全序列 54–441 帧），只产 1 张 R。

本脚本把每个场景的帧序列切成 W 个**互不重叠的连续窗**，每窗按 build_rrw_pool 完全
相同的配方独立提一张 R（活跃帧选择 + 时域中值 + bilateral），产出 W 张候选层：
  R_w = median_{活跃帧∈窗}( clip(frame − s·GT, 0) )        s=1（tone-correct=none，同旧口径）

强度重标定沿用 build_rrw_pool 配方（**批内**统计，与 build_world_soft_pool 同惯例）：
  base=0.06*255=15.3; c=clip((R̄_w/m_nat)^0.5, 0.45, 1.8); rho~logU[1,2.6]（u 由强度×结构
  综合分位决定）; target=clip(base·c·rho, 6, 75); 缩放 clip(R_w·scale, 0, 255), scale≤8。
筛选同配方：native R̄ < --min-r-mean(4.0) 逐窗剔除（噪声退化）。

产物（**新目录，绝不写 r_rrw_strong/**）:
  <out>/raw/{sub}__{scene}__w{i}.png    未缩放 R（源域表示，供泄漏安全网比对）
  <out>/pool/{sub}__{scene}__w{i}.png   重标定后的成品层
  <out>/meta.csv                        逐窗元数据（含 corr_gt 泄漏指标、窗区间）

用法:
  python3 scripts/build_rrw_pool_expand.py            # 全量
  python3 scripts/build_rrw_pool_expand.py --only refD_vivo1/outdoor_plants --workers 1
"""
import argparse, csv, importlib.util, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("brrw", os.path.join(HERE, "build_rrw_pool.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

cv2 = B.cv2


class PipeArgs:
    """build_rrw_pool 管线所需的参数子集（默认值与旧池一致）。"""
    def __init__(self, max_frames=24, max_side=1280, tone_correct="none", denoise=True,
                 sigma_color=15.0, tv_low_pct=20.0, tone_bins=24, tone_q=15.0,
                 tone_fit_lo=30.0, tone_fit_hi=230.0):
        self.max_frames = max_frames; self.max_side = max_side
        self.tone_correct = tone_correct; self.denoise = denoise
        self.sigma_color = sigma_color; self.tv_low_pct = tv_low_pct
        self.tone_bins = tone_bins; self.tone_q = tone_q
        self.tone_fit_lo = tone_fit_lo; self.tone_fit_hi = tone_fit_hi


def corr_gray(R, gt):
    a = R.mean(axis=2).ravel().astype(np.float64)
    b = gt.mean(axis=2).ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def layer_in_window(sub, scene, frames, gt_path, p, w0, w1):
    """窗口 [w0,w1) 内的时域中值 R（活跃帧规则与 build_rrw_pool 完全一致）。"""
    gt = B.downscale(B.imread_rgb(gt_path), p.max_side)
    h, w = gt.shape[:2]
    sub_frames = frames[w0:w1]
    n = len(sub_frames)
    k = min(p.max_frames, n)
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    imgs = [B.fit_size(B.downscale(B.imread_rgb(sub_frames[i]), p.max_side), w, h) for i in idx]
    ts = B.temporal_gains(imgs, gt, p) if p.tone_correct == "tv" else None
    raws = [float(np.clip(f - gt, 0, None).mean()) for f in imgs]
    layers = [B.residual(f, gt, p, None if ts is None else ts[j]) for j, f in enumerate(imgs)]
    stack = np.stack(layers)
    fm = np.asarray(raws)
    thr = max(float(fm.max()) * 0.35, float(np.percentile(fm, 25)))
    act = fm >= thr
    if act.sum() < 6:
        act = fm >= np.percentile(fm, 60)
    R = np.median(stack[act], axis=0)
    if p.denoise:
        R = cv2.bilateralFilter(R, 9, p.sigma_color, 7)
    Rg = R.mean(axis=2)
    return dict(sub=sub, scene=scene, w0=int(w0), w1=int(w1), n_win=n, k=len(idx),
                n_active=int(act.sum()), w_img=w, h_img=h,
                gt_mean=float(gt.mean()), r_mean=float(R.mean()),
                ratio=float(R.mean()) / (float(gt.mean()) + 1e-6),
                bg_p10=float(np.percentile(Rg, 10)),
                hole=B.holes_score(Rg), edge=B.edge_score(Rg),
                corr_gt=corr_gray(R, gt), R=R)


def process_scene(task):
    sub, scene, frames, gt_path, n_win, min_win, p = task
    n = len(frames)
    if n < 2 * min_win:
        return dict(sub=sub, scene=scene, skipped=f"n={n}<2*{min_win}", recs=[])
    lims = np.linspace(0, n, n_win + 1).round().astype(int)
    recs = []
    for i in range(n_win):
        rec = layer_in_window(sub, scene, frames, gt_path, p, lims[i], lims[i + 1])
        rec["win"] = i
        recs.append(rec)
    return dict(sub=sub, scene=scene, skipped="", recs=recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/DataDisk/zmy_code/datasets/RRWDataset")
    ap.add_argument("--out", default=os.path.join(PROJ, "data/pools/r_rrw_expand_0918"))
    ap.add_argument("--windows", type=int, default=4, help="每场景最大窗数")
    ap.add_argument("--min-win-frames", type=int, default=40, help="单窗最小帧数（不足则不切）")
    ap.add_argument("--max-frames", type=int, default=24, help="每窗抽样帧数（同旧配方）")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--tone-correct", choices=["none", "slope", "tv"], default="none",
                    help="none=同旧池口径")
    # 重标定配方（与 build_rrw_pool / build_world_soft_pool 一致）
    ap.add_argument("--r-mean", type=float, default=0.06)
    ap.add_argument("--rho-min", type=float, default=1.0)
    ap.add_argument("--rho-max", type=float, default=2.6)
    ap.add_argument("--comp", type=float, default=0.5)
    ap.add_argument("--comp-lo", type=float, default=0.45)
    ap.add_argument("--comp-hi", type=float, default=1.8)
    ap.add_argument("--clip-lo", type=float, default=6.0)
    ap.add_argument("--clip-hi", type=float, default=75.0)
    ap.add_argument("--scale-cap", type=float, default=8.0)
    ap.add_argument("--min-r-mean", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="逗号分隔 subset/scene 过滤（调试）")
    args = ap.parse_args()

    scenes = B.discover(args.src)
    if args.only:
        want = set(args.only.split(","))
        scenes = [s for s in scenes if f"{s[0]}/{s[1]}" in want]
    print(f"发现 {len(scenes)} 个场景；每场景最多切 {args.windows} 窗（单窗 ≥{args.min_win_frames} 帧）",
          flush=True)

    from scipy.stats import rankdata
    p = PipeArgs(max_frames=args.max_frames, max_side=args.max_side,
                 tone_correct=args.tone_correct, denoise=True)
    tasks = [(sub, sc, fr, gt, args.windows, args.min_win_frames, p) for sub, sc, fr, gt in scenes]
    results = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_scene, tasks)):
                results.append(r)
                print(f"[{i+1}/{len(tasks)}] {r['sub']}/{r['scene']} "
                      f"{'skip:'+r['skipped'] if r['skipped'] else str(len(r['recs']))+' 窗'}",
                      flush=True)
    else:
        for i, t in enumerate(tasks):
            r = process_scene(t)
            results.append(r)
            print(f"[{i+1}/{len(tasks)}] {r['sub']}/{r['scene']} "
                  f"{'skip:'+r['skipped'] if r['skipped'] else str(len(r['recs']))+' 窗'}", flush=True)

    recs = [r for res in results for r in res["recs"]]
    skipped = [f"{res['sub']}/{res['scene']}:{res['skipped']}" for res in results if res["skipped"]]
    print(f"\n窗口层: {len(recs)} 张（跳过场景 {len(skipped)}: {skipped}）")
    for r in recs:
        r["kept"] = bool(r["r_mean"] >= args.min_r_mean)
    kept = [r for r in recs if r["kept"]]
    print(f"native R̄≥{args.min_r_mean}: {len(kept)} 张；剔除 {len(recs)-len(kept)} 张(过弱/噪声退化)")

    # —— 批内重标定（同 build_world_soft_pool 惯例：批内中位 + 批内分位）——
    m_nat = float(np.median([r["r_mean"] for r in kept])) if kept else 1.0
    pr = (rankdata([r["ratio"] for r in kept]) + rankdata([r["hole"] for r in kept])
          + rankdata([r["edge"] for r in kept])) / (3.0 * len(kept))
    rng = np.random.default_rng(args.seed)
    lo, hi = np.log(args.rho_min), np.log(args.rho_max)
    base = args.r_mean * 255.0
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "pool"), exist_ok=True)
    for r, pr_i in zip(kept, pr):
        u = float(np.clip(0.5 * rng.random() + 0.5 * pr_i, 0, 1))
        r["rho"] = float(np.exp(lo + u * (hi - lo)))
        c = float(np.clip((r["r_mean"] / m_nat) ** args.comp, args.comp_lo, args.comp_hi))
        target = float(np.clip(base * c * r["rho"], args.clip_lo, args.clip_hi))
        scale = min(target / max(r["r_mean"], 1e-6), args.scale_cap)
        r["target"] = round(r["r_mean"] * scale, 1)
        r["scale"] = round(scale, 3)
        Rs = np.clip(r["R"] * scale, 0, 255)
        r["clip_frac"] = float((Rs.max(axis=2) >= 254).mean())
        name = f"{r['sub']}__{r['scene']}__w{r['win']}.png"
        r["png"] = name
        B.save_u8(r["R"], os.path.join(args.out, "raw", name))
        B.save_u8(Rs, os.path.join(args.out, "pool", name))

    cols = ["sub", "scene", "win", "w0", "w1", "n_win", "k", "n_active", "w_img", "h_img",
            "gt_mean", "r_mean", "ratio", "kept", "rho", "target", "scale", "clip_frac",
            "bg_p10", "hole", "edge", "corr_gt", "png"]
    for r in recs:
        for k_ in ("rho", "target", "scale", "clip_frac"):
            r.setdefault(k_, 0.0)
        r.setdefault("png", "")
    with open(os.path.join(args.out, "meta.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(recs, key=lambda x: (x["sub"], x["scene"], x["win"])):
            w.writerow({k_: (round(r[k_], 4) if isinstance(r[k_], float) else r[k_]) for k_ in cols})

    tg = np.array([r["target"] for r in kept])
    print(f"\n=== 扩池汇总 ===")
    print(f"入池 {len(kept)} 张 -> {args.out}/pool（native 中位 R̄={m_nat:.1f}）")
    print(f"重标定 target: min={tg.min():.1f} 中位={np.median(tg):.1f} max={tg.max():.1f}（旧池中位 24.4）")
    cg = np.abs(np.array([r["corr_gt"] for r in kept]))
    print(f"|corr(R,GT)|: 中位={np.median(cg):.3f} p90={np.percentile(cg, 90):.3f}（旧场景层中位 0.431）")
    print(f"meta -> {os.path.join(args.out, 'meta.csv')}")


if __name__ == "__main__":
    main()
