#!/usr/bin/env python3
"""三连图训练集生成器: 同场景 × 三档 ᾱ=1.0/0.6/0.3, 共享掩膜 m 与弧度场 f。

与 build_train_pairs.py 的区别(实施方案 §7.1, D0):
  - 每"源场景"(T_src×R_src 对)只采样一次, 同一 rng 流依次生成 T/R 裁剪、
    coverage_mask(m)、arc_gradient(f)、r_overlay_field → 三档复用同一 m/f/R_eff;
  - 每场景三档共用同一噪声种子 → 三档之差 = 纯 α 变化(受控三连图);
  - 落盘: 每样本 I/α, 每场景 T/R + **m/g(fp16, g=1+s·f, 新)**;
  - meta 增加 scene_id, 供跨档一致性 loss(组件①/③)按场景组 batch。

§7.2 选项(已启用): 档 1.0 clip_frac 长尾(来自 R 金属热点, 与 T 源无关)重采样——
整场景重抽 rng, 至多 --clip-max-retry 次, 直到档 1.0 的 clip_frac ≤ --clip-cap;
未达标者保留并计入 meta.redraws.still_over。

α 可由落盘量精确重建: α = clip(ᾱ·g·m + (1−m), 0.01, 1.0)。

用法:
  python3 scripts/build_train_triplets.py --out data/train_sets/train_triplets_0913 --n-scenes 800
  python3 scripts/build_train_triplets.py --check data/train_sets/train_triplets_0913   # 自检(S0 门)
  python3 scripts/build_train_triplets.py --preview 3                   # 目检 sheet
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image, ImageDraw

from synth_tinted_degrade import load, save, coverage_mask, arc_gradient, \
    alpha_from, r_overlay_field, compose_alpha

LEVELS = [1.0, 0.6, 0.3]           # 可被 --levels 覆盖（main 内设全局）
LEVEL_TAGS = {1.0: "l100", 0.6: "l060", 0.3: "l030"}
DEFAULT_T_SRC = [
    ("sir2", "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer"),
    ("nature", "/home/zmy/mycode/datasets/nature/natural_T"),
    ("scenes", "/home/zmy/vehicle_ssd/datasets/scenes"),
]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def glob_images(d, pattern="*"):
    return sorted(f for f in glob.glob(os.path.join(d, pattern))
                  if os.path.splitext(f)[1].lower() in IMG_EXT)


def desat(r, k):
    if k >= 0.999:
        return r
    g = r.mean(axis=2, keepdims=True)
    return np.clip(g + k * (r - g), 0.0, 1.0)


def crop_square(im, size, rng):
    h, w = im.shape[:2]
    if min(h, w) < size:
        sc = size / min(h, w)
        im = np.asarray(Image.fromarray((im * 255).round().astype(np.uint8)).resize(
            (max(size, int(round(w * sc))), max(size, int(round(h * sc)))), Image.LANCZOS),
            np.float32) / 255.0
        h, w = im.shape[:2]
    y0 = int(rng.integers(0, h - size + 1))
    x0 = int(rng.integers(0, w - size + 1))
    return im[y0:y0 + size, x0:x0 + size]


def test_scene_ids(meta_path):
    if not os.path.exists(meta_path):
        return set()
    with open(meta_path) as fp:
        meta = json.load(fp)
    return {os.path.basename(s["src_r"])[:-4] for s in meta["scenes"]}


def pool_files(pool_dir, exclude_ids):
    keep = []
    for f in sorted(glob.glob(os.path.join(pool_dir, "*.png"))):
        stem = os.path.basename(f)[:-4]
        sid = stem[:-4] if stem.endswith("_alt") else stem
        if sid in exclude_ids:
            continue
        keep.append(f)
    return keep


def synth_scene(r_f, t_f, rng, size, s, noise, r_sat, cover_frac, noise_seed):
    """一个源场景: 三档共享 m/f/R_eff/噪声, 只 ᾱ 变化。返回 (t, r_eff, m, g, {level: (a, i)})。"""
    t = crop_square(load(t_f), size, rng)
    r = crop_square(load(r_f), size, rng)
    m = coverage_mask(size, size, rng, frac=tuple(cover_frac))
    f = arc_gradient(size, size, rng)
    g = 1.0 + s * f
    r_eff = desat(r * r_overlay_field(size, size, rng)[..., None], r_sat)
    out = {}
    for lv in LEVELS:
        a = alpha_from(m, f, lv, s)
        out[lv] = (a, compose_alpha(t, r_eff, a, noise, noise_seed))
    return t, r_eff, m, g, out


def make_sheet(rows, path, cw=224):
    pad, lab = 5, 22
    ncol = len(rows[0])
    W = pad + ncol * (cw + pad)
    H = pad + len(rows) * (cw + lab + pad)
    S = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(S)
    for ri, row in enumerate(rows):
        y0 = pad + ri * (cw + lab + pad)
        for ci, (im, name) in enumerate(row):
            x0 = pad + ci * (cw + pad)
            im8 = (np.clip(im, 0, 1) * 255).round().astype(np.uint8)
            S.paste(Image.fromarray(im8), (x0, y0))
            d.text((x0 + 4, y0 + cw + 4), name, fill=(230, 230, 230))
    S.save(path)
    print("sheet ->", path)


def gray3(x):
    return np.repeat(np.clip(x, 0, 1)[..., None], 3, 2)


def run_check(out):
    """自检: ①磁盘上的 α 可由 m/g/level 重建; ②三档 m 一致(同场景单文件);
    ③膜区 α 三档单调分离、裸区三档完全一致; ④分布复核(对齐测试集口径)。"""
    with open(os.path.join(out, "meta.json")) as fp:
        meta = json.load(fp)
    print(f"check <- {out}  scenes={meta['n_scenes']} samples={meta['n']}")
    by_scene = {}
    for smp in meta["samples"]:
        by_scene.setdefault(smp["scene_id"], {})[smp["level"]] = smp

    errs = []
    max_a_err, max_naked_d, min_mono_gap = 0.0, 0.0, 9.9
    film_obs_gap, clip_share = [], []
    n_checked = 0
    for sc_id, lv_map in sorted(by_scene.items()):
        m = np.load(os.path.join(out, "m", sc_id + ".npy")).astype(np.float32)
        g = np.load(os.path.join(out, "g", sc_id + ".npy")).astype(np.float32)
        alphas = {}
        for lv, smp in lv_map.items():
            a_disk = np.load(os.path.join(out, "alpha", smp["sid"] + ".npy")).astype(np.float32)
            a_re = np.clip(lv * g * m + (1.0 - m), 0.01, 1.0)
            max_a_err = max(max_a_err, float(np.abs(a_disk - a_re).max()))
            alphas[lv] = a_disk
        naked = m < 1e-3                       # 膜外: α 应恒为 1, 与档位无关
        film = m > 0.95
        film_obs = film & (alphas[1.0] < 0.95)  # 档1.0 膜区未被 clip 到 1 的可观测像素
        for lv in (0.6, 0.3):
            if naked.any():
                max_naked_d = max(max_naked_d, float(np.abs(alphas[lv] - alphas[1.0])[naked].max()))
            min_mono_gap = min(min_mono_gap, float((alphas[1.0] - alphas[lv]).min()))
        if film.any():
            clip_share.append(float((alphas[1.0][film] >= 0.95).mean()))
        if film_obs.any():
            film_obs_gap.append(float((alphas[1.0] - alphas[0.6])[film_obs].mean()))
        n_checked += 1
        if n_checked >= 50:
            break

    print(f"[a] α 重建最大误差 (fp16 容差 2e-3): {max_a_err:.2e}")
    print(f"[b] 裸区(m<1e-3)三档 α 最大偏差: {max_naked_d:.2e}")
    print(f"[c] 全图单调性 min(α_1.0 − α_0.6, α_1.0 − α_0.3) ≥ −2e-3: {min_mono_gap:.2e}")
    print(f"[d] 档1.0 膜区内 α 被 clip 到 1(不可观测)占比: {np.mean(clip_share):.3f}")
    print(f"    可观测膜区 α_1.0−α_0.6 平均间隔: {np.mean(film_obs_gap):.3f}")
    ok = max_a_err < 2e-3 and max_naked_d < 2e-3 and min_mono_gap > -2e-3
    if not ok:
        errs.append("ABC 项未过")
    print(f"[abc] {'PASS' if ok else 'FAIL'}")

    for lv in LEVELS:
        sel = [s for s in meta["samples"] if s["level"] == lv]
        ae = np.array([s["alpha_eff"] for s in sel])
        cf = np.array([s["clip_frac"] for s in sel])
        print(f"[e] 档{lv:g}: n={len(sel)} α_eff={ae.mean():.3f}  "
              f"clip_frac mean={cf.mean() * 100:.2f}% median={np.median(cf) * 100:.2f}% max={cf.max() * 100:.2f}%")

    print("[f] 测试集 v2.1 参照: α_eff=[0.892, 0.655, 0.410]  "
          "clip_frac mean=[1.41, 1.09, 0.92]% median=[1.04, 0.58, 0.26]%")
    return 0 if not errs else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/DataDisk/zmy_code/TintedGlass_Enhance/data/train_sets/train_triplets_0913")
    ap.add_argument("--n-scenes", type=int, default=800, help="源场景数(× 3 档 = 样本数)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--spatial", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--r-sat", type=float, default=0.45)
    ap.add_argument("--cover-frac", type=float, nargs=2, default=[0.78, 0.92])
    ap.add_argument("--r-pool", default="/DataDisk/zmy_code/TintedGlass_Enhance/data/pools/r_rrw_strong/pool")
    ap.add_argument("--test-meta", default="/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/synth_rrw_v21_0913/meta.json")
    ap.add_argument("--t-weights", default="0.35,0.25,0.40", help="对应 sir2,nature,scenes 抽样权重")
    ap.add_argument("--t-extra-dir", default="", help="附加第 4 个 T 源目录(如域匹配的 augmented 池)")
    ap.add_argument("--t-extra-name", default="extra", help="附加 T 源名称")
    ap.add_argument("--t-extra-weight", type=float, default=1.0, help="附加 T 源权重")
    ap.add_argument("--t-extra-glob", default="*", help="附加 T 源文件通配(如可见光干净图用 '*_label.jpg')")
    ap.add_argument("--t-exclude-json", default="", help="逗号分隔 JSON 列表, 收集其中 scenes[].src_t 并排除(防泄漏)")
    ap.add_argument("--levels", default="1.0,0.6,0.3",
                    help="档位列表(逗号分隔); 例 '1.0,0.6,0.3,0.15,0.1'")
    ap.add_argument("--clip-cap", type=float, default=0.05, help="档 1.0 clip_frac 重采样上限(<=0 关闭)")
    ap.add_argument("--clip-max-retry", type=int, default=4)
    ap.add_argument("--preview", type=int, default=0, help="只出目检 sheet(行=场景)后退出")
    ap.add_argument("--check", default="", help="对已生成目录跑自检后退出")
    args = ap.parse_args()
    global LEVELS, LEVEL_TAGS
    LEVELS = [float(x) for x in args.levels.split(",")]
    LEVEL_TAGS = {lv: f"l{int(round(lv * 100)):03d}" for lv in LEVELS}
    print(f"档位: {LEVELS}  标签: {list(LEVEL_TAGS.values())}")

    if args.check:
        raise SystemExit(run_check(args.check))

    t_pools = list(DEFAULT_T_SRC)
    w = [float(x) for x in args.t_weights.split(",")]
    if args.t_extra_dir:
        t_pools.append((args.t_extra_name, args.t_extra_dir))
        w.append(args.t_extra_weight)
    t_weights = np.array(w)
    t_weights = t_weights / t_weights.sum()
    excl = set()
    for jf in filter(None, (s.strip() for s in args.t_exclude_json.split(","))):
        with open(jf) as fp:
            for s in json.load(fp).get("scenes", []):
                if s.get("src_t"):
                    excl.add(os.path.basename(s["src_t"]))
    t_files = {name: [f for f in glob_images(d, args.t_extra_glob if name == args.t_extra_name else "*")
                      if os.path.basename(f) not in excl]
               for name, d in t_pools}
    for name, fs in t_files.items():
        print(f"T[{name}]: {len(fs)}" + (f" (已排除 {len(excl)} 张泄漏源)" if excl else ""))
        assert fs, f"T 源为空: {name}"
    metas = [m.strip() for m in args.test_meta.split(",") if m.strip()]
    excl = set().union(*(test_scene_ids(m) for m in metas)) if metas else set()
    r_files = pool_files(args.r_pool, excl)
    print(f"R pool: {len(r_files)} (已排除 {len(metas)} 个测试集的 R 源, 共 {len(excl)} 个)")

    def sample_sources(rng):
        ti = int(rng.choice(len(t_pools), p=t_weights))
        t_name = t_pools[ti][0]
        t_f = t_files[t_name][int(rng.integers(0, len(t_files[t_name])))]
        r_f = r_files[int(rng.integers(0, len(r_files)))]
        return t_name, t_f, r_f

    if args.preview:
        os.makedirs(args.out, exist_ok=True)
        rows = []
        for i in range(args.preview):
            rng = np.random.default_rng(args.seed * 1000003 + i)
            t_name, t_f, r_f = sample_sources(rng)
            t, r, m, g, out = synth_scene(r_f, t_f, rng, args.size, args.spatial,
                                          args.noise, args.r_sat, args.cover_frac,
                                          args.seed + 999 + i)
            tag = f"{os.path.basename(t_f)[:14]}|{os.path.basename(r_f)[:14]}"
            row = [(t, f"T {tag}"), (r, "R_eff")]
            for lv in LEVELS:
                a, im = out[lv]
                row += [(im, f"I a={lv:g}"), (gray3(a), f"alpha={lv:g}")]
            row += [(gray3(m), "m"), (gray3((g - g.min()) / (np.ptp(g) + 1e-6)), "g(minmax)")]
            rows.append(row)
        p = os.path.join(os.path.dirname(args.out.rstrip("/")),
                         os.path.basename(args.out.rstrip("/")) + "_preview.png")
        make_sheet(rows, p)
        return

    for sub in ["I", "T", "R", "alpha", "m", "g"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    samples, scenes = [], []
    n_redraws, n_still_over = 0, 0
    for i in range(args.n_scenes):
        sc_id = f"s{i:04d}"
        retry = 0
        while True:
            rng = np.random.default_rng(args.seed * 1000003 + i + retry * 7919)
            t_name, t_f, r_f = sample_sources(rng)
            noise_seed = args.seed + 999 + i + retry * 7919
            t, r, m, g, out = synth_scene(r_f, t_f, rng, args.size, args.spatial,
                                          args.noise, args.r_sat, args.cover_frac, noise_seed)
            clip_l100 = float((out[1.0][1] > 1.0 - 1e-6).mean())
            if args.clip_cap <= 0 or clip_l100 <= args.clip_cap or retry >= args.clip_max_retry:
                break
            retry += 1
        n_redraws += retry
        if clip_l100 > args.clip_cap > 0:
            n_still_over += 1
        save(t, os.path.join(args.out, "T", sc_id + ".png"))
        save(r, os.path.join(args.out, "R", sc_id + ".png"))
        np.save(os.path.join(args.out, "m", sc_id + ".npy"), m.astype(np.float16))
        np.save(os.path.join(args.out, "g", sc_id + ".npy"), g.astype(np.float16))
        t_src = f"{t_name}:{os.path.basename(t_f)}"
        r_src = os.path.basename(r_f)
        for lv in LEVELS:
            a, im = out[lv]
            sid = f"{sc_id}_{LEVEL_TAGS[lv]}"
            save(im, os.path.join(args.out, "I", sid + ".png"))
            np.save(os.path.join(args.out, "alpha", sid + ".npy"), a.astype(np.float16))
            samples.append({"sid": sid, "scene_id": sc_id, "level": lv,
                            "t_src": t_src, "r_src": r_src,
                            "clip_frac": float((im > 1.0 - 1e-6).mean()),
                            "alpha_eff": float(a.mean()), "r_mean": float(r.mean()),
                            "noise_seed": noise_seed})
        scenes.append({"scene_id": sc_id, "t_src": t_src, "r_src": r_src})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{args.n_scenes} scenes")

    meta = {"n_scenes": args.n_scenes, "n": len(samples), "size": args.size,
            "seed": args.seed, "levels": LEVELS, "spatial": args.spatial,
            "noise": args.noise, "r_sat": args.r_sat, "cover_frac": args.cover_frac,
            "r_pool": args.r_pool, "r_pool_n": len(r_files),
            "test_meta": args.test_meta, "t_weights": list(t_weights),
            "physics": "v2.1: I = alpha(x)*T + R_eff + n (p=0); 三档共享 m/f/R_eff/噪声",
            "g_def": "g = 1 + s*f_arc; alpha = clip(level*g*m + (1-m), 0.01, 1.0)",
            "clip_cap": args.clip_cap, "clip_max_retry": args.clip_max_retry,
            "redraws": {"total": n_redraws, "still_over": n_still_over},
            "scenes": scenes, "samples": samples}
    with open(os.path.join(args.out, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=1, ensure_ascii=False)
    print(f"done -> {args.out}  scenes={args.n_scenes} samples={len(samples)}  "
          f"redraws={n_redraws} still_over={n_still_over}")
    for lv in LEVELS:
        sel = [s for s in samples if s["level"] == lv]
        ae = np.mean([s["alpha_eff"] for s in sel])
        cf = np.array([s["clip_frac"] for s in sel])
        print(f"  档{lv:g}: α_eff={ae:.3f}  clip_frac mean={cf.mean() * 100:.2f}% "
              f"median={np.median(cf) * 100:.2f}% max={cf.max() * 100:.2f}%")


if __name__ == "__main__":
    main()
