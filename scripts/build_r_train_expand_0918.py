#!/usr/bin/env python3
"""R 训练扩池清洁（2026-09-18）：旧 601 候选 + alt 层 + RRW 窗口扩池 → 内容簇去重 + 双口径安全网。

与 `build_r_train_pool.py`（9.15 版，产出 r_train_0915 131 张）同款流程，本脚本**只读**旧池、
不改旧池；差异是落实 9.15 评审的已知改进项：

  ① 安全网判定口径提升到**源域 / 128²**：候选与测试 R 源都取"源域表示"（wsoft → 原始
     r_world_soft/*.jpg；RRW 层 → r_rrw_strong/raw/*.png 或扩池 raw/*.png）后 128² 灰度 MSE。
     理由（本机实测）：池域（重标定 PNG/256²）非孪生中位 MSE=0.0015、近孪生簇 <5e-5，仅
     ~30× 分离；源域（未缩放/128²）非孪生中位 MSE=0.0235、近孪生簇 <5e-5，~500× 分离。
     alt 层无源域文件，回退用池文件（本批 31/31，已记录）。
  ② 旧口径（池 PNG / 256², tau=0.002）同时计算并记录，供对照。
  ③ 每个被安全网排除的样本记录其"最小 MSE − tau"余量；保留样本记录最小余量。

规则（同 9.15）：
  簇化 64² z-score 贪心聚类 r>--thresh(0.95) → 含测试源的簇整簇剔除 → 冗余副本每簇留 1 代表
  （代表优先级：RRW 场景层 > RRW 窗口层 > alt > wsoft）→ 安全网复核 → 输出干净训练池。

用法:
  python3 scripts/build_r_train_expand_0918.py [--dry-run] [--no-alt] [--tau-src 0.002]
"""
import argparse, csv, glob, json, os, shutil
import numpy as np
from PIL import Image

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
OLD_POOL = os.path.join(PROJ, "data/pools/r_rrw_strong/pool")
OLD_RAW = os.path.join(PROJ, "data/pools/r_rrw_strong/raw")
EXP_POOL = os.path.join(PROJ, "data/pools/r_rrw_expand_0918/pool")
EXP_RAW = os.path.join(PROJ, "data/pools/r_rrw_expand_0918/raw")
WSOFT_SRC = "/home/zmy/vehicle_ssd/datasets/r_world_soft"
OUT = os.path.join(PROJ, "data/pools/r_train_expand_0918")
TEST_METAS = [os.path.join(PROJ, f"data/test_sets/{t}/meta.json")
              for t in ("synth_rrw_v21_0913", "synth_rrw_v21_indomain",
                        "synth_rrw_v22_scan", "synth_rrw_v23_lowscan")]

# 测试集 r/ 目录里的**渲染域**R（渲染画布 + 掩膜），只作诊断列：
# 实测（9.18）非孪生候选对该域的最小 MSE 地板 0.002062（对源域则是 0.0）→ 无判别力，
# 不进判定，避免引入第二个贴边阈值。判定只用 meta.json scenes[].src_r（源域 128²）。
TEST_RDIRS = [os.path.join(PROJ, f"data/test_sets/{t}/r")
              for t in ("synth_rrw_v21_0913", "synth_rrw_v21_indomain",
                        "synth_rrw_v22_scan", "synth_rrw_v23_lowscan")]


def feat(p, s=64):
    im = Image.open(p).convert("L").resize((s, s), Image.LANCZOS)
    x = np.asarray(im, np.float32)
    x = x - x.mean()
    sd = x.std()
    return ((x / sd).ravel() if sd > 1e-6 else x.ravel())


def gray(p, s):
    return np.asarray(Image.open(p).convert("L").resize((s, s), Image.LANCZOS), np.float32) / 255.0


def provenance(name):
    if name.endswith("_alt.png"):
        return "rrw_alt"
    if name.startswith("wsoft__"):
        return "wsoft"
    if name.count("__") >= 2:
        return "rrw_window"
    return "rrw_scene"


def src_domain(path, name):
    """候选/测试源的源域表示（未重标定域）。"""
    stem = os.path.splitext(name)[0]
    if name.startswith("wsoft__"):
        p = os.path.join(WSOFT_SRC, stem.split("__", 1)[1] + ".jpg")
        return (p, "jpg") if os.path.exists(p) else (path, "pool_fallback")
    for raw_dir in (EXP_RAW, OLD_RAW):
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            return p, "raw"
    return path, "pool_fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.95)
    ap.add_argument("--tau-src", type=float, default=0.002, help="源域 128² MSE 阈值（主判）")
    ap.add_argument("--zmin", type=float, default=0.90, help="MSE 判定的尺度不变确认阈值（z-corr）")
    ap.add_argument("--zhard", type=float, default=0.95, help="z-corr 无条件孪生阈值")
    ap.add_argument("--tau-pool", type=float, default=0.0, help="池域 256² MSE 阈值（对照，0=关）")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--meta-out", default=os.path.join(PROJ, "data/pools/r_train_expand_0918_meta.csv"))
    ap.add_argument("--no-alt", dest="alt", action="store_false", default=True,
                    help="不把旧池 *_alt.png 计入候选")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    test_src = set()
    for m in TEST_METAS:
        with open(m) as fp:
            test_src |= {os.path.basename(s["src_r"]) for s in json.load(fp)["scenes"]}
    print(f"测试 R 源（四集并集）: {len(test_src)} 个")

    groups = [("rrw_scene", sorted(f for f in glob.glob(os.path.join(OLD_POOL, "*.png"))
                                   if not f.endswith("_alt.png") and not os.path.basename(f).startswith("wsoft__"))),
              ("wsoft", sorted(glob.glob(os.path.join(OLD_POOL, "wsoft__*.png"))))]
    if args.alt:
        groups.append(("rrw_alt", sorted(glob.glob(os.path.join(OLD_POOL, "*_alt.png")))))
    groups.append(("rrw_window", sorted(glob.glob(os.path.join(EXP_POOL, "*.png")))))
    cands = [f for _, fs in groups for f in fs]
    print("候选: " + ", ".join(f"{k}={len(fs)}" for k, fs in groups) + f" → 共 {len(cands)}")

    F = {f: feat(f) for f in cands}
    clusters = []
    for f in cands:
        q = F[f]
        hit = None
        for c in clusters:
            if float(F[c["rep"]] @ q) / q.size > args.thresh:
                hit = c
                break
        if hit is None:
            clusters.append(dict(rep=f, members=[f]))
        else:
            hit["members"].append(f)
            if _rank(f) < _rank(hit["rep"]):
                hit["rep"] = f
    print(f"簇数: {len(clusters)}")

    # 测试源张量（两口径）
    T_src128, T_pool256, t100 = {}, {}, []
    for n in sorted(test_src):
        op = os.path.join(OLD_POOL, n)
        if not os.path.exists(op):
            print(f"  [warn] 测试源 {n} 不在旧池中，跳过")
            continue
        sp, kind = src_domain(op, n)
        T_src128[n] = gray(sp, 128)
        T_pool256[n] = gray(op, 256)
        t100.append((n, kind))
    n_fb = sum(1 for _, k in t100 if k == "pool_fallback")
    print(f"测试源张量: 源域128² {len(T_src128)} 个（其中池文件回退 {n_fb} 个）、池域256² {len(T_pool256)} 个")
    T_z = {n: (v - v.mean()) / max(float(v.std()), 1e-8) for n, v in T_src128.items()}

    def zcorr_max(a):
        x = (a - a.mean()) / max(float(a.std()), 1e-8)
        return max(float((x * v).mean()) for v in T_z.values())
    rends = []
    for rd in TEST_RDIRS:
        rends += [os.path.join(rd, f) for f in sorted(os.listdir(rd))]
    R_render = np.stack([gray(p, 128) for p in rends])
    print(f"渲染域诊断张量: {R_render.shape[0]} 张（只记录 min_mse_render，不进判定）")

    rows, kept, excl_test, excl_net, excl_dup = [], [], [], [], []
    for ci, c in enumerate(clusters):
        names = [os.path.basename(m) for m in c["members"]]
        rep_name = os.path.basename(c["rep"])
        if any(n in test_src for n in names):
            excl_test += c["members"]
            for n in names:
                rows.append(dict(cluster=ci, size=len(names), file=n, prov=provenance(n),
                                 action="drop_test_cluster", rep=rep_name, min_mse_src="",
                                 margin_src="", min_mse_pool="", min_mse_render="", best_src=""))
            continue
        # 安全网（双口径，取代表判）
        rp_src, kind = src_domain(c["rep"], rep_name)
        a128 = gray(rp_src, 128)
        m_src = min(((a128 - v) ** 2).mean() for v in T_src128.values())
        best_src = min(T_src128, key=lambda k: ((a128 - T_src128[k]) ** 2).mean())
        a256 = gray(c["rep"], 256)
        m_pool = min(((a256 - v) ** 2).mean() for v in T_pool256.values())
        m_rend = min(((a128 - v) ** 2).mean() for v in R_render)
        z_src = zcorr_max(a128)
        # 新口径：源域 128² MSE 主判 + 尺度不变确认（MSE 对低对比层不具可比性：
        # 实测 tau=0.002 下 38/234 假阳性，zcorr 仅 0.07–0.24；zcorr≥zhard 无条件判孪生）
        hit_src = (m_src < args.tau_src and z_src >= args.zmin) or (z_src >= args.zhard)
        hit_pool = m_pool < args.tau_pool
        if hit_src or hit_pool:
            excl_net += c["members"]
            why = ("drop_twin_src" if hit_src else "") + ("+pool" if hit_pool else "")
            for n in names:
                rows.append(dict(cluster=ci, size=len(names), file=n, prov=provenance(n), action=why,
                                 rep=rep_name, min_mse_src=round(m_src, 7),
                                 margin_src=round(m_src - args.tau_src, 7),
                                 min_mse_pool=round(m_pool, 7), zcorr_src=round(z_src, 4),
                                 min_mse_render=round(m_rend, 6), best_src=best_src))
            continue
        kept.append(c["rep"])
        excl_dup += [m for m in c["members"] if m != c["rep"]]
        for m in c["members"]:
            rows.append(dict(cluster=ci, size=len(names), file=os.path.basename(m),
                             prov=provenance(os.path.basename(m)),
                             action="keep" if m == c["rep"] else "drop_dup", rep=rep_name,
                             min_mse_src=round(m_src, 7), margin_src=round(m_src - args.tau_src, 7),
                             min_mse_pool=round(m_pool, 7), zcorr_src=round(z_src, 4),
                             min_mse_render=round(m_rend, 6), best_src=best_src))

    n_test_files = sum(1 for m in excl_test if os.path.basename(m) in test_src)
    prov_kept = {}
    for f in kept:
        prov_kept[provenance(os.path.basename(f))] = prov_kept.get(provenance(os.path.basename(f)), 0) + 1
    src_kept = np.array([r["min_mse_src"] for r in rows if r["action"] == "keep"], dtype=float)
    pool_kept = np.array([r["min_mse_pool"] for r in rows if r["action"] == "keep"], dtype=float)

    print(f"  测试簇: {sum(1 for c in clusters if any(os.path.basename(m) in test_src for m in c['members']))} "
          f"→ 剔除 {len(excl_test)} 张（其中测试源本身 {n_test_files} 张）")
    print(f"  安全网（源域128² MSE<{args.tau_src} 且 z-corr≥{args.zmin}，或 z-corr≥{args.zhard}）: "
          f"剔除 {len(excl_net)} 张" + (f"（| 池域对照 tau={args.tau_pool} 另剔除若干）"
                                        if args.tau_pool > 0 else ""))
    print(f"  冗余副本剔除: {len(excl_dup)} 张")
    print(f"  **保留: {len(kept)} 张**  构成: {prov_kept}")
    print(f"  保留样本最小余量（源域）: min={src_kept.min() - args.tau_src:.7f} "
          f"（最小 MSE {src_kept.min():.6f}）; 池域对照 min MSE={pool_kept.min():.6f}")
    bad = [r for r in rows if r["action"] == "keep" and float(r["margin_src"]) < 0.0005]
    print(f"  余量 <5e-4 的保留样本: {len(bad)}")
    for r in bad[:10]:
        print(f"     {r['file']:46s} mse_src={r['min_mse_src']} margin={r['margin_src']} twin={r['best_src']}")

    if args.dry_run:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    for f in kept:
        shutil.copy2(f, os.path.join(args.out_dir, os.path.basename(f)))
    with open(args.meta_out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["cluster", "size", "file", "prov", "action", "rep",
                                           "min_mse_src", "margin_src", "min_mse_pool", "zcorr_src",
                                           "min_mse_render", "best_src"])
        w.writeheader(); w.writerows(rows)
    print(f"-> {args.out_dir}（干净训练池）; 清单 {args.meta_out}")


def _rank(path):
    """簇代表优先级（小者优先）。"""
    n = os.path.basename(path)
    return {"rrw_scene": 0, "rrw_window": 1, "rrw_alt": 2, "wsoft": 3}[provenance(n)]


if __name__ == "__main__":
    main()
