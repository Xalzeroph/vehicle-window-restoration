#!/usr/bin/env python3
"""R 训练池内容级去重 + 测试簇剔除（2026-09-15 孪生簇泄漏修复）。

规则（按内容簇切分，不按文件名）：
  ① 对候选 R 文件（现池 38 张 + wsoft 扩池 563 张）与**全部测试 R 源**（三测试集并集）
     计算 64² 灰度 z-score 特征；
  ② 贪心聚类：与已有簇代表 r > --thresh 即入簇（单链近似）；
  ③ 含任一测试 R 源的簇 → 整簇剔除出训练池；
  ④ 其余簇每簇保留一个代表（优先保留原池文件），冗余副本剔除；
  ⑤ 输出 data/pools/r_train_0915/（干净训练池）+ 逐文件清单 CSV。

用法: python3 scripts/build_r_train_pool.py [--thresh 0.98] [--dry-run]
"""
import argparse, csv, glob, json, os, shutil
import numpy as np
from PIL import Image

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
SRC_POOL = os.path.join(PROJ, "data/pools/r_rrw_strong/pool")
OUT = os.path.join(PROJ, "data/pools/r_train_0915")
TEST_METAS = [f"{PROJ}/data/test_sets/{t}/meta.json"
              for t in ("synth_rrw_v21_0913", "synth_rrw_v21_indomain", "synth_rrw_v22_scan")]


def feat(p, s=64):
    im = Image.open(p).convert("L").resize((s, s), Image.LANCZOS)
    x = np.asarray(im, np.float32)
    x = x - x.mean()
    sd = x.std()
    return ((x / sd).ravel() if sd > 1e-6 else x.ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.95)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mse-tau", type=float, default=0.002, help="对测试源的 256² MSE 安全阈值")
    args = ap.parse_args()

    test_src = set()
    for m in TEST_METAS:
        with open(m) as fp:
            test_src |= {os.path.basename(s["src_r"]) for s in json.load(fp)["scenes"]}
    print(f"测试 R 源（三集并集）: {len(test_src)} 个")

    cands = sorted(f for f in glob.glob(os.path.join(SRC_POOL, "*.png"))
                   if not os.path.basename(f).endswith("_alt.png"))
    print(f"候选池文件: {len(cands)}（含测试源文件本身）")
    F = {f: feat(f) for f in cands}

    # 贪心聚类
    clusters = []          # list of dict(rep, members)
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
            # 簇代表优先保留原池（r_rrw_strong 命名）文件
            if (hit["rep"].find("wsoft__") >= 0) and (f.find("wsoft__") < 0):
                hit["rep"] = f

    # 安全网: 对每个保留文件与全部测试源做 256² MSE 复核（<--mse-tau 即剔除）
    def f256(p_):
        return np.asarray(Image.open(p_).convert("L").resize((256, 256), Image.LANCZOS),
                          np.float32) / 255.0
    tset = {}
    for n in test_src:
        q = os.path.join(SRC_POOL, n)
        if os.path.exists(q):
            tset[n] = f256(q)

    rows, kept, dropped_test, dropped_dup = [], [], [], []
    for ci, c in enumerate(clusters):
        names = [os.path.basename(m) for m in c["members"]]
        has_test = any(n in test_src for n in names)
        if has_test:
            dropped_test += c["members"]
            for n, m in zip(names, c["members"]):
                rows.append(dict(cluster=ci, size=len(names), file=n,
                                 action="drop_test_cluster", rep=os.path.basename(c["rep"])))
            continue
        rp = f256(c["rep"])
        twin = [n for n, tv in tset.items() if ((rp - tv) ** 2).mean() < args.mse_tau]
        if twin:
            dropped_test += c["members"]
            for n_, m_ in zip(names, c["members"]):
                rows.append(dict(cluster=ci, size=len(names), file=n_,
                                 action=f"drop_twin_of:{twin[0]}", rep=os.path.basename(c["rep"])))
            continue
        kept.append(c["rep"])
        dropped_dup += [m for m in c["members"] if m != c["rep"]]
        for m in c["members"]:
            rows.append(dict(cluster=ci, size=len(names), file=os.path.basename(m),
                             action="keep" if m == c["rep"] else "drop_dup",
                             rep=os.path.basename(c["rep"])))

    n_test_files = sum(1 for m in dropped_test if os.path.basename(m) in test_src)
    print(f"簇数: {len(clusters)}")
    print(f"  测试簇: {sum(1 for c in clusters if any(os.path.basename(m) in test_src for m in c['members']))} "
          f"→ 剔除 {len(dropped_test)} 张（其中测试源本身 {n_test_files} 张）")
    print(f"  冗余副本剔除: {len(dropped_dup)} 张")
    print(f"  **训练池保留: {len(kept)} 张**")
    if args.dry_run:
        return
    os.makedirs(OUT, exist_ok=True)
    for f in kept:
        shutil.copy2(f, os.path.join(OUT, os.path.basename(f)))
    with open(os.path.join(PROJ, "data/pools/r_train_0915_meta.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["cluster", "size", "file", "action", "rep"])
        w.writeheader(); w.writerows(rows)
    print(f"-> {OUT}（干净训练池）; 清单 data/pools/r_train_0915_meta.csv")


if __name__ == "__main__":
    main()
