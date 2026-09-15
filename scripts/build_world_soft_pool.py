#!/usr/bin/env python3
"""把 r_world_soft（真实软反射层, 575 张）按 build_rrw_pool 的强度配方扩入 R 训练池。

配方（与 build_rrw_pool.py 一致）:
  base=0.06*255=15.3; c=clip((mean_i/median)^0.5, 0.45, 1.8); rho~logU[1,2.6]（u 由
  "强度×结构"综合分位决定）; target=clip(base·c·rho, 6, 75); 缩放 clip(x·target/mean, 0, 255)。
防护: 排除 e1_testset 的测试 R 源（12 个）；对新入池文件与 测试 R / 现有池 做 md5+感知查重。

用法: python3 scripts/build_world_soft_pool.py [--dry-run]
"""
import argparse, csv, glob, hashlib, json, os
import numpy as np
from PIL import Image
from scipy.stats import rankdata

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
SRC = "/home/zmy/vehicle_ssd/datasets/r_world_soft"
POOL = os.path.join(PROJ, "data/pools/r_rrw_strong/pool")
E1_META = "/home/zmy/vehicle_ssd/e1_testset/meta.json"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def feat(p):
    im = Image.open(p).convert("L").resize((64, 64), Image.LANCZOS)
    x = np.asarray(im, np.float32)
    x = x - x.mean()
    s = x.std()
    return (x / s).ravel() if s > 1e-6 else x.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()

    with open(E1_META) as fp:
        e1_test = {os.path.basename(s["src_r"]) for s in json.load(fp)["scenes"]}
    files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
    files = [f for f in files if os.path.basename(f) not in e1_test]
    print(f"源: {len(files)} 张（已排除 e1 测试 R 源 {len(e1_test)} 个）")

    # 读图 + 统计
    arrs, means, darks, stds = [], [], [], []
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), np.float32)
        arrs.append(a); means.append(a.mean()); darks.append(float((a.mean(2) < 12).mean()))
        stds.append(a.std())
    means = np.array(means)
    m_nat = float(np.median(means))
    p = (rankdata(means) + rankdata(darks) + rankdata(stds)) / (3.0 * len(files))
    rng = np.random.default_rng(args.seed)
    lo, hi = np.log(1.0), np.log(2.6)
    base = 0.06 * 255.0

    # 查重基准: 测试 R 源 + 现有池
    pool_old = [f for f in glob.glob(os.path.join(POOL, "*.png"))]
    ref_md5 = {md5(f): f for f in pool_old}
    ref_feat = {f: feat(f) for f in pool_old}
    dup, near = [], []
    n_written = 0
    rows = []
    for i, (f, a, m, pr) in enumerate(zip(files, arrs, means, p)):
        u = float(np.clip(0.5 * rng.random() + 0.5 * pr, 0, 1))
        rho = float(np.exp(lo + u * (hi - lo)))
        c = float(np.clip((m / m_nat) ** 0.5, 0.45, 1.8))
        target = float(np.clip(base * c * rho, 6.0, 75.0))
        scale = target / max(m, 1e-6)
        out = np.clip(a * scale, 0, 255).round().astype(np.uint8)
        name = f"wsoft__{os.path.splitext(os.path.basename(f))[0]}.png"
        outp = os.path.join(POOL, name)
        if not args.dry_run:
            Image.fromarray(out).save(outp)
            n_written += 1
        h = hashlib.md5(out.tobytes()).hexdigest()
        if h in ref_md5:
            dup.append((name, ref_md5[h]))
        fv = None
        for rp, rf in ref_feat.items():
            if fv is None:
                x = np.asarray(Image.open(outp if not args.dry_run else f)
                               .convert("L").resize((64, 64), Image.LANCZOS), np.float32)
                x = x - x.mean(); s = x.std()
                fv = (x / s).ravel() if s > 1e-6 else x.ravel()
            r = float(fv @ rf / fv.size)
            if r > 0.95:
                near.append((name, os.path.basename(rp), round(r, 3)))
        rows.append(dict(src=os.path.basename(f), out=name, mean_in=round(m, 1),
                         rho=round(rho, 2), target=round(target, 1),
                         scale=round(scale, 3), clip_frac=round(float((out >= 254).mean()), 4)))

    with open(os.path.join(PROJ, "data/pools/r_world_soft_expand_meta.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    tgt = np.array([r["target"] for r in rows])
    print(f"入池: {n_written} 张 -> {POOL}/wsoft__*.png")
    print(f"强度: 中位 target={np.median(tgt):.1f}  范围 [{tgt.min():.1f},{tgt.max():.1f}]"
          f"（现有池中位 24.4 / 范围 [13.2,49.3]）")
    print(f"查重: 字节级重复 {len(dup)} 对; 感知近重复(r>0.95) {len(near)} 对")
    for n in near[:5]:
        print("   ", n)


if __name__ == "__main__":
    main()
