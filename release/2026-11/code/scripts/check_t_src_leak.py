#!/usr/bin/env python3
"""训练 T 源池 vs 测试集 T 的**内容级**泄漏核查（2026-09-15 升级：md5 + 缩放派生）。

背景：augmented 池曾发生"516 个文件只有 12 种内容、且全部 = 测试集 T（gt 为其精确缩放）"
的泄漏事故；事故前按**文件名**排除测试源完全无效 → 本脚本一切以**内容**为准。

检查项:
  [0] 池去重统计: N 文件 → K 唯一内容(md5); 有重复即警告（禁止按文件名排除/留出）
  [1] 字节级重复: 测试源文件 md5 命中池中任意文件（别名文件全部列出）
  [2] 缩放派生匹配: 测试图 vs 唯一池内容, 池内容缩放到测试图尺寸后 MSE < 阈值 → 泄漏
      （事故复现: v2.1 gt = label 精确缩放, MSE=0）
  [3] 感知近重复: 64² 灰度 z-score, 8 个 D4 变换最大归一化相关 r > 阈值
  [4] 裁剪级: 测试图是否为候选训练源的 64² 窗口裁剪

用法:
  # 默认: 富池(sir2/nature/scenes) vs e1_testset 测试源
  python3 scripts/check_t_src_leak.py
  # 任意池(如 augmented) vs 测试集 gt 图:
  python3 scripts/check_t_src_leak.py \
      --t-src aug=/home/zmy/vehicle_ssd/datasets/augmented --t-glob '*_label.jpg' \
      --ts-dir data/test_sets/synth_rrw_v21_0913/gt
退出码: 1 = 发现泄漏
"""
import argparse, glob, hashlib, json, os
import numpy as np
from PIL import Image

DEFAULT_T_SRC = [
    ("sir2", "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer"),
    ("nature", "/home/zmy/mycode/datasets/nature/natural_T"),
    ("scenes", "/home/zmy/vehicle_ssd/datasets/scenes"),
]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def resolve(p):
    """旧机路径重写（e1_testset/meta.json 在项目外，未随 2026-09-14 搬迁同步）。"""
    if not os.path.exists(p) and p.startswith("/home/courseliu/"):
        alt = p.replace("/home/courseliu/", "/home/zmy/", 1)
        if os.path.exists(alt):
            return alt
    return p


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def unique_contents(files):
    """md5 -> [文件...]（首元素为展示用代表）。"""
    groups = {}
    for f in files:
        groups.setdefault(md5(f), []).append(f)
    return groups


def feat64(path):
    im = Image.open(path).convert("L").resize((64, 64), Image.LANCZOS)
    x = np.asarray(im, np.float32)
    x = x - x.mean()
    s = x.std()
    return (x / s) if s > 1e-6 else x * 0.0


def d4(x):
    out = []
    for k in range(4):
        y = np.rot90(x, k)
        out.append(y)
        out.append(np.flipud(y))
    return out


def parse_pools(pairs):
    if not pairs:
        return list(DEFAULT_T_SRC)
    out = []
    for s in pairs:
        name, sep, d = s.partition("=")
        assert sep and d, f"--t-src 需 NAME=DIR 形式: {s}"
        out.append((name, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-src", action="append", default=[], metavar="NAME=DIR",
                    help="训练 T 池（可重复）；默认 sir2/nature/scenes")
    ap.add_argument("--t-glob", default="*", help="池文件通配，如 '*_label.jpg'")
    ap.add_argument("--e1-meta", default="/home/zmy/vehicle_ssd/e1_testset/meta.json")
    ap.add_argument("--ts-dir", default="", help="测试图目录（如 data/test_sets/synth_rrw_v21_0913/gt）；优先于 e1-meta")
    ap.add_argument("--thresh", type=float, default=0.95, help="感知/裁剪级 r 阈值")
    ap.add_argument("--mse-thresh", type=float, default=1.0, help="[2] 缩放派生 MSE 阈值 (0-255)")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--skip-crop", action="store_true", help="跳过 [4] 裁剪级（大池快速跑）")
    args = ap.parse_args()

    if args.ts_dir:
        test_files = sorted(f for f in glob.glob(os.path.join(args.ts_dir, "*"))
                            if os.path.splitext(f)[1].lower() in IMG_EXT)
        print(f"测试图（--ts-dir）: {len(test_files)}")
    else:
        with open(args.e1_meta) as fp:
            test_files = [resolve(s["src_t"]) for s in json.load(fp)["scenes"]]
        missing = [f for f in test_files if not os.path.exists(f)]
        test_files = [f for f in test_files if os.path.exists(f)]
        print(f"测试 T 源（e1 meta）: {len(test_files)}")
        if missing:
            print(f"    ⚠️ {len(missing)} 个路径不存在，例: {missing[0]}")
    assert test_files, "测试文件为空"

    train_files = []
    for name, d in parse_pools(args.t_src):
        fs = sorted(f for f in glob.glob(os.path.join(d, args.t_glob), recursive=True)
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
        print(f"T[{name}]: {len(fs)}  ({d})")
        assert fs, f"T 源为空: {name}"
        train_files += fs

    # [0] 池内容去重统计（事故根因检测：内容重复 ⇒ 文件名排除/留出无效）
    groups = unique_contents(train_files)
    n_file, n_uniq = len(train_files), len(groups)
    print(f"\n[0] 池去重统计: {n_file} 文件 → {n_uniq} 种内容")
    dup_leak0 = n_uniq < n_file
    if dup_leak0:
        top = sorted(groups.values(), key=len, reverse=True)[:5]
        print(f"    ⚠️ 存在内容重复（每种内容最多 {len(top[0])} 个文件名）——"
              f"按文件名排除测试源无效！重复内容代表:")
        for fs in top:
            print(f"      x{len(fs):3d}  {os.path.basename(fs[0])}  例: "
                  + ", ".join(os.path.basename(f) for f in fs[:3]))
    else:
        print("    内容与文件一一对应")
    uniq_reps = [fs[0] for fs in groups.values()]

    # [1] 字节级重复
    test_md5 = {md5(f): f for f in test_files}
    dup = [(f, test_md5[md5(f)]) for f in train_files if md5(f) in test_md5]
    print(f"\n[1] 字节级重复: {len(dup)} 对")
    for f, t in dup[:20]:
        print(f"    {os.path.basename(t)}  ==  {f}")
    if len(dup) > 20:
        print(f"    ... 及另外 {len(dup)-20} 对")

    # [2] 缩放派生匹配（内容级：两侧统一缩到 96² 再逐像素比；派生对在任何尺度都吻合，
    #     事故对在 96² 下 MSE≈0.1（0-255 尺度），远低于阈值）。分块向量化避免爆内存。
    SZ = 96
    def small(p):
        return np.asarray(Image.open(p).convert("RGB").resize((SZ, SZ), Image.LANCZOS),
                          np.float32)
    tests_small = [small(f) for f in test_files]
    best_mse = np.full(len(test_files), np.inf, np.float64)
    best_rep = [""] * len(test_files)
    for i0 in range(0, len(uniq_reps), 512):
        chunk = uniq_reps[i0:i0 + 512]
        arr = np.stack([small(r) for r in chunk])
        for ti in range(len(test_files)):
            m = ((arr - tests_small[ti]) ** 2).mean(axis=(1, 2, 3))
            k = int(np.argmin(m))
            if m[k] < best_mse[ti]:
                best_mse[ti], best_rep[ti] = float(m[k]), chunk[k]
    print(f"\n[2] 缩放派生匹配（96² 统一尺度, MSE<{args.mse_thresh} 判泄漏）:")
    leak2 = []
    for ti, tf in enumerate(test_files):
        flag = "  <<< 泄漏!" if best_mse[ti] < args.mse_thresh else ""
        print(f"    {os.path.basename(tf):16s} best MSE={best_mse[ti]:9.4f}  "
              f"{os.path.basename(best_rep[ti])}{flag}")
        if best_mse[ti] < args.mse_thresh:
            leak2.append((tf, best_rep[ti]))

    # [3] 感知近重复（对唯一内容，去重池不重复计分）
    tr = np.stack([feat64(f) for f in uniq_reps]).reshape(len(uniq_reps), -1)
    te = []
    for f in test_files:
        x = feat64(f)
        te += [np.asarray(y, np.float32).ravel() for y in d4(x)]
    te = np.stack(te)
    scores = (te @ tr.T) / tr.shape[1]
    scores = scores.reshape(len(test_files), 8, len(uniq_reps)).max(1)

    best = scores.max(1)
    print(f"\n[3] 感知近重复: 最优朝向最大相关 r: mean={best.mean():.3f} "
          f"p50={np.median(best):.3f} max={best.max():.3f}")
    q = np.percentile(scores, [50, 99, 99.9, 100])
    print(f"    全配对分布: p50={q[0]:.3f} p99={q[1]:.3f} p99.9={q[2]:.3f} max={q[3]:.3f}")
    for t in (0.90, 0.95, 0.99):
        print(f"    r>{t}: {(best > t).sum()}/{len(best)} 场景")

    n_over = int((best > args.thresh).sum())
    if n_over:
        print(f"--- 超阈配对 (r>{args.thresh}), 每场景 top{args.topk}:")
        for i in np.argsort(-best)[:n_over]:
            print(f"  {os.path.basename(test_files[i])}  r={best[i]:.4f}")
            for j in np.argsort(-scores[i])[:args.topk]:
                if scores[i, j] < 0.80:
                    break
                print(f"      {scores[i, j]:.4f}  {uniq_reps[j]}")

    # [4] 裁剪级
    worst = (0.0, "", "")
    if args.skip_crop:
        print("\n[4] 裁剪级: 已跳过 (--skip-crop)")
    else:
        print("[4] 裁剪级检测 (top-3 候选 × 64² 窗口):")
        for i in range(len(test_files)):
            qf = feat64(test_files[i])
            cands = np.argsort(-scores[i])[:args.topk]
            rmax, who = 0.0, ""
            for j in cands:
                im = Image.open(uniq_reps[j]).convert("L")
                for scale in (128, 256, 512):
                    c = np.asarray(im.resize((scale, scale), Image.LANCZOS), np.float32)
                    for o in d4(qf):
                        o = np.asarray(o, np.float32)
                        pw = c - c.mean()
                        feats, pos = [], []
                        for y in range(0, scale - 64 + 1, 8):
                            for x in range(0, scale - 64 + 1, 8):
                                w = pw[y:y + 64, x:x + 64]
                                s = w.std()
                                if s < 1e-3:
                                    continue
                                feats.append((w / s).ravel())
                                pos.append((y, x))
                        if not feats:
                            continue
                        feats = np.stack(feats)
                        r = (feats @ o.ravel()) / o.size
                        k = int(np.argmax(r))
                        if r[k] > rmax:
                            rmax = float(r[k])
                            who = f"{os.path.basename(uniq_reps[j])}@{scale}px,win{pos[k]}"
            if rmax > worst[0]:
                worst = (rmax, os.path.basename(test_files[i]), who)
            print(f"    {os.path.basename(test_files[i])}: max r={rmax:.3f}  <- {who}")
        print(f"[4] 裁剪级最大 r={worst[0]:.4f} ({worst[1]} <- {worst[2]})")

    # 去重统计只作卫生警告；泄漏判定以"池内容 ∩ 测试内容"为准
    leaked = bool(dup) or bool(leak2) or n_over > 0 or (worst[0] > args.thresh)
    if leaked:
        print("\n=== 结论: ⚠️ 发现泄漏（见上方 <<< 标记） ===")
    else:
        msg = "未发现泄漏"
        if dup_leak0:
            msg += f"（但池有内容重复 {n_file}→{n_uniq}，不可按文件名做排除/留出）"
        print(f"\n=== 结论: {msg} ===")
    raise SystemExit(1 if leaked else 0)


if __name__ == "__main__":
    main()
