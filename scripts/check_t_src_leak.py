#!/usr/bin/env python3
"""D0.2 泄漏核查: 训练 T 源池(sir2/nature/scenes) vs v2.1 测试场景 T 源(augmented)。

方法:
  ① 文件字节 md5 精确重复;
  ② 感知近重复: 64×64 灰度 z-score 归一化, 对测试图做全部 8 个 D4 变换
     (90° 旋转 × 翻转)后与训练源取最大归一化相关 r; r 对亮度/对比度/朝向不敏感。
输出: 每测试场景 top-3 匹配与全分布分位; 存在 r>--thresh 配对时退出码 1。

用法:
  python3 scripts/check_t_src_leak.py
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


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-meta", default="/home/zmy/vehicle_ssd/e1_testset/meta.json")
    ap.add_argument("--thresh", type=float, default=0.95)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    with open(args.e1_meta) as fp:
        test_files = [s["src_t"] for s in json.load(fp)["scenes"]]
    test_files = [f for f in test_files if os.path.exists(f)]
    print(f"测试 T 源: {len(test_files)}")
    assert test_files, "e1 meta 中 src_t 全部缺失"

    train_files = []
    for name, d in DEFAULT_T_SRC:
        fs = sorted(f for f in glob.glob(os.path.join(d, "*"))
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
        print(f"T[{name}]: {len(fs)}")
        assert fs, f"T 源为空: {name}"
        train_files += fs

    test_md5 = {md5(f): f for f in test_files}
    dup = [f for f in train_files if md5(f) in test_md5]
    print(f"[①] 字节级重复: {len(dup)}"
          + ("" if not dup else "  -> " + ", ".join(os.path.basename(f) for f in dup[:5])))

    tr = np.stack([feat64(f) for f in train_files]).reshape(len(train_files), -1)  # N×4096
    te = []
    for f in test_files:
        x = feat64(f)
        te += [np.asarray(y, np.float32).ravel() for y in d4(x)]
    te = np.stack(te)                                                              # M×4096
    scores = (te @ tr.T) / tr.shape[1]                                             # 归一化相关 M×N
    scores = scores.reshape(len(test_files), 8, len(train_files)).max(1)           # 每测试图取最优朝向

    best = scores.max(1)
    print(f"[②] 最优朝向最大相关 r: mean={best.mean():.3f} p50={np.median(best):.3f} "
          f"max={best.max():.3f}")
    q = np.percentile(scores, [50, 99, 99.9, 100])
    print(f"    全配对分布: p50={q[0]:.3f} p99={q[1]:.3f} p99.9={q[2]:.3f} max={q[3]:.3f}")

    cum = np.sort(best)
    for t in (0.90, 0.95, 0.99):
        print(f"    r>{t}: {(best > t).sum()}/{len(best)} 场景")

    n_over = int((best > args.thresh).sum())
    if n_over:
        print(f"--- 超阈配对 (r>{args.thresh}), 每场景 top{args.topk}:")
        for i in np.argsort(-best)[:n_over]:
            order = np.argsort(-scores[i])[:args.topk]
            print(f"  {os.path.basename(test_files[i])}  r={best[i]:.4f}")
            for j in order:
                if scores[i, j] < 0.80:
                    break
                print(f"      {scores[i, j]:.4f}  {train_files[j]}")
    else:
        print(f"--- 无超阈配对 (r>{args.thresh}); 最大 {best.max():.4f} "
              f"({os.path.basename(test_files[int(np.argmax(best))])} vs "
              f"{train_files[int(np.argmax(scores[int(np.argmax(best))]))]})")

    # ③ 裁剪级: 测试图是否为候选训练源的某块裁剪(64² 窗口, stride 8, D4)
    print("[③] 裁剪级检测 (top-3 候选 × 64² 窗口):")
    worst = (0.0, "", "")
    for i in range(len(test_files)):
        q = feat64(test_files[i])
        cands = np.argsort(-scores[i])[:args.topk]
        rmax, who = 0.0, ""
        for j in cands:
            im = Image.open(train_files[j]).convert("L")
            for scale in (128, 256, 512):
                c = np.asarray(im.resize((scale, scale), Image.LANCZOS), np.float32)
                if scale < 64:
                    continue
                for o in d4(q):
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
                        who = f"{os.path.basename(train_files[j])}@{scale}px,win{pos[k]}"
        if rmax > worst[0]:
            worst = (rmax, os.path.basename(test_files[i]), who)
        print(f"    {os.path.basename(test_files[i])}: max r={rmax:.3f}  <- {who}")
    print(f"[③] 裁剪级最大 r={worst[0]:.4f} ({worst[1]} <- {worst[2]})")
    raise SystemExit(1 if (n_over or worst[0] > args.thresh) else 0)


if __name__ == "__main__":
    main()
