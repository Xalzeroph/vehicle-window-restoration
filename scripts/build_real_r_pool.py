#!/usr/bin/env python3
"""从标准 SIRR 数据集提取真实反射残差池 R = clip(blended − transmission_layer, 0, 1)。

依据: nature20/real20/SIR2 的 (blended, transmission_layer) 是同机位实测对,
残差近似真实玻璃反射层(含对齐/曝光残差,负值截断)。输出 PNG 存入 out/ 供合成器当 R 池。
"""
import argparse, glob, os
import numpy as np
from PIL import Image

PAIRS = [
    ("nature20", "/home/courseliu/mycode/datasets/nature/nature20/blended",
     "/home/courseliu/mycode/datasets/nature/nature20/transmission"),
    ("real20", "/home/courseliu/mycode/datasets/real/real20/blended",
     "/home/courseliu/mycode/datasets/real/real20/transmission_layer"),
    ("sir2_solid200", "/home/courseliu/mycode/datasets/SIR2/SolidObjectDataset/solid200/blended",
     "/home/courseliu/mycode/datasets/SIR2/SolidObjectDataset/solid200/transmission_layer"),
    ("sir2_wild55", "/home/courseliu/mycode/datasets/SIR2/Wildscene/wild55/blended",
     "/home/courseliu/mycode/datasets/SIR2/Wildscene/wild55/transmission_layer"),
    ("sir2_postcard199", "/home/courseliu/mycode/datasets/SIR2/postcard/postcard199/blended",
     "/home/courseliu/mycode/datasets/SIR2/postcard/postcard199/transmission_layer"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for name, bd, td in PAIRS:
        if not os.path.isdir(bd) or not os.path.isdir(td):
            print(f"[skip] {name}: dir 缺失")
            continue
        bs = {os.path.basename(f): f for f in glob.glob(bd + "/*")}
        ts = {os.path.basename(f): f for f in glob.glob(td + "/*")}
        keys = sorted(set(bs) & set(ts))
        n = 0
        for k in keys:
            B = np.asarray(Image.open(bs[k]).convert("RGB"), np.float32) / 255.0
            T = np.asarray(Image.open(ts[k]).convert("RGB"), np.float32) / 255.0
            if B.shape != T.shape:
                continue
            R = np.clip(B - T, 0.0, None)
            Image.fromarray((R * 255).round().astype(np.uint8)).save(
                os.path.join(args.out, f"{name}_{n:04d}.png"))
            n += 1
        total += n
        print(f"{name}: {n} 张 (keys={len(keys)})")
    print(f"total: {total} -> {args.out}")


if __name__ == "__main__":
    main()
