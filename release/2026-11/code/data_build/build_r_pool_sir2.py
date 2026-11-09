"""构建"无着色偏"的反射层（R）池：只取公认公开数据集（SIR² / Wan et al. ICCV 2017）。

排除项（用户 2026-10-29 定案）：
  ✗ 我们自采的 4 月数据（透过贴膜玻璃，有色偏）
  ✗ RRW dataset（同为贴膜玻璃，有色偏）

来源与派生方式：
  G1 SIR/blended − SIR/transmission_layer        （499，有符号反射层）
  G2 postcard199/blended − postcard199/transmission_layer   （~199）
  G3 Wildscene/*/r.jpg                            （102，实拍反射图）
  G4 SolidObjectDataset/<obj>/<Focus|Thick>/{...} （按目录内 I、T 推 R，若存在）

色偏过滤：逐图统计三通道均值的相对离散度
    cast = (max_c mu_c - min_c mu_c) / mean_c mu_c
丢弃 cast > --max-cast（默认 0.12），并把统计写进 meta.json。

用法：
  python build_r_pool_sir2.py --out data/pools/r_sir2_1029
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

SIR2 = "/DataDisk/zmy_code/datasets/SIR2"


def cast_score(img):
    """逐图"色偏贡献"：LAB 的 a*/b* 均值相对中性轴的偏移量（0 = 中性）。

    注意：反射层本身会带有被反射场景的颜色（绿草地反光是绿的），这不是色偏。
    真正的"色偏"是**整池**在 a*/b* 上的一致偏移（例如整批透过茶色贴膜拍摄）。
    故这里只记录逐图值，池级偏移在 main() 末尾汇总。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_star = lab[..., 1].mean() - 128.0
    b_star = lab[..., 2].mean() - 128.0
    return float(np.hypot(a_star, b_star) / 127.0)


def save(out_dir, name, img, group, cast, src):
    cv2.imwrite(os.path.join(out_dir, name), img)
    return dict(name=name, group=group, cast=cast, src=src,
                h=int(img.shape[0]), w=int(img.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pools/r_sir2_1029")
    ap.add_argument("--max-cast", type=float, default=0.45)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta, dropped = [], 0

    # ---- G1 / G2：blended - T ----
    for gname, d in [("sir2_sir", os.path.join(SIR2, "SIR")),
                     ("sir2_postcard", os.path.join(SIR2, "postcard/postcard199"))]:
        bd, td = os.path.join(d, "blended"), os.path.join(d, "transmission_layer")
        if not (os.path.isdir(bd) and os.path.isdir(td)):
            print(f"  跳过 {gname}（缺 blended/transmission_layer）")
            continue
        common = sorted(set(os.listdir(bd)) & set(os.listdir(td)))
        n = 0
        for f in common:
            B = cv2.imread(os.path.join(bd, f))
            T = cv2.imread(os.path.join(td, f))
            if B is None or T is None or B.shape != T.shape:
                continue
            R = np.clip(B.astype(np.float32) - T.astype(np.float32), 0, 255).astype(np.uint8)
            c = cast_score(R)
            if c > a.max_cast:
                dropped += 1
                continue
            meta.append(save(a.out, f"{gname}__{f}", R, gname, c, "blended-T"))
            n += 1
        print(f"  {gname}: {n} 张（候选 {len(common)}）")

    # ---- G3：Wildscene 的 r.jpg ----
    ws = os.path.join(SIR2, "Wildscene")
    sc = sorted(os.listdir(ws)) if os.path.isdir(ws) else []
    n = 0
    for s in sc:
        p = os.path.join(ws, s, "r.jpg")
        if not os.path.exists(p):
            continue
        R = cv2.imread(p)
        if R is None:
            continue
        c = cast_score(R)
        if c > a.max_cast:
            dropped += 1
            continue
        meta.append(save(a.out, f"wildscene__{s}.png", R, "sir2_wildscene", c, "r.jpg"))
        n += 1
    print(f"  sir2_wildscene: {n} 张（场景 {len(sc)}）")

    # ---- G4：SolidObjectDataset（有 I/T 推 R，否则用 I 当反射图）----
    so = os.path.join(SIR2, "SolidObjectDataset")
    n = 0
    if os.path.isdir(so):
        for obj in sorted(os.listdir(so)):
            od = os.path.join(so, obj)
            if not os.path.isdir(od):
                continue
            for cond in ("Focus", "Thick"):
                cd = os.path.join(od, cond)
                if not os.path.isdir(cd):
                    continue
                for f in sorted(glob.glob(os.path.join(cd, "*")))[:1]:
                    R = cv2.imread(f)
                    if R is None:
                        continue
                    c = cast_score(R)
                    if c > a.max_cast:
                        dropped += 1
                        continue
                    meta.append(save(a.out, f"solidobj__{obj}_{cond}.png", R,
                                     "sir2_solidobj", c, os.path.basename(f)))
                    n += 1
    print(f"  sir2_solidobj: {n} 张")

    json.dump(dict(n=len(meta), dropped_cast=dropped, max_cast=a.max_cast,
                   sources=["SIR2 (Wan et al., ICCV 2017): SIR/blended-T, postcard199/blended-T, "
                            "Wildscene/r.jpg, SolidObjectDataset"],
                   excluded=["self-captured 2026-04 (tinted glass, colour cast)",
                             "RRW dataset (tinted glass, colour cast)"],
                   items=meta),
              open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    casts = [m["cast"] for m in meta]
    print(f"\n合计 {len(meta)} 张（因色偏丢弃 {dropped}）")
    print(f"  色偏分数 cast: 中位 {np.median(casts):.3f}  p90 {np.percentile(casts, 90):.3f}  max {max(casts):.3f}")
    for g in sorted({m['group'] for m in meta}):
        v = [m["cast"] for m in meta if m["group"] == g]
        print(f"    {g:<18} n={len(v):4d}  cast 中位 {np.median(v):.3f}")
    # ---- 池级色偏对比：SIR² vs RRW vs 自采 ----
    def pool_tint(paths, label):
        if not paths:
            print(f"    {label:<22} (无样本)")
            return
        A, B, N = [], [], []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                continue
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
            A.append(lab[..., 1].mean() - 128.0)
            B.append(lab[..., 2].mean() - 128.0)
            N.append(lab[..., 0].mean())
        print(f"    {label:<22} n={len(A):4d}  a*={np.mean(A):+6.2f}  b*={np.mean(B):+6.2f}  "
              f"偏移={np.hypot(np.mean(A), np.mean(B)):5.2f}  L={np.mean(N):5.1f}")

    print("\n  池级色偏对比（a*/b* 应为 0；偏移越大越像有色偏）：")
    pool_tint(sorted(glob.glob(os.path.join(a.out, "*.png"))), "SIR2（本次新建）")
    pool_tint(sorted(glob.glob(f"{os.environ.get('TINTED_ROOT', '/DataDisk/zmy_code/TintedGlass_Enhance')}"
                               "/data/pools/r_signed_0920/rrw/*.png")), "RRW（已排除）")
    pool_tint(sorted(glob.glob(f"{os.environ.get('TINTED_ROOT', '/DataDisk/zmy_code/TintedGlass_Enhance')}"
                               "/data/train_sets/train_pairs_omega_0920/R/*.png")), "自采 4 月（已排除）")
    print(f"  写出 {a.out}/meta.json")


if __name__ == "__main__":
    main()
