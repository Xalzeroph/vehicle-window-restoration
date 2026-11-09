#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1019 权威评测表：PSNR / SSIM / HF-PSNR / 梯度余弦 / LapVar + 洞区（v13）分区域（2026-10-20）。

为什么要有这一版
---------------
此前各指标分散在多个脚本里、口径不一，且 LapVar 会把"反射边缘"算成"细节"。
本脚本把**能反映不同失效面的一整套指标**一次算齐，并对 v13 洞区单独报。

指标
----
| 指标 | 衡量什么 | 说明 |
|---|---|---|
| PSNR / SSIM | 综合保真 | 与文献可比 |
| **HF-PSNR** | **高频内容对不对** | 只在高通带（σ=2 高斯）算 PSNR，不受"反射边缘多少"干扰 |
| **梯度余弦** | **结构方向保没保对** | ∇T̂ 与 ∇T 在强梯度处的方向一致性，1=完全一致 |
| LapVar | 高频**能量**多少 | 辅助；会混入反射边缘 |
| masked PSNR（洞区/非洞区） | 反射弱但有边缘的困难区 | 掩膜由 v13 定义；**必须配 identity 基线看相对值**（否则循环） |

输出
----
eval/table_final_1019.json / .csv / .md
eval/fig_1019/table_final.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
EVAL = os.path.join(P, "eval")
FIG = os.path.join(EVAL, "fig_1019")

ARMS_DEFAULT = [("identity", None),
                ("rdnet_zs", "real_pair_1019_rdnet_zs"),
                ("ft_s2main", "real_pair_1019_ft_s2main"),
                ("ft_mixbest", "real_pair_1019_ft_mixbest"),
                ("lightrr", "real_pair_1019_lightrr")]

# 由 --pred-map 覆盖：JSON {"arm_name": "pred_subdir_or_abs_path", ...}
# 新增臂（GCF / 归属头 / 级联）只需生成 runs/pred/<subdir>/ 并在 pred-map 里登记，
# 不必改本文件。identity 恒在首位（基线）。
ARMS = list(ARMS_DEFAULT)

# 洞区（v13）参数
LAMBDA_R = 12.0
T_FLOOR = 20.0
T_HAS_REFL = 0.15
SMOOTH_R = 5
MIN_AREA = 200
ERODE_PX = 2

SIG_LP = 2.0      # 高通用的高斯尺度


# ---------------- 基础 ----------------
def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def gray(x):
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def highpass(g):
    return g - cv2.GaussianBlur(g, (0, 0), SIG_LP)


def psnr_gray(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def psnr_rgb(a, b):
    return float(np.mean([psnr_gray(a[..., c], b[..., c]) for c in range(3)]))


def psnr_masked(a, b, mask):
    if mask.sum() < 50:
        return float("nan")
    v = []
    for c in range(3):
        av = a[..., c][mask].astype(np.float64)
        bv = b[..., c][mask].astype(np.float64)
        mse = float(np.mean((av - bv) ** 2))
        v.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


def ssim_rgb(a, b):
    return float(np.mean([ssim_fn(a[..., c], b[..., c], data_range=255)
                          for c in range(3)]))


def lapvar(x):
    return float(cv2.Laplacian(gray(x), cv2.CV_32F).var(dtype=np.float64))


def grad_cos(a_g, b_g, topk=0.15):
    ax, ay = np.gradient(a_g)
    bx, by = np.gradient(b_g)
    ma, mb = np.hypot(ax, ay), np.hypot(bx, by)
    m = (ma >= np.percentile(ma, (1 - topk) * 100)) & \
        (mb >= np.percentile(mb, (1 - topk) * 100))
    if m.sum() < 100:
        return float("nan")
    dot = (ax * bx + ay * by)[m]
    den = (ma * mb)[m].clip(1e-6)
    return float(np.mean(dot / den))


# ---------------- 洞区（v13）----------------
def robust_affine(src, ref, keep=0.7, iters=3):
    A = np.ones(3); B = np.zeros(3)
    for c in range(3):
        x = src[..., c].ravel().astype(np.float64)
        y = ref[..., c].ravel().astype(np.float64)
        for _ in range(iters):
            V = np.vstack([y, np.ones_like(y)]).T
            coef, *_ = np.linalg.lstsq(V, x, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            rr = np.abs(x - (a * y + b))
            k = rr <= np.percentile(rr, keep * 100)
            x, y = x[k], y[k]
            if len(x) < 500:
                break
        A[c], B[c] = a, b
    return A, B


def hole_mask(I, G):
    A, B = robust_affine(I, G)
    I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
    T = gray(G)
    R_s = (I_al - G).mean(axis=2).astype(np.float32)
    r = cv2.blur((R_s / np.maximum(T, T_FLOOR)).astype(np.float32),
                 (SMOOTH_R, SMOOTH_R))
    has = r >= T_HAS_REFL
    vals = r[has]
    if vals.size < 500:
        return np.zeros(r.shape, bool)
    lo, hi = float(vals.min()), float(vals.max())
    v8 = (np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    thr_n, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tau = lo + (thr_n / 255.0) * (hi - lo)
    m = cv2.morphologyEx((has & (r <= tau)).astype(np.uint8),
                         cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) >= MIN_AREA:
            keep |= (lab == i)
    if ERODE_PX > 0 and keep.any():
        k = np.ones((ERODE_PX * 2 + 1,) * 2, np.uint8)
        e = cv2.erode(keep.astype(np.uint8), k).astype(bool)
        if e.sum() > 0:
            keep = e
    return keep


# ---------------- 主流程 ----------------
def _resolve_pred_dir(sub: str) -> str:
    """pred-map 里可写相对子目录（相对 PRED）或绝对路径。"""
    if not sub:
        return ""
    return sub if os.path.isabs(sub) else os.path.join(PRED, sub)


def main() -> int:
    global ARMS

    ap = argparse.ArgumentParser(description="1019 权威评测表")
    ap.add_argument("--pred-map", default="",
                    help='JSON {"arm": "subdir_or_abs"}; 覆盖默认 ARMS（identity 自动置首）')
    ap.add_argument("--pairs", default=PAIR, help="配对集目录")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免覆盖主表")
    args = ap.parse_args()

    if args.pred_map:
        with open(args.pred_map, encoding="utf-8") as fh:
            m = json.load(fh)
        arms = [("identity", None)]
        for k, v in m.items():
            if k == "identity":
                continue
            arms.append((k, _resolve_pred_dir(v)))
        ARMS = arms
        print(f"[pred-map] {len(ARMS)} 臂: {[a for a, _ in ARMS]}\n")

    os.makedirs(FIG, exist_ok=True)
    pairs = []
    for sc in sorted(os.listdir(args.pairs)):
        d = os.path.join(args.pairs, sc)
        if not os.path.isdir(d):
            continue
        G = read(os.path.join(d, "GT.png"))
        if G is None:
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                I = read(os.path.join(d, f))
                if I is not None:
                    pairs.append((sc, os.path.splitext(f)[0], I, G))
    print(f"配对 {len(pairs)} 组\n")

    gt_lap = float(np.mean([lapvar(G) for _, _, _, G in pairs]))

    rows = []
    for sc, stem, I, G in pairs:
        hole = hole_mask(I, G)
        for arm, sub in ARMS:
            if arm == "identity":
                Pr = I
            else:
                pth = os.path.join(PRED, sub, f"{sc}__{stem}", "t_hat.png")
                if not os.path.exists(pth):
                    continue
                Pr = read(pth)
                if Pr is None:
                    continue
                if Pr.shape[:2] != I.shape[:2]:
                    Pr = cv2.resize(Pr, (I.shape[1], I.shape[0]),
                                    interpolation=cv2.INTER_AREA)
            rows.append({
                "scene": sc, "input": stem, "arm": arm,
                "hole_frac": round(float(hole.mean()), 4),
                "psnr": round(psnr_rgb(Pr, G), 3),
                "ssim": round(ssim_rgb(Pr, G), 4),
                "hf_psnr": round(psnr_gray(highpass(gray(Pr)), highpass(gray(G))), 3),
                "grad_cos": round(grad_cos(gray(Pr), gray(G)), 4),
                "lapvar": round(lapvar(Pr), 1),
                "psnr_hole": round(psnr_masked(Pr, G, hole), 3),
                "psnr_nonhole": round(psnr_masked(Pr, G, ~hole), 3),
            })
            # 注意：此处不能 break —— 一旦 break 会把后续所有臂都跳过
            # （曾因此只输出 identity 一行）

    # 汇总
    arms = [a for a, _ in ARMS]
    summ = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        agg = {}
        for k in ("psnr", "ssim", "hf_psnr", "grad_cos", "lapvar",
                  "psnr_hole", "psnr_nonhole"):
            agg[k] = round(float(np.nanmean([r[k] for r in rs])), 4)
        agg["n"] = len(rs)
        agg["lapvar_pct"] = round(agg["lapvar"] / gt_lap * 100, 1)
        agg["gap"] = round(agg["psnr_hole"] - agg["psnr_nonhole"], 3)
        summ[arm] = agg

    base = summ.get("identity", {})
    for arm in summ:
        s = summ[arm]
        s["d_psnr"] = round(s["psnr"] - base.get("psnr", 0), 3)
        s["d_hf"] = round(s["hf_psnr"] - base.get("hf_psnr", 0), 3)
        s["d_gc"] = round(s["grad_cos"] - base.get("grad_cos", 0), 4)
        s["d_gap"] = round(s["gap"] - base.get("gap", 0), 3)

    # 打印
    hdr = (f"{'臂':<12}{'PSNR':>7}{'Δ':>6}{'SSIM':>7}{'HF-PSNR':>9}{'ΔHF':>7}"
           f"{'梯度余弦':>9}{'Δgc':>7}{'LapVar':>7}{'%GT':>6}")
    print(hdr); print("-" * len(hdr))
    for arm in arms:
        if arm not in summ:
            continue
        s = summ[arm]
        print(f"{arm:<12}{s['psnr']:>7.2f}{s['d_psnr']:>+6.2f}{s['ssim']:>7.4f}"
              f"{s['hf_psnr']:>9.2f}{s['d_hf']:>+7.2f}{s['grad_cos']:>9.3f}"
              f"{s['d_gc']:>+7.3f}{s['lapvar']:>7.1f}{s['lapvar_pct']:>5.0f}%")

    print(f"\n{'臂':<12}{'洞区':>8}{'非洞区':>9}{'gap':>8}{'Δgap':>8}   （洞区占 "
          f"{np.mean([r['hole_frac'] for r in rows if r['arm']=='identity'])*100:.2f}%）")
    print("-" * 52)
    for arm in arms:
        if arm not in summ:
            continue
        s = summ[arm]
        print(f"{arm:<12}{s['psnr_hole']:>8.2f}{s['psnr_nonhole']:>9.2f}"
              f"{s['gap']:>8.2f}{s['d_gap']:>+8.2f}")

    # 落盘（--tag 避免覆盖主表）
    sfx = f"_{args.tag}" if args.tag else ""
    json.dump({"gt_lapvar": round(gt_lap, 1), "summary": summ, "per_pair": rows},
              open(os.path.join(EVAL, f"table_final_1019{sfx}.json"), "w",
                   encoding="utf-8"),
              ensure_ascii=False, indent=1)
    cp = os.path.join(EVAL, f"table_final_1019{sfx}.csv")
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Markdown
    md = ["| 模型 | PSNR↑ | SSIM↑ | HF-PSNR↑ | 梯度余弦↑ | LapVar | 洞区 PSNR | 非洞区 | Δgap |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        if arm not in summ:
            continue
        s = summ[arm]
        md.append(f"| {arm} | {s['psnr']:.2f} | {s['ssim']:.4f} | {s['hf_psnr']:.2f} | "
                  f"{s['grad_cos']:.3f} | {s['lapvar']:.1f} | {s['psnr_hole']:.2f} | "
                  f"{s['psnr_nonhole']:.2f} | {s['gap']:+.2f} |")
    md_txt = "\n".join(md)
    open(os.path.join(EVAL, f"table_final_1019{sfx}.md"), "w", encoding="utf-8").write(
        f"# 1019 权威评测表（n={len(pairs)} 对，GT LapVar={gt_lap:.1f}）\n\n{md_txt}\n")
    print("\n" + md_txt)

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [a for a in arms if a in summ]
        lbl = ks
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, (key, ttl) in zip(ax, [("psnr", "PSNR (dB)"),
                                      ("hf_psnr", "HF-PSNR (dB)"),
                                      ("grad_cos", "gradient cosine")]):
            a.bar(lbl, [summ[k][key] for k in ks], color="#4C72B0")
            a.set_title(ttl); a.grid(axis="y", alpha=.3)
            a.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        p = os.path.join(FIG, f"table_final{sfx}.png")
        plt.savefig(p, dpi=140)
        print(f"\n图 -> {p}")
    except Exception as e:
        print(f"[跳过出图] {e}")
    print(f"写入 {cp} / table_final_1019{sfx}.json / table_final_1019{sfx}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
