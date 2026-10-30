"""逐检查点评测：PSNR / SSIM / HF-PSNR / LapVar / 反射掩码 F1 / FPS / 综合分。

与 table_final_1019.py 的区别：
  * 那个脚本只评最终 ckpt，且区域列用 v13 反射掩码（不是训练期 Ω）
  * 本脚本面向「训练过程」：给定一组 ckpt，逐个评，出一张随迭代变化的曲线表
  * 额外给 FPS，以及 f1_resid（残差落点诊断，见下方警告），并合成一个用于早停/选点的综合分

综合分定义（固定锚点，非 z-score，便于跨臂比较）：
    psnr_n = (PSNR - 18.44) / (22.0 - 18.44)      # identity 地板 -> 22 dB 天花板
    ssim_n = (SSIM - 0.69) / (0.78 - 0.69)
    hf_n   = (HF   - 32.9) / (35.0 - 32.9)
    lap_n  = 1 - |LapVar - 39.6| / 39.6            # GT LapVar = 39.6
    fps_n  = min(FPS / 2.0, 1.0)                   # 2 FPS 记满分（原生分辨率）
    score  = 0.40*psnr_n + 0.20*ssim_n + 0.25*hf_n + 0.10*lap_n + 0.05*fps_n
权重是先验设定的（PSNR 主导、结构其次、速度只作小幅惩罚），不做事后调参。

警告：f1_resid 的语义（别当质量分用）：
    它是「残差热点」与「GT 反射热点」的 F1。完美去反射的模型残差接近 0，热点为空，F1 也接近 0，
    所以它衡量的是残差落在哪里，不是做得好不好，**不进入综合分**。
    只有模型输出反射掩码（门控头 / RRW 定位图）时 F1 才是质量指标，
    那种情况请用 scripts/gate_heads.py 里对真值 R 的 IoU/F1 评测。

用法：
    python eval_progress.py --ckpts runs/X/ckpt_500.pth,runs/X/ckpt_1000.pth --tag X
    python eval_progress.py --ckpts runs/X/ckpt_latest.pth --tag X_single --fps-only
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
import time

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PANEL = os.path.join(P, "data/test_sets/real_pair_1019")
TAU_REFL = 5.0          # 灰阶阈值：反射存在判据（与 v13 的 r 判据无关，仅用于 F1）
GT_LAPVAR = 39.6


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_simple(a, b):
    a = gray(a); b = gray(b)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    ma, mb = cv2.blur(a, k), cv2.blur(b, k)
    va, vb = cv2.blur(a * a, k) - ma * ma, cv2.blur(b * b, k) - mb * mb
    cov = cv2.blur(a * b, k) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))
    return float(np.mean(s))


def hf_psnr(a, b):
    ka = gray(a) - cv2.GaussianBlur(gray(a), (0, 0), 2.0)
    kb = gray(b) - cv2.GaussianBlur(gray(b), (0, 0), 2.0)
    return psnr(ka, kb)


def lapvar(x):
    return float(cv2.Laplacian(gray(x), cv2.CV_32F).var())


def robust_affine(i, g):
    """逐通道 a,b 的最小二乘（与 table_final_1019 一致），把 I 对到 G 的量纲。"""
    A = []; B = []
    for c in range(3):
        x = i[..., c].astype(np.float64).ravel()
        y = g[..., c].astype(np.float64).ravel()
        n = x.size
        mx, my = x.mean(), y.mean()
        vx = ((x - mx) ** 2).sum() / max(n, 1)
        cov = ((x - mx) * (y - my)).sum() / max(n, 1)
        a = cov / max(vx, 1e-9)
        A.append(a); B.append(my - a * mx)
    return np.asarray(A, np.float32), np.asarray(B, np.float32)


def refl_f1(i, t_hat, t_gt):
    """反射掩码 F1：|I - T̂| 与 |I - T| 同时超阈值即判为反射像素。"""
    a, b = robust_affine(i, t_hat)
    r_hat = np.abs((i.astype(np.float32) - b.reshape(1, 1, 3)) / a.reshape(1, 1, 3)
                   - t_hat.astype(np.float32)).mean(axis=2)
    r_gt = np.abs(i.astype(np.float32) - t_gt.astype(np.float32)).mean(axis=2)
    mp, mg = r_hat > TAU_REFL, r_gt > TAU_REFL
    tp = float((mp & mg).sum()); fp = float((mp & ~mg).sum()); fn = float((~mp & mg).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, prec, rec, float(mg.mean())


def composite(m):
    psnr_n = (m["psnr"] - 18.44) / (22.0 - 18.44)
    ssim_n = (m["ssim"] - 0.69) / (0.78 - 0.69)
    hf_n = (m["hf_psnr"] - 32.9) / (35.0 - 32.9)
    lap_n = 1.0 - abs(m["lapvar"] - GT_LAPVAR) / GT_LAPVAR
    fps_n = min(m["fps"] / 2.0, 1.0)
    return (0.40 * psnr_n + 0.20 * ssim_n + 0.25 * hf_n + 0.10 * lap_n + 0.05 * fps_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="逗号分隔的 ckpt 路径，或含通配符的单个模式")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-fps", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    paths = []
    for p in a.ckpts.split(","):
        paths += sorted(glob.glob(p)) if "*" in p else [p]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("没有可评的 ckpt")

    sys.path.insert(0, os.path.join(P, "scripts"))
    sys.path.insert(0, P)
    import torch
    # 与 eval_arm_1019.py 完全同一套加载/推理（保证与既有表同口径）
    from eval_1019_multi import (  # noqa: E402
        build_rdnet, load_img, load_rdnet, predict_rdnet,
    )
    from eval_arm_1019 import detect_mona  # noqa: E402

    scenes = sorted(d for d in os.listdir(PANEL) if os.path.isdir(os.path.join(PANEL, d)))
    rows = []
    for ck in paths:
        mona = detect_mona(ck)
        net, net_c = build_rdnet(mona)
        load_rdnet(net, ck, mona)
        net.to(a.device).eval(); net_c.to(a.device).eval()
        dev = torch.device(a.device)
        per = []
        fps = float("nan")
        for sc in scenes:
            d = os.path.join(PANEL, sc)
            ip = sorted(glob.glob(os.path.join(d, "I_*.png")))
            gp = os.path.join(d, "GT.png")
            if not ip or not os.path.exists(gp):
                continue
            im = load_img(ip[0])                    # RGB uint8，与训练/推理一致
            gt = load_img(gp)                       # RGB uint8
            if im is None or gt is None:
                continue
            t0 = time.time()
            TH = predict_rdnet(net, net_c, im, dev)  # RGB float
            dt = time.time() - t0
            TH = np.clip(np.asarray(TH), 0, 255).astype(np.uint8)
            # 全部在 RGB 空间算，避免 BGR/RGB 混用导致 F1 与 PSNR 口径不一致
            f1, prec, rec, frac = refl_f1(im, TH, gt)
            per.append(dict(scene=sc, psnr=psnr(TH, gt), ssim=ssim_simple(TH, gt),
                            hf_psnr=hf_psnr(TH, gt), lapvar=lapvar(TH),
                            f1_resid=f1, prec=prec, rec=rec, gt_frac=frac, sec=dt))
        m = {k: st.mean([r[k] for r in per]) for k in
             ("psnr", "ssim", "hf_psnr", "lapvar", "f1_resid", "prec", "rec", "sec")}
        fps = 1.0 / m["sec"] if m["sec"] > 0 else float("nan")
        m["fps"] = fps if not a.no_fps else float("nan")
        m["score"] = composite({**m, "fps": m["fps"] if not a.no_fps else 0.0})
        m["ckpt"] = os.path.basename(ck)
        m["n"] = len(per)
        rows.append(m)
        print(f"{m['ckpt']:<22} PSNR {m['psnr']:.3f} SSIM {m['ssim']:.4f} HF {m['hf_psnr']:.2f} "
              f"Lap {m['lapvar']:.1f} F1res {m['f1_resid']:.3f} FPS {fps:.2f} -> score {m['score']:+.4f}")

    out = a.out or os.path.join(P, "eval", f"progress_{a.tag}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv
    ks = ["ckpt", "n", "psnr", "ssim", "hf_psnr", "lapvar", "f1_resid", "prec", "rec", "fps", "score"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ks, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: r["score"])
    print(f"\n写入 {out}")
    print(f"按综合分最佳点: {best['ckpt']} (score {best['score']:+.4f}, PSNR {best['psnr']:.3f})")


if __name__ == "__main__":
    main()
