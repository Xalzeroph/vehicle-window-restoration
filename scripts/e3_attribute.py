#!/usr/bin/env python3
"""E3 归因: 把 RDNet 增益匹配后残差 E = g·T̂ − T 按像素归因。

三桶互斥分解(能量占比):
  D  暗区丢失: α(x) < 0.12 的像素(膜最黑处, 信息在特征层面可能已丢)
  Rr 反射残留: α≥0.12 且 R 局部能量高(R_blur > 1.5×R 全局均值)
  N  其余
外加: corr(|E|, R_blur), corr(|E|, 1−α), corr(|E|, |∇f|) 与线性回归分解,
以及"平均|E| vs α(x)"分箱曲线。

用法:
  python3 e3_attribute.py --ts <e1_testset> --pred <e1_pred> --model rdnet \
      --out-prefix /path/to/E3
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_DARK = 0.12
ALPHAS = [1.0, 0.6, 0.3, 0.15, 0.1, 0.05]


def load(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float32)


def gain_match(q, g):
    q = np.asarray(q, np.float64)
    s = float((q * g).sum() / (q * q).sum())
    return np.clip(q * s, 0, 255)


def alpha_field(f_file, alpha_bar, s):
    f = np.load(f_file).astype(np.float32)
    a = alpha_bar * (1.0 + s * f)[..., None] * np.ones((1, 1, 3), np.float32)
    return np.clip(a, 0.01, 1.0)[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--model", default="rdnet")
    ap.add_argument("--out-prefix", default="/home/courseliu/vehicle_ssd/E3")
    args = ap.parse_args()

    with open(os.path.join(args.ts, "meta.json")) as fp:
        meta = json.load(fp)
    s = meta["spatial"]
    scenes = sorted(os.path.basename(f)[:-4] for f in
                    glob.glob(os.path.join(args.ts, "gt", "*.png")))

    rows = []
    for lv in ALPHAS:
        E = eD = eRr = eN = None
        pixD = pixRr = []
        corr = {"R": [], "one_minus_a": [], "grad_f": []}
        for stem in scenes:
            t = load(os.path.join(args.ts, "gt", stem + ".png"))
            p = load(os.path.join(args.pred, f"{args.model}_alpha_{lv:g}",
                                  stem, "ytmt_ucs_sirs_l.png"))
            r = load(os.path.join(args.ts, "r", stem + ".png"))
            a = alpha_field(os.path.join(args.ts, "f", stem + ".npy"), lv, s)
            q = gain_match(p, t)
            e = np.abs(q - t)                      # 每像素绝对误差 (0-255)
            lum = e.mean(-1)
            r_blur = uniform_filter(r.mean(-1), size=15, mode="nearest")
            grad = np.abs(np.gradient(a))
            grad_f = (grad[0] ** 2 + grad[1] ** 2) ** 0.5

            # 场景内相对桶: D=该场景 α 最低 25% 像素(膜最黑处);
            # Rr=非D 且 R 局部能量 >1.5×场景 R 均值; N=其余
            a_thr = np.quantile(a, 0.25)
            mD = a <= a_thr
            mRr = (~mD) & (r_blur > 1.5 * r.mean())
            mN = ~(mD | mRr)
            e2 = (e ** 2).sum(-1)

            E = e2 if E is None else E + e2
            eD = (e2 * mD).sum() if eD is None else eD + (e2 * mD).sum()
            eRr = (e2 * mRr).sum() if eRr is None else eRr + (e2 * mRr).sum()
            eN = (e2 * mN).sum() if eN is None else eN + (e2 * mN).sum()
            pixD.append(mD.mean())
            pixRr.append(mRr.mean())

            corr["R"].append(np.corrcoef(lum.ravel(), r_blur.ravel())[0, 1])
            corr["one_minus_a"].append(np.corrcoef(lum.ravel(), (1 - a).ravel())[0, 1])
            corr["grad_f"].append(np.corrcoef(lum.ravel(), grad_f.ravel())[0, 1])

        tot = E.sum()
        rows.append(dict(
            alpha=lv,
            D=eD / tot * 100, Rr=eRr / tot * 100, N=eN / tot * 100,
            pix_D=np.mean(pixD) * 100, pix_Rr=np.mean(pixRr) * 100,
            corr_R=np.nanmean(corr["R"]),
            corr_1a=np.nanmean(corr["one_minus_a"]),
            corr_grad=np.nanmean(corr["grad_f"])))

    print(f"{'a_bar':>6} {'D%':>6} {'Rr%':>6} {'N%':>6} "
          f"{'pixD%':>7} {'pixRr%':>7} {'corrR':>7} {'corr(1-a)':>9} {'corrGrad':>8}")
    for r in rows:
        print(f"{r['alpha']:>6} {r['D']:6.1f} {r['Rr']:6.1f} {r['N']:6.1f} "
              f"{r['pix_D']:7.1f} {r['pix_Rr']:7.1f} {r['corr_R']:7.3f} "
              f"{r['corr_1a']:9.3f} {r['corr_grad']:8.3f}")

    # ---- 低档合并: 线性回归分解 + α 分箱曲线 ----
    X, y, a_all = [], [], []
    for lv in [0.3, 0.15, 0.1, 0.05]:
        for stem in scenes:
            t = load(os.path.join(args.ts, "gt", stem + ".png"))
            p = load(os.path.join(args.pred, f"{args.model}_alpha_{lv:g}",
                                  stem, "ytmt_ucs_sirs_l.png"))
            r = load(os.path.join(args.ts, "r", stem + ".png"))
            a = alpha_field(os.path.join(args.ts, "f", stem + ".npy"), lv, s)
            q = gain_match(p, t)
            e = np.abs(q - t).mean(-1)
            r_blur = uniform_filter(r.mean(-1), size=15, mode="nearest")
            grad = np.abs(np.gradient(a))
            grad_f = (grad[0] ** 2 + grad[1] ** 2) ** 0.5
            X.append(np.stack([r_blur.ravel(), (1 - a).ravel(),
                               grad_f.ravel(), np.ones(r_blur.size)], 1))
            y.append(e.ravel())
            a_all.append(a.ravel())
    X = np.concatenate(X)
    y = np.concatenate(y)
    a_all = np.concatenate(a_all)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    xs = X.std(0) + 1e-9
    std_coef = coef * xs / y.std()
    names = ["R_blur", "1-alpha", "|grad alpha|", "const"]
    print("\n低档(ᾱ≤0.3)线性回归 |E| ~ Σβ·x (标准化系数):")
    for n, c, sc in zip(names, coef, std_coef):
        print(f"  {n:>12}: beta={c:7.4f}  std_beta={sc:6.3f}")

    bins = np.array([0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0])
    bm = np.digitize(a_all, bins)
    bin_mean, bin_frac = [], []
    for i in range(1, len(bins)):
        sel = bm == i
        bin_mean.append(float(y[sel].mean()) if sel.any() else np.nan)
        bin_frac.append(float(np.sum(y[sel] ** 2) / np.sum(y ** 2) * 100))
    centers = 0.5 * (bins[:-1] + bins[1:])
    print("\n|E| vs alpha(x) 分箱 (低档合并):")
    for c, m, f in zip(centers, bin_mean, bin_frac):
        print(f"  alpha~{c:5.2f}: mean|E|={m:6.2f}  能量占比={f:5.1f}%")

    # ---- 图1: 分箱曲线 ----
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(centers, bin_mean, "o-", color="tab:blue", label="mean |E| (0-255)")
    ax1.set_xlabel("alpha(x) (local, low-alpha-bar scenes merged)")
    ax1.set_ylabel("mean |E|", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.bar(centers, bin_frac, width=0.04, alpha=0.35, color="tab:red",
            label="error energy share %")
    ax2.set_ylabel("error energy share %", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("RDNet residual vs local alpha (gain-matched, a_bar<=0.3)")
    fig.tight_layout()
    fig.savefig(args.out_prefix + "_error_vs_alpha.png", dpi=130)
    print("curve ->", args.out_prefix + "_error_vs_alpha.png")

    # ---- 图2: 残差结构拼图 (t00, ᾱ=0.05) ----
    stem = scenes[0]
    lv = 0.05
    t = load(os.path.join(args.ts, "gt", stem + ".png"))
    p = load(os.path.join(args.pred, f"{args.model}_alpha_{lv:g}", stem,
                          "ytmt_ucs_sirs_l.png"))
    r = load(os.path.join(args.ts, "r", stem + ".png"))
    a = alpha_field(os.path.join(args.ts, "f", stem + ".npy"), lv, s)
    q = gain_match(p, t)
    e = np.abs(q - t).mean(-1)
    e_img = np.clip(e * 4, 0, 255).astype(np.uint8)          # |E| 放大显示
    a_img = (np.clip(a / 0.15, 0, 1) * 255).astype(np.uint8)  # α 热图(0.15 满量程)
    r_img = (np.clip(r / 0.15, 0, 1) * 255).astype(np.uint8)  # R 热图

    cols = [("T-hat (gm)", q.astype(np.uint8)), ("GT", t.astype(np.uint8)),
            ("|E| x4", e_img), ("R layer", r_img), ("alpha map", a_img)]
    cw, ch, pad, lab = 330, 152, 6, 26
    W = cw * len(cols) + pad * (len(cols) + 1)
    H = lab + ch + pad * 2
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for ci, (name, im) in enumerate(cols):
        x0 = pad + ci * (cw + pad)
        d.text((x0 + 8, 4), name, fill=(255, 255, 255), font=font)
        sheet.paste(Image.fromarray(im).resize((cw, ch)), (x0, lab + pad))
    p_out = args.out_prefix + "_structure.png"
    sheet.save(p_out)
    print("structure ->", p_out)


if __name__ == "__main__":
    main()
