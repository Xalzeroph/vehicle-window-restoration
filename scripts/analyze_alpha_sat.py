#!/usr/bin/env python3
"""α 头饱和测试汇总: 各 run 的 α̂ 分区统计 + 训练轨迹。

统计 (均按 12 场景聚合, 像素级):
  naked 区 (α_GT>=0.95):  α̂ 均值 (目标 >=0.9) / corr
  film 区 (非边界非 naked): α̂ 均值 vs GT / MAE / corr
  边界带 (|∇α|>0.02):       MAE / band-share
数据源: pilot_pred{tag}/{run}/alpha_X/*_a.npy 与测试集 alpha/{stem}_aX.npy。

用法:
  python3 scripts/analyze_alpha_sat.py
"""
import glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
TS = os.path.join(PROJ, "synth_rrw_v21_0913")
LEVELS = [1.0, 0.6, 0.3]
RUNS = [  # (显示名, pred 根目录, run 目录名)
    ("A_r1 (1000)", "pilot_pred", "A"),
    ("A_r3 (4000)", "pilot_pred_r3", "A"),
    ("A_sat (8000, official-init)", "pilot_pred_sat", "A_sat"),
    ("A_sat_r3 (8000, warm from A_r3)", "pilot_pred_sat_r3", "A_sat_r3"),
    ("A_sat_cc (4000, code-cond)", "pilot_pred_sat_cc", "A_sat_cc"),
]


def grad_mag(a):
    gx = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
    gy = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
    return gx + gy


def region_stats(preds, gts):
    """preds/gts: list of 2D arrays (已按区展平前的整图)."""
    out = {}
    for name, mask_fn in (
            ("naked", lambda a: a >= 0.95),
            ("film", lambda a: ~((a >= 0.95) | (grad_mag(a) > 0.02))),
            ("band", lambda a: grad_mag(a) > 0.02)):
        p = np.concatenate([pr[m] for pr, g, m in zip(preds, gts, [mask_fn(g) for g in gts])])
        g = np.concatenate([gg[m] for gg, m in zip(gts, [mask_fn(g) for g in gts])])
        if p.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            corr = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-8 and g.std() > 1e-8 else np.nan
        out[name] = dict(mean=float(p.mean()), gt=float(g.mean()),
                         mae=float(np.abs(p - g).mean()), corr=corr, frac=float(len(p)))
    return out


def main():
    for lv in LEVELS:
        stem_files = sorted(glob.glob(os.path.join(TS, "alpha", f"*_a{lv:g}.npy")))
        gts = [np.load(f).astype(np.float32) for f in stem_files]
        print(f"\n===== ᾱ={lv:g} (12 场景像素级聚合) =====")
        hdr = f"{'run':34s} {'naked α̂':>9s} {'GT':>5s} {'MAE':>6s} {'corr':>6s} | " \
              f"{'film α̂':>8s} {'GT':>5s} {'MAE':>6s} {'corr':>6s} | {'band MAE':>8s}"
        print(hdr)
        for name, root, run in RUNS:
            pred_dir = os.path.join(PROJ, root, run, f"alpha_{lv:g}")
            if not glob.glob(os.path.join(pred_dir, "*_a.npy")):
                print(f"{name:34s} (缺, 未评测)")
                continue
            preds = [np.load(os.path.join(pred_dir, os.path.basename(f).replace(f"_a{lv:g}.npy", "_a.npy")))
                     .astype(np.float32) for f in stem_files]
            st = region_stats(preds, gts)
            n, f_, b = st["naked"], st["film"], st["band"]
            print(f"{name:34s} {n['mean']:9.3f} {n['gt']:5.2f} {n['mae']:6.3f} {n['corr']:6.3f} | "
                  f"{f_['mean']:8.3f} {f_['gt']:5.2f} {f_['mae']:6.3f} {f_['corr']:6.3f} | "
                  f"{b['mae']:8.3f}")

    # 训练轨迹
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    logs = [(os.path.join(PROJ, "pilot_ckpt/A_sat/train_log.csv"), "sat (official-init)"),
            (os.path.join(PROJ, "pilot_ckpt/A_sat_r3/train_log.csv"), "sat_r3 (warm from A_r3)")]
    for ax, (tag, ref) in zip(axes, [("naked α̂ (目标≥0.9)", "naked"), ("L_α", "L_alpha"), ("α̂ MAE", "alpha_mae")]):
        for path, label in logs:
            if not os.path.exists(path):
                continue
            d = np.genfromtxt(path, delimiter=",", names=True)
            y = {"naked": d["alpha_hat_mean"], "L_alpha": d["L_alpha"], "alpha_mae": d["alpha_mae"]}[ref]
            ax.plot(d["iter"], y, label=label, lw=1.2)
        ax.set_title(tag)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].axhline(0.9, color="r", ls=":", lw=1)
    fig.suptitle("α-head saturation (alpha-only, 8000 iter, batch 4)")
    fig.tight_layout()
    outp = os.path.join(PROJ, "fig", "alpha_sat_curves.png")
    fig.savefig(outp, dpi=130)
    print("\n图 ->", outp)


if __name__ == "__main__":
    main()
