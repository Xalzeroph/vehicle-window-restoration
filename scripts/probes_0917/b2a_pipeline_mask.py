#!/usr/bin/env python3
"""批次2a：掩膜头自动带 α 锚管线（零训练，合成集 v21 三档 + v23_lowscan 四档）。

问题：批次 1a 的边界台阶锚用 **GT α 图** 定义带（膜侧带 = 距裸区 6–12px，裸侧带 = 距膜 ≤6px），
MAE 0.080（12–24 场景聚类）。本探针把"带定义"换成**不依赖 GT 的自动检测**：
档位头（fac_taps_rlr 等）前向得膜掩膜 m̂，m̂ < thr 视为裸区，再按同一距离规则取带，代入
   α̂ = (mean(I_膜带) − mean(R̂_膜带)) / (mean(I_裸带) − mean(R̂_裸带))
（R̂ = 冻结分离器 rdt_plain 的 r_hat.png）。对照：同样本 GT α 带（复现批次 1 口径）并列同表。

阈值口径声明（重要）：
  * 报告主口径用 **thr=0.5**（mask 头 sigmoid 的自然决策边界，训练口径，**未用测试 GT**）；
  * 扩展扫描 0.3/0.5/0.7/0.9/0.95/0.99 全部列出；"扫出的最优阈值"在测试集 GT 上挑选，
    属 **oracle 口径上限**，单独标注，不得当作实用管线数字。

真值口径（同批次 1a）：α_true = 带内 GT α 无权重均值，带 = GT α>0.995 定义（batch-1 convention）。
样本：v2.1 = 12 场景 × 3 档（36 行 / 12 簇, t₂₀.₉₇₅=2.201, df=11）；
      v23_lowscan = 48 场景 × 4 档（0.3/0.2/0.1/0.05，192 行 / 48 簇, t=2.0117, df=47）。
      两集均**不含** alpha_1 档用于 v23（其 α 图大面积 clip 到 1.0，带定义退化，见探针记录）。
CI 口径：逐行 e=|α̂−α_true| → 场景内跨档先平均 → 跨场景 t 分布 95%CI（同 b1a2）。

用法: python3 scripts/probes_0917/b2a_pipeline_mask.py [--ckpt runs/ckpt/fac_taps_rlr/ckpt_final.pth] [--gpu 1]
产物: eval/probe_b2a_mask_pipeline.csv（逐样本）+ 控制台表
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import t as tdist

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
import eval_iar  # noqa: E402  (import 时 chdir 到 RDNet；以下一律用绝对路径)

BARE_T = 0.995        # GT 裸区阈值（批次 1a 口径）
FILM_T2 = 0.99
D_LO, D_HI, D_NEAR = 6, 12, 6   # 膜侧带 dist(裸)∈[6,12]；裸侧带 dist(膜)≤6
MIN_BAND = 30
THRS = (0.3, 0.5, 0.7, 0.9, 0.95, 0.99)   # 主口径 0.5；其余为扫描（最优=oracle）
PRIMARY = 0.5
LEVELS = {  # 按任务书：v21 三档；v23 四档（排除 alpha_1）
    "synth_rrw_v21_0913": ["1", "0.6", "0.3"],
    "synth_rrw_v23_lowscan": ["0.3", "0.2", "0.1", "0.05"],
}


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def bands(bare):
    """bare: bool 裸区掩膜 → (膜侧带, 裸侧带)，同批次 1a 距离规则。"""
    film = ~bare
    if bare.sum() == 0 or film.sum() == 0:
        return None, None
    d_bare = ndimage.distance_transform_edt(film)   # 膜像素→最近裸像素
    d_film = ndimage.distance_transform_edt(bare)   # 裸像素→最近膜像素
    film_b = film & (d_bare >= D_LO) & (d_bare <= D_HI)
    bare_b = bare & (d_film <= D_NEAR)
    return film_b, bare_b


def a_hat(Im, Rm, film_b, bare_b):
    """锚式估计（I/R 均为逐通道均值图）。"""
    if film_b is None or film_b.sum() < MIN_BAND or bare_b.sum() < MIN_BAND:
        return np.nan
    If, Rf = float(Im[film_b].mean()), float(Rm[film_b].mean())
    Ib, Rb = float(Im[bare_b].mean()), float(Rm[bare_b].mean())
    den = Ib - Rb
    return (If - Rf) / den if abs(den) > 1e-6 else np.nan


def fmt(v, nd=3):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), nd)


def ci95(vals):
    v = np.asarray(vals, float)
    n = len(v)
    m, s = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 1 else 0.0
    tc = float(tdist.ppf(0.975, n - 1)) if n > 1 else np.nan
    return dict(n=n, mean=m, sd=s, t=tc, df=n - 1, lo=m - tc * se, hi=m + tc * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ckpt/fac_taps_rlr/ckpt_final.pth")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--sets", default=",".join(LEVELS))
    ap.add_argument("--out", default="eval/probe_b2a_mask_pipeline.csv")
    args = ap.parse_args()
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJ, args.ckpt)
    out_csv = args.out if os.path.isabs(args.out) else os.path.join(PROJ, args.out)

    import torch
    device = torch.device(f"cuda:{args.gpu}")
    net = eval_iar.build("taps", ckpt, device)
    net.eval()
    os.chdir(PROJ)
    print(f"[b2a] ckpt={os.path.relpath(ckpt, PROJ)}  主阈值 {PRIMARY}（无测试 GT）；"
          f"扫描 {THRS}（最优=oracle 口径）")

    rows = []
    for s in args.sets.split(","):
        root = os.path.join(PROJ, "data/test_sets", s)
        scenes = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "gt")))
        for key in LEVELS[s]:
            lvdir = f"alpha_{key}"
            for sid in scenes:
                A = np.load(os.path.join(root, "alpha", f"{sid}_a{key}.npy")).astype(np.float32)
                if A.ndim == 3:
                    A = A[:, :, 0]
                I = img(os.path.join(root, "levels", lvdir, sid + ".png"))
                Rm_p = os.path.join(PROJ, "runs/pred/rdt_plain", s, lvdir, sid, "r_hat.png")
                assert os.path.exists(Rm_p), Rm_p
                R = img(Rm_p)
                assert R.shape[:2] == I.shape[:2], (s, key, sid, R.shape, I.shape)
                x = torch.from_numpy(I.transpose(2, 0, 1) / 255.0)[None].to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    _, m, _, _ = net.forward_alpha_only(x)
                mh = m.float()[0, 0].cpu().numpy()
                Im, Rm = I.mean(2), R.mean(2)

                # --- 对照：GT α 带（批次 1 口径）---
                bare_gt = A > BARE_T
                film_gt = A < FILM_T2
                fb_gt, bb_gt = bands(bare_gt)
                a_true = float(A[fb_gt].mean()) if fb_gt is not None and fb_gt.sum() > 0 else np.nan
                a_gtband = a_hat(Im, Rm, fb_gt, bb_gt)

                # --- 掩膜路线：m̂ < thr 为裸区 ---
                res = {}
                for thr in THRS:
                    fb, bb = bands(mh < thr)
                    tag = f"t{int(round(thr * 100)):03d}"
                    res[tag] = (a_hat(Im, Rm, fb, bb), fb, bb)

                lab = A < BARE_T          # 与后面 m_iou 的 GT 膜定义一致
                pm = mh >= PRIMARY
                inter, union = (pm & lab).sum(), (pm | lab).sum()
                naked = A > BARE_T
                film = A < FILM_T2
                row = dict(set=s, level=key, scene=sid,
                           alpha_true=fmt(a_true), a_hat_gtband=fmt(a_gtband),
                           n_fb_gt=int(fb_gt.sum()) if fb_gt is not None else "",
                           n_bb_gt=int(bb_gt.sum()) if bb_gt is not None else "",
                           m_iou=fmt(inter / max(union, 1)),
                           m_nrec=fmt((naked & ~pm).sum() / max(naked.sum(), 1)),
                           m_frec=fmt((film & pm).sum() / max(film.sum(), 1)),
                           m_cov=fmt(pm.mean()), gt_cov=fmt(A.mean()),
                           note="")
                for thr in THRS:
                    tag = f"t{int(round(thr * 100)):03d}"
                    av, fb, bb = res[tag]
                    row[f"a_hat_{tag}"] = fmt(av)
                    row[f"n_fb_{tag}"] = int(fb.sum()) if fb is not None else ""
                    row[f"n_bb_{tag}"] = int(bb.sum()) if bb is not None else ""
                if res["t050"][1] is None or res["t050"][1].sum() < MIN_BAND:
                    row["note"] = "掩膜带过小"
                rows.append(row)
                print(f"[{s[:18]} {lvdir} {sid}] α_true={a_true:.3f} gt带={a_gtband if np.isfinite(a_gtband) else float('nan'):.3f} "
                      f"mask@0.5={res['t050'][0] if np.isfinite(res['t050'][0]) else float('nan'):.3f} "
                      f"IoU={row['m_iou']:.3f} nrec={row['m_nrec']:.3f}")

    ok = [r for r in rows if r["note"] == ""]
    print(f"\n有效样本 {len(ok)}/{len(rows)}")

    # ---------------- 汇总：MAE + 场景聚类 CI ----------------
    def band_names():
        names = [("a_hat_gtband", "GT带(批次1对照)")]
        for thr in THRS:
            tag = f"t{int(round(thr * 100)):03d}"
            tagd = "主口径(无GT)" if abs(thr - PRIMARY) < 1e-9 else "扫描"
            names.append((f"a_hat_{tag}", f"mask@{thr:g}({tagd})"))
        return names

    summary = {}
    print("\n================ 表：α̂ MAE（场景均值口径 e=|α̂−α_true|） ================")
    print(f"{'集':<24}{'带定义':<20}{'n行':>5}{'n簇':>5}{'naive':>8}{'聚类均值':>9}  聚类 95%CI (t, df){'':>10}")
    for s in args.sets.split(","):
        sub = [r for r in ok if r["set"] == s]
        if not sub:
            continue
        clusters = sorted({r["scene"] for r in sub})
        for key, name in band_names():
            sel = [r for r in sub if r[key] != ""]
            if not sel:
                print(f"{s:<24}{name:<20}   无有效样本")
                continue
            naive = float(np.mean([abs(float(r[key]) - float(r["alpha_true"])) for r in sel]))
            cvals = []
            for c in clusters:
                cr = [abs(float(r[key]) - float(r["alpha_true"])) for r in sel if r["scene"] == c]
                if cr:
                    cvals.append(float(np.mean(cr)))
            c = ci95(cvals)
            summary[(s, key)] = dict(naive=naive, **c)
            print(f"{s:<24}{name:<20}{len(sel):>5}{c['n']:>5}{naive:>8.3f}{c['mean']:>9.3f}  "
                  f"[{c['lo']:.3f}, {c['hi']:.3f}]  (t={c['t']:.3f}, df={c['df']})")

    # ---------------- 分档 MAE（每集 × 带定义 × 档） ----------------
    print("\n================ 分档 MAE ================")
    for s in args.sets.split(","):
        for key, name in band_names():
            out = []
            for lvk in LEVELS[s]:
                sel = [r for r in ok if r["set"] == s and r["level"] == lvk and r[key] != ""]
                if sel:
                    mae = float(np.mean([abs(float(r[key]) - float(r["alpha_true"])) for r in sel]))
                    out.append(f"α{lvk}:{mae:.3f}")
            if out:
                print(f"{s[:22]:<22} {name:<20} " + "  ".join(out))

    # ---------------- 掩膜质量 ----------------
    print("\n================ 掩膜质量 vs GT 膜 (A<0.995, m̂≥0.5) ================")
    for s in args.sets.split(","):
        for lvk in LEVELS[s]:
            sel = [r for r in ok if r["set"] == s and r["level"] == lvk]
            if sel:
                f = lambda k: float(np.mean([r[k] for r in sel]))
                print(f"{s[:22]:<22} α{lvk:<5} IoU={f('m_iou'):.3f} 裸recall={f('m_nrec'):.3f} "
                      f"膜recall={f('m_frec'):.3f} pred_cov={f('m_cov'):.3f}")

    # ---------------- oracle 阈值 ----------------
    print("\n================ 阈值扫描（最优=oracle 口径，用了测试 GT，上限参考） ================")
    for s in args.sets.split(","):
        best = min(((summary[(s, f'a_hat_t{int(round(t * 100)):03d}')]["mean"], t)
                    for t in THRS if (s, f'a_hat_t{int(round(t * 100)):03d}') in summary))
        print(f"{s:<24} oracle 最优阈值 {best[1]:g} → 聚类均值 MAE {best[0]:.3f}；"
              f"主口径 0.5 → {summary[(s, 'a_hat_t050')]['mean']:.3f}")

    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {os.path.relpath(out_csv, PROJ)}")
    print(f"   注：主口径 thr=0.5 未用测试 GT；表中 thr≠0.5 的扫描列与 oracle 结论均为 oracle 口径。")


if __name__ == "__main__":
    main()
