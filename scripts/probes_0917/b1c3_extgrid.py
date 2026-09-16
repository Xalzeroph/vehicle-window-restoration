#!/usr/bin/env python3
"""批次1c 遗留：补两个自动判据（对比度匹配 / NR-IQA 清晰度）并把 5 判据按双口径读数。

判据 4 sel_std  : std(候选灰度) 最接近"训练集 T 的 std 中位数"（data/train_sets/train_triplets_rdt_0915/T 实测，
                  全图 per-image std 中位数为主口径；另附 256×128 随机裁片口径 sel_stdc 作变体）
判据 5 sel_iqa  : skimage.measure.blur_effect（0=不糊 1=最糊）取**最小**=最清晰（NR-IQA 风格正解）；
                  同时记录字面"取最大"变体 sel_iqamax（该度量越大越糊，仅作对照）
另含口径 A（命中率 top1/top2）与口径 B（悔值均值 / 90 分位 / ≤0.91dB 占比）两组读数，
覆盖上批三判据（读 eval/probe_b1c_candidates.csv）。悔值口径与跨度定义见 docs/2026.9.17_批次1遗留_1c门口径.md。

取带与候选生成逻辑复制自 b1c_candidate_family.py（原脚本未改动）。
用法: python3 scripts/probes_0917/b1c2_selectors.py
产物: eval/probe_b1c3_extgrid.csv
"""
import csv, os
import numpy as np
from PIL import Image
from skimage.measure import blur_effect

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
ALPHAS = [0.75, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05]
SETS = ["synth_rrw_v21_0913", "synth_rrw_v22_scan"]
TRAIN_T_DIR = os.path.join(PROJ, "data/train_sets/train_triplets_rdt_0915/T")
N_STD_SAMPLE = 400          # 裁片口径采样图数（每图 4 片）
SPAN_REF = 9.13             # dB 跨度：oracle@k 均值 − 恒选 α=0.2 均值（见报告）
GATE_REGRET = 0.91          # 10% × 9.13 dB


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def trainT_std_prior():
    """训练集 T 灰度 std 先验：全图 per-image 中位数 + 256×128 随机裁片中位数。"""
    fs = sorted(os.listdir(TRAIN_T_DIR))
    rng = np.random.default_rng(0)
    full, crops = [], []
    for i, f in enumerate(fs):
        y = img(os.path.join(TRAIN_T_DIR, f)).mean(2)
        full.append(float(y.std()))
        if i < N_STD_SAMPLE:
            for _ in range(4):
                y0 = int(rng.integers(0, y.shape[0] - 128))
                x0 = int(rng.integers(0, y.shape[1] - 256))
                crops.append(float(y[y0:y0 + 128, x0:x0 + 256].std()))
    return float(np.median(full)), float(np.median(crops)), len(fs)


def main():
    tau_full, tau_crop, n_tr = trainT_std_prior()
    print(f"训练集 T（{n_tr} 图）std 先验: 全图 per-image 中位数 {tau_full:.2f} | 256×128 裁片中位数 {tau_crop:.2f}")

    rows, n_tie = [], 0
    for s in SETS:
        root = os.path.join(PROJ, "data/test_sets", s)
        for lv in sorted(os.listdir(os.path.join(root, "levels")), key=lambda x: -float(x.split("_")[-1])):
            key = lv.split("_")[-1]
            for sc in sorted(os.listdir(os.path.join(root, "gt"))):
                sid = os.path.splitext(sc)[0]
                A = np.load(os.path.join(root, "alpha", f"{sid}_a{key}.npy")).astype(np.float32)
                A = A[:, :, 0] if A.ndim == 3 else A
                if (A < 0.9).sum() < 5000:
                    continue
                pred = os.path.join(PROJ, "runs/pred/rdt_plain", s, lv, sid)
                Rh = img(os.path.join(pred, "r_hat.png"))
                I = img(os.path.join(root, "levels", lv, sc))
                gt = img(os.path.join(root, "gt", sc))
                ys, xs = np.nonzero(A < 0.9)
                cy, cx = int(ys.mean()), int(xs.mean())
                y0, x0 = np.clip(cy - 64, 0, A.shape[0] - 128), np.clip(cx - 128, 0, A.shape[1] - 256)
                sl = (slice(y0, y0 + 128), slice(x0, x0 + 256))
                if (A[sl] < 0.9).mean() < 0.98:
                    continue
                Tc = (I[sl] - Rh[sl]).clip(0, 255)
                gc = gt[sl]
                ps, stds, blurs = [], [], []
                for a in ALPHAS:
                    c = (Tc / a).clip(0, 255)
                    y = c.mean(2)
                    ps.append(psnr(c, gc))
                    stds.append(float(y.std()))
                    blurs.append(float(blur_effect(y)))
                ps, stds, blurs = np.array(ps), np.array(stds), np.array(blurs)
                if np.all(stds == stds[0]) or np.ptp(blurs) == 0:
                    n_tie += 1
                At = A[sl]
                c_gt = ((I[sl] - Rh[sl]) / np.maximum(At, 0.05)[..., None]).clip(0, 255)
                r_gt = np.asarray(Image.open(os.path.join(root, "r", sc)).convert("RGB"), np.float32)[sl]
                c_perf = ((I[sl] - r_gt) / np.maximum(At, 0.05)[..., None]).clip(0, 255)
                o = int(ps.argmax())
                i_std = int(np.argmin(np.abs(stds - tau_full)))
                i_stdc = int(np.argmin(np.abs(stds - tau_crop)))
                i_iqa = int(np.argmin(blurs))       # 最清晰
                i_iqamax = int(np.argmax(blurs))    # 字面"选最大"变体（越大越糊）
                rows.append(dict(set=s, level=lv, scene=sid, a_true=round(float(A[sl].mean()), 3),
                                 oracle_psnr=round(float(ps[o]), 2), oracle_a=ALPHAS[o],
                                 sel_std_psnr=round(float(ps[i_std]), 2), sel_std_a=ALPHAS[i_std],
                                 sel_stdc_psnr=round(float(ps[i_stdc]), 2), sel_stdc_a=ALPHAS[i_stdc],
                                 sel_iqa_psnr=round(float(ps[i_iqa]), 2), sel_iqa_a=ALPHAS[i_iqa],
                                 sel_iqamax_psnr=round(float(ps[i_iqamax]), 2), sel_iqamax_a=ALPHAS[i_iqamax],
                                 gtalpha_psnr=round(float(psnr(c_gt, gc)), 2),
                                 perfect_psnr=round(float(psnr(c_perf, gc)), 2),
                                 gt_std=round(float(gc.mean(2).std()), 2),
                                 gt_blur=round(float(blur_effect(gc.mean(2))), 4),
                                 cand_psnrs="/".join(f"{v:.1f}" for v in ps),
                                 cand_stds="/".join(f"{v:.1f}" for v in stds),
                                 cand_blurs="/".join(f"{v:.4f}" for v in blurs)))
    # ---------- 合并上批判据（逐样本），一并写出 ----------
    base = {(r["set"], r["level"], r["scene"]): r
            for r in csv.DictReader(open(os.path.join(PROJ, "eval/probe_b1c_candidates.csv")))}
    for r in rows:
        b = base[(r["set"], r["level"], r["scene"])]
        r["sel_ent_psnr"], r["sel_ent_a"] = b["sel_ent_psnr"], b["sel_ent_a"]
        r["sel_grad_psnr"], r["sel_grad_a"] = b["sel_grad_psnr"], b["sel_grad_a"]
        r["sel_mean_psnr"], r["sel_mean_a"] = b["sel_mean_psnr"], b["sel_mean_a"]
    out = os.path.join(PROJ, "eval/probe_b1c3_extgrid.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"样本 {len(rows)}（全膜裁片，k={len(ALPHAS)}）；stds 或 blur 全等的退化样本 {n_tie} 个")
    print(f"与上批 CSV 对齐: 本批 {len(rows)} / 上批 {len(base)}（差 {len(base) - len(rows)}）")
    print(f"-> {os.path.relpath(out, PROJ)}\n")

    orc = np.array([float(r["oracle_psnr"]) for r in rows])
    sel_keys = [("sel_ent", "熵最大"), ("sel_grad", "梯度能量最大"), ("sel_mean", "亮度先验96.4"),
                ("sel_std", "对比度匹配std先验"), ("sel_stdc", "对比度匹配std裁片先验"),
                ("sel_iqa", "IQA-blur最小(最清晰)"), ("sel_iqamax", "IQA-blur最大(字面变体)")]
    idx = lambda r, k: ALPHAS.index(float(r[f"{k}_a"]))
    oidx = lambda r: ALPHAS.index(float(r["oracle_a"]))

    def stats(sub, k):
        v = np.array([float(r[f"{k}_psnr"]) for r in sub])
        o = np.array([float(r["oracle_psnr"]) for r in sub])
        reg = o - v
        hit1 = np.mean([idx(r, k) == oidx(r) for r in sub])
        hit2 = np.mean([abs(idx(r, k) - oidx(r)) <= 1 for r in sub])
        return v.mean(), reg.mean(), np.percentile(reg, 90), np.mean(reg <= GATE_REGRET), hit1, hit2

    def table(title, sub):
        print(f"\n== {title}（n={len(sub)}） ==")
        print(f"{'判据':22s} {'PSNR':>6s} {'悔值':>6s} {'悔p90':>6s} {'悔≤0.91':>8s} {'top1':>5s} {'top2':>5s}")
        for k, name in sel_keys:
            p, m, q, f, h1, h2 = stats(sub, k)
            print(f"{name:22s} {p:6.2f} {m:+6.2f} {q:6.2f} {f:8.1%} {h1:5.0%} {h2:5.0%}")
        print(f"{'[上界] oracle@k':22s} {orc_sub(sub).mean():6.2f} {'-':>6s} {'-':>6s} {'-':>8s} {'-':>5s} {'-':>5s}")

    def orc_sub(sub):
        return np.array([float(r["oracle_psnr"]) for r in sub])

    table("口径A+B 总表", rows)
    for s in SETS:
        table(f"集 {s}", [r for r in rows if r["set"] == s])
    levels = sorted({r["level"] for r in rows}, key=lambda x: -float(x.split("_")[-1]))
    table("全档位", rows)
    print("\n按档位（set×level，均按上表同口径）:")
    for s in SETS:
        for lv in [l for l in levels if any(r['set'] == s and r['level'] == l for r in rows)]:
            sub = [r for r in rows if r["set"] == s and r["level"] == lv]
            line = f"  {s[:18]:18s} {lv:9s} n={len(sub):2d} oracle {orc_sub(sub).mean():5.2f} |"
            for k, name in sel_keys:
                _, m, _, f, h1, h2 = stats(sub, k)
                line += f" {name[:4]}:悔{m:+.1f}/t2{h2:.0%}"
            print(line)
    print(f"\n参考: 常数字段（恒选单档）PSNR / 悔值:")
    for i, a in enumerate(ALPHAS):
        v = np.array([float(r["cand_psnrs"].split("/")[i]) for r in rows])
        print(f"  恒选 α={a}: {v.mean():6.2f} dB, 悔值 {(orc - v).mean():+.2f} dB")
    print(f"  真α候选 {np.mean([float(r['gtalpha_psnr']) for r in rows]):.2f} | 真R+真α {np.mean([float(r['perfect_psnr']) for r in rows]):.2f}")
    print(f"\n诊断: GT 裁片 std 中位数 {np.median([float(r['gt_std']) for r in rows]):.2f} "
          f"(vs 训练先验 {tau_full:.2f}/{tau_crop:.2f}); GT 裁片 blur 中位数 {np.median([float(r['gt_blur']) for r in rows]):.4f}")


if __name__ == "__main__":
    main()
