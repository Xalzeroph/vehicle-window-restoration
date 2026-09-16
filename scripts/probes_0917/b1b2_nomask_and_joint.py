#!/usr/bin/env python3
"""批次1b 遗留两项复核 (P4 审阅意见):

(a) 不变量三口径: 差值比 D_j/D_k = (alpha_j-alpha_ref)/(alpha_k-alpha_ref) 的相对误差
    - gt   : GT alpha 掩膜 (alpha_l < 0.995, 复现上批 0.31%/0.25%)
    - none : 无掩膜
    - proxy: **可检测代理掩膜**——只用 I 的可观测量: 相邻档链式比例一致性
             chain(l_hi,l_mid,l_lo): a1=I_lhi-I_lmid, a2=I_lmid-I_llo,
             无 clip 时 a1/a2 = (l_hi-l_mid)/(l_mid-l_lo) 逐像素精确成立;
             clip 到 1 破坏之 => 保留 |a1/a2-rt| <= tau*rt 且 |a2|>floor 的像素。
             (由档位单调性, 该检验通过即保证该对涉及的更低档位都未 clip)
    - naive: 仅 |I_j-I_k|>eps 的近退化像素剔除 (任务书给的朴素代理, 作对照)

(b) 小邻域联合锚定: 对膜侧像素 p, q=其最近裸区像素(m<1e-3),
    step(p)=I_j(q)-I_j(p), diff(p)=I_j(p)-I_k(p), 稳健回归估 r=step/diff 整体斜率,
    与真值 r_true=(1-alpha_j)/(alpha_j-alpha_k) 比。三种真值聚合口径:
      bmean  : (1-mean alpha_j)/(mean alpha_j - mean alpha_k)  (上批口径)
      pxmed  : 逐像素比值的带内中位
      gtslope: 用 GT 重建 I=alpha*T+R 后按同一估计器算的斜率 (同聚合口径 oracle)
    估计器: meanratio(Σstep/Σdiff, 上批"分带均值"同族) / medratio(逐像素比值中位) /
            theilsen / huber0(Huber 过原点 IRLS)
    配置: prev_bug(复现上批配对 bug: fb=dm∈[6,12] 且 |p-q|<=6) / prev_fix(同带, q=最近裸区) /
          nbh4(dm<=4, 任务书小邻域) / band_{4,6,8,10}(dm∈d±1, 取带位置敏感性)
    另附 GT 诊断: |T_q-T_p| 与 step 的 rms 比 (T 剖面误差占比)。

用法: python3 scripts/probes_0917/b1b2_nomask_and_joint.py [--n 24]
产物: eval/probe_b1b2_nomask_joint.csv + 控制台表
"""
import argparse, csv, math, os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRIP = os.path.join(PROJ, "data/train_sets/train_triplets_rdt_0915")
TAGS = {1.0: "l100", 0.75: "l075", 0.6: "l060", 0.45: "l045", 0.3: "l030"}
LEVELS = [1.0, 0.75, 0.6, 0.45, 0.3]
PAIRS = [(1.0, 0.6), (0.6, 0.3)]          # (j, k); ref 固定 0.45 (同上批)
REF = 0.45
QDIFF = 1.5 / 255.0                        # diff 量化噪声底 (共享噪声 => diff 仅量化误差)
FLOOR = 4.0 / 255.0                        # 代理掩膜链式检验的分母下限
FIELDS = ["scene", "section", "pair", "config", "estimator", "n_px", "n_ok",
          "r_hat", "r_true_bmean", "r_true_pxmed", "r_true_gtslope", "r_true_prevconv",
          "err_bmean", "err_pxmed", "err_gtslope",
          "relerr_bmean", "relerr_pxmed", "relerr_gtslope", "relerr_prevconv",
          "degenerate", "note"]


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def med_iqr(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), float(v.max()), v.size


# ---------------- 估计器 (x=diff, y=step, 过原点) ----------------
def meanratio(x, y):
    return float(y.mean() / max(x.mean(), 1e-9))


def medratio(x, y):
    return float(np.median(np.divide(y, x, out=np.full_like(y, np.nan), where=np.abs(x) > 1e-9)))


def theilsen(x, y, cap=2000, seed=0):
    n = len(x)
    if n < 8:
        return np.nan
    if n > cap:
        idx = np.random.default_rng(seed).choice(n, cap, replace=False)
        x, y = x[idx], y[idx]
        n = cap
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    sel = np.triu(np.ones((n, n), bool), 1) & (np.abs(dx) > QDIFF)
    if sel.sum() < 8:
        return np.nan
    return float(np.median(dy[sel] / dx[sel]))


def huber0(x, y, c=1.345, iters=40):
    den = max((x * x).sum(), 1e-12)
    b = float((x * y).sum() / den)
    for _ in range(iters):
        r = y - b * x
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.abs(r) / s
        w = np.clip(u, 0, c) / np.maximum(u, 1e-12)
        den = max((w * x * x).sum(), 1e-12)
        b = float((w * x * y).sum() / den)
    return b


EST = {"meanratio": meanratio, "medratio": medratio, "theilsen": theilsen, "huber0": huber0}


# ---------------- (a) 不变量: 掩膜 ----------------
def chain_ok(Is, triple, tau, floor=FLOOR):
    hi, mid, lo = triple
    a1 = (Is[hi] - Is[mid]).mean(-1)
    a2 = (Is[mid] - Is[lo]).mean(-1)
    rt = (hi - mid) / (mid - lo)
    ok = (np.abs(a2) > floor) & (np.abs(a1) > floor)
    ratio = np.divide(a1, a2, out=np.zeros_like(a1), where=np.abs(a2) > 1e-9)
    return ok & (np.abs(ratio - rt) <= tau * abs(rt))


def proxy_mask(Is, pair, tau):
    """相邻三档链式检验; 单仪器 (j, j下邻档, j的次下邻档) 经单调性即可保证对中所有档未 clip。"""
    j = pair[0]
    i = LEVELS.index(j)
    if i + 2 >= len(LEVELS):
        return np.ones_like(Is[j], bool)
    return chain_ok(Is, tuple(LEVELS[i:i + 3]), tau)


def invariant_beta(num, den, mask, robust=False):
    n = num[mask].astype(np.float64)
    d = den[mask].astype(np.float64)
    if len(n) < 100:
        return np.nan, 0
    if not robust:
        return float((n * d).sum() / max((d * d).sum(), 1e-9)), len(n)
    return huber0(d, n), len(n)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.25, help="代理掩膜链式比例容差")
    args = ap.parse_args()

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(TRIP, "T")))[::max(1, 4000 // args.n)][:args.n]
    rows = []

    for sid in ids:
        T = img(os.path.join(TRIP, "T", f"{sid}.png"))
        R = img(os.path.join(TRIP, "R", f"{sid}.png"))
        Is, As = {}, {}
        for lv, tag in TAGS.items():
            Is[lv] = img(os.path.join(TRIP, "I", f"{sid}_{tag}.png"))
            A = np.load(os.path.join(TRIP, "alpha", f"{sid}_{tag}.npy")).astype(np.float32)
            As[lv] = A[:, :, 0] if A.ndim == 3 else A
        m = np.load(os.path.join(TRIP, "m", f"{sid}.npy")).astype(np.float32)
        bare_m = m < 1e-3
        bm = np.stack(np.nonzero(bare_m), 1)
        fm = np.stack(np.nonzero(~bare_m), 1)
        dm_m, near_m = cKDTree(bm).query(fm)        # 物理裸区(m<1e-3) 的距离/最近裸区

        # ================= (a) 不变量三口径 =================
        for j, k in PAIRS:
            Dj = (Is[j] - Is[REF]).mean(-1)
            Dk = (Is[k] - Is[REF]).mean(-1)
            true_beta = (j - REF) / (k - REF)
            masks = {}
            masks["gt"] = (As[j] < 0.995) & (As[k] < 0.995) & (As[REF] < 0.995)
            masks["none"] = np.ones_like(bare_m)
            masks["proxy"] = proxy_mask(Is, (j, k), args.tau)
            masks["naive"] = (np.abs(Is[j] - Is[k]).mean(-1) > QDIFF)   # 朴素: 只剔近退化像素
            for mode, msk in masks.items():
                for estr, rob in [("ls", False), ("huber", True)]:
                    beta, npx = invariant_beta(Dj, Dk, msk, robust=rob)
                    iou = np.nan
                    if mode == "proxy" and masks["gt"].sum() > 0:
                        inter = (msk & masks["gt"]).sum()
                        iou = float(inter / max((msk | masks["gt"]).sum(), 1))
                    row = dict.fromkeys(FIELDS, "")
                    row.update(dict(scene=sid, section="invariant", pair=f"{j:g}/{k:g}", config=mode,
                                    estimator=estr, n_px=npx, n_ok=npx,
                                    r_hat=round(beta, 6) if np.isfinite(beta) else "",
                                    r_true_bmean=round(true_beta, 6),
                                    err_bmean=round(abs(beta - true_beta), 6) if np.isfinite(beta) else "",
                                    relerr_bmean=round(abs(beta - true_beta) / abs(true_beta), 6) if np.isfinite(beta) else "",
                                    degenerate=0, note=f"iou_gt={round(iou,3)}" if np.isfinite(iou) else ""))
                    rows.append(row)

        # ================= (b) 小邻域联合锚定 =================
        for j, k in PAIRS:
            abare = (As[j] > 0.995) & (As[k] > 0.995)          # 上批口径 (alpha>=0.995 视作裸)
            ba = np.stack(np.nonzero(abare), 1)
            fa = np.stack(np.nonzero(~abare), 1)
            dm_a, near_a = cKDTree(ba).query(fa)
            Ij, Ik = Is[j], Is[k]
            # GT 重建 (无噪声, 生成器同在 [0,1] 截断)
            Igt_j = np.clip(As[j][..., None] * T + R, 0, 1).mean(-1)
            Igt_k = np.clip(As[k][..., None] * T + R, 0, 1).mean(-1)

            configs = {}
            # 上批配对 (保留其 bug 以对照): fb=dm_a in [6,12] 且 最近裸区距离<=6
            fb = (dm_a >= 6) & (dm_a <= 12)
            px_fb = fa[fb]; q_fb = ba[near_a[fb]]
            dd = np.sqrt(((px_fb - q_fb) ** 2).sum(1))
            configs["prev_bug0"] = (px_fb[dd <= 6], q_fb[dd <= 6], fa[fb], True)   # 上批原始口径(无 floor)
            configs["prev_bug"] = (px_fb[dd <= 6], q_fb[dd <= 6], fa[fb], False)  # 第3项=整个 fb 带坐标(上批真值口径)
            configs["prev_fix"] = (px_fb, q_fb, None, False)
            # 新口径: 物理裸区 (m<1e-3) 为 q
            for d in [4, 6, 8, 10]:
                sel = np.abs(dm_m - d) <= 1
                configs[f"band{d}"] = (fm[sel], bm[near_m[sel]], None, False)
            sel = dm_m <= 4
            configs["nbh4"] = (fm[sel], bm[near_m[sel]], None, False)

            for cname, (px, q, fb_sel, nofloor) in configs.items():
                if len(px) < (8 if nofloor else 50):
                    continue
                pr, pc = px[:, 0], px[:, 1]
                qr, qc = q[:, 0], q[:, 1]
                step = (Ij[qr, qc] - Ij[pr, pc]).mean(-1)
                diff = (Ij[pr, pc] - Ik[pr, pc]).mean(-1)
                step_g = (Igt_j[qr, qc] - Igt_j[pr, pc])
                diff_g = Igt_j[pr, pc] - Igt_k[pr, pc]
                ok = np.ones(len(px), bool) if nofloor else (np.abs(diff) > QDIFF)
                n_ok = int(ok.sum())
                if n_ok < (8 if nofloor else 40):
                    continue
                aj = As[j][pr, pc][ok]; ak = As[k][pr, pc][ok]
                r_px = (1.0 - aj) / np.maximum(aj - ak, 1e-6)
                r_true_pxmed = float(np.median(r_px))
                r_true_bmean = (1.0 - aj.mean()) / max(aj.mean() - ak.mean(), 1e-9)
                r_true_prevconv = np.nan
                if fb_sel is not None:      # 上批口径: 真值取整个 fb 带 (dm 6-12) 的 alpha 均值
                    aj_fb = As[j][fb_sel[:, 0], fb_sel[:, 1]].mean()
                    ak_fb = As[k][fb_sel[:, 0], fb_sel[:, 1]].mean()
                    r_true_prevconv = (1.0 - aj_fb) / max(aj_fb - ak_fb, 1e-9)
                degenerate = int(r_true_bmean < 0.05)
                Tp = T[pr, pc].mean(-1)[ok]; Tq = T[qr, qc].mean(-1)[ok]
                tgrad_ratio = float(np.sqrt(((Tq - Tp) ** 2).mean()) / max(np.sqrt((step_g[ok] ** 2).mean()), 1e-9))
                # GT 复算斜率 (同估计器, 全部估计器都算作 oracle)
                r_gt = {}
                xg, yg = diff_g[ok], step_g[ok]
                for en, fn in EST.items():
                    r_gt[en] = fn(xg, yg)
                x, y = diff[ok], step[ok]
                for en, fn in EST.items():
                    rh = fn(x, y)
                    rgs = r_gt[en]
                    row = dict.fromkeys(FIELDS, "")
                    row.update(dict(scene=sid, section="joint", pair=f"{j:g}/{k:g}", config=cname,
                               estimator=en, n_px=len(px), n_ok=n_ok,
                               r_hat=round(rh, 6) if np.isfinite(rh) else "",
                               r_true_bmean=round(r_true_bmean, 6),
                               r_true_pxmed=round(r_true_pxmed, 6),
                               r_true_gtslope=round(rgs, 6) if np.isfinite(rgs) else "",
                               r_true_prevconv=round(r_true_prevconv, 6) if np.isfinite(r_true_prevconv) else "",
                               err_bmean=round(abs(rh - r_true_bmean), 6) if np.isfinite(rh) else "",
                               err_pxmed=round(abs(rh - r_true_pxmed), 6) if np.isfinite(rh) else "",
                               err_gtslope=round(abs(rh - rgs), 6) if np.isfinite(rh) and np.isfinite(rgs) else "",
                               degenerate=degenerate,
                               note=f"tgrad/step={tgrad_ratio:.2f}"))
                    for tgt, ref_v in [("bmean", r_true_bmean), ("pxmed", r_true_pxmed), ("gtslope", r_gt[en]),
                                       ("prevconv", r_true_prevconv)]:
                        re_ = np.nan
                        if np.isfinite(rh) and np.isfinite(ref_v) and abs(ref_v) > 1e-6:
                            re_ = abs(rh - ref_v) / abs(ref_v)
                        row[f"relerr_{tgt}"] = round(re_, 6) if np.isfinite(re_) else ""
                    rows.append(row)

    out = os.path.join(PROJ, "eval/probe_b1b2_nomask_joint.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {os.path.relpath(out, PROJ)}  ({len(rows)} rows)")

    # ================= 控制台汇总 =================
    inv = [r for r in rows if r["section"] == "invariant"]
    def re_of(r, key="relerr_bmean"):
        v = r.get(key, "")
        return float(v) if v != "" else np.nan
    print("\n[A] 不变量 D_j/D_k 相对误差 (|<r>| = 中位/最大; LS 与 Huber 两种估计):")
    for pair in [f"{j:g}/{k:g}" for j, k in PAIRS]:
        for est in ["ls", "huber"]:
            for mode in ["gt", "none", "proxy", "naive"]:
                es = [re_of(r) for r in inv if r["pair"] == pair and r["config"] == mode and r["estimator"] == est]
                if not es:
                    continue
                es = np.array(es)
                print(f"  pair {pair} [{est:5s}] {mode:5s}: median {np.nanmedian(es):.4f}  max {np.nanmax(es):.4f}  mean {np.nanmean(es):.4f}")
        ious = [float(r["note"].split("=")[1]) for r in inv if r["pair"] == pair and r["config"] == "proxy" and r["note"]]
        if ious:
            print(f"  pair {pair} proxy vs gt 掩膜 IoU: median {np.median(ious):.3f}")
    # 上批 CSV 对照: prev_bug 的 r_hat 应≈上批 m_hat (上批只做 0.6/0.3); relerr_prevconv 复现上批口径
    old = os.path.join(PROJ, "eval/probe_b1b_pairing.csv")
    if os.path.exists(old):
        with open(old) as fp:
            oc = {r["scene"]: r for r in csv.DictReader(fp)}
        dif = []
        for r in [r for r in rows if r["section"] == "joint" and r["config"] == "prev_bug0"
                  and r["pair"] == "0.6/0.3" and r["estimator"] == "meanratio"]:
            o = oc.get(r["scene"])
            if o and o["m_hat"] and r["r_hat"] != "":
                dif.append(abs(float(o["m_hat"]) - float(r["r_hat"])))
        if dif:
            dif = np.array(dif)
            print(f"\n[对照] 复现上批原始口径 prev_bug0: |Δ m_hat| 中位 {np.median(dif):.3f}, "
                  f"≤0.01 的 {int((dif <= 0.01).sum())}/{len(dif)} 场景 (巨大差仅来自 diff̄≈0 的退化场景)")
        es_all = []
        for r in [r for r in rows if r["section"] == "joint" and r["config"] == "prev_bug0"
                  and r["pair"] == "0.6/0.3" and r["estimator"] == "meanratio"]:
            o = oc.get(r["scene"])
            if o and o["m_hat"] and o["m_true"] and r["r_hat"] != "":
                es_all.append(abs(float(r["r_hat"]) - float(o["m_true"])) / abs(float(o["m_true"])))
        if es_all:
            md, q1, q3, mx, n = med_iqr(es_all)
            print(f"[对照] 上批口径复现 (prev_bug0 r_hat 对旧 CSV m_true 真值): 中位 {md:.3f} [{q1:.3f},{q3:.3f}] 最大 {mx:.1f} (n={n})")

    jt = [r for r in rows if r["section"] == "joint"]
    print("\n[B] 联合回归 r=step/diff 相对误差 (中位 [q25,q75] / 最大; 剔除 r_true<0.05 的退化场景):")
    for pair in [f"{j:g}/{k:g}" for j, k in PAIRS]:
        print(f"  pair {pair}:")
        for cname in ["prev_bug0", "prev_bug", "prev_fix", "nbh4", "band4", "band6", "band8", "band10"]:
            line = f"    {cname:9s}"
            for en in ["meanratio", "medratio", "theilsen", "huber0"]:
                sub = [r for r in jt if r["pair"] == pair and r["config"] == cname and r["estimator"] == en]
                es = [re_of(r) for r in sub if r["degenerate"] == 0]
                if not es:
                    line += f" {en[:4]}: n/a"
                    continue
                md, q1, q3, mx, n = med_iqr(es)
                line += f" {en[:4]}: {md:.3f} [{q1:.3f},{q3:.3f}] n={n}"
            print(line)
        # 三口径对照 (最佳配置)
        for cname, en in [("prev_fix", "meanratio"), ("band10", "meanratio"), ("band10", "medratio")]:
            sub = [r for r in jt if r["pair"] == pair and r["config"] == cname and r["estimator"] == en
                   and r["degenerate"] == 0]
            if not sub:
                continue
            txt = f"    ({cname}/{en} 对三口径 中位:"
            for tgt in ["bmean", "pxmed", "gtslope"]:
                md, q1, q3, mx, n = med_iqr([re_of(r, f"relerr_{tgt}") for r in sub])
                txt += f" {tgt} {md:.3f}[{q1:.3f},{q3:.3f}]" if n else f" {tgt} n/a"
            print(txt + ")")
    print("\n[B2] T 剖面误差占比 |T_q-T_p|_rms / |step_g|_rms (诊断, 按 config):")
    for pair in [f"{j:g}/{k:g}" for j, k in PAIRS]:
        for cname in ["nbh4", "band4", "band6", "band8", "band10", "prev_fix"]:
            vals = [float(r["note"].split("=")[1]) for r in jt
                    if r["pair"] == pair and r["config"] == cname and r["estimator"] == "meanratio" and "tgrad" in r["note"]]
            if vals:
                print(f"  pair {pair} {cname:9s}: median {np.median(vals):.2f}")
    print("\n[B3] 退化场景数 (r_true_bmean<0.05, 即档 j 在该带内几乎全 clip):")
    for pair in [f"{j:g}/{k:g}" for j, k in PAIRS]:
        for cname in ["nbh4", "band6", "band8", "prev_fix"]:
            sub = [r for r in jt if r["pair"] == pair and r["config"] == cname and r["estimator"] == "meanratio"]
            if sub:
                print(f"  pair {pair} {cname:9s}: {sum(r['degenerate'] for r in sub)}/{len(sub)}")


if __name__ == "__main__":
    main()
