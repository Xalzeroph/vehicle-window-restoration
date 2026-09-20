#!/usr/bin/env python3
"""批次2a（图像启发式路线）：零模型膜/裸分割 → 边界台阶锚 α̂。

背景：批次 1a 用 **GT α 图** 定义带（膜侧带 = 距裸区 6–12px，裸侧带 = 距膜 ≤6px）时
   α̂ = (mean(I_膜带) − mean(R̂_膜带)) / (mean(I_裸带) − mean(R̂_裸带))
MAE ≈ 0.080（R̂ = 冻结分离器 rdt_plain 的 r_hat.png）。本探针回答：
   **完全不用 GT（也不用任何模型，除已有 R̂）**，仅从图像本身找膜/裸/边界，还剩多少精度。

方法（每图独立、逐档独立）：
  1) 候选生成：灰度大核低通 S=Gaussian(I,σ=12) 上做分位数扫描 q∈[0.05,0.42]，
     取亮侧分量最大连通域（+填洞）→ 一批候选裸区掩膜（面积先验 0.05–0.35）；
     另附一路 R̂ 辅助（D=I−R̂ 同扫描）作次要变体 imgD（用到了 r_hat）。
  2) 候选评分（无 GT）：沿候选轮廓法线 ±9px 的台阶中位数 med（在 σ=4 模糊图量）
     与轮廓粗糙度 rough（沿切向坐标的二阶差分 RMS，生成器边界是平滑曲线，属先验知识），
     score = med（取 argmax）。规则对照（v21 全档聚类 MAE）：med 0.112 最优；
     med/(1+rough/15) 0.130；min(med_I, med_D) 0.127；med×posf 0.115——均无增益（P4 记录）。
     边界精化（refine_mask：分箱法线中点穿越 + 稳健多项式重栅格化）实测对 img 无增益、
     对 oracle 反而变差（见 P4 记录），故默认关闭，--refine 可选开启。
  3) 对照变体：
     - gt   ：GT α>0.995 定义带（批次 1a 口径，同表并列）；
     - oracle：在同一候选族内用 GT IoU 选最优候选（该族的上限，非实用管线数字）；
     - imgD ：候选在 D=I−R̂ 上生成（R̂ 辅助，次要变体）。
  4) 带定义（各变体统一）：膜侧带 = 膜像素且 dist(裸)∈[6,12]；裸侧带 = 裸像素且 dist(膜)≤6。
结果（P4，聚类 MAE / 95%CI）：v21 img 0.112 [0.068,0.155]，GT 带对照 0.114 [0.075,0.152]；
   v23 img 0.160 [0.134,0.187]，对照 0.091 [0.070,0.111]。img 掩膜 IoU：v21 中位 0.29（28/36<0.5，
   强反射/高 α 场景掩膜落在内容边缘），v23 中位 0.77。oracle（族内最优）MAE 反而 0.155/0.160，
   说明 e 在带内定义时会与掩膜位置自洽、低估定位错误（故附 e_gb 诊断）。
真值口径：α_true = 图内所定义带内 GT α 均值（与 1a 一致），另附诊断列
   e_gb = |α̂ − α_true(GT 带)|（高 α 档候选整块落在内容边缘时，带内 α_true 与带定义同错，
   e 会低估误差，e_gb 以 canonical GT 带为参照）。误差 e=|α̂−α_true| 逐行 → 场景内跨档平均
→ 跨场景 t 95%CI。样本：v21 = 12 场景×3 档（df=11, t=2.201）；v23 = 48 场景×4 档
（0.3/0.2/0.1/0.05；df=47, t=2.0117；alpha_1 档 α 图大面积 clip 到 1，带定义退化，仅入 CSV 标注不入门）。
  注：v21 的 alpha_1 档同样 clip 退化（t00: 46% 画面被 clip 进"裸区"，见 P4），
  故 v21 另报 0.6+0.3 两档聚类 CI 作非退化口径。

用法: python3 scripts/probes_0917/b2a_pipeline_img.py [--sets ...] [--limit N] [--refine]
产物: eval/probe_b2a_img_pipeline.csv + 控制台表
"""
import argparse, csv, os, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import t as tdist

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BARE_T, FILM_T2 = 0.995, 0.99
D_LO, D_HI, D_NEAR = 6, 12, 6
MIN_BAND = 30
LEVELS = {"synth_rrw_v21_0913": ["1", "0.6", "0.3"],
          "synth_rrw_v23_lowscan": ["0.3", "0.2", "0.1", "0.05"]}
GATE_LEVELS = {"synth_rrw_v21_0913": ["1", "0.6", "0.3"],
               "synth_rrw_v23_lowscan": ["0.3", "0.2", "0.1", "0.05"]}
SIG_COARSE, QGRID = 12.0, np.arange(0.05, 0.42, 0.01)
ROUGH_TAU = 15.0


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def gray1(x):
    return x.mean(2)


# ---------------- 候选生成 ----------------
def gen_candidates(S, frac_lo=0.05, frac_hi=0.35):
    out = []
    for q in QGRID:
        thr = np.quantile(S, 1 - q)
        m = S >= thr
        lab, nl = ndi.label(m)
        if nl == 0:
            continue
        sizes = ndi.sum(m, lab, range(1, nl + 1))
        mm = ndi.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)
        if frac_lo <= mm.mean() <= frac_hi:
            out.append((float(q), mm))
    return out


def cand_feats(mm, Ss):
    """轮廓台阶中位数 + 轮廓粗糙度（沿切向的 s 二阶差分 RMS）。"""
    h, w = mm.shape
    b = mm & ~ndi.binary_erosion(mm, np.ones((3, 3), bool))
    b[:4, :] = False; b[-4:, :] = False; b[:, :4] = False; b[:, -4:] = False
    ys, xs = np.nonzero(b)
    if len(ys) < 60:
        return None
    ms = ndi.gaussian_filter(mm.astype(np.float32), 2.5)
    gym, gxm = np.gradient(ms)
    nv = np.stack([gxm[ys, xs], gym[ys, xs]], 1)
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9)

    def samp(off):
        yy = np.clip((ys + nv[:, 1] * off).round().astype(int), 0, h - 1)
        xx = np.clip((xs + nv[:, 0] * off).round().astype(int), 0, w - 1)
        return Ss[yy, xx]
    step = samp(9) - samp(-9)
    med = float(np.median(step))
    # 粗糙度：PCA 切向排序后的法向残差二阶差分
    pts = np.stack([xs, ys], 1).astype(np.float64)
    P = pts - pts.mean(0)
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    t = P @ vt[0]; r = P @ vt[1]
    o = np.argsort(t)
    ro = r[o]
    rough = float(np.sqrt(np.mean(np.diff(ro, 2) ** 2))) if len(ro) > 5 else 1e9
    return dict(med=med, rough=rough, frac=float(mm.mean()),
                posf=float(np.mean(step > 0.02)), nstep=int(len(step)))


def select_candidate(cands, Ss, rule="med"):
    """选择规则（无 GT）：med = 轮廓法线 ±9px 台阶中位数最大（实测最优，见 P4 分析）。
    rough 变体（med/(1+rough/15)，粗糙度先验）作对照记录。"""
    best, best_sc, infos = None, -1e18, []
    for q, mm in cands:
        f = cand_feats(mm, Ss)
        if f is None:
            infos.append(dict(q=q, score=np.nan, med=np.nan, rough=np.nan, frac=float(mm.mean())))
            continue
        sc = f["med"] if rule == "med" else f["med"] / (1.0 + f["rough"] / ROUGH_TAU)
        infos.append(dict(q=q, score=sc, med=f["med"], rough=f["rough"], frac=f["frac"]))
        if sc > best_sc:
            best_sc, best = sc, (q, mm, f)
    return best, infos


def oracle_candidate(cands, bare_gt):
    best, iou = None, -1.0
    for q, mm in cands:
        i = (mm & bare_gt).sum() / max((mm | bare_gt).sum(), 1)
        if i > iou:
            iou, best = i, (q, mm)
    return best, iou


# ---------------- 边界精化 ----------------
def refine_mask(mm, g, n_bins=64, deg=2):
    """候选掩膜 → 逐切向分箱法线剖面中点穿越 → 稳健多项式边界 → 重栅格化。"""
    h, w = mm.shape
    Sf = ndi.gaussian_filter(g, 2.5)
    b = mm & ~ndi.binary_erosion(mm, np.ones((3, 3), bool))
    b[:4, :] = False; b[-4:, :] = False; b[:, :4] = False; b[:, -4:] = False
    ys, xs = np.nonzero(b)
    if len(ys) < 60:
        return mm, dict(ok=False, why="contour small")
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    P = pts - c
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    v0, v1 = vt[0], vt[1]                      # v0 切向, v1 法向（符号待定）
    tc = P @ v0; sc = P @ v1
    tlo, thi = tc.min(), tc.max()
    edges = np.linspace(tlo, thi, n_bins + 1)
    tcen, sstar = [], []
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    T = (X - c[0]) * v0[0] + (Y - c[1]) * v0[1]
    Sc = (X - c[0]) * v1[0] + (Y - c[1]) * v1[1]
    for i in range(n_bins):
        sel = (tc >= edges[i]) & (tc < edges[i + 1])
        if sel.sum() < 20:
            continue
        s0 = float(np.median(sc[sel]))
        # 相邻箱也用上（像素少时更稳）：取该箱 t 邻域 ±1 箱
        m2 = (T >= edges[i]) & (T < edges[i + 1])
        if m2.sum() < 40:
            continue
        sv = Sc[m2]; vv = Sf[m2]
        offs = np.arange(-24, 25, 1.0)
        prof = np.full(len(offs), np.nan)
        j = np.clip(np.digitize(sv - s0, np.concatenate([offs - 0.5, [offs[-1] + 0.5]])) - 1, 0, len(offs) - 1)
        for k in range(len(offs)):
            v = vv[j == k]
            if len(v) >= 8:
                prof[k] = v.mean()
        ok = ~np.isnan(prof)
        if ok.sum() < 30:
            continue
        deep_p = prof[offs >= 12]; deep_n = prof[offs <= -12]
        if np.isnan(deep_n).all() or np.isnan(deep_p).all():
            continue
        mid = 0.5 * (np.nanmean(deep_p) + np.nanmean(deep_n))
        # 找离 o=0 最近的中点穿越
        order = np.argsort(np.abs(offs))
        cs = None
        for k in order:
            k2 = k - 1 if k > 0 else k + 1
            if not (ok[k] and ok[k2]):
                continue
            a, bq = prof[k], prof[k2]
            if (a - mid) * (bq - mid) <= 0:
                d = bq - a
                fr = 0.5 if abs(d) < 1e-9 else (mid - a) / d
                cs = s0 + offs[k] + fr * (offs[1] - offs[0]) * (1 if k2 > k else -1)
                break
        if cs is not None:
            tcen.append(0.5 * (edges[i] + edges[i + 1]))
            sstar.append(cs)
    if len(sstar) < 12:
        return mm, dict(ok=False, why=f"bins {len(sstar)}")
    tcen = np.array(tcen); sstar = np.array(sstar)
    keep = np.ones(len(tcen), bool)
    for _ in range(4):
        if keep.sum() < deg + 2:
            break
        coef = np.polyfit(tcen[keep], sstar[keep], deg)
        resid = sstar - np.polyval(coef, tcen)
        sd = resid[keep].std()
        keep = np.abs(resid) < max(1.5, 2.5 * sd)
    if keep.sum() < deg + 2:
        return mm, dict(ok=False, why="fit fail")
    coef = np.polyfit(tcen[keep], sstar[keep], deg)
    s_poly = np.polyval(coef, T.ravel())
    r = (Sc.ravel() - s_poly).reshape(h, w)
    # 方向判定：使重栅格化掩膜与候选掩膜 IoU 较大的一侧为裸区
    m_pos = r > 0
    iou_pos = (m_pos & mm).sum() / max((m_pos | mm).sum(), 1)
    mask_new = m_pos if iou_pos >= 0.5 else ~m_pos
    return mask_new, dict(ok=True, n_bins=len(sstar), n_keep=int(keep.sum()),
                          resid_sd=float(np.abs(resid[keep]).std()),
                          flip=bool(iou_pos < 0.5))


# ---------------- 带 + α̂ ----------------
def bands(bare):
    film = ~bare
    if bare.sum() < 50 or film.sum() < 50:
        return None, None
    d_bare = ndi.distance_transform_edt(film)
    d_film = ndi.distance_transform_edt(bare)
    film_b = film & (d_bare >= D_LO) & (d_bare <= D_HI)
    bare_b = bare & (d_film <= D_NEAR)
    if film_b.sum() < MIN_BAND or bare_b.sum() < MIN_BAND:
        return None, None
    return film_b, bare_b


def alpha_hat(I, Rg, film_b, bare_b):
    If = I[film_b].mean(); Ib = I[bare_b].mean()
    Rf = Rg[film_b].mean(); Rb = Rg[bare_b].mean()
    den = Ib - Rb
    if abs(den) < 1e-6:
        return np.nan
    return float((If - Rf) / den)


def ci95(vals):
    v = np.asarray(vals, float); n = len(v)
    m = v.mean(); sd = v.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tc = tdist.ppf(0.975, n - 1) if n > 1 else np.nan
    return dict(n=n, mean=m, sd=sd, se=se, t=tc, lo=m - tc * se, hi=m + tc * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="synth_rrw_v21_0913,synth_rrw_v23_lowscan")
    ap.add_argument("--limit", type=int, default=0, help="每集场景数上限（调试用）")
    ap.add_argument("--refine", action="store_true", help="开启边界精化（实测无增益，默认关）")
    args = ap.parse_args()

    rows = []
    t00 = time.time()
    for sname in args.sets.split(","):
        root = os.path.join(PROJ, "data/test_sets", sname)
        scenes = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "gt")))
        if args.limit:
            scenes = scenes[:args.limit]
        for lv in LEVELS[sname]:
            for sc in scenes:
                Ip = os.path.join(root, "levels", f"alpha_{lv}", sc + ".png")
                A = np.load(os.path.join(root, "alpha", f"{sc}_a{lv}.npy")).astype(np.float32)
                if A.ndim == 3:
                    A = A[:, :, 0]
                rp = os.path.join(PROJ, "runs/pred/rdt_plain", sname, f"alpha_{lv}", sc, "r_hat.png")
                if not (os.path.exists(Ip) and os.path.exists(rp)):
                    rows.append(dict(set=sname, level=lv, scene=sc, method="img", status="missing", alpha_hat="", alpha_true="", e="", note=""))
                    continue
                I = load_rgb(Ip); R = load_rgb(rp)
                g = gray1(I); Rg = gray1(R)
                S = ndi.gaussian_filter(g, SIG_COARSE)
                Ss = ndi.gaussian_filter(g, 4.0)
                bare_gt = A > BARE_T
                cands = gen_candidates(S)
                # ---- 选择（无 GT）----
                sel, infos = select_candidate(cands, Ss)
                # ---- oracle（族上限，用 GT IoU 选）----
                orc, orc_iou = oracle_candidate(cands, bare_gt)
                # ---- 各变体算 α̂ ----
                def eval_mask(mm, tag, q=None):
                    if mm is None:
                        return None
                    if args.refine:
                        mm2, rinfo = refine_mask(mm, g)
                    else:
                        mm2, rinfo = mm, dict(ok=True)
                    fb, bb = bands(mm2)
                    if fb is None:
                        return None
                    ah = alpha_hat(I, Rg, fb, bb)
                    at = float(A[fb].mean())
                    iou = (mm2 & bare_gt).sum() / max((mm2 | bare_gt).sum(), 1)
                    return dict(tag=tag, alpha_hat=ah, alpha_true=at, iou=iou, q=q,
                                film_frac=float((~mm2).mean()), rinfo=rinfo)
                # ---- GT 带对照（canonical，批次 1a 口径；同时提供 e_gb 参照）----
                at_gt = np.nan
                out_gt = None
                bgt = np.stack(np.nonzero(bare_gt), 1)
                if bare_gt.sum() > 50:
                    from scipy.spatial import cKDTree
                    f_all = np.stack(np.nonzero(A < FILM_T2), 1)
                    d_bare, _ = cKDTree(bgt).query(f_all)
                    fb = f_all[(d_bare >= D_LO) & (d_bare <= D_HI)]
                    d_film, _ = cKDTree(f_all).query(bgt)
                    bb = bgt[d_film <= D_NEAR]
                    if len(fb) >= MIN_BAND and len(bb) >= MIN_BAND:
                        fy, fx = fb[:, 0], fb[:, 1]; by, bx = bb[:, 0], bb[:, 1]
                        fb_m = np.zeros(A.shape, bool); fb_m[fy, fx] = True
                        bb_m = np.zeros(A.shape, bool); bb_m[by, bx] = True
                        at_gt = float(A[fb_m].mean())
                        out_gt = dict(tag="gt", alpha_hat=alpha_hat(I, Rg, fb_m, bb_m),
                                      alpha_true=at_gt, iou=np.nan, q=None,
                                      film_frac=float((A < FILM_T2).mean()), rinfo=dict(ok=True))
                out = {}
                if sel is not None:
                    out["img"] = eval_mask(sel[1], "img", sel[0])
                if orc is not None:
                    out["oracle"] = eval_mask(orc[1], "oracle", orc[0])
                # ---- imgD 变体：D=I−R̂ 上同扫描同选择（次要变体，记录其增/减益）----
                Sd = ndi.gaussian_filter(g - Rg, SIG_COARSE)
                cands_d = gen_candidates(Sd)
                seld, _ = select_candidate(cands_d, Ss)
                if seld is not None:
                    out["imgD"] = eval_mask(seld[1], "imgD", seld[0])
                if out_gt is not None:
                    out["gt"] = out_gt
                for tag in ("img", "imgD", "oracle", "gt"):
                    o = out.get(tag)
                    if o is None:
                        rows.append(dict(set=sname, level=lv, scene=sc, method=tag, status="na",
                                         alpha_hat="", alpha_true="", e="", e_gb="", iou="", film_frac="",
                                         q="", note=""))
                        continue
                    e = abs(o["alpha_hat"] - o["alpha_true"]) if o["alpha_hat"] == o["alpha_hat"] else np.nan
                    egb = (abs(o["alpha_hat"] - at_gt) if (o["alpha_hat"] == o["alpha_hat"] and at_gt == at_gt)
                           else np.nan)
                    rows.append(dict(set=sname, level=lv, scene=sc, method=tag, status="ok",
                                     alpha_hat=round(o["alpha_hat"], 4) if o["alpha_hat"] == o["alpha_hat"] else "",
                                     alpha_true=round(o["alpha_true"], 4), e=round(e, 4) if e == e else "",
                                     e_gb=round(egb, 4) if egb == egb else "",
                                     iou=round(o["iou"], 3) if o["iou"] == o["iou"] else "",
                                     film_frac=round(o["film_frac"], 3),
                                     q=round(o["q"], 2) if o.get("q") is not None else "",
                                     note="" if o["rinfo"].get("ok") else o["rinfo"].get("why", "")))
        print(f"  [{sname}] {time.time()-t00:.0f}s")

    # ---------------- 汇总 ----------------
    def report(method):
        print(f"\n================ 方法 {method} ================")
        for sname in args.sets.split(","):
            sub = [r for r in rows if r["set"] == sname and r["method"] == method and r["status"] == "ok"]
            if not sub:
                continue
            for lv in LEVELS[sname]:
                lvr = [r for r in sub if r["level"] == lv]
                ev = np.array([float(r["e"]) for r in lvr if r["e"] != ""], float)
                at = np.array([float(r["alpha_true"]) for r in lvr if r["e"] != ""], float)
                eg = np.array([float(r["e_gb"]) for r in lvr if r["e_gb"] != ""], float)
                if len(ev) == 0:
                    continue
                egs = f" | e_gb {eg.mean():.3f}" if len(eg) else ""
                print(f"  {sname[:22]:>22} a{lv:>4}: n={len(ev):2d} α_真均值 {at.mean():.3f} MAE {ev.mean():.3f} "
                      f"（逐行 max {ev.max():.3f}）{egs}")
            # gate levels only
            gsub = [r for r in sub if r["level"] in GATE_LEVELS[sname] and r["e"] != ""]
            print(f"  [{sname}] （四档/三档口径 + 场景聚类 CI）")
            for lv in GATE_LEVELS[sname]:
                lvr = [r for r in sub if r["level"] == lv and r["e"] != ""]
                c = ci95([float(r["e"]) for r in lvr])
                eg = [float(r["e_gb"]) for r in lvr if r["e_gb"] != ""]
                egs = f" e_gb {np.mean(eg):.3f}" if eg else ""
                print(f"    a{lv:>4}: 12/48-场景口径 n={c['n']:2d} MAE {c['mean']:.3f} 95%CI [{c['lo']:.3f},{c['hi']:.3f}]{egs}")
            vals = [r for r in gsub]
            byc = {}
            for r in vals:
                byc.setdefault(r["scene"], []).append(float(r["e"]))
            cm = [float(np.mean(v)) for v in byc.values()]
            c = ci95(cm)
            df = c["n"] - 1
            print(f"    ★场景聚类（{c['n']} 簇, df={df}, t={c['t']:.4f}）整体 MAE {c['mean']:.3f} "
                  f"95%CI [{c['lo']:.3f},{c['hi']:.3f}]  逐行 MAE {np.mean([float(r['e']) for r in gsub]):.3f}")
            # e_gb 口径（以 canonical GT 带 α_true 为参照）
            if any(r["e_gb"] != "" for r in gsub):
                byc2 = {}
                for r in gsub:
                    if r["e_gb"] != "":
                        byc2.setdefault(r["scene"], []).append(float(r["e_gb"]))
                cm2 = [float(np.mean(v)) for v in byc2.values()]
                c2 = ci95(cm2)
                print(f"    ★场景聚类 e_gb 口径（对照 canonical GT 带）：MAE {c2['mean']:.3f} "
                      f"95%CI [{c2['lo']:.3f},{c2['hi']:.3f}]（{c2['n']} 簇）")
            # v21 非退化两档口径（a1 被 clip 污染）
            if sname == "synth_rrw_v21_0913":
                sub2 = [r for r in sub if r["level"] in ("0.6", "0.3") and r["e"] != ""]
                byc3 = {}
                for r in sub2:
                    byc3.setdefault(r["scene"], []).append(float(r["e"]))
                if byc3:
                    c3 = ci95([float(np.mean(v)) for v in byc3.values()])
                    print(f"    ○v21 非退化两档（0.6+0.3）场景聚类 MAE {c3['mean']:.3f} "
                          f"95%CI [{c3['lo']:.3f},{c3['hi']:.3f}]（{c3['n']} 簇, df={c3['n']-1}, t={c3['t']:.4f}）")
            ious = [(r["scene"], float(r["iou"])) for r in sub if r.get("iou") not in ("", None, "nan")]
            if ious:
                iv = np.array([x[1] for x in ious])
                print(f"    掩膜 IoU：中位 {np.median(iv):.3f}  min {iv.min():.3f}  <0.5 场景数 {int((iv<0.5).sum())}/{len(iv)}")

    for m in ("gt", "img", "oracle", "imgD"):
        report(m)

    # 失效模式明细（img 变体；FAIL 判据 max(e, e_gb) > 0.15，或掩膜 IoU < 0.5）
    print("\n---- img 变体失效明细（max(e,e_gb)>0.15 或 IoU<0.5）----")
    nf = 0
    for r in rows:
        if r["method"] == "img" and r["status"] == "ok":
            e = float(r["e"]) if r["e"] != "" else np.nan
            eg = float(r["e_gb"]) if r["e_gb"] != "" else np.nan
            iou = float(r["iou"]) if r["iou"] != "" else np.nan
            fail = (e == e and e > 0.15) or (eg == eg and eg > 0.15) or (iou == iou and iou < 0.5)
            if fail:
                nf += 1
                print(f"  {r['set'][:18]:>18} a{r['level']:>4} {r['scene']}: e={r['e']} e_gb={r['e_gb']} "
                      f"a_hat={r['alpha_hat']} a_true={r['alpha_true']} iou={r.get('iou','')} q={r.get('q','')}")
    print(f"  失效行合计 {nf}")

    out = os.path.join(PROJ, "eval/probe_b2a_img_pipeline.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {os.path.relpath(out, PROJ)}  总耗时 {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
