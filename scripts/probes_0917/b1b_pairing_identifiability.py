#!/usr/bin/env python3
"""批次1b：多档配对的可辨识性探针（零训练；训练集同场景 5 档，T/R 完全相同）。

三步：
 ① gauge 族构造性验证：T→aT, R→R+cT, α→(α−c)/a 应产生逐位相同的 I_j（观测不可分）
 ② 不变量：D_jk = I_j − I_k = (α_j−α_k)T（R 自动抵消）→ 差值比 (α_j−α_k)/(α_j−α_l) 可恢复
 ③ 绝对化：已知档位时 U := m·T = D_jk/(level_j−level_k) 可解（与 GT 比）；
    裸区锚 + 边界 T 连续 → m̂ = U_膜/U_裸（R 已抵消，预期优于 1a 的 I 比）
用法: python3 scripts/probes_0917/b1b_pairing_identifiability.py [--n 24]
产物: eval/probe_b1b_pairing.csv + 控制台
"""
import argparse, csv, json, os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRIP = os.path.join(PROJ, "data/train_sets/train_triplets_rdt_0915")
TAGS = {1.0: "l100", 0.75: "l075", 0.6: "l060", 0.45: "l045", 0.3: "l030"}


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(TRIP, "T")))[::max(1, 4000 // args.n)][:args.n]
    rows = []
    for sid in ids:
        T = img(os.path.join(TRIP, "T", f"{sid}.png"))
        R = img(os.path.join(TRIP, "R", f"{sid}.png"))
        Is, As, lvls = {}, {}, []
        for lv, tag in TAGS.items():
            Is[lv] = img(os.path.join(TRIP, "I", f"{sid}_{tag}.png"))
            A = np.load(os.path.join(TRIP, "alpha", f"{sid}_{tag}.npy")).astype(np.float32)
            As[lv] = A[:, :, 0] if A.ndim == 3 else A
            lvls.append(lv)
        # ① gauge 族验证
        rng = np.random.default_rng(0)
        a_g, c_g = float(rng.uniform(0.7, 1.3)), float(rng.uniform(-0.15, 0.15))
        maxdiff = 0.0
        for lv in lvls:
            Tp, Rp = a_g * T, R + c_g * T
            Ap = (As[lv] - c_g) / a_g
            Irec = np.clip(Ap[..., None] * Tp + Rp, 0, 1)      # 与生成器一致：clip 后才落盘
            maxdiff = max(maxdiff, float(np.abs(Irec - Is[lv]).max()) * 255)
        # ② 不变量：D_1j 与 D_1k 的比值（数值上取全图最小二乘斜率字段的正则化估计）
        ref = 0.45 if 0.45 in lvls else lvls[0]
        D = {lv: Is[lv] - Is[ref] for lv in lvls if lv != ref}
        pairs = [(1.0, 0.6), (0.75, 0.45), (0.6, 0.3)]
        # 未截断掩膜：α 未顶到 1（clip 破坏档间比例性）
        unclip = np.stack([As[lv] < 0.995 for lv in lvls]).all(0)
        inv_err = []
        for j, k in pairs:
            if j not in D or k not in D:
                continue
            num, den = D[j][unclip].astype(np.float64), D[k][unclip].astype(np.float64)
            beta = float((num * den).sum() / max((den * den).sum(), 1e-9))
            true_beta = (j - ref) / (k - ref)      # 真档位已知 → 比值真值
            inv_err.append((f"{j}/{k}", round(abs(beta - true_beta) / abs(true_beta), 4)))
        # ③ U = m·T 解 + 裸区锚边界 m̂ = U_膜/U_裸
        j, k = 0.6, 0.3
        U = (Is[j] - Is[k]) / (j - k)
        U_gt = (As[j] - As[k])[:, :, None] / (j - k) * T   # = m·T（若 α=level·m(x)）
        u_p = psnr(np.clip(U, 0, 1) * 255, np.clip(U_gt, 0, 1) * 255)
        bare = (As[j] > 0.995) & (As[k] > 0.995)
        if bare.sum() > 200:
            b_all = np.stack(np.nonzero(bare), 1)
            f_all = np.stack(np.nonzero(~bare), 1)
            d_bare, _ = cKDTree(b_all).query(f_all)
            dm = np.full(As[j].shape, 1e9, np.float32)
            dm[~bare] = d_bare
            fb = (~bare) & (dm >= 6) & (dm <= 12)
            dbf, _ = cKDTree(f_all).query(b_all)
            bb = np.zeros(As[j].shape, bool)
            bb[b_all[dbf <= 6][:, 0], b_all[dbf <= 6][:, 1]] = True
            if fb.sum() > 50 and bb.sum() > 50:
                # 联合锚（配对版）：步长/档差 = (1−α_j)/(α_j−α_k)；配对使 T 漂移与 R 都抵消
                fy_, fx_ = np.nonzero(fb)
                bb_all = np.stack(np.nonzero(bb), 1)
                dd, jj = cKDTree(bb_all).query(np.stack([fy_, fx_], 1))
                keep = dd <= 6
                fy2, fx2 = fy_[keep], fx_[keep]
                by2, bx2 = bb_all[jj[keep], 0], bb_all[jj[keep], 1]
                step = float(np.mean(Is[j][by2, bx2].mean(-1) - Is[j][fy2, fx2].mean(-1)))
                diff = float(np.mean(Is[j][fy2, fx2].mean(-1) - Is[k][fy2, fx2].mean(-1)))
                r_hat = step / max(diff, 1e-6)
                a_j = float(As[j][fb].mean()); a_k = float(As[k][fb].mean())
                r_true = (1 - a_j) / max(a_j - a_k, 1e-6)
                m_hat, m_true = r_hat, r_true
                rows.append(dict(scene=sid, gauge_maxdiff_255=round(maxdiff, 2),
                                 inv_relerr=";".join(f"{a}:{b}" for a, b in inv_err),
                                 U_psnr=round(u_p, 2),
                                 m_hat=round(m_hat, 3), m_true=round(m_true, 3),
                                 m_abs_err=round(abs(m_hat - m_true), 3)))
                continue
        rows.append(dict(scene=sid, gauge_maxdiff_255=round(maxdiff, 2),
                         inv_relerr=";".join(f"{a}:{b}" for a, b in inv_err),
                         U_psnr=round(u_p, 2), m_hat="", m_true="", m_abs_err=""))
    ok = [r for r in rows if r["m_hat"] != ""]
    print(f"样本 {len(rows)}（含裸区可评 m̂ 的 {len(ok)}）")
    print(f"① gauge 族最大重构误差（0-255）: max {max(r['gauge_maxdiff_255'] for r in rows):.2f}（应≈0）")
    per = {}
    for r in rows:
        for kv in r["inv_relerr"].split(";"):
            a, b = kv.split(":")
            per.setdefault(a, []).append(float(b))
    for a, v in per.items():
        print(f"② 不变量相对误差 [{a}]: 均值 {np.mean(v):.4f} 最大 {np.max(v):.4f}")
    print(f"③ U=mT 恢复 PSNR: 均值 {np.mean([r['U_psnr'] for r in rows]):.1f} dB")
    if ok:
        e = np.array([r["m_abs_err"] for r in ok])
        rh = np.array([r["m_hat"] for r in ok], float); rt = np.array([r["m_true"] for r in ok], float)
        rel = np.abs(rh - rt) / np.maximum(np.abs(rt), 1e-6)
        print(f"③' 联合锚 比值 (1−α_j)/(α_j−α_k): 估计 vs 真 相对误差 均值 {rel.mean():.3f} 最大 {rel.max():.3f}")
    out = os.path.join(PROJ, "eval/probe_b1b_pairing.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
