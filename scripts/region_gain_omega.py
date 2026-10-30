"""TODO 1：用**训练期 Ω 口径**算「相对 identity 的区域内增益」，同时给出 v13 反射掩码口径做对照。

为什么必须两种口径都给：两种划分给出**相反**的结论（我们自己的数据就能演示），
这正是"raw Δgap 不能当证据"的直接证据。

输出：
  A. Ω 口径（omega_of，权威定义）：每方法 Ω 内 / Ω 外 PSNR，以及相对 identity 的增益
  B. v13 口径（table_final_1019 的 hole_mask）：同样的增益，用作对照

用法：python region_gain_omega.py
"""
import glob
import os
import statistics as st
import sys

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
sys.path.insert(0, os.path.join(P, "scripts"))
from build_omega_v2_0920 import omega_of  # noqa: E402

PANEL = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")

METHODS = {
    "identity": "real_pair_1019_identity",
    "official RDNet": "real_pair_1019_rdnet_zs",
    "RRW (Zhu)": "real_pair_1019_zhu",
    "DSRNet": "real_pair_1019_dsrnet",
    "LightRR": "real_pair_1019_lightrr",
    "ERRNet": "real_pair_1019_errnet",
    "IBCLN": "real_pair_1019_ibcln",
}


def psnr_masked(pred, gt, m):
    """m: (H,W) bool。逐像素平方误差先在通道上平均，再按掩码取子集（避免 2D 掩码错切 3D 数组）。"""
    if m.sum() < 100:
        return float("nan")
    sq = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(axis=2)
    mse = float(sq[m].mean())
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    scenes = []
    for d in sorted(glob.glob(os.path.join(PANEL, "*"))):
        ip = sorted(glob.glob(os.path.join(d, "I_*.png")))
        gp = os.path.join(d, "GT.png")
        if ip and os.path.exists(gp):
            scenes.append((os.path.basename(d), ip[0], gp))
    print(f"面板场景 {len(scenes)}")

    res = {k: {"in": [], "out": [], "frac": [], "all": []} for k in METHODS}
    used = 0
    for sc, ip, gp in scenes:
        # 选定一个输入 stem：优先 I_000000，且必须在各方法预测目录里都存在
        stems = sorted({os.path.basename(os.path.dirname(f)).split("__")[-1]
                        for f in glob.glob(os.path.join(PRED, METHODS["identity"], sc + "__*", "t_hat.png"))})
        if not stems:
            continue
        stem = "I_000000" if "I_000000" in stems else stems[0]
        ip = os.path.join(os.path.dirname(ip), stem + ".png")
        if not os.path.exists(ip):
            continue
        I = cv2.imread(ip)
        G = cv2.imread(gp)
        if I is None or G is None or I.shape != G.shape:
            continue
        used += 1
        try:
            out = omega_of(I, G)   # dict: Om / Rg / S / tau / s / gate
            om = np.asarray(out["Om"]).astype(bool)
        except Exception as e:  # noqa: BLE001
            print("  omega_of 失败", sc, e)
            continue
        if om.shape != I.shape[:2]:
            om = cv2.resize(om.astype(np.uint8), (I.shape[1], I.shape[0])) > 0
        for name, dname in METHODS.items():
            f = os.path.join(PRED, dname, f"{sc}__{stem}", "t_hat.png")
            if not os.path.exists(f):
                continue
            TH = cv2.imread(f)
            if TH is None or TH.shape != G.shape:
                continue
            res[name]["in"].append(psnr_masked(TH, G, om))
            res[name]["out"].append(psnr_masked(TH, G, ~om))
            res[name].setdefault("all", []).append(
                psnr_masked(TH, G, np.ones(om.shape, bool)))
            res[name]["frac"].append(float(om.mean()))

    base_in = st.mean(res["identity"]["in"])
    base_out = st.mean(res["identity"]["out"])
    print(f"实际使用场景数 {used}")
    print(f"\nΩ 面积中位 {st.median(res['identity']['frac']) * 100:.1f}%")
    print(f"identity：Ω 内 {base_in:.2f} / Ω 外 {base_out:.2f}")
    print(f"\n{'method':<15}{'PSNR in Ω':>10}{'gain in':>9}{'PSNR out':>10}{'gain out':>10}{'all':>10}")
    for name in METHODS:
        r = res[name]
        if not r["in"]:
            print(f"{name:<15}  (缺预测)")
            continue
        vi, vo = st.mean(r["in"]), st.mean(r["out"])
        va = st.mean(r["all"]) if r.get("all") else float("nan")
        print(f"{name:<15}{vi:>10.2f}{vi - base_in:>+9.2f}{vo:>10.2f}{vo - base_out:>+10.2f}"
              f"{va:>10.2f}")


if __name__ == "__main__":
    main()
