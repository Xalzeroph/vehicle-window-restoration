#!/usr/bin/env python3
"""同估计器对齐: 合成图跑 RDNet 测 R̂/T̂, 与实拍锚点(同 RDNet 口径)比较。

实拍锚点: 4.26 户外强反射 0.230 / 162901 0.065 / 164653 0.030。
合成侧 R̂/T̂ = mean(ytmt_ucs_sirs_r)/mean(ytmt_ucs_sirs_l), 逐场景记录再汇总。
"""
import argparse, glob, os
import numpy as np
from PIL import Image

ANCHORS = {"real426": 0.230, "real162901": 0.065, "real164653": 0.030}


def load(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float32).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True,
                    help="含 rdnet_alpha_X/<scene>/{ytmt_ucs_sirs_l,ytmt_ucs_sirs_r,m_input}.png")
    args = ap.parse_args()

    dirs = sorted(os.listdir(args.pred_root), key=lambda s: float(s.split("_")[-1]))
    print(f"{'level':>9} {'in̄':>7} {'R̂̄':>7} {'T̂̄':>7} {'R̂/T̂ mean':>10} {'median':>7}  ratio per scene")
    for lv in dirs:
        rows = []
        din = []
        for d in sorted(glob.glob(os.path.join(args.pred_root, lv, "*"))):
            l = os.path.join(d, "ytmt_ucs_sirs_l.png")
            r = os.path.join(d, "ytmt_ucs_sirs_r.png")
            if not (os.path.exists(l) and os.path.exists(r)):
                continue
            rl, tl = load(r), load(l)
            rows.append((os.path.basename(d), rl, tl, rl / max(tl, 1e-6)))
            mi = os.path.join(d, "m_input.png")
            if os.path.exists(mi):
                din.append(load(mi))
        if not rows:
            continue
        rs = np.array([x[3] for x in rows])
        print(f"{lv:>9} {np.mean(din):7.1f} {np.mean([x[1] for x in rows]):7.1f} "
              f"{np.mean([x[2] for x in rows]):7.1f} {rs.mean():10.3f} {np.median(rs):7.3f}  "
              + " ".join(f"{x[0]}:{x[3]:.2f}" for x in rows))
    print("\n实拍锚点(同 RDNet 口径):", ", ".join(f"{k}={v}" for k, v in ANCHORS.items()))


if __name__ == "__main__":
    main()
