"""汇报版对比图：3 个最好场景 + 2× 放大，去掉 identity 列，格子更大。

列：input | GT | ours | RRW | DSRNet | RDNet official
用法：python make_demo_report.py [--tile 380] [--out /tmp/demo_report.png]
"""
import argparse
import glob
import os

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PANEL = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
OURS = ("Ours (may+jun, s101)", "real_pair_1019_mayjun15_s101")
COLS = [("RRW (CVPR'24)", "real_pair_1019_zhu"),
        ("DSRNet (ICCV'23)", "real_pair_1019_dsrnet"),
        ("RDNet official", "real_pair_1019_rdnet_zs")]
SCENES = ["VID_20261019_165126", "VID_20261019_164257", "VID_20261019_164730"]


def psnr(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    m = float(d.mean())
    return 99.0 if m <= 1e-12 else 10.0 * np.log10(255.0 ** 2 / m)


def pred(dirname, sc):
    p = os.path.join(PRED, dirname, sc + "__I_000000", "t_hat.png")
    if os.path.exists(p):
        return cv2.imread(p)
    fs = glob.glob(os.path.join(PRED, dirname, sc + "__*", "t_hat.png"))
    return cv2.imread(sorted(fs)[0]) if fs else None


def fit(img, w):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def tag(img, text, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=380)
    ap.add_argument("--out", default="/tmp/demo_report.png")
    a = ap.parse_args()
    T = a.tile
    rows = []
    for sc in SCENES:
        d = os.path.join(PANEL, sc)
        ip = sorted(glob.glob(os.path.join(d, "I_*.png")))
        gp = os.path.join(d, "GT.png")
        if not ip or not os.path.exists(gp):
            continue
        I, G = cv2.imread(ip[0]), cv2.imread(gp)
        th = pred(OURS[1], sc)
        if th is None or I is None or G is None:
            continue
        cells = [tag(fit(I, T), f"input   {psnr(I, G):.2f}"),
                 tag(fit(G, T), "GT")]
        cells.append(tag(fit(th, T), f"{OURS[0]}   {psnr(th, G):.2f}", (120, 255, 120)))
        for name, dn in COLS:
            t = pred(dn, sc)
            if t is not None:
                cells.append(tag(fit(t, T), f"{name}   {psnr(t, G):.2f}"))
        hh = min(c.shape[0] for c in cells)
        rows.append(np.hstack([c[:hh] for c in cells]))
        print(f"  {sc}: ours {psnr(th, G):.2f}  " + "  ".join(
            f"{n.split(' (')[0]} {psnr(pred(dn, sc), G):.2f}" for n, dn in COLS if pred(dn, sc) is not None))

    # 2× 放大（第一个场景的中心 1/3）
    sc = SCENES[0]
    d = os.path.join(PANEL, sc)
    G = cv2.imread(os.path.join(d, "GT.png"))
    I = cv2.imread(sorted(glob.glob(os.path.join(d, "I_*.png")))[0])
    h, w = G.shape[:2]
    y0, x0, cw, ch = h // 3, w // 3, w // 3, h // 3
    crop = lambda im: cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (T, T), interpolation=cv2.INTER_NEAREST)
    z = [tag(crop(I), f"zoom: input"), tag(crop(G), "zoom: GT"),
         tag(crop(pred(OURS[1], sc)), f"zoom: ours", (120, 255, 120))]
    for name, dn in COLS:
        t = pred(dn, sc)
        if t is not None:
            z.append(tag(crop(t), f"zoom: {name}"))
    hh = min(c.shape[0] for c in z)
    zrow = np.hstack([c[:hh] for c in z])

    grid = np.vstack(rows)
    wmax = max(grid.shape[1], zrow.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, max(0, wmax - x.shape[1])), (0, 0)))
    out = np.vstack([pad(grid), pad(zrow)])
    hdr = np.zeros((34, out.shape[1], 3), np.uint8)
    cv2.putText(hdr, "Tinted-glass reflection removal | frozen 15-scene panel | baselines are ZERO-SHOT (no fine-tuning on our data)",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 230, 255), 1, cv2.LINE_AA)
    out = np.vstack([hdr, out])
    cv2.imwrite(a.out, out)
    print(f"\n写出 {a.out}  {out.shape}")


if __name__ == "__main__":
    main()
