"""出 Demo 对比图：输入 / GT / 我们的最好权重 / 公开基线，逐场景标 PSNR。

选场景规则：按 (ours − 最强基线) 的逐场景增益排序，取前 N 个作为"好例子"；
再额外补 1 个我们输的场景（诚实展示），并给最好的那一例加 2× 局部放大。

用法：python make_demo_grid.py [--n 3] [--out /tmp/demo_grid.png]
"""
import argparse
import glob
import os
import statistics as st

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PANEL = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")

OURS = ("ours (may+jun, s101)", "real_pair_1019_mayjun15_s101")
BASELINES = [
    ("RRW (CVPR'24)", "real_pair_1019_zhu"),
    ("DSRNet (ICCV'23)", "real_pair_1019_dsrnet"),
    ("RDNet official", "real_pair_1019_rdnet_zs"),
    ("identity", "real_pair_1019_identity"),
]


def psnr(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    m = float(d.mean())
    return 99.0 if m <= 1e-12 else 10.0 * np.log10(255.0 ** 2 / m)


def load_pair(sc):
    d = os.path.join(PANEL, sc)
    ip = sorted(glob.glob(os.path.join(d, "I_*.png")))
    gp = os.path.join(d, "GT.png")
    if not ip or not os.path.exists(gp):
        return None, None
    I, G = cv2.imread(ip[0]), cv2.imread(gp)
    if I is None or G is None or I.shape != G.shape:
        return None, None
    return I, G


def load_pred(dirname, sc):
    fs = glob.glob(os.path.join(PRED, dirname, sc + "__*", "t_hat.png"))
    if not fs:
        return None
    # 与评测脚本一致：取同一输入 stem
    want = os.path.join(PRED, dirname, sc + "__I_000000", "t_hat.png")
    p = want if os.path.exists(want) else sorted(fs)[0]
    return cv2.imread(p)


def fit(img, w):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tile", type=int, default=300)
    ap.add_argument("--out", default="/tmp/demo_grid.png")
    a = ap.parse_args()

    scenes = sorted(os.listdir(PANEL))
    rows = []
    for sc in scenes:
        I, G = load_pair(sc)
        if I is None:
            continue
        th = load_pred(OURS[1], sc)
        if th is None:
            continue
        r = {"scene": sc, "I": I, "G": G, "ours": th,
             "p_ours": psnr(th, G), "p_id": psnr(I, G), "preds": {}, "ps": {}}
        for name, d in BASELINES:
            t = load_pred(d, sc)
            if t is None:
                continue
            r["preds"][name] = t
            r["ps"][name] = psnr(t, G)
        base_best = max([v for k, v in r["ps"].items()])
        r["gain"] = r["p_ours"] - base_best
        rows.append(r)

    rows.sort(key=lambda z: -z["gain"])
    print(f"共 {len(rows)} 个场景；逐场景增益（ours − 最强基线）：")
    for r in rows:
        print(f"  {r['scene'][:26]:<28} ours {r['p_ours']:5.2f}  best-base {max(r['ps'].values()):5.2f}  "
              f"gain {r['gain']:+.2f}  id {r['p_id']:5.2f}")

    picks = rows[:a.n]
    worst = rows[-1]
    if worst not in picks:
        picks.append(worst)

    T = a.tile
    blocks = []
    for r in picks:
        cells = [label(fit(r["I"], T), f'input  {r["p_id"]:.2f}'), label(fit(r["G"], T), "GT")]
        cells.append(label(fit(r["ours"], T), f'{OURS[0]}  {r["p_ours"]:.2f}', (0, 255, 0)))
        for name, _ in BASELINES:
            if name in r["preds"]:
                cells.append(label(fit(r["preds"][name], T), f'{name}  {r["ps"][name]:.2f}'))
        hh = min(c.shape[0] for c in cells)
        blocks.append(np.hstack([c[:hh] for c in cells]))
    w = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0))) for b in blocks]
    grid = np.vstack(blocks)

    # 最好那例的 2× 局部放大（中心 1/3）
    r0 = picks[0]
    y0, x0 = r0["G"].shape[0] // 3, r0["G"].shape[1] // 3
    crop = lambda im: cv2.resize(im[y0:y0 + r0["G"].shape[0] // 3, x0:x0 + r0["G"].shape[1] // 3],
                                 (T, T), interpolation=cv2.INTER_NEAREST)
    zc = [label(crop(r0["I"]), "zoom: input"), label(crop(r0["G"]), "zoom: GT"),
          label(crop(r0["ours"]), f'zoom: ours {r0["p_ours"]:.2f}', (0, 255, 0))]
    for name, _ in BASELINES:
        if name in r0["preds"]:
            zc.append(label(crop(r0["preds"][name]), f'zoom: {name} {r0["ps"][name]:.2f}'))
    hh = min(c.shape[0] for c in zc)
    zrow = np.hstack([c[:hh] for c in zc])
    w2 = grid.shape[1]
    zrow = np.pad(zrow, ((0, 0), (0, max(0, w2 - zrow.shape[1])), (0, 0)))
    out = np.vstack([grid, zrow])

    hdr = np.zeros((30, out.shape[1], 3), np.uint8)
    cv2.putText(hdr, f"Tinted-glass SIRR | 15-scene frozen panel | rows: 3 best-gain scenes + 1 worst | bottom: 2x zoom",
                (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 225, 255), 1, cv2.LINE_AA)
    out = np.vstack([hdr, out])
    cv2.imwrite(a.out, out)
    print(f"\n写出 {a.out}  {out.shape}")


if __name__ == "__main__":
    main()
