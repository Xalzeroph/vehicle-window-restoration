"""从 Penn-Fudan 行人掩码库抽出「单个行人剪影」，整理成可复用的形状先验库。

来源：Penn-Fudan Pedestrian Database（170 张图 / 约 345 个行人，像素级标注）
       https://www.cis.upenn.edu/~jshi/ped_html/
用法：python build_silhouette_lib.py  → data_ext/silhouettes/lib/*.png（256×256 二值）+ contact_sheet.png

注意：掩码里 0=背景，>0 是行人 ID；一张图可能有多个行人，逐个抽出。
      行人可能被遮挡成断片，故只保留最大的连通域（并按需合并 ≥20% 面积的邻近块）。
"""
import glob
import os

import cv2
import numpy as np

ROOT = r"D:\c_course\paper-writing\reflection_removal\data_ext\silhouettes"
SRC = os.path.join(ROOT, "PennFudanPed", "PedMasks")
OUT = os.path.join(ROOT, "lib")
SIZE = 256
MIN_AREA = 400          # 原始掩码里的最小像素数
KEEP_FRAC = 0.20        # 只保留面积 ≥ 最大块 20% 的连通域


def extract(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    out = []
    for lab in [int(v) for v in np.unique(m) if v > 0]:
        b = (m == lab).astype(np.uint8)
        if b.sum() < MIN_AREA:
            continue
        n, cc, st, _ = cv2.connectedComponentsWithStats(b, 8)
        if n <= 1:
            continue
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = np.zeros_like(b)
        big = areas.max()
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= KEEP_FRAC * big:
                keep |= (cc == i).astype(np.uint8)
        ys, xs = np.nonzero(keep)
        if len(ys) < MIN_AREA:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = keep[y0:y1, x0:x1]
        # 等比缩放到 SIZE 内，居中放到 SIZE×SIZE 画布
        h, w = crop.shape
        s = (SIZE - 8) / max(h, w)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((SIZE, SIZE), np.uint8)
        oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
        canvas[oy:oy + nh, ox:ox + nw] = r
        out.append((canvas * 255, lab))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.png")))
    print(f"掩码文件 {len(files)} 张")
    n = 0
    for f in files:
        stem = os.path.basename(f).replace("_mask", "").replace(".png", "")
        for img, lab in extract(f):
            cv2.imwrite(os.path.join(OUT, f"{stem}_p{lab:02d}.png"), img)
            n += 1
    print(f"抽出剪影 {n} 个 -> {OUT}")

    # 拼 contact sheet（10 列）
    libs = sorted(glob.glob(os.path.join(OUT, "*.png")))
    cols = 10
    rows = (len(libs) + cols - 1) // cols
    T = 96
    sheet = np.full((rows * T, cols * T), 30, np.uint8)
    for i, p in enumerate(libs):
        im = cv2.imread(p, 0)
        im = cv2.resize(im, (T - 6, T - 6), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        sheet[r * T + 3:r * T + 3 + im.shape[0], c * T + 3:c * T + 3 + im.shape[1]] = im
    cv2.imwrite(os.path.join(ROOT, "contact_sheet.png"), sheet)
    print(f"contact sheet: {sheet.shape} -> {os.path.join(ROOT, 'contact_sheet.png')}")
    areas = [cv2.imread(p, 0).mean() / 255 for p in libs]
    print(f"剪影面积占比（在 256 画布内）中位 {np.median(areas)*100:.1f}%")


if __name__ == "__main__":
    main()
