"""浠?Penn-Fudan 琛屼汉鎺╃爜搴撴娊鍑恒€屽崟涓浜哄壀褰便€嶏紝鏁寸悊鎴愬彲澶嶇敤鐨勫舰鐘跺厛楠屽簱銆?
鏉ユ簮锛歅enn-Fudan Pedestrian Database锛?70 寮犲浘 / 绾?345 涓浜猴紝鍍忕礌绾ф爣娉級
       https://www.cis.upenn.edu/~jshi/ped_html/
鐢ㄦ硶锛歱ython build_silhouette_lib.py  鈫?data_ext/silhouettes/lib/*.png锛?56脳256 浜屽€硷級+ contact_sheet.png

娉ㄦ剰锛氭帺鐮侀噷 0=鑳屾櫙锛?0 鏄浜?ID锛涗竴寮犲浘鍙兘鏈夊涓浜猴紝閫愪釜鎶藉嚭銆?      琛屼汉鍙兘琚伄鎸℃垚鏂墖锛屾晠鍙繚鐣欐渶澶х殑杩為€氬煙锛堝苟鎸夐渶鍚堝苟 鈮?0% 闈㈢Н鐨勯偦杩戝潡锛夈€?"""
import glob
import os

import cv2
import numpy as np

ROOT = os.environ.get("TINTED_SILHOUETTES", os.path.join("data_ext", "silhouettes"))  # override with env var / absolute path as needed
SRC = os.path.join(ROOT, "PennFudanPed", "PedMasks")
OUT = os.path.join(ROOT, "lib")
SIZE = 256
MIN_AREA = 400          # 鍘熷鎺╃爜閲岀殑鏈€灏忓儚绱犳暟
KEEP_FRAC = 0.20        # 鍙繚鐣欓潰绉?鈮?鏈€澶у潡 20% 鐨勮繛閫氬煙


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
        # 绛夋瘮缂╂斁鍒?SIZE 鍐咃紝灞呬腑鏀惧埌 SIZE脳SIZE 鐢诲竷
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
    print(f"鎺╃爜鏂囦欢 {len(files)} 寮?)
    n = 0
    for f in files:
        stem = os.path.basename(f).replace("_mask", "").replace(".png", "")
        for img, lab in extract(f):
            cv2.imwrite(os.path.join(OUT, f"{stem}_p{lab:02d}.png"), img)
            n += 1
    print(f"鎶藉嚭鍓奖 {n} 涓?-> {OUT}")

    # 鎷?contact sheet锛?0 鍒楋級
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
    print(f"鍓奖闈㈢Н鍗犳瘮锛堝湪 256 鐢诲竷鍐咃級涓綅 {np.median(areas)*100:.1f}%")


if __name__ == "__main__":
    main()
