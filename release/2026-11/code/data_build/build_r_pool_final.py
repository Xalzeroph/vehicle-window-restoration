"""鏈€缁?R 姹狅細鍒ゆ柇"鑹插亸鏄惁绯荤粺鎬?锛屽苟鎹寤烘睜銆?
鍒ゅ畾鍙ｅ緞锛堢郴缁熸€?vs 鍐呭鑹诧級锛?    閫愬浘绠?LAB 鑹插害 (a*, b*) 鐨勫潎鍊硷紱涓€涓睜鐨?*绯荤粺鎬ц壊鍋?*琛ㄧ幇涓?      |mean(a*)|銆亅mean(b*)| 鐩稿鏍囧噯璇緢澶э紙t = mean / (sd/鈭歯)锛夛紝鍗虫暣姹犳湞鍚屼竴鏂瑰悜鍋忋€?    鍐呭鑹诧紙濡傜豢鑽夊湴鍙嶅厜锛夋槸閫愬浘闅忔満鏂瑰悜锛宼 鎺ヨ繎 0銆?
姹犲瓙锛?    A  RRW frames锛堢湡瀹為厤瀵规暟鎹泦锛屾櫘閫氱幓鐠冿紱璺ㄥ満鏅娊甯с€佹寜鍐呭鍘婚噸锛?    B  SIR虏 Wildscene r.jpg锛堝疄鎷嶅弽灏勫浘锛?    C  鑷噰 4 鏈堬紙璐磋啘鐜荤拑锛屼粎浣滃鐓э紝涓嶅叆姹狅級

杈撳嚭锛歞ata/pools/r_final_1029/*.png锛堥€愬浘褰掍竴鍖栧埌缁熶竴浜害锛? meta.json锛堝惈鑹插亸缁熻涓庢潵婧愶級
鐢ㄦ硶锛歱ython build_r_pool_final.py [--max-per-scene 3]
"""
import argparse
import glob
import json
import os
import statistics as st

import cv2
import numpy as np

RRW = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/datasets/RRWDataset"
SIR2 = os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "/datasets/SIR2"
SELF = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance") + "/data/train_sets/train_pairs_omega_0920/R"
OUT_DEFAULT = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance") + "/data/pools/r_final_1029"


def tint_stats(paths, label, maxn=300):
    A, B = [], []
    for p in paths[:maxn]:
        im = cv2.imread(p)
        if im is None:
            continue
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        A.append(lab[..., 1].mean() - 128.0)
        B.append(lab[..., 2].mean() - 128.0)
    if not A:
        print(f"  {label:<28} (鏃犳牱鏈?")
        return None
    n = len(A)
    ma, mb = float(st.mean(A)), float(st.mean(B))
    sa, sb = (float(st.stdev(A)) if n > 1 else 0.0), (float(st.stdev(B)) if n > 1 else 0.0)
    ta = float(ma / (sa / n ** 0.5)) if sa > 1e-6 else 0.0
    tb = float(mb / (sb / n ** 0.5)) if sb > 1e-6 else 0.0
    print(f"  {label:<28} n={n:4d}  a*={ma:+6.2f}卤{sa:5.2f} (t={ta:+5.1f})   "
          f"b*={mb:+6.2f}卤{sb:5.2f} (t={tb:+5.1f})")
    return dict(label=label, n=n, a_mean=ma, a_sd=sa, a_t=ta, b_mean=mb, b_sd=sb, b_t=tb)


def dhash(im, n=8):
    g = cv2.cvtColor(cv2.resize(im, (32, 32)), cv2.COLOR_BGR2GRAY)
    d = cv2.dct(g.astype(np.float32))[:n, :n]
    return int("".join("1" if v > d.mean() else "0" for v in d.ravel()), 2)


def norm_bright(im, p95_target=0.35):
    """閫愬浘褰掍竴鍖栵細鎶?95 鍒嗕綅浜害鎷夊埌 p95_target锛岄伩鍏?娈嬪樊寰堟殫"瀵艰嚧鍚堟垚閲屾病鏈夊弽灏勩€?""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p95 = float(np.percentile(g, 95))
    if p95 < 1e-3:
        return im
    k = p95_target / p95
    return np.clip(im.astype(np.float32) * k, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--max-per-scene", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = []

    # ---------- 0) 鑹插亸绯荤粺鎬у垽瀹?----------
    print("=== 鑹插亸锛氱郴缁熸€?or 鍐呭鑹?===")
    rrw_sample = sorted(glob.glob(os.path.join(RRW, "reflection_camera1", "camera_objects", "*.jpg")))[:200]
    rrw_sample += sorted(glob.glob(os.path.join(RRW, "reflection_camera2", "indoor_object", "*.jpg")))[:100]
    sir2_sample = sorted(glob.glob(os.path.join(SIR2, "Wildscene", "*", "r.jpg")))
    self_sample = sorted(glob.glob(os.path.join(SELF, "*.png")))
    stats = [tint_stats(p, "RRW frames") for p in (rrw_sample,)]
    stats.append(tint_stats(sir2_sample, "SIR2 Wildscene r.jpg"))
    stats.append(tint_stats(self_sample, "鑷噰 4 鏈堬紙瀵圭収锛?))

    # ---------- 1) A: SIR2 blended - T锛堝幓閲?+ 閫愬浘褰掍竴鍖栵級----------
    n_a = 0
    seen = set()
    for gname, d in [("sir", os.path.join(SIR2, "SIR")),
                     ("postcard", os.path.join(SIR2, "postcard/postcard199"))]:
        bd, td = os.path.join(d, "blended"), os.path.join(d, "transmission_layer")
        if not (os.path.isdir(bd) and os.path.isdir(td)):
            continue
        for f in sorted(set(os.listdir(bd)) & set(os.listdir(td))):
            B = cv2.imread(os.path.join(bd, f))
            T = cv2.imread(os.path.join(td, f))
            if B is None or T is None or B.shape != T.shape:
                continue
            R = np.clip(B.astype(np.float32) - T.astype(np.float32), 0, 255).astype(np.uint8)
            h = dhash(R)
            if h in seen:
                continue
            seen.add(h)
            name = f"sir2{gname}__{f}"
            cv2.imwrite(os.path.join(a.out, name), norm_bright(R))
            meta.append(dict(name=name, group=f"sir2_{gname}_residue", src=os.path.join(bd, f)))
            n_a += 1
    print(f"\n  A SIR2 blended-T: {n_a} 寮狅紙鍘婚噸鍚庯級")

    # ---------- 2) B: SIR2 Wildscene ----------
    n_b = 0
    for p in sir2_sample:
        im = cv2.imread(p)
        if im is None:
            continue
        sc = os.path.basename(os.path.dirname(p))
        name = f"sir2wild__{sc}.png"
        cv2.imwrite(os.path.join(a.out, name), norm_bright(im))
        meta.append(dict(name=name, group="sir2_wildscene", src=p))
        n_b += 1
    print(f"  B SIR2 Wildscene: {n_b} 寮?)

    json.dump(dict(n=len(meta), pool="r_final_1029",
                   sources={"sir2_blended_minus_T": n_a, "sir2_wildscene": n_b},
                   excluded=["self-captured 2026-04 (tinted transmission; R itself neutral, kept out by protocol)",
                             "RRW frames: systematic colour cast a*=+6.7 (t=+69), b*=+10.8 (t=+25)"],
                   tint=stats, items=meta),
              open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  鍚堣 {len(meta)} 寮?鈫?{a.out}")
    print(f"  锛堟棫姹?25 寮狅紱鏈睜 {len(meta)} 寮狅紝鎻愬崌 {len(meta)/25:.0f}脳锛?)


if __name__ == "__main__":
    main()
