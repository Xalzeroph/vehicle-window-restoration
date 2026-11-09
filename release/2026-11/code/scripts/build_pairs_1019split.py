#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 `data/raw_1019` 构建 1019 内部划分的**域内训练配对**（2026-10-22）。

来源与口径
---------
· GT  : `data/test_sets/real_pair_1019/<scene>/GT.png`（已核定的无反射帧）
· 候选 I : `data/raw_1019/<scene>/*.jpg`（含反射的原始帧）
· 只处理 `eval/dom_v1/split_1019.json` 里冻结的 **3 个训练场景**；测试 12 场景绝不进入

QC（与 `build_train_real_pairs.py` 完全同口径，不过则丢）
------------------------------------------------------
1. `phaseCorrelate(I, GT)` 位移 ≤ `--max-shift`（默认 3.0 px；1019 相机有小幅运动，比 apr 的 1.5 放宽）
2. 逐通道 `|g − 1| ≤ --max-gain-dev`（默认 0.25）
3. `omega_of(I, GT)` 非 None 且 Ω 占比 ≥ 1e-4

产出
----
`data/train_sets/real_pairs_1019split/{I,T,R,omega}/` + `meta.json`（dict 型，键=case）
  · `R = I − g·T` 存 +128；`omega` 存 0/255 单通道
  · 每项含 domain="1019split" / scene / src / g / C_I / omega_frac / R_abs_mean / shift
  · `C_I` 口径同 `build_train_omega_multi.py:101`

用法
----
  python3 scripts/build_pairs_1019split.py                 # 默认每场景最多 120 帧
  python3 scripts/build_pairs_1019split.py --cap-per-scene 60 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_protocol import robust_gain      # noqa: E402
from eval_g2_baseline import omega_of      # noqa: E402

RAW = os.path.join(PROJ, "data/raw_1019")
TEST = os.path.join(PROJ, "data/test_sets/real_pair_1019")
SPLIT = os.path.join(PROJ, "eval/dom_v1/split_1019.json")
OUT_DEFAULT = os.path.join(PROJ, "data/train_sets/real_pairs_1019split")


def list_frames(scene_dir):
    out = []
    for f in os.listdir(scene_dir):
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"):
            out.append(f)
    return sorted(out)


def phase_shift(a_gray, b_gray):
    h = min(a_gray.shape[0], b_gray.shape[0])
    w = min(a_gray.shape[1], b_gray.shape[1])
    a = cv2.resize(a_gray, (w, h))
    b = cv2.resize(b_gray, (w, h))
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    return float(np.hypot(dx, dy))


def c_i_of(I_bgr, om):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(om.astype(np.uint8), ker) > 0) & (~om)
    if not (om.any() and ring.any()):
        return 0.0
    lI = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(lI[ring].mean() - lI[om].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cap-per-scene", type=int, default=120)
    ap.add_argument("--max-shift", type=float, default=3.0)
    ap.add_argument("--max-gain-dev", type=float, default=0.25)
    ap.add_argument("--domain", default="1019split")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.split):
        raise SystemExit(f"缺划分文件 {args.split}；先跑 make_test_split12.py")
    sp = json.load(open(args.split, encoding="utf-8"))
    train = sp["train_scenes"]
    print(f"冻结划分 sha256={sp['sha256'][:16]}…  训练场景 {len(train)}: {train}")

    if not args.dry_run:
        for sub in ("I", "T", "R", "omega"):
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    meta = {}
    tot_ok = tot_try = 0
    hdr = f"{'场景':<24}{'帧数':>6}{'候选':>6}{'接受':>6}{'位移拒':>7}{'增益拒':>7}{'omega拒':>9}"
    print(hdr); print("-" * len(hdr))
    for sc in train:
        gtp = os.path.join(TEST, sc, "GT.png")
        rd = os.path.join(RAW, sc)
        gt = cv2.imread(gtp, cv2.IMREAD_COLOR)
        if gt is None or not os.path.isdir(rd):
            print(f"{sc:<24} ！缺 raw 或 GT，跳过")
            continue
        frames = list_frames(rd)
        # 等距抽样，最多 cap*2 个候选（留淘汰余量）
        pool_n = min(len(frames), args.cap_per_scene * 2)
        idx = np.linspace(0, len(frames) - 1, pool_n).astype(int) if pool_n else []
        pool = [frames[i] for i in sorted(set(idx.tolist()))]
        gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

        n_ok = n_shift = n_gain = n_omega = 0
        for fn in pool:
            if n_ok >= args.cap_per_scene:
                break
            tot_try += 1
            im = cv2.imread(os.path.join(rd, fn), cv2.IMREAD_COLOR)
            if im is None:
                continue
            tgt = gt
            if im.shape[:2] != gt.shape[:2]:
                tgt = cv2.resize(gt, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
            sh = phase_shift(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY))
            if sh > args.max_shift:
                n_shift += 1; continue
            g = robust_gain(im, tgt)
            gdev = float(np.abs(g - 1).mean())
            if gdev > args.max_gain_dev:
                n_gain += 1; continue
            om_rec = omega_of(im, tgt)
            if om_rec is None:
                n_omega += 1; continue
            om = om_rec["Om"]
            if om.mean() < 1e-4:
                n_omega += 1; continue

            case = f"{args.domain}__{sc}__{os.path.splitext(fn)[0]}"
            R = im.astype(np.float64) - tgt.astype(np.float64) * g
            if not args.dry_run:
                cv2.imwrite(os.path.join(args.out, "I", case + ".png"), im)
                cv2.imwrite(os.path.join(args.out, "T", case + ".png"), tgt)
                cv2.imwrite(os.path.join(args.out, "R", case + ".png"),
                            np.clip(R + 128.0, 0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(args.out, "omega", case + ".png"),
                            (om.astype(np.uint8) * 255))
            meta[case] = {"domain": args.domain, "scene": sc, "src": fn,
                          "g": round(float(g.mean()), 4),
                          "gain_dev": round(gdev, 4), "shift": round(sh, 3),
                          "C_I": round(c_i_of(im, om), 2),
                          "omega_frac": round(float(om.mean()), 4),
                          "R_abs_mean": round(float(np.abs(R).mean()), 3),
                          "size": [int(im.shape[1]), int(im.shape[0])]}
            n_ok += 1; tot_ok += 1
        flag = "" if n_ok >= 80 else "  ← 不足 80，需报警"
        print(f"{sc:<24}{len(frames):>6}{len(pool):>6}{n_ok:>6}{n_shift:>7}"
              f"{n_gain:>7}{n_omega:>9}{flag}")

    if not args.dry_run and meta:
        json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        cis = sorted(v["C_I"] for v in meta.values())
        oms = sorted(v["omega_frac"] for v in meta.values())
        print(f"\n共接受 {tot_ok} 对（尝试 {tot_try}，接受率 "
              f"{tot_ok/max(tot_try,1)*100:.1f}%）")
        print(f"  C_I 中位 {cis[len(cis)//2]:.2f} | Ω 占比 中位 {oms[len(oms)//2]:.3f}"
              f" | 场景 {len({v['scene'] for v in meta.values()})}")
        print(f"写入 {args.out}/meta.json")

    # 断言：绝不含测试场景
    leak = sorted({v["scene"] for v in meta.values()} & set(sp["test_scenes"]))
    if leak:
        raise SystemExit(f"！训练池含测试场景 {leak}，中止")
    print("断言通过：训练池不含任何测试场景")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
