#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建真实配对训练集（一 GT 配多 I）—— 2026-10-20。

动机
----
`My_Reflection_Videos_cropped/` 里有 24 个场景共 **3904 帧**完整实拍，但历史训练集
（`pairs_apr/may/jun`）只用了其中 16/14/13 个特定帧，**绝大部分帧从未进入训练**。
一个场景只有一张 GT，但 GT 可以配该场景的**任意**反射帧 → 一对多变多对。

GT 来源（不重新解析 label.txt，避免格式陷阱）
------------------------------------------
label.txt 有三种格式（`gt_frame: N`、含 "gt" 的行、裸数字），且 jun 每场景有 4 个标注档
（反射低光 / 无反射低光 / 反射带加光 / gt 无反射带加光），语义不同易错。
→ **直接复用已核定的 GT 图**，与现有 `pairs_*` 训练集完全同源：
    apr : data/test_sets/real_pair_apr_0917/<scene>/GT.png
    may : data/test_sets/real_pair_may_0917/<scene>/GT.png
    jun : data/test_sets/real_pair_0917/<scene>/T.png   （= 无反射低光）

GT 帧索引：用降采样 MSE 匹配 `My_Reflection_Videos_cropped/<scene>/frame_*.jpg` 自动定位
（不依赖 label.txt），用于排除 GT 帧及其后帧。

配对规则（保守）
---------------
只取 **帧号 < GT帧号 − margin** 的帧（GT 前、确定仍是反射态）。
在这批帧里**均匀抽样**（覆盖不同反射强度），再逐帧 QC，直到凑够 `--cap-per-scene`。

QC（不过则丢）
--------------
1. `phaseCorrelate(I, GT)` 位移 ≤ `--max-shift`（默认 1.5 px）
2. 逐通道 |g − 1| ≤ `--max-gain-dev`（默认 0.10）

产出
----
`data/train_sets/real_pairs_1020/{I,T,R,omega}/` + `meta.json`
  · meta 为 **dict 型式**（与 `pairs_*` 一致，键 = case 名）
  · `R = I − T` 存 **+128**；`omega` 存 0/255 单通道
  · 每项含 domain / scene / src / gt_frame / g / omega_frac / R_abs_mean / shift

另出 `data/splits/split_v2.json`（场景清单，供评测脚本引用）

用法
----
  python3 scripts/build_train_real_pairs.py                 # 全量默认参数
  python3 scripts/build_train_real_pairs.py --cap-per-scene 40 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "scripts"))   # eval_protocol / eval_g2_baseline 在 scripts/

from eval_protocol import robust_gain      # noqa: E402
from eval_g2_baseline import omega_of      # noqa: E402

FRAMES_ROOT = os.path.join(PROJ, "data/My_Reflection_Videos_cropped")
GT_DIRS = {
    "apr": (os.path.join(PROJ, "data/test_sets/real_pair_apr_0917"), "GT.png"),
    "may": (os.path.join(PROJ, "data/test_sets/real_pair_may_0917"), "GT.png"),
    "jun": (os.path.join(PROJ, "data/test_sets/real_pair_0917"), "T.png"),
}
OUT_DEFAULT = os.path.join(PROJ, "data/train_sets/real_pairs_1020")


def list_frames(scene_dir):
    out = []
    for f in os.listdir(scene_dir):
        if f.startswith("frame_") and f.endswith(".jpg"):
            try:
                out.append((int(f[6:-4]), f))
            except ValueError:
                pass
    return sorted(out)


def phase_shift(a_gray, b_gray):
    """a 相对 b 的平移（px）。尺寸不一致时先缩到同一尺寸。"""
    h = min(a_gray.shape[0], b_gray.shape[0])
    w = min(a_gray.shape[1], b_gray.shape[1])
    a = cv2.resize(a_gray, (w, h))
    b = cv2.resize(b_gray, (w, h))
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    return float(np.hypot(dx, dy))


def parse_gt_from_label(label_path: str):
    """从 label.txt 解析 GT 帧号。优先按**语义**，比 MSE 匹配可靠得多。

    实测 MSE 匹配会挑错帧（17628 实际 1303 被匹配成 1535），进而把候选帧跨度拉大、
    导致大量帧因相机漂移被 QC 拒掉。

    格式与语义（三种格式并存）
    ------------------------
    apr : 裸数字            `000875`
    may : `000750 gt低光无反射`
    jun : 四档并存
          000000 反射低光            ← 输入类
          000427 无反射低光          ← **GT**（本题）
          000718 反射带加光
          001866 gt 无返射带加光     ← 不是 GT（带加光，光照不同）
    规则：取同时含（无反射|无返射）且**不含**（补光|加光）的那一行；
          再退化到含 "gt" 的行；再退化到裸数字行。
    """
    if not os.path.exists(label_path):
        return None, "no-label"
    txt = open(label_path, errors="replace").read()
    best = None
    for line in txt.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^(\d{4,7})\b(.*)$", s)
        if not m:
            continue
        num, desc = int(m.group(1)), m.group(2)
        if ("无反射" in desc or "无返射" in desc) and ("补光" not in desc and "加光" not in desc):
            return num, "no-reflection"
        if "gt" in desc.lower() and best is None:
            best = (num, "gt-keyword")
        if desc.strip() == "" and best is None:
            best = (num, "bare-number")
    return best if best else (None, "unparsed")


def find_gt_index(scene_dir, frames, gt_img):
    """用降采样 MSE 在帧里找与 GT 最像的一帧（= GT 帧）。

    性能要点：必须用 IMREAD_REDUCED_GRAYSCALE_4 解码 —— 全分辨率解码 3904 帧
    实测要 10+ 分钟；降采样解码后 ~1 分钟。
    """
    small_gt = cv2.resize(cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY), (96, 64)).astype(np.float32)
    best, best_idx = 1e18, None
    for num, fn in frames:
        im = cv2.imread(os.path.join(scene_dir, fn), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if im is None:
            continue
        s = cv2.resize(im, (96, 64)).astype(np.float32)
        d = float(np.mean((s - small_gt) ** 2))
        if d < best:
            best, best_idx = d, num
    return best_idx, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cap-per-scene", type=int, default=60)
    ap.add_argument("--margin", type=int, default=15, help="只用 GT 帧号 − margin 之前的帧")
    ap.add_argument("--window", type=int, default=250,
                    help="候选窗口下界 = GT帧号 − window（限制时间跨度以控制相机漂移）")
    ap.add_argument("--max-shift", type=float, default=1.5, help="相位位移上限 px")
    ap.add_argument("--max-gain-dev", type=float, default=0.10, help="逐通道 |g-1| 上限")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--domains", default="apr,may,jun", help="comma separated: apr,may,jun")
    args = ap.parse_args()

    for sub in ("I", "T", "R", "omega"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    scenes = sorted(s for s in os.listdir(FRAMES_ROOT)
                    if s.startswith("VID_") and os.path.isdir(os.path.join(FRAMES_ROOT, s)))
    print(f"场景 {len(scenes)} 个；输出 {args.out}\n")

    meta, split = {}, {"version": "v2_20261020",
                       "rule": "scene-level + capture-date isolation",
                       "train_real": {}, "test_1019": {}, "notes": []}
    tot_ok = tot_try = 0

    hdr = (f"{'场景':<26}{'域':>4}{'帧':>5}{'GT帧':>8}{'候选':>5}"
           f"{'接受':>5}{'位移拒':>7}{'增益拒':>7}{'omega拒':>8}")
    print(hdr); print("-" * len(hdr))

    _want = {d.strip() for d in args.domains.split(",") if d.strip()}
    def _dom(s):
        return ("apr" if s.startswith("VID_20260426") else
                "may" if s.startswith("VID_20260512") else
                "jun" if s.startswith("VID_20260609") else "other")
    scenes = [s for s in scenes if _dom(s) in _want]
    for sc in scenes:
        dom = "apr" if sc.startswith("VID_20260426") else (
            "may" if sc.startswith("VID_20260512") else (
                "jun" if sc.startswith("VID_20260609") else "other"))
        if dom not in GT_DIRS:
            print(f"{sc:<26} 跳过（未知域）")
            continue
        gt_root, gt_name = GT_DIRS[dom]
        gt_path = os.path.join(gt_root, sc, gt_name)
        if not os.path.exists(gt_path):
            print(f"{sc:<26} 跳过（无 GT: {gt_path}）")
            continue
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if gt is None:
            print(f"{sc:<26} 跳过（GT 读失败）")
            continue

        sdir = os.path.join(FRAMES_ROOT, sc)
        frames = list_frames(sdir)
        if not frames:
            continue
        gt_idx, gt_src = parse_gt_from_label(os.path.join(sdir, "label.txt"))
        if gt_idx is None:
            gt_idx, _ = find_gt_index(sdir, frames, gt)      # 兜底
            gt_src = f"mse-fallback({gt_src})"
        if gt_idx is None:
            print(f"{sc:<26}{dom:>4}{len(frames):>5}{-1:>8}{0:>5}"
                  f"{0:>5}{0:>7}{0:>7}{0:>8}  [跳过：无法确定 GT 帧]")
            continue

        # 候选窗口：**GT 前 [margin, window]** 之间。
        # 下界 margin 排除 GT 帧与布段；上界 window 控制时间跨度，
        # 避免离 GT 太远的帧因相机漂移而全部 QC 不过（实测 jun 曾 100% 被拒）。
        lo = gt_idx - args.window
        his = [n for n, _ in frames]
        lo = max(lo, min(his))
        cand = [(n, f) for n, f in frames if lo <= n <= gt_idx - args.margin]
        if not cand:                                          # GT 在开头时放宽
            cand = [(n, f) for n, f in frames if n != gt_idx][-args.cap_per_scene * 2:]
        # 均匀抽样一个池（留 2 倍余量给 QC 淘汰）
        pool_n = min(len(cand), args.cap_per_scene * 2)
        idx = np.linspace(0, len(cand) - 1, pool_n).astype(int) if pool_n else []
        pool = [cand[i] for i in sorted(set(idx.tolist()))]

        gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        n_ok = n_shift = n_gain = n_omega = 0
        for num, fn in pool:
            if n_ok >= args.cap_per_scene:
                break
            tot_try += 1
            im = cv2.imread(os.path.join(sdir, fn), cv2.IMREAD_COLOR)
            if im is None:
                continue
            tgt = gt
            if im.shape[:2] != gt.shape[:2]:
                tgt = cv2.resize(gt, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
            shift = phase_shift(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY))
            if shift > args.max_shift:
                n_shift += 1
                continue
            g = robust_gain(im, tgt)
            gdev = float(np.abs(g - 1).mean())
            if gdev > args.max_gain_dev:
                n_gain += 1
                continue
            om_rec = omega_of(im, tgt)
            # omega_of 返回 dict(Om=..., S=..., bg=...) 或 None（整帧无反射）
            if om_rec is None:
                n_omega += 1
                continue
            om = om_rec["Om"]
            if om.mean() < 1e-4:
                n_omega += 1
                continue

            case = f"{dom}__{sc}__f{num:06d}"
            # R 用对齐口径 I − g·T（与既有 pairs_* 训练集一致；色调项抵消）
            R = im.astype(np.float64) - tgt.astype(np.float64) * g
            if not args.dry_run:
                cv2.imwrite(os.path.join(args.out, "I", case + ".png"), im)
                cv2.imwrite(os.path.join(args.out, "T", case + ".png"), tgt)
                cv2.imwrite(os.path.join(args.out, "R", case + ".png"),
                            np.clip(R + 128.0, 0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(args.out, "omega", case + ".png"),
                            (om.astype(np.uint8) * 255))
            meta[case] = {"domain": dom, "scene": sc, "src": fn, "gt_frame": gt_idx,
                          "gt_src": gt_src,
                          "g": round(float(g.mean()), 4),
                          "gain_dev": round(gdev, 4), "shift": round(shift, 3),
                          "omega_frac": round(float(om.mean()), 4),
                          "R_abs_mean": round(float(np.abs(R).mean()), 3),
                          "size": [int(im.shape[1]), int(im.shape[0])]}
            n_ok += 1
            tot_ok += 1

        split["train_real"].setdefault(dom, []).append(sc)
        print(f"{sc:<26}{dom:>4}{len(frames):>5}{gt_idx:>8}{len(pool):>5}"
              f"{n_ok:>5}{n_shift:>7}{n_gain:>7}{n_omega:>8}")

    # 1019 全留作测试
    t19 = os.path.join(PROJ, "data/raw_1019")
    if os.path.isdir(t19):
        split["test_1019"]["scenes"] = sorted(
            s for s in os.listdir(t19) if s.startswith("VID_20261019"))
    split["notes"] = [
        "apr/may/jun 全部作训练；1019 全部作测试（不参与梯度）",
        "GT 来源与既有 pairs_* 训练集同源（apr/may=GT.png, jun=T.png）",
        "只取 GT 帧号-margin 之前的帧，排除 GT 帧与布段帧",
    ]
    split["counts"] = {"train_pairs": tot_ok, "tried": tot_try}

    if not args.dry_run:
        with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        spdir = os.path.join(PROJ, "data/splits")
        os.makedirs(spdir, exist_ok=True)
        with open(os.path.join(spdir, "split_v2.json"), "w", encoding="utf-8") as fh:
            json.dump(split, fh, ensure_ascii=False, indent=1)

    print("-" * len(hdr))
    print(f"共接受 {tot_ok} 对（尝试 {tot_try}，接受率 "
          f"{tot_ok/max(tot_try,1)*100:.1f}%）")
    if not args.dry_run:
        print(f"写入 {args.out}/meta.json 与 data/splits/split_v2.json")
    else:
        print("[dry-run] 未写盘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
