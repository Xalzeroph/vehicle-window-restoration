#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 10-19 最终数据集转成配对测试集 real_pair_1019（2026-10-20）。

配对口径
--------
每个场景有 1 张人工标注的 GT（无反射）和若干有反射帧。GT 取你自己标的帧，
I 取 **GT 帧之前 30/60/90 帧**（30 fps 下约 1/2/3 秒前）。选多帧是因为单帧统计量
太弱（n=15），3 帧能把 n 提到 45，且能看"离 GT 越远是否越差"。

输出（与既有的 data/test_sets/real_pair_* 结构一致）
----------------------------------------------------
<proj>/data/test_sets/real_pair_1019/{scene}/
    GT.png              # 无反射真值（人工标注帧）
    I_{帧号}.png         # 有反射输入（最多 3 张）
    meta.json           # scene / gt_frame / 各 I 帧号与距离 / 尺寸
"""
from __future__ import annotations

import json
import os
import re
import shutil

import cv2

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
RAW = os.path.join(P, "data/raw_1019")
OUT = os.path.join(P, "data/test_sets/real_pair_1019")

# 目标对齐尺寸：1299x858 -> 32 的倍数（1280x832），保证所有模型口径一致
TARGET = (1280, 832)   # (W, H)
OFFSETS = (30, 60, 90)


def frames_of(d: str):
    out = []
    for f in os.listdir(d):
        m = re.match(r"frame_(\d{6})\.jpg$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main() -> int:
    scenes = sorted(s for s in os.listdir(RAW)
                    if s.startswith("VID_20261019_") and os.path.isdir(os.path.join(RAW, s)))
    print(f"场景 {len(scenes)} 个 -> {OUT}")
    os.makedirs(OUT, exist_ok=True)

    total_pairs = 0
    for sc in scenes:
        src = os.path.join(RAW, sc)
        lb = os.path.join(src, "label.txt")
        gt = None
        if os.path.exists(lb):
            m = re.search(r"^gt_frame:\s*(\d+)", open(lb, encoding="utf-8").read(), re.M)
            if m:
                gt = int(m.group(1))
        if gt is None:
            print(f"  [跳过] {sc}: 无 gt_frame")
            continue

        frames = frames_of(src)
        have = {n for n, _ in frames}
        if gt not in have:
            print(f"  [跳过] {sc}: GT 帧 {gt} 不在目录")
            continue

        # 选 I 帧：GT 之前 offset 帧，取最接近的已存在帧（且不重复）
        picks = []
        for off in OFFSETS:
            target = gt - off
            cand = [n for n in have if n <= target]
            if not cand:
                continue
            n = max(cand)
            if n == gt or n in picks:
                continue
            picks.append(n)
        picks = sorted(picks)
        if not picks:
            print(f"  [跳过] {sc}: 找不到 GT 之前的 I 帧")
            continue

        dst = os.path.join(OUT, sc)
        os.makedirs(dst, exist_ok=True)

        # GT
        gsrc = os.path.join(src, f"frame_{gt:06d}.jpg")
        gimg = cv2.imread(gsrc)
        if gimg is None:
            print(f"  [跳过] {sc}: GT 读不到")
            continue
        gimg = cv2.resize(gimg, TARGET, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dst, "GT.png"), gimg)

        # I 帧
        imeta = []
        for n in picks:
            isrc = os.path.join(src, f"frame_{n:06d}.jpg")
            iimg = cv2.imread(isrc)
            if iimg is None:
                continue
            iimg = cv2.resize(iimg, TARGET, interpolation=cv2.INTER_AREA)
            name = f"I_{n:06d}.png"
            cv2.imwrite(os.path.join(dst, name), iimg)
            imeta.append({"frame": n, "file": name, "dist_to_gt": gt - n})
            total_pairs += 1

        meta = {"scene": sc, "gt_frame": gt, "gt_file": "GT.png",
                "size": list(TARGET), "inputs": imeta,
                "note": "I 取 GT 前 30/60/90 帧；GT 为人工标注无反射帧"}
        with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        print(f"  {sc}: GT=f{gt:06d}  I={[x['frame'] for x in imeta]}  "
              f"距离={[x['dist_to_gt'] for x in imeta]}")

    print(f"\n共生成 {total_pairs} 对 (I, GT) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
