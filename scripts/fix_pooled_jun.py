#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 jun 的像素文件从 `real_pairs_1020_jun/` 链接回 `real_pairs_1020/`（2026-10-22）。

背景
----
上一轮 `prep_ablation_data.py` 为隔离 jun，从 `real_pairs_1020/` 删了 176 个 jun 文件；
`restore_jun.py` 只把 **meta 合并回去**，没有恢复像素。
结果：meta 有 44 条 jun，但 `I/T/R/omega` 里没有对应文件 → 补 `C_I` 时 imread 返回 None。

本脚本按 meta 里的 jun case 逐个硬链（失败则复制），并清理"meta 有、文件无"的僵尸条目。
"""
from __future__ import annotations

import collections
import json
import os
import shutil

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")
JUN = os.path.join(P, "data/train_sets/real_pairs_1020_jun")
SUBS = ("I", "T", "R", "omega")


def link_or_copy(a, b):
    if os.path.exists(b):
        return "exists"
    try:
        os.link(a, b)
        return "hardlink"
    except OSError:
        shutil.copy2(a, b)
        return "copy"


def main() -> int:
    mp = os.path.join(POOL, "meta.json")
    meta = json.load(open(mp, encoding="utf-8"))
    jun_cases = [c for c, v in meta.items() if str(v.get("domain")) == "jun"]
    print(f"meta 共 {len(meta)} 对，其中 jun {len(jun_cases)} 对")

    stat = collections.Counter()
    missing_src = []
    for case in jun_cases:
        for sub in SUBS:
            src = os.path.join(JUN, sub, case + ".png")
            dst = os.path.join(POOL, sub, case + ".png")
            if not os.path.exists(src):
                missing_src.append(src)
                continue
            stat[link_or_copy(src, dst)] += 1
    print(f"链接结果: {dict(stat)}")
    if missing_src:
        print(f"！jun 源缺 {len(missing_src)} 个文件，例: {missing_src[:2]}")

    # 清理僵尸条目：四个子目录文件不全的 case 从 meta 移除
    dead = []
    for case in list(meta.keys()):
        if not all(os.path.exists(os.path.join(POOL, s, case + ".png")) for s in SUBS):
            dead.append(case)
            meta.pop(case)
    if dead:
        json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"移除僵尸条目 {len(dead)} 条，例: {dead[:3]}")
    else:
        print("无僵尸条目")

    by = collections.Counter(str(v.get("domain")) for v in meta.values())
    print(f"\n池现状: {len(meta)} 对  域分布 {dict(by)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
