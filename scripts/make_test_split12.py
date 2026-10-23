#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结 1019 的域内划分 + 建 12 场景测试集（2026-10-22）。

划分（预登记，写死）
------------------
训练 3 场景 : 165400 / 170406 / 171126
  选取理由：三者抽样对齐 6/6 全通过；互不构成 r>0.9 的高相似对；覆盖整场早/中/晚。
测试 12 场景 : 其余全部（含 165530 这个最难场景，含 164730~164901 与 164257~164555 两对高相似）
  关键约束：高相似对必须同侧，故训练场景只能从不在那两对里的 11 个中挑。

为什么 3 而不是 5
----------------
1019 是唯一干净测试集，训练场景越少污染越小。3 场景 × ~110 帧 ≈ 300+ 对，够用；
测试保留 12 场景 → MDE = 1.0 × √(15/12) ≈ 1.12 dB。

产出
----
eval/dom_v1/split_1019.json                 冻结清单（含 sha256，后续一律读它）
data/test_sets/real_pair_1019_testsplit12/  12 个场景的**目录符号链接**
"""
from __future__ import annotations

import hashlib
import json
import os

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
TEST = os.path.join(P, "data/test_sets/real_pair_1019")
OUTDIR = os.path.join(P, "eval/dom_v1")
SPLIT = os.path.join(OUTDIR, "split_1019.json")
LINK = os.path.join(P, "data/test_sets/real_pair_1019_testsplit12")

TRAIN = ["VID_20261019_165400", "VID_20261019_170406", "VID_20261019_171126"]
HISIM = [("VID_20261019_164730", "VID_20261019_164901"),
         ("VID_20261019_164257", "VID_20261019_164555")]


def digest(train, test):
    s = "|".join(sorted(train)) + "||" + "|".join(sorted(test))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    all_scenes = sorted(x for x in os.listdir(TEST) if os.path.isdir(os.path.join(TEST, x)))
    print(f"1019 共 {len(all_scenes)} 场景")

    # ---- 划分合法性校验 ----
    for t in TRAIN:
        if t not in all_scenes:
            raise SystemExit(f"训练场景不存在: {t}")
    test = [s for s in all_scenes if s not in TRAIN]
    for a, b in HISIM:
        if (a in TRAIN) != (b in TRAIN):
            raise SystemExit(f"高相似对 {a}~{b} 被拆到两侧，违反预登记约束")
    print(f"训练 {len(TRAIN)} / 测试 {len(test)}")

    h = digest(TRAIN, test)
    if os.path.exists(SPLIT):
        old = json.load(open(SPLIT, encoding="utf-8"))
        if old.get("sha256") != h:
            raise SystemExit(f"！已存在的 split_1019.json 与本脚本不一致（{old.get('sha256')[:12]}"
                             f" vs {h[:12]}），拒绝覆盖。若确需改划分请人工处理并记录修订。")
        print("split_1019.json 已存在且一致，不覆盖")
    else:
        json.dump({"train_scenes": TRAIN, "test_scenes": test, "sha256": h,
                   "n_all": len(all_scenes), "note": "预登记冻结；后续工作一律读本文件"},
                  open(SPLIT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"写入 {SPLIT}  sha256={h[:16]}…")

    # ---- 建 12 场景测试集（目录符号链接）----
    os.makedirs(LINK, exist_ok=True)
    made = skipped = 0
    for sc in test:
        dst = os.path.join(LINK, sc)
        src = os.path.join(TEST, sc)
        if os.path.islink(dst) or os.path.exists(dst):
            skipped += 1
            continue
        os.symlink(src, dst)
        made += 1
    n_in = len([x for x in os.listdir(LINK) if os.path.isdir(os.path.join(LINK, x))])
    print(f"测试集链接: 新建 {made}，已存在 {skipped}，现有 {n_in} 个场景")

    # 抽查结构
    bad = []
    for sc in test:
        d = os.path.join(LINK, sc)
        if not (os.path.exists(os.path.join(d, "GT.png"))):
            bad.append(sc)
    print(f"结构抽查：缺 GT 的场景 {bad or '无'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
