#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 `real_pairs_1020/meta.json` 的全部 300 对补 `C_I`（2026-10-22）。

为什么必须补
-----------
`finetune_rdnet_s2.py:132` 读 `m['C_I']` 算 hinge 阈值 `δ = clamp(0.5·C_I, min=4)/255`。
`real_pairs_1020` 是 `build_train_real_pairs.py` 建的，**没写 `C_I`**（300/0）；
只有它的 apr 子集（`real_pairs_1020_apr`）在上一轮被补过。
本轮要按域过滤（`--omega-domains apr|may|jun`），元数据必须在**同一个池**里齐全。

口径（与 `build_train_omega_multi.py:101` 完全一致）
    ker  = getStructuringElement(MORPH_ELLIPSE, (21,21))
    ring = (dilate(om, ker) > 0) & ~om
    luma = cvtColor(I, BGR2GRAY)
    C_I  = mean(luma[ring]) − mean(luma[om])      # om 或 ring 为空 → 0.0

自检
----
apr 的 126 个 case 重算值须与 `real_pairs_1020_apr/meta.json` 已有 `C_I` 相符（容差 0.01）。
"""
from __future__ import annotations

import collections
import json
import os
import shutil

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")
APR_REF = os.path.join(P, "data/train_sets/real_pairs_1020_apr/meta.json")
BAK = os.path.join(POOL, "meta.json.bak_20261022")


def c_i_of(i_path, om_path):
    I = cv2.imread(i_path)
    om_u8 = cv2.imread(om_path, 0)
    if I is None or om_u8 is None:
        return None
    om = om_u8 > 127
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(om.astype(np.uint8), ker) > 0) & (~om)
    if not (om.any() and ring.any()):
        return 0.0
    lI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(lI[ring].mean() - lI[om].mean())


def main() -> int:
    mp = os.path.join(POOL, "meta.json")
    meta = json.load(open(mp, encoding="utf-8"))
    print(f"池 {len(meta)} 对")

    if not os.path.exists(BAK):
        shutil.copy2(mp, BAK)
        print(f"备份 -> {os.path.basename(BAK)}")
    else:
        print(f"备份已存在，跳过备份")

    n_new = n_skip = n_fail = 0
    recalc = {}
    for case, v in meta.items():
        if "C_I" in v:
            n_skip += 1
            recalc[case] = v["C_I"]
            continue
        ci = c_i_of(os.path.join(POOL, "I", case + ".png"),
                    os.path.join(POOL, "omega", case + ".png"))
        if ci is None:
            n_fail += 1
            continue
        v["C_I"] = round(ci, 2)
        recalc[case] = v["C_I"]
        n_new += 1

    json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    by = collections.defaultdict(list)
    for case, v in meta.items():
        by[str(v.get("domain", "?"))].append(v["C_I"])
    print(f"\n新写入 {n_new}  已有跳过 {n_skip}  读失败 {n_fail}")
    print(f"{'域':<8}{'n':>5}{'p10':>8}{'中位':>8}{'p90':>8}")
    print("-" * 37)
    for dom in sorted(by):
        v = sorted(by[dom])
        print(f"{dom:<8}{len(v):>5}{v[len(v)//10]:>8.2f}{v[len(v)//2]:>8.2f}"
              f"{v[int(len(v)*0.9)]:>8.2f}")

    # 自检：apr 与上一轮已补的子集比对
    if os.path.exists(APR_REF):
        ref = json.load(open(APR_REF, encoding="utf-8"))
        diffs, missing = [], 0
        for case, rv in ref.items():
            if "C_I" not in rv:
                continue
            if case not in recalc:
                missing += 1
                continue
            diffs.append(abs(float(recalc[case]) - float(rv["C_I"])))
        if diffs:
            bad = sum(1 for d in diffs if d > 0.01)
            print(f"\n自检（apr 与 real_pairs_1020_apr 比对）: 比了 {len(diffs)} 个，"
                  f"最大差 {max(diffs):.4f}，超差(>0.01) {bad} 个，缺失 {missing}")
            if bad:
                print("  ！有超差，说明口径不一致，需复查")
                return 1
            print("  ✅ 口径一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
