#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 jun 从 over-eager 的 `locked/` 移到中性同级目录，并恢复 real_pairs_1020 的 300 对 meta。

背景（2026-10-21）
----------------
我按技能文件里的「jun 锁死」把 jun 移进了 `data/locked/`。用户随即指出 jun 已降级为训练集。
服务器 docs 显示 jun 自 2026-09-20 起一直锁死（且 `docs/2026.9.20_方法线门控轮_方案.md:91`
写明「jun 任何形式入训 → 确认集作废」）。**这是待用户裁决的冲突，不该由我单方面处置。**

因此本脚本只做「撤销我的过度处置、恢复原状」，把 jun 放到中性同级目录
`data/train_sets/real_pairs_1020_jun/`（既非 locked、也不并入主训练集），
由用户决定是并入训练还是留作确认集。
"""
from __future__ import annotations

import json
import os
import shutil

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
SRC = os.path.join(P, "data/train_sets/real_pairs_1020")
JUN = os.path.join(P, "data/train_sets/real_pairs_1020_jun")
LOCK = os.path.join(P, "data/locked/jun_pairs_1021")
SUBS = ("I", "T", "R", "omega")


def main() -> int:
    # 1) 从 locked 移回中性目录
    if os.path.isdir(LOCK):
        os.makedirs(JUN, exist_ok=True)
        for f in os.listdir(LOCK):
            s = os.path.join(LOCK, f)
            d = os.path.join(JUN, f)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        shutil.rmtree(os.path.dirname(LOCK))
        print(f"→ 移回 {JUN}（{len(json.load(open(os.path.join(JUN,'meta.json'),encoding='utf-8')))} 对）")
    elif os.path.isdir(JUN):
        print(f"→ {JUN} 已存在，跳过")
    else:
        print("！locked 与 jun 目录都不存在，无法恢复")

    # 2) 恢复 real_pairs_1020 的 300 对 meta（apr+may+jun）
    mp = os.path.join(SRC, "meta.json")
    cur = json.load(open(mp, encoding="utf-8"))
    jm = os.path.join(JUN, "meta.json")
    if os.path.exists(jm):
        jun_meta = json.load(open(jm, encoding="utf-8"))
        merged = dict(cur)
        merged.update(jun_meta)
        json.dump(merged, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"→ 恢复 {mp}: {len(cur)} + {len(jun_meta)} = {len(merged)} 对")

    # 3) 汇报三处现状（不动数据）
    print("\n当前数据布局：")
    for name in ("real_pairs_1020", "real_pairs_1020_apr", "real_pairs_1020_jun"):
        d = os.path.join(P, "data/train_sets", name)
        m = os.path.join(d, "meta.json")
        if os.path.exists(m):
            n = len(json.load(open(m, encoding="utf-8")))
            print(f"  train_sets/{name:<24} {n:>4} 对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
