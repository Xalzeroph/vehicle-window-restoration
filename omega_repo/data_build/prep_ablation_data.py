#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为损失消融准备数据：隔离域、锁死 jun、补 C_I、暂存 1019 GT（2026-10-21）。

做四件事（全部可重入）
--------------------
1. `real_pairs_1020` 按 domain 拆出 **apr 子集** → `data/train_sets/real_pairs_1020_apr/`
   （硬链接，不复制像素；含 I/T/R/omega + meta.json）
2. **把 jun 移出训练目录** → `data/locked/jun_pairs_1021/`
   理由：课题红线「jun 数据锁死（不进训练/调参/探针）」，靠目录结构强制，而非靠命令行自觉
3. 给 apr 子集补 `C_I`（源码 `finetune_rdnet_s2.py:132` 读 `m['C_I']`，缺则 KeyError）
   口径与 `build_train_omega_multi.py:101` 完全一致：
   `ring = dilate(om, ellipse21) & ~om`；`C_I = mean(luma[ring]) − mean(luma[om])`，
   luma = `cv2.cvtColor(BGR, COLOR_BGR2GRAY)`
4. 把 1019 的 15 张 GT 拷进扁平目录 `/tmp/ts1019/`，供 `check_t_src_leak.py --ts-dir` 做**内容级**泄漏核查
   （该脚本 --ts-dir 只吃扁平目录）
"""
from __future__ import annotations

import json
import os
import shutil

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
SRC = os.path.join(P, "data/train_sets/real_pairs_1020")
APR = os.path.join(P, "data/train_sets/real_pairs_1020_apr")
LOCK = os.path.join(P, "data/locked/jun_pairs_1021")
TS = "/tmp/ts1019"
TEST19 = os.path.join(P, "data/test_sets/real_pair_1019")
SUBS = ("I", "T", "R", "omega")


def link_or_copy(a, b):
    if os.path.exists(b):
        os.remove(b)
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def c_i_of(i_path, om_path):
    I = cv2.imread(i_path)
    om_u8 = cv2.imread(om_path, 0)
    if I is None or om_u8 is None:
        return None, None
    om = om_u8 > 127
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ring = (cv2.dilate(om.astype(np.uint8), ker) > 0) & (~om)
    if not (om.any() and ring.any()):
        return 0.0, float(om.mean())
    lI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(lI[ring].mean() - lI[om].mean()), float(om.mean())


def main() -> int:
    meta = json.load(open(os.path.join(SRC, "meta.json"), encoding="utf-8"))
    print(f"源 real_pairs_1020: {len(meta)} 对")

    by_dom = {}
    for case, info in meta.items():
        by_dom.setdefault(info.get("domain", "?"), []).append(case)
    print("域分布:", {k: len(v) for k, v in by_dom.items()})

    # ---- 1+3: apr 子集 + C_I ----
    for sub in SUBS:
        os.makedirs(os.path.join(APR, sub), exist_ok=True)
    apr_meta, n_no_ci = {}, 0
    for case in by_dom.get("apr", []):
        info = dict(meta[case])
        for sub in SUBS:
            link_or_copy(os.path.join(SRC, sub, case + ".png"),
                         os.path.join(APR, sub, case + ".png"))
        ci, omf = c_i_of(os.path.join(APR, "I", case + ".png"),
                         os.path.join(APR, "omega", case + ".png"))
        if ci is None:
            n_no_ci += 1
            continue
        info["C_I"] = round(ci, 2)
        info["omega_frac"] = round(omf, 4)
        apr_meta[case] = info
    json.dump(apr_meta, open(os.path.join(APR, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ {APR}: {len(apr_meta)} 对（读失败 {n_no_ci}）")

    cis = sorted(v["C_I"] for v in apr_meta.values())
    if cis:
        print(f"  C_I: p10 {cis[len(cis)//10]:.2f}  中位 {cis[len(cis)//2]:.2f}  "
              f"p90 {cis[int(len(cis)*0.9)]:.2f}  "
              f"为 0 的 {sum(1 for x in cis if x == 0)} 个")
    print(f"  场景数 {len(set(v['scene'] for v in apr_meta.values()))}")

    # ---- 2: jun 锁死（移出训练目录）----
    for sub in SUBS:
        os.makedirs(os.path.join(LOCK, sub), exist_ok=True)
    jun = by_dom.get("jun", [])
    jun_meta = {}
    for case in jun:
        for sub in SUBS:
            link_or_copy(os.path.join(SRC, sub, case + ".png"),
                         os.path.join(LOCK, sub, case + ".png"))
        jun_meta[case] = meta[case]
    if jun_meta:
        json.dump(jun_meta, open(os.path.join(LOCK, "meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    # 从训练目录删除 jun 的 I/T/R/omega（保留 may+apr）
    n_del = 0
    for case in jun:
        for sub in SUBS:
            p = os.path.join(SRC, sub, case + ".png")
            if os.path.exists(p):
                os.remove(p)
                n_del += 1
    # 同步 SRC 的 meta（去掉 jun）
    for case in jun:
        meta.pop(case, None)
    json.dump(meta, open(os.path.join(SRC, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ {LOCK}: {len(jun_meta)} 对（从训练目录移出，删了 {n_del} 个文件）")
    print(f"  训练目录 real_pairs_1020 现在剩 {len(meta)} 对（不含 jun）")

    # ---- 4: 暂存 1019 GT（扁平目录）----
    os.makedirs(TS, exist_ok=True)
    for f in os.listdir(TS):
        os.remove(os.path.join(TS, f))
    n_gt = 0
    for sc in sorted(os.listdir(TEST19)):
        p = os.path.join(TEST19, sc, "GT.png")
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(TS, f"{sc}.png"))
            n_gt += 1
    print(f"→ {TS}: {n_gt} 张 1019 GT（供 check_t_src_leak.py --ts-dir）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
