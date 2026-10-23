#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实测 S2 训练集**按分支**的反射强度（2026-10-21）。

S2Dataset.__getitem__ 返回 5 项，第 5 项是
`tensor([has_R, has_om, C_I, w, src])`，其中 **src 就是分支标签**（0/1/2）。
所以可以按分支分别统计 |I−T|，直接回答"模型在每条数据上见到的反射有多强"。

对照：1019 测试集 |I−GT| 中位 = 19.18（p10 15.64 / p90 22.37）
"""
from __future__ import annotations

import os
import sys
import collections

import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
os.chdir(os.path.join(P, "scripts"))

from finetune_rdnet_s2 import S2Dataset   # noqa: E402

N = 600
TEST_STRENGTH = 19.18


def run(arm: str, omega_dir: str | None, label: str):
    ds = S2Dataset(arm=arm, size=256, omega_dir=omega_dir)
    print(f"\n=== {label} ===")
    print(f"  omega_dir={os.path.basename(str(ds.omega_dir))}   p={ds.p}")
    by = collections.defaultdict(list)
    err = None
    for i in range(N):
        try:
            I, T, R, om, meta = ds[i]
        except Exception as e:
            if err is None:
                err = f"{type(e).__name__}: {e}"
            continue
        src = int(round(float(meta[4])))
        by[src].append(float((I - T).abs().mean()) * 255.0)
    if err:
        print(f"  [首个异常] {err[:160]}")
    names = {0: "omega(实拍Ω对)", 1: "synth(在线合成)", 2: "opensrc(开源直督)"}
    allv = []
    print(f"  {'分支':<18}{'n':>5}{'p10':>8}{'中位':>8}{'p90':>8}{'vs 1019':>9}")
    print("  " + "-" * 56)
    for src in sorted(by):
        v = np.array(by[src])
        allv.extend(v.tolist())
        print(f"  {names.get(src, str(src)):<18}{len(v):>5}"
              f"{np.percentile(v,10):>8.2f}{np.median(v):>8.2f}"
              f"{np.percentile(v,90):>8.2f}{TEST_STRENGTH/np.median(v):>8.2f}x")
    if allv:
        a = np.array(allv)
        print(f"  {'混合后(实际看到)':<18}{len(a):>5}"
              f"{np.percentile(a,10):>8.2f}{np.median(a):>8.2f}"
              f"{np.percentile(a,90):>8.2f}{TEST_STRENGTH/np.median(a):>8.2f}x")
    return np.median(allv) if allv else float("nan")


print("=" * 66)
print(f"参照：1019 测试集 |I−GT| 中位 = {TEST_STRENGTH}")
print("=" * 66)

m1 = run("main", None, "rdt_s2_main_s101（默认 omega_dir）")
m2 = run("main", os.path.join(P, "data/train_sets/pairs_may"), "mix_b2_may_apr（pairs_may）")
m3 = run("b1", None, "arm=b1（0:57:43，几乎全合成）")

print("\n" + "=" * 66)
print(f"{'配置':<26}{'训练(混合)中位':>16}{'1019 中位':>12}{'比值':>8}")
print("-" * 62)
for lab, r in (("rdt_s2_main_s101", m1), ("mix_b2_may_apr", m2), ("arm=b1", m3)):
    print(f"{lab:<26}{r:>16.2f}{TEST_STRENGTH:>12.2f}{TEST_STRENGTH/r:>8.2f}x")
