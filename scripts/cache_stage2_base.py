#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""级联 stage-2 的 T̂_base 缓存（2026-10-21）。

为什么要缓存
-----------
级联第二阶段训练 `(I, T̂_base) → T`，其中 `T̂_base` 是**冻结**的 RDNet 输出。
若在训练循环里现算，每个 iter 要跑一次 266 M 的 RDNet（1.5–2 s），
5000 iter 就是 2.5 h —— 而缓存后训练只用 17 M 的 NAFNet（0.3 s/iter）。

数据
----
· real_pairs_1020：300 对真实配对（apr/may/jun），已 QC
· opensrc：从 train_pairs_opensrc_0920 抽 N 个，**默认剔除 SIR2**（项目纪律红线）

分辨率
------
统一把**长边缩到 ≤ 640** 再缓存。理由：训练只裁 256² 随机块，640 足够；
而 1280×832 逐图推理要 1.7 s，2300 图就是 65 min，640 只要 ~19 min。

输出
----
data/train_sets/stage2_cache_1021/
  I/ T/ base/  <key>.png
  meta.json    list 型：[{key, I, T, base, src, scene}]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_1019_multi import build_rdnet, predict_rdnet   # noqa: E402

REAL = os.path.join(PROJ, "data/train_sets/real_pairs_1020")
OPENSRC = os.path.join(PROJ, "data/train_sets/train_pairs_opensrc_0920/meta.json")
OUT = os.path.join(PROJ, "data/train_sets/stage2_cache_1021")
OFFICIAL = "/DataDisk/zmy_code/RDNet-main/rdnet.pth"
MAXSIDE = 640


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)


def fit(im, maxside=MAXSIDE):
    h, w = im.shape[:2]
    m = max(h, w)
    if m <= maxside:
        return im
    s = maxside / m
    return cv2.resize(im, (max(2, int(round(w * s))), max(2, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-opensrc", type=int, default=2000)
    ap.add_argument("--exclude-src", default="sir2", help="逗号分隔，默认剔除 sir2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    for sub in ("I", "T", "base"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    # ---- 采集清单 ----
    items = []
    rmeta = json.load(open(os.path.join(REAL, "meta.json"), encoding="utf-8"))
    for case, info in rmeta.items():
        items.append({"key": "real__" + case,
                      "I": os.path.join(REAL, "I", case + ".png"),
                      "T": os.path.join(REAL, "T", case + ".png"),
                      "src": "real_" + str(info.get("domain", "?")),
                      "scene": info.get("scene", "?")})
    print(f"真实配对 {len(items)}")

    if os.path.exists(OPENSRC) and args.n_opensrc > 0:
        ometa = json.load(open(OPENSRC, encoding="utf-8"))
        import collections
        dist = collections.Counter(str(x.get("src", "?")).lower() for x in ometa)
        print(f"opensrc 池 {len(ometa)}，src 分布 {dict(dist)}")
        ban = {s.strip().lower() for s in args.exclude_src.split(",") if s.strip()}
        pool = [x for x in ometa if str(x.get("src", "?")).lower() not in ban]
        print(f"剔除 {sorted(ban)} 后剩 {len(pool)}")
        random.Random(args.seed).shuffle(pool)
        for x in pool[:args.n_opensrc]:
            items.append({"key": "osrc__" + str(x.get("name", len(items))),
                          "I": x["I"], "T": x["T"],
                          "src": str(x.get("src", "opensrc")),
                          "scene": str(x.get("name", "?"))})
        print(f"+ opensrc 抽样 {min(args.n_opensrc, len(pool))}")

    print(f"\n合计待缓存 {len(items)} 张", flush=True)

    # ---- 推理缓存 ----
    dev = torch.device(args.device)
    net, net_c = build_rdnet("none")
    sd = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get("icnn", sd), strict=False)
    net.to(dev).eval(); net_c.to(dev).eval()

    meta, n_ok, t0 = [], 0, time.time()
    for i, it in enumerate(items, 1):
        if not (os.path.exists(it["I"]) and os.path.exists(it["T"])):
            continue
        I = read(it["I"]); T = read(it["T"])
        if I is None or T is None:
            continue
        I = fit(I); T = fit(T)
        if I.shape[:2] != T.shape[:2]:
            T = cv2.resize(T, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
        base = predict_rdnet(net, net_c, I, dev)
        base = np.clip(base, 0, 255).astype(np.uint8)
        k = it["key"]
        cv2.imwrite(os.path.join(args.out, "I", k + ".png"),
                    cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out, "T", k + ".png"),
                    cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out, "base", k + ".png"),
                    cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        meta.append({"key": k, "src": it["src"], "scene": it["scene"],
                     "I": os.path.join(args.out, "I", k + ".png"),
                     "T": os.path.join(args.out, "T", k + ".png"),
                     "base": os.path.join(args.out, "base", k + ".png"),
                     "size": [int(I.shape[1]), int(I.shape[0])]})
        n_ok += 1
        if i % 100 == 0 or i == len(items):
            el = time.time() - t0
            print(f"  {i}/{len(items)}  已缓存 {n_ok}  {el:.0f}s  "
                  f"({el/max(i,1):.2f}s/图)  预计剩余 "
                  f"{el/max(i,1)*(len(items)-i)/60:.1f} min", flush=True)

    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    import collections
    dist = collections.Counter(m["src"] for m in meta)
    print(f"\n缓存完成 {n_ok} 张，来源 {dict(dist)}")
    print(f"写入 {args.out}/meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
