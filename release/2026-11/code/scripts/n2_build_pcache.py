#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N2 探针用的**训练域 P 缓存**（只做推理，不训练、不改任何既有脚本）。

为什么需要
----------
审查要求"训练域拟合 → 面板直测"的直接迁移证据：α 幅度诊断与共享 RGB 仿射对照都必须在
**训练域**拟合。训练域此前没有任何 F0 预测缓存，因此这两项只能记"未判"。本脚本用冻结的
F0（三个 may+jun 种子）在原训练池上跑一次前向，把 P 落盘，供 n2_probe_cpu.py 的
`transfer` 动作读取。

口径（与面板缓存逐项对齐，别自己发明）
--------------------------------------
  * 输入原样（不 resize；与 scripts/eval_1019_multi.predict_rdnet 同一函数、同一 pad 规则）
  * 输出 uint8 = rint(clip(P,0,255))，RGB→BGR 后写 PNG（与面板 t_hat.png 同口径）
  * 只用 I（推理不需要 GT）；GT 只在探针评分时读，不进算子
  * 一个种子一个目录：<out>/s101/<case>.png ...

用法（GPU 单卡，几分钟；**不是训练**）
  cd /DataDisk
  export HOME=/DataDisk/zmy_code/.home_zmy
  CUDA_VISIBLE_DEVICES=0 /DataDisk/miniconda/envs/xreflection/bin/python3 \
    /DataDisk/zmy_code/TintedGlass_Enhance/scripts/n2_build_pcache.py \
    --seeds 101,202,303 --out runs/n2_1031/pcache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_1019_multi import build_rdnet, load_rdnet, predict_rdnet   # noqa: E402

CKPT_TMPL = os.path.join(PROJ, "runs/dom_v1/R-mayjun_s{seed}/ckpt_final.pth")
POOL = os.path.join(PROJ, "data/train_sets/real_pairs_1020")


def log(*a):
    print(*a, flush=True)


def cases(pool_dir, domains):
    with open(os.path.join(pool_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    keep = {d.strip().lower() for d in str(domains).split(",") if d.strip()}
    out = []
    for k in sorted(meta):
        m = meta[k]
        if keep and str(m.get("domain", "")).lower() not in keep:
            continue
        out.append((k, os.path.join(pool_dir, "I", k + ".png")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--ckpt-tmpl", default=CKPT_TMPL)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--domains", default="may,jun")
    ap.add_argument("--out", default=os.path.join(PROJ, "runs/n2_1031/pcache"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--mona", default="both")
    ap.add_argument("--limit", type=int, default=0, help="调试用：只跑前 N 例")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cs = cases(args.pool, args.domains)
    if args.limit:
        cs = cs[: args.limit]
    log(f"[pcache] {len(cs)} 例 / pool={os.path.relpath(args.pool, PROJ)} domains={args.domains} "
        f"dev={dev} out={os.path.relpath(args.out, PROJ)}")
    t0 = time.time()
    for seed in [int(x) for x in str(args.seeds).split(",") if x.strip()]:
        ck = args.ckpt_tmpl.format(seed=seed)
        if not os.path.exists(ck):
            raise SystemExit(f"[pcache] 缺 ckpt {ck}")
        net, net_c = build_rdnet(args.mona)
        load_rdnet(net, ck, args.mona)
        net.eval().to(dev)
        net_c.eval().to(dev)          # net_c 由 build_rdnet 建在 CPU 上，必须一起搬
        d = os.path.join(args.out, f"s{seed}")
        os.makedirs(d, exist_ok=True)
        with torch.no_grad():
            for k, (case, ip) in enumerate(cs):
                im = cv2.imread(ip, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)   # predict_rdnet 要 float32
                t = predict_rdnet(net, net_c, img, dev)
                q = np.rint(np.clip(t, 0, 255)).astype(np.uint8)
                cv2.imwrite(os.path.join(d, case + ".png"), q[..., ::-1])
                if (k + 1) % 10 == 0:
                    log(f"  s{seed} {k + 1}/{len(cs)} {time.time() - t0:.0f}s")
        del net, net_c
        torch.cuda.empty_cache()
        log(f"[pcache] s{seed} done -> {d}  累计 {time.time() - t0:.0f}s")
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(pool=os.path.relpath(args.pool, PROJ), domains=args.domains,
                       seeds=[int(x) for x in str(args.seeds).split(",") if x.strip()],
                       ckpt_tmpl=args.ckpt_tmpl, n_case=len(cs),
                       built=time.strftime("%Y-%m-%d %H:%M:%S"),
                       quant="uint8 = rint(clip(P,0,255)); RGB->BGR PNG"), fh,
                  ensure_ascii=False, indent=1)
    log(f"[pcache] all done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
