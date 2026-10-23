#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 runs/ckpt 下全部训练臂：训练配置、损失组成、耗时、已有评测（2026-10-21）。

为什么要 mmap
------------
每个 ckpt ~1.6 GB，30+ 个臂全量 `torch.load` 要读 50 GB。
`torch.load(..., mmap=True)`（torch≥2.1）把张量映射而不读入，只取 `args`/`iter`
这类小字段时快一个数量级。

损失判定
--------
`train_log.csv` 的表头直接说明该臂启用了哪些损失列；再看末行数值是否非零，
区分"有该列但权重为 0"与"真的在用"。

输出
----
eval/train_inventory.json      结构化全量
eval/train_inventory.md        Markdown 汇总表
"""
from __future__ import annotations

import collections
import csv
import json
import os

import torch

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
ROOT = os.path.join(P, "runs/ckpt")
EVAL = os.path.join(P, "eval")
PRED = os.path.join(P, "runs/pred")

LOSS_COLS = ["L_t", "L_r", "L_om", "L_hinge", "L_dists", "L_recons", "L_excl",
             "L_grad", "L_lap", "L_gc"]
ARG_KEYS = ["iters", "batch", "lr", "freeze", "mona", "peft", "lam_r", "lam_recons",
            "lam_grad", "lam_lap", "lam_gc", "lam_wlow", "arm", "omega_dir",
            "triplets", "data", "seed"]


def read_log(d: str):
    p = os.path.join(d, "train_log.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
    if not rows:
        return None
    hdr = list(rows[0].keys())
    last = rows[-1]
    # 各损失列的末值（有列才取）
    vals = {}
    for c in LOSS_COLS:
        if c in last:
            try:
                vals[c] = float(last[c])
            except (TypeError, ValueError):
                pass
    return {"n_iters_logged": len(rows), "cols": hdr,
            "last_iter": last.get("iter"), "last_sec": last.get("sec"),
            "loss_last": vals, "total_last": last.get("total")}


def read_args(ckpt: str):
    if not os.path.exists(ckpt):
        return None, None
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        try:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            return None, None
    if not isinstance(ck, dict):
        return None, None
    a = {k: v for k, v in ck.get("args", {}).items() if k in ARG_KEYS}
    return a, ck.get("iter")


def main() -> int:
    os.makedirs(EVAL, exist_ok=True)
    arms = sorted(x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x)))
    out = {}
    print(f"共 {len(arms)} 个臂\n")
    hdr = (f"{'臂':<24}{'iter':>7}{'bs':>4}{'freeze':>9}{'mona':>6}"
           f"{'grad':>6}{'lap':>6}{'gc':>5}{'耗时h':>7}{'1019评测':>9}")
    print(hdr); print("-" * len(hdr))

    for a in arms:
        d = os.path.join(ROOT, a)
        ckpts = sorted([f for f in os.listdir(d) if f.startswith("ckpt_") and f.endswith(".pth")],
                       key=lambda f: (f != "ckpt_final.pth", f))
        log = read_log(d)
        # 取 ckpt_final 优先，否则最大的那个
        pick = None
        for cand in ("ckpt_final.pth", "ckpt_latest.pth"):
            if cand in ckpts:
                pick = os.path.join(d, cand)
                break
        if pick is None and ckpts:
            pick = os.path.join(d, ckpts[-1])
        args, it = read_args(pick) if pick else (None, None)

        # 损失是否真在用：有列且末值非零
        loss_last = (log or {}).get("loss_last", {})
        def on(name):
            return "yes" if loss_last.get(name, 0.0) > 1e-9 else (
                "-" if name not in loss_last else "0")
        # 已有 1019 预测？
        has_pred = any(x.startswith(f"real_pair_1019_{a}") or x == f"real_pair_1019_{a}"
                       for x in os.listdir(PRED)) if os.path.isdir(PRED) else False

        secs = None
        if log and log.get("last_sec"):
            try:
                secs = float(log["last_sec"])
            except ValueError:
                pass

        out[a] = {"ckpt": pick, "n_ckpts": len(ckpts), "iter": it, "args": args or {},
                  "log": log, "has_1019_pred": has_pred,
                  "train_hours": round(secs / 3600, 2) if secs else None}
        ar = args or {}
        print(f"{a:<24}{(it if it is not None else -1):>7}{str(ar.get('batch', '-')):>4}"
              f"{str(ar.get('freeze', '-'))[:8]:>9}{str(ar.get('mona', '-')):>6}"
              f"{on('L_grad'):>6}{on('L_lap'):>6}{on('L_gc'):>5}"
              f"{(f'{secs/3600:.1f}' if secs else '-'):>7}"
              f"{('有' if has_pred else '-'):>9}")

    json.dump(out, open(os.path.join(EVAL, "train_inventory.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 汇总 Markdown ----
    def fmt(a):
        r = out[a]
        ar = r["args"]
        ll = (r["log"] or {}).get("loss_last", {})
        losses = [c for c in ("L_grad", "L_lap", "L_gc") if ll.get(c, 0) > 1e-9]
        return (f"| {a} | {r['iter'] if r['iter'] is not None else '-'} | "
                f"{ar.get('batch', '-')} | {ar.get('freeze', '-')} | {ar.get('mona', '-')} | "
                f"{ar.get('lam_r', '-')} | {'/'.join(losses) if losses else '—'} | "
                f"{r['train_hours'] if r['train_hours'] is not None else '-'} | "
                f"{'是' if r['has_1019_pred'] else '否'} |")

    md = ["| 臂 | iter | bs | freeze | mona | lam_r | 额外损失(L1 形式) | 训练h | 已有1019评测 |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [fmt(a) for a in arms]
    txt = "\n".join(md)
    open(os.path.join(EVAL, "train_inventory.md"), "w", encoding="utf-8").write(
        f"# 训练臂清单（{len(arms)} 个）\n\n{txt}\n")
    print("\n" + txt)
    print(f"\n写入 {EVAL}/train_inventory.json / .md")

    # 损失启用统计
    c = collections.Counter()
    for a in arms:
        ll = (out[a]["log"] or {}).get("loss_last", {})
        for k in ("L_grad", "L_lap", "L_gc"):
            if ll.get(k, 0) > 1e-9:
                c[k] += 1
    print(f"\n启用统计：{dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
