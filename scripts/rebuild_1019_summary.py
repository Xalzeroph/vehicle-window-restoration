#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建 1019 多臂汇总 JSON（2026-10-20）。

背景：`eval_1019_multi.py` 每次运行都会**覆盖** `real_pair_1019_summary.json`，
导致多次分臂跑完后只留下最后一臂。本脚本从各臂的逐对 CSV 重新汇总，
不重跑推理（零 GPU）。

输出：eval/real_pair_1019_summary.json（合并全臂）+ eval/fig_1019/table_1019.md
"""
from __future__ import annotations

import csv
import glob
import json
import os

PROJ = "/DataDisk/zmy_code/TintedGlass_Enhance"
EVAL = os.path.join(PROJ, "eval")
OUT = os.path.join(EVAL, "fig_1019")

DISPLAY = {
    "identity": "INPUT (identity)",
    "rdnet_zs": "RDNet official",
    "ft_s2main": "Ours ft_s2main (rdt_s2_main_s101/10k)",
    "ft_mixbest": "Ours ft_mixbest (mix_b2_may_apr)",
    "lightrr": "LightRR (CVPR26, w/o Nature)",
}
ORDER = ["identity", "rdnet_zs", "ft_s2main", "ft_mixbest", "lightrr"]


def summarise(csv_path: str) -> dict:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return {}
    ps = [float(r["psnr"]) for r in rows]
    ss = [float(r["ssim"]) for r in rows]
    pg = [float(r["psnr_gm"]) for r in rows]
    return {"n": len(rows),
            "psnr": round(sum(ps) / len(ps), 3),
            "ssim": round(sum(ss) / len(ss), 4),
            "psnr_gm": round(sum(pg) / len(pg), 3)}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    summ = {}
    for k in ORDER:
        p = os.path.join(EVAL, f"real_pair_1019_{k}.csv")
        if os.path.exists(p):
            summ[k] = summarise(p)
            print(f"  {k:<12} n={summ[k]['n']}  PSNR={summ[k]['psnr']:.2f}  "
                  f"SSIM={summ[k]['ssim']:.4f}")
        else:
            print(f"  {k:<12} [缺 CSV] {p}")

    mx = os.path.join(EVAL, "real_pair_1019_mix_arms.json")
    if os.path.exists(mx):
        d = json.load(open(mx, encoding="utf-8"))
        summ["_mix_arms"] = d.get("arms", {})
        summ["_mix_best"] = d.get("best")

    sp = os.path.join(EVAL, "real_pair_1019_summary.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    print(f"\n写入 {sp}")

    base = summ.get("identity", {}).get("psnr")
    lines = ["| 模型 | n | PSNR↑ | SSIM↑ | PSNR(gm)↑ | ΔPSNR vs INPUT |",
             "|---|---|---|---|---|---|"]
    for k in ORDER:
        if k not in summ:
            continue
        s = summ[k]
        d = f"{s['psnr']-base:+.2f}" if base is not None else "-"
        lines.append(f"| {DISPLAY[k]} | {s['n']} | {s['psnr']:.2f} | {s['ssim']:.4f} | "
                     f"{s['psnr_gm']:.2f} | {d} |")
    tbl = "\n".join(lines)
    print("\n" + tbl)
    with open(os.path.join(OUT, "table_1019.md"), "w", encoding="utf-8") as fh:
        fh.write("# 10-19 户外车窗配对集（n=%d）多模型对比\n\n%s\n\n"
                 "> gt(gm) = 对预测做最优全局乘性匹配后再算 PSNR，口径同 eval_e1.py。\n"
                 % (summ.get("identity", {}).get("n", 0), tbl))
    print(f"\n写入 {OUT}/table_1019.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
