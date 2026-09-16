#!/usr/bin/env python3
"""汇总 eval/eval_rdt_*_summary.json → 对照表 + G-R1/G-R2 预登记门判决。

G-R1（提升门，raw 主口径，freeze 臂 2-seed 均值）:
  v2.1 raw@ᾱ=0.3 ≥ 17.5；raw 全档 ≥ pilot B_r2 − 0.2（19.42/17.87/16.31）
G-R2（方法门，w 组件，freeze − plain 同 seed 配对 2-seed 平均）:
  ᾱ=0.3 ≥ +0.3 dB 且 ᾱ=1.0 ≥ −0.2 dB（never-worse）

用法: python3 scripts/summarize_rdt_evals.py [--out docs/表.md]
"""
import argparse, glob, json, os

V21 = "synth_rrw_v21_0913"
IND = "synth_rrw_v21_indomain"
V22 = "synth_rrw_v22_scan"
LV3 = ["alpha_1", "alpha_0.6", "alpha_0.3"]          # 降序后缀数值：1 / 0.6 / 0.3
PILOT_RAW = {"alpha_1": 19.62, "alpha_0.6": 18.07, "alpha_0.3": 16.51}
GR1_RAW03 = 17.5
GR1_SLACK = 0.2
GR2_D03, GR2_D10 = 0.3, -0.2
ARM_SEEDS = {"freeze": "20260915", "freeze_s101": "101", "plain": "20260915", "plain_s101": "101"}


def load(glob_pat="eval/eval_rdt_*_summary.json"):
    tab = {}
    for f in sorted(glob.glob(glob_pat)):
        d = json.load(open(f))
        tab.setdefault(d["tag"], {})[d["set"]] = {r["level"]: r for r in d["per_level"]}
    return tab


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="eval/eval_rdt_*_summary.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tab = load(args.glob)
    L = []

    def sec(title):
        L.append("")
        L.append(title)

    # 表 1/2: v2.1 与 indomain 三档
    for name, s in (("v2.1", V21), ("indomain", IND)):
        sec(f"### {name}（3 档，12 场景均值）")
        L.append("| tag | raw 1.0/0.6/0.3 | gm 1.0/0.6/0.3 | in_gm | R̂raw 1.0/0.6/0.3 | ssim(raw) |")
        L.append("|---|---|---|---|---|---|")
        for tag, sets in tab.items():
            if s not in sets:
                continue
            r = sets[s]
            L.append("| " + tag + " | "
                     + "/".join(f(r[lv]["raw"]) for lv in LV3) + " | "
                     + "/".join(f(r[lv]["gm"]) for lv in LV3) + " | "
                     + f(r[LV3[0]]["in_gm"]) + " | "
                     + "/".join(f(r[lv]["r_raw"]) for lv in LV3) + " | "
                     + "/".join(f(r[lv]["raw_ssim"], 3) for lv in LV3) + " |")

    # 表 3: v22 scan 曲线（raw）
    lvs22 = sorted(tab[next(iter(tab))][V22].keys(), key=lambda x: -float(x.split("_")[-1]))
    sec("### v2.2 scan（raw，8 档，α 降序）")
    L.append("| tag | " + " | ".join(lv.split("_")[-1] for lv in lvs22) + " |")
    L.append("|---" * (len(lvs22) + 1) + "|")
    for tag, sets in tab.items():
        if V22 not in sets:
            continue
        L.append("| " + tag + " | " + " | ".join(f(sets[V22][lv]["raw"]) for lv in lvs22) + " |")
    L.append("| in_gm(基线) | " + " | ".join(f(tab[next(iter(tab))][V22][lv]["in_gm"]) for lv in lvs22) + " |")

    # 判决
    sec("### 判决（预登记门）")
    def arm(tag, lv, key="raw"):
        try:
            return tab[tag][V21][lv][key]
        except KeyError:
            return None

    L.append("**G-R1（提升门，raw，freeze 臂）**")
    for tag in ("freeze", "freeze_s101"):
        if tag not in tab:
            continue
        parts = []
        for lv in LV3:
            v = arm(tag, lv)
            thr = GR1_RAW03 if lv == "alpha_0.3" else round(PILOT_RAW[lv] - GR1_SLACK, 2)
            parts.append(f"{lv.split('_')[-1]} {f(v)}（{'≥' if v is not None and v >= thr else '<'} 门 {thr}）")
        L.append(f"- {tag}: " + "，".join(parts))
    means = {lv: None for lv in LV3}
    if "freeze" in tab and "freeze_s101" in tab:
        for lv in LV3:
            means[lv] = (arm("freeze", lv) + arm("freeze_s101", lv)) / 2
        ok = (means["alpha_0.3"] >= GR1_RAW03 and
              all(means[lv] >= PILOT_RAW[lv] - GR1_SLACK for lv in LV3))
        L.append(f"- **2-seed 均值**: " + "，".join(f"{lv.split('_')[-1]} {f(means[lv])}" for lv in LV3)
                 + f" → **G-R1 {'PASS' if ok else 'FAIL'}**")
        spread = {lv: abs(arm("freeze", lv) - arm("freeze_s101", lv)) for lv in LV3}
        L.append("- seed 极差: " + "，".join(f"{lv.split('_')[-1]} {f(spread[lv])}" for lv in LV3))

    L.append("")
    L.append("**G-R2（方法门，w 组件：freeze − plain）**")
    for lv in LV3:
        if "freeze" not in tab or "plain" not in tab:
            continue
        d1 = arm("freeze", lv) - arm("plain", lv)
        d2 = arm("freeze_s101", lv) - arm("plain_s101", lv)
        dm = (d1 + d2) / 2
        thr = GR2_D03 if lv == "alpha_0.3" else (GR2_D10 if lv == "alpha_1" else None)
        mark = "" if thr is None else ("（门 ≥%+.1f → %s）" % (thr, "PASS" if dm >= thr else "FAIL"))
        L.append(f"- {lv.split('_')[-1]}: seed1 {d1:+.2f} / seed101 {d2:+.2f} / 均值 {dm:+.2f}{mark}")
    if "freeze" in tab and "plain" in tab:
        dm03 = ((arm("freeze", "alpha_0.3") - arm("plain", "alpha_0.3")) +
                (arm("freeze_s101", "alpha_0.3") - arm("plain_s101", "alpha_0.3"))) / 2
        dm10 = ((arm("freeze", "alpha_1") - arm("plain", "alpha_1")) +
                (arm("freeze_s101", "alpha_1") - arm("plain_s101", "alpha_1"))) / 2
        ok = dm03 >= GR2_D03 and dm10 >= GR2_D10
        L.append(f"- **G-R2 {'PASS' if ok else 'FAIL'}**（0.3 档 {dm03:+.2f} / 1.0 档 {dm10:+.2f}）")

    text = "\n".join(L)
    if args.out:
        with open(args.out, "w") as fp:
            fp.write(text + "\n")
        print(f"-> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
