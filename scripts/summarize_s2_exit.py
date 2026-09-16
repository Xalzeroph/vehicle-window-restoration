#!/usr/bin/env python3
"""S2 出口轮判决：E/G 臂 vs plain 基线 → 对照表 + G-S2a/G-S2b 预登记门。

预登记门（Phase A 后定稿）：
  G-S2a 兑现门：v2.1 raw α=1.0 档 2-seed 均值 ≥ plain + 0.5；全档 ≥ plain − 0.2；
                v2.2 α=1.0/0.75 Δ ≥ 0；关键判据 = 配对逐图 CI（n=24）下界 > 0
  G-S2b 选型门：前置"至少一臂 α=1.0 ≥ plain+0.5"；|E − G| ≤ 0.3 → 取 E；G 胜出 >0.3 → 主臂改 G
用法: python3 scripts/summarize_s2_exit.py
"""
import csv, glob, json, os
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V21, V22, IND = "synth_rrw_v21_0913", "synth_rrw_v22_scan", "synth_rrw_v21_indomain"
ARMS = {"E": ["exit_E_s20260915", "exit_E_s101"], "G": ["exit_G_s20260915", "exit_G_s101"]}
T95 = {12: 2.201, 24: 2.069}


def rows(tag, s):
    p = os.path.join(PROJ, f"eval/eval_rdt_{tag}_{s}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def img_raw(tag, s, lv):
    r = rows(tag, s)
    if r is None:
        return None
    return {x["scene"]: float(x["raw"]) for x in r if x["level"] == lv}


def mean_raw(tag, s, lv):
    d = img_raw(tag, s, lv)
    return None if d is None else float(np.mean(list(d.values())))


def paired_delta(tags, s, lv, base="plain"):
    """tags 内每臂与 base 的逐图配对差（合并 → 均值/95%CI/n）。"""
    b = img_raw(base, s, lv)
    ds = []
    for t in tags:
        a = img_raw(t, s, lv)
        if a:
            ds += [a[k] - b[k] for k in a]
    d = np.array(ds)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t95 = T95.get(n, 2.201)
    return d.mean(), d.mean() - t95 * se, d.mean() + t95 * se, n


def main():
    L = []
    lv21 = ["alpha_1", "alpha_0.6", "alpha_0.3"]
    lv22 = ["alpha_1", "alpha_0.75", "alpha_0.6", "alpha_0.45", "alpha_0.3"]

    for name, s, lvs in (("v2.1", V21, lv21), ("v2.2 scan", V22, lv22)):
        L.append(f"### {name}（raw）")
        L.append("| 臂 | " + " | ".join(l.split("_")[-1] for l in lvs) + " |")
        L.append("|---" * (len(lvs) + 1) + "|")
        base = [mean_raw("plain", s, l) for l in lvs]
        L.append("| plain | " + " | ".join(f"{v:.2f}" if v else "-" for v in base) + " |")
        for tag in ["exit_E_s20260915", "exit_E_s101", "exit_G_s20260915", "exit_G_s101"]:
            vals = [mean_raw(tag, s, l) for l in lvs]
            L.append(f"| {tag} | " + " | ".join(
                f"{v:.2f}" if v else "-" for v in vals) + " |")
        for arm, tags in ARMS.items():
            m = []
            for l in lvs:
                vs = [mean_raw(t, s, l) for t in tags]
                vs = [v for v in vs if v is not None]
                m.append(np.mean(vs) if vs else None)
            L.append(f"| **{arm} 2-seed 均值** | " + " | ".join(
                f"{v:.2f}" if v else "-" for v in m) + " |")
        L.append("")

    L.append("### 配对差（2 seed 合并，n=24；括号 = 95%CI）")
    L.append("| 集·档 | E − plain | G − plain | E − G |")
    L.append("|---|---|---|---|")
    for s, lvs in ((V21, lv21), (V22, lv22)):
        for l in lvs:
            e = paired_delta(ARMS["E"], s, l)
            g = paired_delta(ARMS["G"], s, l)
            # E−G：先按 seed 配对
            eg = []
            bE = img_raw("exit_E_s20260915", s, l); bG = img_raw("exit_G_s20260915", s, l)
            if bE and bG:
                eg += [bE[k] - bG[k] for k in bE]
            bE = img_raw("exit_E_s101", s, l); bG = img_raw("exit_G_s101", s, l)
            if bE and bG:
                eg += [bE[k] - bG[k] for k in bE]
            egm = float(np.mean(eg)) if eg else float("nan")
            L.append(f"| {s[:18]}·{l.split('_')[-1]} | {e[0]:+.2f} [{e[1]:+.2f},{e[2]:+.2f}] | "
                     f"{g[0]:+.2f} [{g[1]:+.2f},{g[2]:+.2f}] | {egm:+.2f} |")
    L.append("")

    L.append("### 判决")
    for arm, tags in ARMS.items():
        p10 = mean_raw("plain", V21, "alpha_1")
        v10 = np.mean([mean_raw(t, V21, "alpha_1") for t in tags])
        dl = {l: np.mean([mean_raw(t, V21, l) or 0 for t in tags]) - mean_raw("plain", V21, l)
              for l in lv21}
        d10 = paired_delta(tags, V21, "alpha_1")
        d22 = {l: np.mean([mean_raw(t, V22, l) or 0 for t in tags]) - mean_raw("plain", V22, l)
               for l in ("alpha_1", "alpha_0.75")}
        ok_a = (v10 >= p10 + 0.5) and all(dl[l] >= -0.2 for l in lv21) \
            and all(d22[l] >= 0 for l in d22) and d10[1] > 0
        L.append(f"- **{arm}**：v2.1 α=1.0 {v10:.2f}（plain {p10:.2f}，Δ{dl['alpha_1']:+.2f}，"
                 f"配对 CI [{d10[1]:+.2f},{d10[2]:+.2f}]）；全档 Δ "
                 + "/".join(f"{dl[l]:+.2f}" for l in lv21)
                 + "；v2.2 高档 Δ " + "/".join(f"{d22[l]:+.2f}" for l in d22)
                 + f" → **G-S2a {'PASS' if ok_a else 'FAIL'}**")

    e10 = np.mean([mean_raw(t, V21, "alpha_1") for t in ARMS["E"]])
    g10 = np.mean([mean_raw(t, V21, "alpha_1") for t in ARMS["G"]])
    if max(e10, g10) - mean_raw("plain", V21, "alpha_1") > 0.5:
        verdict = "取 E（α 参数化不劣）" if abs(e10 - g10) <= 0.3 else \
                  ("G 胜出 → 主臂改 G" if g10 > e10 else "E 胜出")
        L.append(f"- **G-S2b**：E {e10:.2f} vs G {g10:.2f}（Δ {e10-g10:+.2f}）→ {verdict}")
    else:
        L.append(f"- **G-S2b**：前置未满足（两臂 α=1.0 均未达 plain+0.5：E {e10:.2f} / G {g10:.2f}）"
                 f" → 两臂塌缩，无胜出方")

    text = "\n".join(L)
    out = os.path.join(PROJ, "eval/summarize_s2_exit.md")
    with open(out, "w") as fp:
        fp.write(text + "\n")
    print(text)
    print(f"\n-> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
