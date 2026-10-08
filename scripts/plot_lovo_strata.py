# -*- coding: utf-8 -*-
"""LOVO 分层统计图（2026-10-08）输入 eval/probe_lovo_strata_{may,apr}.csv -> fig/lovo_strata.png"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for c in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"):
    try:
        font_manager.findfont(c, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [c]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ARMS = ["off", "k2", "k4", "k6"]
LABEL = {"off": "官方 RDNet", "k2": "k=2 (6 对)", "k4": "k=4 (12 对)", "k6": "k=6 (16 对)"}
COLOR = {"off": "#444444", "k2": "#1f77b4", "k4": "#ff7f0e", "k6": "#2ca02c"}
ROBS = [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .8), (.8, 1.2), (1.2, 9)]
LUM = [(0, 25), (25, 40), (40, 60), (60, 90), (90, 140), (140, 255)]


def load(name):
    with open(f"eval/probe_lovo_strata_{name}.csv", encoding="utf-8") as fh:
        return [{k: (v if k in ("set", "scene", "stem") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def med(rows, key):
    return float(np.median([r[key] for r in rows])) if rows else np.nan


def grouped(ax, rows, bins, keyf, field, title, ylabel, nmin=8, hline=None, subset=None):
    xs = np.arange(len(bins)); w = .2; labels = []
    top = 0
    for i, (a, b) in enumerate(bins):
        sub = [r for r in rows if a <= r[keyf] < b and (subset is None or subset(r))]
        labels.append(f"{a:g}–{b:g}\nn={len(sub)}")
        for j, arm in enumerate(ARMS):
            v = med(sub, field + arm) if len(sub) >= nmin else np.nan
            if not np.isnan(v):
                top = max(top, abs(v))
            ax.bar(xs[i] + (j - 1.5) * w, v, w, color=COLOR[arm],
                   label=LABEL[arm] if i == 0 else None, edgecolor="white", linewidth=.4)
    if hline is not None:
        ax.axhline(hline, color="#999", lw=1, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, pad=8); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=.25, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return top


may, apr = load("may"), load("apr")
fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.6))
ax = axes[0][0]
t = grouped(ax, may, ROBS, "r_obs", "dpsnr_", "A. may（跨域）· 分反射强度档：块级 ΔPSNR", "ΔPSNR (dB)")
ax.axvspan(4.5, 8, color="#ffe8e8", alpha=.6, zorder=0)
ax.text(6.2, t * .80, "亮反射档：官方明显更好", ha="center", fontsize=9, color="#a33")
ax = axes[0][1]
grouped(ax, may, ROBS, "r_obs", "lap_", "B. may（跨域）· 分反射强度档：细节保留", "细节保留（臂/GT）")
ax.axhline(1.0, color="#999", lw=1, ls="--")
ax = axes[1][0]
grouped(ax, may, LUM, "luma_gt", "dpsnr_", "C. may · Ω 外（真阴影/暗区）：块级 ΔPSNR", "ΔPSNR (dB)",
        subset=lambda r: r["in_omega"] == 0)
ax = axes[1][1]
t2 = grouped(ax, apr, ROBS, "r_obs", "dpsnr_", "D. apr（域内对照）· 块级 ΔPSNR", "ΔPSNR (dB)")
ax.text(5.5, t2 * .62, "域内：微调臂全面优于官方", ha="center", fontsize=9, color="#2a6")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.suptitle("LOVO 分层统计（逐图增益对齐，64×64 块）：跨域时官方在亮反射与暗区都优于微调臂；域内反之",
             fontsize=12.5, y=.985)
fig.tight_layout(rect=[0, .045, 1, .96])
os.makedirs("fig", exist_ok=True)
fig.savefig("fig/lovo_strata.png", dpi=140)
print("saved fig/lovo_strata.png")
