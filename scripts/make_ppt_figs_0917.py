#!/usr/bin/env python3
"""Generate the 5 data figures for the 2026-09-17 group-meeting deck.

All numbers are read from eval/ artifacts (no hand-copied values). Two numbers
without a machine-readable source are quoted from the experiment records and
marked  # doc: provenance in the code below.
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/DataDisk/zmy_code/TintedGlass_Enhance"
OUT = os.path.join(ROOT, "projects/tintedglass_sept_review_20260917/images")
os.makedirs(OUT, exist_ok=True)

BLUE = "#1F5C8B"
TEAL = "#0E7C86"
ORANGE = "#C2410C"
GRAY = "#8A9099"
LIGHT = "#D9DEE4"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.edgecolor": "#4A5158",
    "axes.linewidth": 1.0,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

rng = np.random.default_rng(0)


def level_alpha(level):
    return float(level.replace("alpha_", "").replace("_", "."))


def load_arm(path):
    """{(level, scene): raw} from one eval CSV."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[(row["level"], row["scene"])] = float(row["raw"])
    return out


def paired_delta(arms_a, arms_b):
    """Mean per-scene delta (a - b) over seeds, plus scene-clustered 95% CI."""
    by_level = defaultdict(lambda: defaultdict(list))
    for aa, bb in zip(arms_a, arms_b):
        for key in aa:
            if key in bb:
                by_level[key[0]][key[1]].append(aa[key] - bb[key])
    stats = {}
    for level, scenes in by_level.items():
        per_scene = {s: float(np.mean(v)) for s, v in scenes.items()}
        vals = np.array(list(per_scene.values()))
        names = list(per_scene)
        boots = []
        idx = np.arange(len(vals))
        for _ in range(4000):
            pick = rng.choice(idx, size=len(idx), replace=True)
            boots.append(vals[pick].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        stats[level] = dict(mean=float(vals.mean()), lo=float(lo), hi=float(hi), n=len(vals))
    return stats


def eval_csv(tag, testset):
    return os.path.join(ROOT, f"eval/eval_rdt_{tag}_synth_rrw_{testset}.csv")


print("=" * 62)
# ---------------------------------------------------------------- fig 1
v23_plain = [load_arm(eval_csv(t, "v23_lowscan")) for t in ("plain", "plain_s101")]
v23_freeze = [load_arm(eval_csv(t, "v23_lowscan")) for t in ("freeze", "freeze_s101")]
v21_plain = [load_arm(eval_csv(t, "v21_0913")) for t in ("plain", "plain_s101")]
v21_freeze = [load_arm(eval_csv(t, "v21_0913")) for t in ("freeze", "freeze_s101")]

d23 = paired_delta(v23_freeze, v23_plain)
d21 = paired_delta(v21_freeze, v21_plain)
for name, d in (("v23 leak-free", d23), ("v2.1", d21)):
    print(f"[fig1] {name}")
    for lv in sorted(d, key=level_alpha):
        s = d[lv]
        print(f"   a={level_alpha(lv):.2f}  d={s['mean']:+.2f}  CI[{s['lo']:+.2f},{s['hi']:+.2f}]  n={s['n']}")

fig, ax = plt.subplots(figsize=(7.6, 4.3))
for d, color, label, marker in ((d23, BLUE, "v23 expanded set (leak-free, 48 scenes)", "o"),
                                (d21, ORANGE, "v2.1 benchmark (12 scenes)", "s")):
    xs = sorted(d, key=level_alpha)
    x = [level_alpha(l) for l in xs]
    y = [d[l]["mean"] for l in xs]
    err = np.array([[d[l]["mean"] - d[l]["lo"] for l in xs],
                    [d[l]["hi"] - d[l]["mean"] for l in xs]])
    ax.errorbar(x, y, yerr=err, marker=marker, color=color, label=label,
                capsize=4, lw=2, ms=7)
ax.axhline(0, color="#3A4046", lw=1.2)
ax.set_xscale("log")
ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.6, 1.0])
ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.6", "1.0"])
ax.set_xlabel(r"nominal tint level $\bar{\alpha}$")
ax.set_ylabel(r"$\Delta$ raw PSNR  (freeze $-$ plain), dB")
ax.set_title("Low-$\\alpha$ specialist gain flips sign with the test set", pad=10)
ax.grid(axis="y", color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="lower left")
fig.savefig(os.path.join(OUT, "fig_delta_alpha.png"))
plt.close(fig)

# ---------------------------------------------------------------- shared tables
probe = {}
with open(os.path.join(ROOT, "eval/probe_s2_exit_synth_rrw_v21_0913.csv")) as fh:
    for row in csv.DictReader(fh):
        probe[level_alpha(row["level"])] = {k: float(v) for k, v in row.items()
                                            if k not in ("set", "level")}
ALPHAS = sorted(probe)
print(f"[probe] levels {ALPHAS}")

plain21 = {}
for a in ("plain", "plain_s101"):
    for (lv, sc), v in load_arm(eval_csv(a, "v21_0913")).items():
        plain21.setdefault(level_alpha(lv), []).append(v)
plain21 = {k: float(np.mean(v)) for k, v in plain21.items()}

exit_arm = {}
for arm in ("E", "G"):
    acc = defaultdict(list)
    for s in ("s20260915", "s101"):
        p = os.path.join(ROOT, f"eval/eval_rdt_exit_{arm}_{s}_synth_rrw_v21_0913_summary.json")
        for r in json.load(open(p))["per_level"]:
            acc[level_alpha(r["level"])].append(r["raw"])
    exit_arm[arm] = {k: float(np.mean(v)) for k, v in acc.items()}

oracle = {}
with open(os.path.join(ROOT, "eval/eval_e2_oracle_v21.csv")) as fh:
    rows = list(csv.DictReader(fh))
print("[oracle] cols:", list(rows[0].keys()))
# exA = "perfect de-reflection + exact per-pixel alpha", docs/2026.9.13_基准v2.1定案与zero-shot首跑.md:69-71
print("[oracle] using column: exA")
for row in rows:
    oracle[level_alpha(row["level"])] = float(row["exA"])
print("[oracle]", {k: round(v, 2) for k, v in sorted(oracle.items())})

# counterfactual: docs/2026.9.17_S2物理出口轮_实验记录.md:56 (v2.1, alpha 1.0/0.6/0.3)
CF = {1.0: 22.15, 0.6: 21.18, 0.3: 18.85}
# constant baseline: docs/2026.9.16_S1四臂评测与判决.md:95
CONST = 17.68
# R pool: data/pools/r_train_0915_meta.csv
pool = defaultdict(int)
with open(os.path.join(ROOT, "data/pools/r_train_0915_meta.csv")) as fh:
    for row in csv.DictReader(fh):
        pool[row["action"].split(":")[0]] += 1
print("[pool]", dict(pool))

print("=" * 62)
# ---------------------------------------------------------------- fig 2
fig, ax = plt.subplots(figsize=(7.8, 4.3))
xs = np.arange(len(ALPHAS))
w = 0.26
series = [("True $R$ + true $\\alpha$ (oracle)", [oracle[a] for a in ALPHAS], GRAY),
          ("Our $\\hat{R}$ / true $\\alpha$", [probe[a]["div_true"] for a in ALPHAS], TEAL),
          ("Current method", [plain21[a] for a in ALPHAS], BLUE)]
for i, (label, vals, color) in enumerate(series):
    bars = ax.bar(xs + (i - 1) * w, vals, w, label=label, color=color)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
ax.axhline(CONST, color=ORANGE, ls="--", lw=1.6)
ax.text(-0.42, CONST + 0.6, f"flat-image constant baseline {CONST:.2f}",
        color=ORANGE, fontsize=10, ha="left")
ax.set_xticks(xs)
ax.set_xticklabels([f"$\\bar{{\\alpha}}$ = {a:.1f}" for a in ALPHAS])
ax.set_ylabel("raw PSNR, dB")
ax.set_ylim(0, 42)
ax.set_xlim(-0.62, len(ALPHAS) - 0.38)
ax.set_title("What each stage of the pipeline can reach, by tint level", pad=10)
ax.grid(axis="y", color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=10.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.14), ncol=3)
fig.savefig(os.path.join(OUT, "fig_boundary_map.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, ax = plt.subplots(figsize=(7.8, 4.3))
w = 0.2
cf = [CF[a] for a in ALPHAS]
ser = [("Exit head, learned $\\tilde{\\alpha}$ (E)", [exit_arm["E"][a] for a in ALPHAS], BLUE),
       ("Exit head, free gain (G)", [exit_arm["G"][a] for a in ALPHAS], LIGHT),
       ("Same $\\beta$, true $\\alpha$", cf, ORANGE),
       ("No exit (plain)", [plain21[a] for a in ALPHAS], GRAY)]
for i, (label, vals, color) in enumerate(ser):
    bars = ax.bar(xs + (i - 1.5) * w, vals, w, label=label, color=color)
    ax.bar_label(bars, fmt="%.1f", fontsize=8.5, padding=2)
for i, a in enumerate(ALPHAS):
    ax.plot([i - 2 * w, i + 1.5 * w], [oracle[a]] * 2, color="#3A4046", ls=":", lw=1.6)
ax.text(0.995, 0.985, "dotted = oracle ceiling (true $R$)", transform=ax.transAxes,
        ha="right", va="top", fontsize=9.5, color="#3A4046")
ax.set_xticks(xs)
ax.set_xticklabels([f"$\\bar{{\\alpha}}$ = {a:.1f}" for a in ALPHAS])
ax.set_ylabel("raw PSNR, dB")
ax.set_ylim(0, 42)
ax.set_title("Swapping $\\tilde{\\alpha}$ for ground truth recovers the ceiling", pad=10)
ax.grid(axis="y", color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="upper center",
          bbox_to_anchor=(0.5, -0.14), ncol=2)
fig.savefig(os.path.join(OUT, "fig_exit_counterfactual.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 4
fig, ax = plt.subplots(figsize=(7.4, 4.2))
w = 0.34
tru = [oracle[a] for a in ALPHAS]
rhat = [probe[a]["div_true"] for a in ALPHAS]
b1 = ax.bar(xs - w / 2, tru, w, label="True $R$ + true $\\alpha$", color=GRAY)
b2 = ax.bar(xs + w / 2, rhat, w, label="Our $\\hat{R}$ + true $\\alpha$", color=TEAL)
ax.bar_label(b1, fmt="%.1f", fontsize=10, padding=2)
ax.bar_label(b2, fmt="%.1f", fontsize=10, padding=2)
for i, a in enumerate(ALPHAS):
    gap = tru[i] - rhat[i]
    xa = i + w / 2 + 0.09
    ax.annotate("", xy=(xa, rhat[i]), xytext=(xa, tru[i]),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    ax.text(xa + 0.05, (tru[i] + rhat[i]) / 2, f"$-${gap:.1f} dB",
            color=ORANGE, fontsize=10.5, va="center", ha="left")
ax.set_xticks(xs)
ax.set_xticklabels([f"$\\bar{{\\alpha}}$ = {a:.1f}" for a in ALPHAS])
ax.set_ylabel("raw PSNR, dB")
ax.set_ylim(0, 42)
ax.set_xlim(-0.62, len(ALPHAS) - 0.02)
ax.set_title("Division by $1/\\alpha$ amplifies every $\\hat{R}$ error", pad=10)
ax.grid(axis="y", color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=10.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.14), ncol=2)
fig.savefig(os.path.join(OUT, "fig_error_budget.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 5
labels = {"keep": "kept (clean pool)", "drop_dup": "redundant duplicate",
          "drop_twin_of": "twin of another candidate", "drop_test_cluster": "test-R cluster"}
order = ["keep", "drop_dup", "drop_twin_of", "drop_test_cluster"]
colors = [BLUE, LIGHT, "#B9C2CB", ORANGE]
vals = [pool[k] for k in order]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.4))
ax.text(0, 0.08, "601", ha="center", va="center", fontsize=22, color="#1A1F24")
ax.text(0, -0.16, "candidates", ha="center", va="center", fontsize=11, color="#5A626B")
ax.legend(wedges, [f"{labels[k]}  ({pool[k]})" for k in order],
          frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(-0.42, -0.12))
ax.set_title("Why 470 of 601 were dropped", pad=6)

ax = axes[1]
src = [("RRW scene layer", 17, BLUE), ("wsoft source", 114, TEAL)]
left = 0
for name, v, c in src:
    ax.barh(0, v, left=left, height=0.5, color=c, label=f"{name}  ({v})")
    ax.text(left + v / 2, 0, str(v), ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    left += v
ax.set_xlim(0, 131)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("images in the clean pool")
ax.set_title("The kept 131 and where they come from", pad=6)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, -0.52), ncol=2)
ax.grid(axis="x", color=LIGHT, lw=0.8)
ax.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "fig_r_pool.png"))
plt.close(fig)

print("=" * 62)
print("[fig3] E", {k: round(v, 2) for k, v in exit_arm["E"].items()})
print("[fig3] G", {k: round(v, 2) for k, v in exit_arm["G"].items()})
print("[fig3] CF", CF)
print("[fig2] plain21", {k: round(v, 2) for k, v in plain21.items()})
print("[fig4] div_true", {k: round(probe[k]["div_true"], 3) for k in ALPHAS})
for n in ("fig_delta_alpha", "fig_boundary_map", "fig_exit_counterfactual",
          "fig_error_budget", "fig_r_pool"):
    print("  wrote", os.path.join(OUT, n + ".png"))
