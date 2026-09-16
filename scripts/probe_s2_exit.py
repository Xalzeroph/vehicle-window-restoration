#!/usr/bin/env python3
"""S2 Phase A 零训练侦察：物理出口的三分解归因 + α̃ 估计器敏感度（raw 口径）。

分解（用 plain 的 R̂ / T̂）：
  Δ1 换估计器：T0 = I−R̂（= R 支路输出） vs T̂_head
  Δ2 ÷真 α   ：T0/α_true vs T0
  Δ3 ÷估 α̃  ：T0/α̃ vs T0/α_true；α̃ ∈ {scene α_eff, 档位名义值 α_nominal, α_true·(1+ε)}
另报：除法输出 clip 占比（饱和）、T̂_head 的 gm 口径（对照）。

用法: python3 scripts/probe_s2_exit.py [--sets ...] [--level alpha_0.3,...]
产物: eval/probe_s2_exit_<set>.csv + 控制台表
"""
import argparse, csv, json, os
import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ["synth_rrw_v21_0913", "synth_rrw_v21_indomain", "synth_rrw_v22_scan"]
EPS = [-0.2, -0.1, 0.1, 0.2]
ALPHA_MIN = 0.05


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def gmatch(q, g):
    q = q.astype(np.float64); g = g.astype(np.float64)
    s = float((q * g).sum() / (q * q).sum().clip(1e-8))
    return np.clip(q * s, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--levels", default="")          # 逗号分隔，空=全部
    args = ap.parse_args()
    for s in args.sets.split(","):
        root = os.path.join(PROJ, "data/test_sets", s)
        meta = json.load(open(os.path.join(root, "meta.json")))
        aeff = {sc["id"]: sc["alpha_eff"] for sc in meta["scenes"]}
        scenes = sorted(os.listdir(os.path.join(root, "gt")))
        levels = sorted(os.listdir(os.path.join(root, "levels")),
                        key=lambda x: -float(x.split("_")[-1]))
        if args.levels:
            keep = set(args.levels.split(","))
            levels = [lv for lv in levels if lv in keep]
        rows = []
        for lv in levels:
            key = lv.split("_")[-1]
            nom = float(key)
            A = {k: [] for k in ["head", "head_gm", "T0", "div_true", "div_eff", "div_nom",
                                 "clip_true"] + [f"div_eps{e:+}" for e in EPS]}
            for sc in scenes:
                sid = os.path.splitext(sc)[0]
                gt = img(os.path.join(root, "gt", sc))
                I = img(os.path.join(root, "levels", lv, sc))
                Atrue = np.load(os.path.join(root, "alpha", f"{sid}_a{key}.npy")).astype(np.float32)
                if Atrue.ndim == 3:
                    Atrue = Atrue[:, :, 0]
                pred = os.path.join(PROJ, "runs/pred", f"rdt_{args.tag}", s, lv, sid)
                Rh = img(os.path.join(pred, "r_hat.png"))
                Th = img(os.path.join(pred, "t_hat.png"))
                T0 = I - Rh
                div_t = np.clip(T0 / np.maximum(Atrue, ALPHA_MIN)[..., None], 0, 255)
                div_e = np.clip(T0 / max(aeff[sid][key], ALPHA_MIN), 0, 255)
                div_n = np.clip(T0 / max(nom, ALPHA_MIN), 0, 255)
                A["head"].append(psnr(Th, gt))
                A["head_gm"].append(psnr(gmatch(Th, gt), gt))
                A["T0"].append(psnr(np.clip(T0, 0, 255), gt))
                A["div_true"].append(psnr(div_t, gt))
                A["div_eff"].append(psnr(div_e, gt))
                A["div_nom"].append(psnr(div_n, gt))
                A["clip_true"].append(float(((div_t >= 254.5).sum(axis=2) > 0).mean()))
                for e in EPS:
                    Ae = np.maximum(Atrue * (1 + e), ALPHA_MIN)
                    A[f"div_eps{e:+}"].append(psnr(np.clip(T0 / Ae[..., None], 0, 255), gt))
            r = dict(set=s, level=lv, n=len(scenes))
            r.update({k: round(float(np.mean(v)), 3) for k, v in A.items()})
            rows.append(r)
            print(f"{s:>22} {lv:>11} head {r['head']:6.2f} gm {r['head_gm']:6.2f} | T0 {r['T0']:6.2f} "
                  f"| ÷真α {r['div_true']:6.2f} ÷α_eff {r['div_eff']:6.2f} ÷名义 {r['div_nom']:6.2f} "
                  f"| ε±0.1 {r['div_eps-0.1']:6.2f}/{r['div_eps+0.1']:6.2f} "
                  f"| clip {r['clip_true']*100:5.1f}%")
        out = os.path.join(PROJ, f"eval/probe_s2_exit_{s}.csv")
        with open(out, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  -> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
