#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finish E2-dark14 seed runs: wait -> eval -> rebuild tables -> re-aggregate -> test.

Pre-registration: docs/2026.10.24__E2dark14.md
ASCII-only source on purpose (survives CRLF/console-codepage mangling).

Usage:
    python3 scripts/finish_e2_dark14.py --wait-h 6
    python3 scripts/finish_e2_dark14.py --wait-h 0 --skip-eval   # 
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PY = "/DataDisk/miniconda/envs/xreflection/bin/python3"
OUT = os.path.join(P, "runs/dom_v1")
PRED = os.path.join(P, "runs/pred")
T12 = "data/test_sets/real_pair_1019_testsplit12"
SEEDS = (202, 303)
ARM = "E2-dark14"


def sh(cmd, gpu=None, log=None):
    env = dict(os.environ)
    env.setdefault("TORCH_HOME", "/home/zmy/data/cache/torch")
    env.setdefault("HF_HOME", "/home/zmy/data/cache/huggingface")
    env.setdefault("MPLBACKEND", "Agg")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(cmd), flush=True)
    if log:
        with open(log, "w") as fh:
            return subprocess.run(cmd, cwd=P, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=P, env=env).returncode


def free_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return None, 0
    best, bestfree = None, -1
    for line in out.strip().split("\n"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3:
            continue
        i, tot, used = int(p[0]), int(p[1]), int(p[2])
        if i in (6, 7):          # Isaac Sim reserved
            continue
        if tot - used > bestfree:
            best, bestfree = i, tot - used
    return best, bestfree


def wait_ckpts(wait_h):
    todo = [s for s in SEEDS
            if not os.path.exists(os.path.join(OUT, "%s_s%d" % (ARM, s), "ckpt_final.pth"))]
    if not todo:
        print("[wait] all ckpts present")
        return True
    print("[wait] missing ckpt for seeds %s; waiting up to %.1f h" % (todo, wait_h))
    t0 = time.time()
    while time.time() - t0 < wait_h * 3600:
        left = [s for s in todo
                if not os.path.exists(os.path.join(OUT, "%s_s%d" % (ARM, s), "ckpt_final.pth"))]
        if not left:
            print("[wait] all ckpts present after %.0f min" % ((time.time() - t0) / 60))
            return True
        time.sleep(120)
    print("[wait] TIMEOUT; still missing %s" % left)
    return False


def ensure_predmap():
    for tag, pre in (("dom_v1", "dom_"), ("dom_v1_12", "dom12_")):
        fn = os.path.join(P, "eval/dom_v1/predmap_%s.json" % tag)
        m = json.load(open(fn, encoding="utf-8")) if os.path.exists(fn) else {}
        before = len(m)
        for s in (101,) + SEEDS:
            key = "%s%s_s%d" % (pre, ARM, s)
            if os.path.isdir(os.path.join(PRED, "real_pair_1019_" + key)):
                m.setdefault(key, "real_pair_1019_" + key)
        json.dump(m, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[predmap] %s: %d -> %d arms" % (tag, before, len(m)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-h", type=float, default=6.0)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--gpu", type=int, default=-1)
    args = ap.parse_args()

    if not args.skip_eval:
        if not wait_ckpts(args.wait_h):
            return 2
        gpu, free = free_gpu()
        if args.gpu >= 0:
            gpu = args.gpu
        print("[eval] using GPU%s (free %d MB)" % (gpu, free))
        for s in SEEDS:
            ck = os.path.join(OUT, "%s_s%d" % (ARM, s), "ckpt_final.pth")
            for pre, pairs in (("dom_", None), ("dom12_", T12)):
                name = "%s%s_s%d" % (pre, ARM, s)
                pdir = os.path.join(PRED, "real_pair_1019_" + name)
                if os.path.exists(os.path.join(pdir, "arm_info.json")):
                    print("[eval] skip %s (done)" % name)
                    continue
                cmd = [PY, "-u", "scripts/eval_arm_1019.py", "--ckpt", ck,
                       "--name", name, "--device", "cuda:0"]
                if pairs:
                    cmd += ["--pairs", pairs]
                rc = sh(cmd, gpu=gpu, log="/tmp/eval_%s.log" % name)
                print("[eval] %s rc=%d" % (name, rc))

    ensure_predmap()
    print("\n===== rebuild tables =====")
    sh([PY, "scripts/table_final_1019.py", "--pred-map",
        "eval/dom_v1/predmap_dom_v1.json", "--tag", "dom_v1"])
    sh([PY, "scripts/table_final_1019.py", "--pred-map",
        "eval/dom_v1/predmap_dom_v1_12.json", "--pairs", T12, "--tag", "dom_v1_12"])
    print("\n===== re-aggregate =====")
    sh([PY, "scripts/agg_dom_v1.py"])

    # ---------- pre-registered test P / Q ----------
    sys.path.insert(0, os.path.join(P, "scripts"))
    import agg_dom_v1 as A  # noqa: E402
    d12 = A.load("table_final_1019_dom_v1_12.csv")

    def seed_means(base):
        return [float(sum(float(v["psnr"]) for v in sc.values()) / len(sc))
                for sc in d12[base].values()]

    def mean(x):
        return sum(x) / len(x)

    def var(x):
        m = mean(x)
        return sum((v - m) ** 2 for v in x) / (len(x) - 1) if len(x) > 1 else 0.0

    print("\n===== pre-registered test =====")
    for base in (ARM, "R-apr"):
        if base not in d12:
            print("  MISSING arm %s" % base)
            return 3
        sm = seed_means(base)
        print("  %-10s seeds=%d  seed PSNR=%s  mean=%.3f  seed std=%.3f"
              % (base, len(sm), ["%.3f" % v for v in sm], mean(sm),
                 math.sqrt(var(sm))))
    a, b = seed_means(ARM), seed_means("R-apr")
    d = mean(a) - mean(b)
    na, nb = len(a), len(b)
    va, vb = var(a), var(b)
    se = math.sqrt(va / na + vb / nb)
    # Welch-Satterthwaite
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) \
        if (va + vb) > 0 else float(na + nb - 2)
    try:
        from scipy import stats
        tc = float(stats.t.ppf(0.975, df))
    except Exception:
        tc = 2.776
    lo, hi = d - tc * se, d + tc * se
    print("  P: delta = %+.3f dB   seed-level SE = %.3f   df = %.2f   t_crit = %.3f" % (d, se, df, tc))
    print("     95%% CI = [%+.3f, %+.3f]" % (lo, hi))
    print("     MDE (95%%) = %.3f dB" % (tc * se))
    verdict = "PASS (CI excludes 0)" if lo > 0 else ("FAIL (CI covers 0)" if hi > 0 else "NEGATIVE")
    print("     VERDICT P: %s" % verdict)
    print("     VERDICT Q (seed std <= 0.6): %s" % ("PASS" if math.sqrt(var(a)) <= 0.6 else "FAIL"))
    print("  note: dose ordering still NOT testable (8%%/11%% arms have 1 seed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
