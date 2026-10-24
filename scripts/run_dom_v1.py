#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""域消融 v1 运行编排（2026-10-22）—— 自适应选卡版。

为什么要自适应
-------------
本机同时有 `aj`（6 个 gmpl.train）、root（2 个 isaaclab）等任务在抢显存。
第一版「启动时固定卡号 + 每卡串行队列」两次都撞上 OOM：检查时空闲 9.2 GB，
真正启动时那块卡被别的进程吃掉 7 GB。所以改为**每个任务启动前重新挑卡**。

机制
----
· N 个 worker 线程共享一个任务队列；每个 worker：取任务 → 挑一张空闲显存 ≥ `--need-free` 的卡
  → 跑 → 释放该卡（用集合记录本进程占用的卡，避免自己撞自己）
· 没有符合条件的卡时轮询等待（默认最多 6 h）
· 固定损失 `--lam-grad 5 --lam-lap 2 --legacy-fidelity`，6000 iter / bs 6 / lr 1e-4，采样比 30:40:30
· **唯一自变量 = `--omega-domains`（或 omega_dir）**

用法
----
  python3 scripts/run_dom_v1.py --stage 1 --dry-run
  python3 scripts/run_dom_v1.py --stage 1 --workers 3
  python3 scripts/run_dom_v1.py --stage 2
  python3 scripts/run_dom_v1.py --stage 1 --only R-jun,R-in19
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PY = "/DataDisk/miniconda/envs/xreflection/bin/python3"
OUT = os.path.join(P, "runs/dom_v1")
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")
IN19 = os.path.join(P, "data/train_sets/real_pairs_1019split")
MAY14 = os.path.join(P, "data/train_sets/pairs_may")
APR16 = os.path.join(P, "data/train_sets/train_pairs_omega_0920")
APR_DARK14 = os.path.join(P, "data/train_sets/real_pairs_1020_apr_dark14")

FID = ["--lam-grad", "5", "--lam-lap", "2", "--legacy-fidelity"]
COMMON = ["--arm", "main", "--iters", "6000", "--batch", "6",
          "--save-every", "6000", "--log-every", "200"]

# 臂 -> (omega_dir, domains, seeds_stage1, seeds_stage2)
# 顺序即队列优先级。doms == "-" 表示不加 --omega-domains 过滤（旧目录没有 domain 字段）
ARMS = {
    "R-mayjun":  (POOL, "may,jun",    [101], [202, 303]),
    "R-may":     (POOL, "may",        [101], [202, 303]),
    "R-in19":    (IN19, "1019split",  [101], [202, 303]),
    "R-jun":     (POOL, "jun",        [101], [202, 303]),
    "R-all":     (POOL, "apr,may,jun", [101], [202, 303]),
    "R-aprmay":  (POOL, "apr,may",    [101], [202, 303]),
    "R-aprjun":  (POOL, "apr,jun",    [101], [202, 303]),
    # 选取规则对照臂：只补种子，阶段 1 已在上一轮跑过
    "R-apr16":   (APR16, "-",         [],    [202, 303]),
    # E2 暗化剂量 14.3%（2026-10-24 补种子）：s101 已跑，只补 202/303。
    # 见 docs/2026.10.24_预登记_E2dark14补种子.md（判据跑前写死）
    "E2-dark14": (APR_DARK14, "apr_dark14", [], [202, 303]),
}

_lock = threading.Lock()
_busy = {}          # gpu -> arm tag（本进程正在用）
_stop = threading.Event()


def gpu_free_map():
    """{gpu_index: free_mb}；查询失败返回 {}。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f"！nvidia-smi 失败: {e}", file=sys.stderr)
        return {}
    res = {}
    for line in out.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            res[int(parts[0])] = int(parts[1]) - int(parts[2])
        except ValueError:
            continue
    return res


def acquire_gpu(need_mb, max_wait_s, tag):
    """等一张空闲 ≥ need_mb 且本进程未占用的卡；返回 gpu 号，超时返回 None。"""
    t0 = time.time()
    last_note = 0.0
    while not _stop.is_set():
        fm = gpu_free_map()
        with _lock:
            cand = [(i, f) for i, f in fm.items() if f >= need_mb and i not in _busy]
            if cand:
                cand.sort(key=lambda x: -x[1])
                gpu = cand[0][0]
                _busy[gpu] = tag
                print(f"[pick] {tag} -> GPU{gpu}（空闲 {cand[0][1]} MB）", flush=True)
                return gpu
        el = time.time() - t0
        if el > max_wait_s:
            print(f"[pick] {tag} 等待超时 {el/3600:.1f}h，放弃", flush=True)
            return None
        if el - last_note > 300:
            last_note = el
            free_desc = " ".join(f"G{i}:{f//1024}G" for i, f in sorted(fm.items()))
            print(f"[pick] {tag} 等待中 {el/60:.0f}min  空闲: {free_desc}", flush=True)
        time.sleep(45)
    return None


def release_gpu(gpu, tag):
    with _lock:
        if _busy.get(gpu) == tag:
            _busy.pop(gpu, None)


def run_task(task, need_mb, max_wait_s, max_attempts=4):
    """跑一个任务；若因 OOM 失败则换卡重试。

    2026-10-23 加：显存"检查→启动"之间存在竞态 —— 检查时空闲 ≥7 GB，真分配时
    已被别人拿走（本轮 8 个任务里 3 个如此，全部 OOM 在只剩 23–55 MB 的卡上）。
    所以 rc!=0 且日志含 out of memory 时，释放该卡并重新挑卡重试。
    """
    name, seed, wdir, doms = task
    tag = f"{name}_s{seed}"
    for attempt in range(1, max_attempts + 1):
        gpu = acquire_gpu(need_mb, max_wait_s, tag)
        if gpu is None:
            return
        try:
            out_dir = os.path.join(OUT, tag)
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, "train.log")
            cmd = [PY, "-u", os.path.join(P, "scripts/finetune_rdnet_s2.py")] + COMMON + [
                "--seed", str(seed), "--gpu", "0",
                "--omega-dir", wdir,
                "--assert-under", wdir,
                "--manifest", os.path.join(out_dir, "manifest.json"),
                "--out", out_dir] + FID
            if doms and doms != "-":
                cmd += ["--omega-domains", doms]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("TORCH_HOME", "/home/zmy/data/cache/torch")
            env.setdefault("HF_HOME", "/home/zmy/data/cache/huggingface")
            env["OMP_NUM_THREADS"] = "8"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            t0 = time.time()
            with open(log_path, "w") as fh:
                rc = subprocess.run(cmd, cwd=P, env=env, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            wall = time.time() - t0
            oom = False
            try:
                txt = open(log_path, errors="replace").read()[-4000:]
                oom = ("out of memory" in txt.lower()) or ("OutOfMemoryError" in txt)
            except Exception:
                pass
            if rc != 0 and oom and attempt < max_attempts:
                print(f"[retry] {tag} gpu{gpu} OOM（第 {attempt} 次），换卡重试", flush=True)
                time.sleep(20)
                continue
            rec = dict(arm=name, seed=seed, gpu=gpu, out=out_dir, rc=rc,
                       attempt=attempt, start=int(t0), end=int(time.time()),
                       wall_sec=int(wall), cmd=" ".join(cmd[1:]))
            with _lock:
                with open(os.path.join(OUT, "launch_log.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[done] {tag} gpu{gpu} rc={rc} attempt={attempt} "
                  f"wall={wall/60:.0f}min", flush=True)
            return
        finally:
            release_gpu(gpu, tag)


def worker(tasks, need_mb, max_wait_s):
    while not _stop.is_set():
        with _lock:
            if not tasks:
                return
            task = tasks.pop(0)
        try:
            run_task(task, need_mb, max_wait_s)
        except Exception as e:
            print(f"[err] {task[0]}_s{task[1]}: {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--need-free", type=int, default=7000, help="需要的最小空闲显存(MB)")
    ap.add_argument("--max-wait-h", type=float, default=6.0)
    ap.add_argument("--skip-done", action="store_true",
                    help="跳过已有 ckpt_final.pth 的任务（重试/续跑时用，避免重复训练）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    tasks = []
    for name, (wdir, doms, s1, s2) in ARMS.items():
        if only and name not in only:
            continue
        for sd in (s1 if args.stage == 1 else s2):
            if args.skip_done and os.path.exists(
                    os.path.join(OUT, f"{name}_s{sd}", "ckpt_final.pth")):
                continue
            tasks.append((name, sd, wdir, doms))
    if not tasks:
        print("没有任务（可能全部已完成）")
        return 0

    print(f"阶段 {args.stage}：{len(tasks)} 个任务，{args.workers} 个 worker，"
          f"每任务需 ≥{args.need_free} MB 空闲显存")
    for t in tasks:
        print(f"  {t[0]:<10} s{t[1]}  domains={t[3]}")
    if args.dry_run:
        return 0

    fm = gpu_free_map()
    print("当前空闲: " + "  ".join(f"G{i}:{f}MB" for i, f in sorted(fm.items())), flush=True)

    ths = [threading.Thread(target=worker,
                            args=(tasks, args.need_free, args.max_wait_h * 3600),
                            daemon=False) for _ in range(args.workers)]
    for t in ths:
        t.start()
        time.sleep(3)
    for t in ths:
        t.join()

    rows = []
    lp = os.path.join(OUT, "launch_log.jsonl")
    if os.path.exists(lp):
        rows = [json.loads(l) for l in open(lp, encoding="utf-8") if l.strip()]
    bad = [r for r in rows if r["rc"] != 0]
    print("\n=== 汇总 ===")
    for r in rows:
        print(f"  {r['arm']:<10} s{r['seed']} gpu{r['gpu']} rc={r['rc']} "
              f"wall={r['wall_sec']//60}min")
    print(f"\n共 {len(rows)} 条，rc!=0 的 {len(bad)} 条")
    return 0 if rows and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
