#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常驻实验队列 autoqueue（2026-10-23）。

它做什么
-------
把「占住卡」变成「卡一直在跑真实验」：从 backlog 里取任务，谁空闲显存够就领谁，
跑完接着下一条；backlog 空了就自己退出（不空转）。

与「占卡脚本」的区别：跑的是真训练/真分析，日志与产物齐全，名字不需要伪装。

安全约束（按用户要求）
--------------------
1. **保留卡**：`--reserve-idx`（默认 6,7）永不使用 —— 留给 Isaac Sim
2. **自动避让**：任何显卡上若有非本用户进程占用 ≥1 GB，则该卡本任务不用
3. **并发上限**：`--max-concurrent`（默认 4），符合项目守则「他人 <3 卡时用 4 卡」
4. **准入阈值**：需要 ≥`--need-gb`（默认 7.5）空闲才上
5. **优雅停机**：出现 STOP 文件或收到 SIGTERM → 不再领新任务，等当前任务收尾
6. **OOM 自愈**：rc!=0 且日志含 out of memory → 换卡重试（最多 3 次）

日志与状态
----------
runs/queue/<task>/<run>.log    每次运行的 stdout
runs/queue/state.json          每个任务的状态（pending/running/ok/fail/skip）
eval/queue/backlog.json        任务清单（本脚本只读）

用法
----
  python3 scripts/autoqueue.py --dry-run
  python3 scripts/autoqueue.py --include-dom --max-concurrent 4
  touch eval/queue/STOP          # 优雅停机
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PY = os.environ.get("TINTED_PYTHON", sys.executable)
QROOT = os.path.join(P, "runs/queue")
STATE = os.path.join(QROOT, "state.json")
QD = os.path.join(P, "eval/queue")
BACKLOG = os.path.join(QD, "backlog.json")
STOP = os.path.join(QD, "STOP")

_lock = threading.Lock()
_busy = {}           # gpu -> task name
_stop = threading.Event()
_owner = None


def owner_user():
    global _owner
    if _owner is None:
        try:
            _owner = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
        except Exception:
            _owner = "zmy"
    return _owner


def gpu_snapshot():
    """[{idx, total, used, ours, others}]；失败返回 []。"""
    try:
        g = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        u = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f"！nvidia-smi 失败: {e}", file=sys.stderr)
        return []
    idx2uuid = {}
    for line in u.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            idx2uuid[parts[1]] = int(parts[0])
    uuid2others = {}
    me = owner_user()
    for line in a.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid, mem = parts[0], parts[1], int(parts[2])
        try:
            own = subprocess.run(["ps", "-o", "user=", "-p", pid],
                                 capture_output=True, text=True).stdout.strip()
        except Exception:
            own = ""
        if own and own != me:
            uuid2others[uuid] = uuid2others.get(uuid, 0) + mem
    out = []
    for line in g.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        idx, tot, used = int(parts[0]), int(parts[1]), int(parts[2])
        uuid = next((k for k, v in idx2uuid.items() if v == idx), None)
        out.append(dict(idx=idx, total=tot, used=used, free=tot - used,
                        others=uuid2others.get(uuid, 0) if uuid else 0))
    return out


def pick_gpu(need_gb, reserve):
    snap = gpu_snapshot()
    with _lock:
        cand = [g for g in snap
                if g["idx"] not in reserve
                and g["idx"] not in _busy
                and g["free"] >= need_gb * 1024
                and g["others"] < 1024]        # 别人在用就不碰
        if not cand:
            return None, snap
        cand.sort(key=lambda x: -x["free"])
        g = cand[0]
        return g["idx"], snap


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(st):
    with _lock:
        os.makedirs(QROOT, exist_ok=True)
        json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def dom_tasks():
    """把 run_dom_v1 里还缺 ckpt 的域消融任务转成队列任务。

    注意：**跳过正在跑的**（train.log 在 20 min 内被改过），否则会与 run_dom_v1 重复训练。
    """
    sys.path.insert(0, os.path.join(P, "scripts"))
    try:
        from run_dom_v1 import ARMS, FID, COMMON
    except Exception as e:
        print(f"！导入 run_dom_v1 失败: {e}")
        return []
    out = []
    now = time.time()
    for name, (wdir, doms, _s1, s2) in ARMS.items():
        for sd in s2:
            d = os.path.join(P, "runs/dom_v1", f"{name}_s{sd}")
            if os.path.exists(os.path.join(d, "ckpt_final.pth")):
                continue
            lg = os.path.join(d, "train.log")
            if os.path.exists(lg) and (now - os.path.getmtime(lg)) < 1200:
                print(f"  [skip] {name}_s{sd} 正在跑（log 新鲜），交给 run_dom_v1")
                continue
            cmd = [PY, "-u", os.path.join(P, "scripts/finetune_rdnet_s2.py")] + COMMON + [
                "--seed", str(sd), "--gpu", "0", "--omega-dir", wdir,
                "--assert-under", wdir, "--manifest", os.path.join(d, "manifest.json"),
                "--out", d] + FID
            if doms and doms != "-":
                cmd += ["--omega-domains", doms]
            out.append(dict(name=f"dom_{name}_s{sd}", kind="train", cmd=cmd,
                            need_gb=7.5, est_h=3.2,
                            done_marker=os.path.join(d, "ckpt_final.pth")))
    return out


def load_backlog(include_dom):
    tasks = []
    if os.path.exists(BACKLOG):
        # utf-8-sig：PowerShell 生成的 JSON 常带 BOM
        tasks += [t for t in json.load(open(BACKLOG, encoding="utf-8-sig"))
                  if not t.get("_disabled")]
    if include_dom:
        tasks += dom_tasks()
    return tasks


def run_shell(task, st):
    name = task["name"]
    d = os.path.join(QROOT, name)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "run.log")
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(task["cmd"], cwd=P, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    st[name] = dict(kind="shell", rc=rc, gpu=None, wall_sec=int(time.time() - t0),
                    log=log)
    save_state(st)
    print(f"[shell] {name} rc={rc} wall={(time.time()-t0)/60:.1f}min", flush=True)


def run_train(task, args, st):
    """跑一个训练任务。

    2026-10-24 修：早期版本把「没有可用卡」也算作一次尝试，3 次轮询（3 min）后
    就把任务静默丢掉 —— 曾因此在 25 min 内丢弃全部 8 个任务然后退出。
    现在 **等卡不计入尝试**，只有真正跑起来并 OOM 才算一次失败尝试。
    """
    name = task["name"]
    d = os.path.join(QROOT, name)
    os.makedirs(d, exist_ok=True)
    attempt = 0
    while attempt < 3 and not _stop.is_set():
        gpu, snap = pick_gpu(task.get("need_gb", 7.5), args.reserve)
        if gpu is None:
            free_desc = " ".join(f"G{g['idx']}:{g['free']//1024}G(他人{g['others']//1024}G)"
                                 for g in snap)
            print(f"[wait] {name} 无可用卡（保留 {sorted(args.reserve)}）: {free_desc}",
                  flush=True)
            # 等卡：不计入 attempt；被 STOP 打断则退出
            for _ in range(max(1, args.poll // 5)):
                if _stop.is_set():
                    st[name] = dict(kind="train", state="aborted_by_stop")
                    save_state(st)
                    return
                time.sleep(5)
            continue
        attempt += 1
        with _lock:
            _busy[gpu] = name
        try:
            log = os.path.join(d, f"attempt{attempt}.log")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("TORCH_HOME", "/home/zmy/data/cache/torch")
            env.setdefault("HF_HOME", "/home/zmy/data/cache/huggingface")
            env["OMP_NUM_THREADS"] = "8"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            t0 = time.time()
            print(f"[run] {name} -> GPU{gpu} (attempt {attempt})", flush=True)
            with open(log, "w") as fh:
                rc = subprocess.run(task["cmd"], cwd=P, env=env, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            wall = int(time.time() - t0)
            oom = False
            try:
                txt = open(log, errors="replace").read()[-4000:].lower()
                oom = "out of memory" in txt or "outofmemoryerror" in txt
            except Exception:
                pass
            st[name] = dict(kind="train", rc=rc, gpu=gpu, attempt=attempt,
                            wall_sec=wall, log=log)
            save_state(st)
            print(f"[train] {name} gpu{gpu} rc={rc} wall={wall/60:.1f}min", flush=True)
            if rc == 0 or not oom:
                return
            print(f"[retry] {name} OOM，换卡重试（{attempt}/3）", flush=True)
            time.sleep(20)
        finally:
            with _lock:
                _busy.pop(gpu, None)
    st[name] = dict(kind="train", state="gave_up_after_retries")
    save_state(st)


def worker(tasks, args, st):
    while not _stop.is_set():
        with _lock:
            pend = [t for t in tasks
                    if t["name"] not in st
                    and not (t.get("done_marker") and os.path.exists(t["done_marker"]))]
            if not pend:
                return
            task = pend[0]
            st[task["name"]] = dict(kind=task.get("kind", "train"),
                                    state="running", started=time.strftime("%F %T"))
            tasks.remove(task)
        try:
            if task.get("kind") == "shell":
                run_shell(task, st)
            else:
                run_train(task, args, st)
        except Exception as e:
            print(f"[err] {task['name']}: {e}", flush=True)
            with _lock:
                st[task["name"]] = dict(kind=task.get("kind"), state="error", err=str(e))
            save_state(st)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve-idx", default="6,7",
                    help="永不动用的卡号（默认 6,7，留给 Isaac Sim）")
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--need-gb", type=float, default=7.5)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--include-dom", action="store_true",
                    help="把 run_dom_v1 里还缺 ckpt 的域消融任务也排进来")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.reserve = {int(x) for x in args.reserve_idx.split(",") if x.strip()}

    os.makedirs(QROOT, exist_ok=True)
    os.makedirs(QD, exist_ok=True)
    if os.path.exists(STOP):
        print(f"！发现 {STOP}，删掉它再启动（上次是优雅停机）")

    tasks = load_backlog(args.include_dom)
    st = load_state()
    # 清掉上一轮的 running 状态（进程已不在）
    for k, v in list(st.items()):
        if isinstance(v, dict) and v.get("state") == "running":
            st[k] = dict(v, state="interrupted")
    save_state(st)

    print(f"=== autoqueue ===")
    print(f"  保留卡（不动）: {sorted(args.reserve)}")
    print(f"  并发上限: {args.max_concurrent}   准入空闲: {args.need_gb} GB")
    print(f"  任务总数: {len(tasks)}（已完成会被跳过）")
    for t in tasks[:20]:
        mark = "✓" if (t.get("done_marker") and os.path.exists(t["done_marker"])) else "·"
        print(f"    {mark} [{t.get('kind','train'):<5}] {t['name']}")
    if len(tasks) > 20:
        print(f"    … 另 {len(tasks)-20} 条")
    pend = [t for t in tasks
            if not (t.get("done_marker") and os.path.exists(t["done_marker"]))
            and t["name"] not in st]
    print(f"  待跑: {len(pend)}")
    if args.dry_run:
        snap = gpu_snapshot()
        print("\n  当前显存:")
        for g in snap:
            flag = "  <= 保留" if g["idx"] in args.reserve else (
                "  <= 他人占用" if g["others"] >= 1024 else "")
            print(f"    GPU{g['idx']} 空闲 {g['free']:>5} MB  他人 {g['others']:>5} MB{flag}")
        return 0
    if not pend:
        print("没有待跑任务，退出")
        return 0

    def on_sig(signum, frame):
        print(f"\n[stop] 收到信号 {signum}，不再领新任务，等当前任务收尾", flush=True)
        _stop.set()

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    ths = []
    for i in range(args.max_concurrent):
        th = threading.Thread(target=worker, args=(tasks, args, st), daemon=False)
        th.start()
        ths.append(th)
        time.sleep(4)
        if _stop.is_set():
            break
    # 主循环同时监视 STOP 文件
    while any(t.is_alive() for t in ths):
        if os.path.exists(STOP):
            print(f"[stop] 发现 STOP 文件，优雅停机", flush=True)
            _stop.set()
        time.sleep(10)
    for t in ths:
        t.join()

    ok = sum(1 for v in st.values() if isinstance(v, dict) and v.get("rc") == 0)
    bad = [k for k, v in st.items() if isinstance(v, dict) and v.get("rc") not in (0, None)]
    print(f"\n=== 结束 ===  成功 {ok}，失败 {len(bad)}")
    for k in bad:
        print(f"  ✗ {k}: {st[k]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
