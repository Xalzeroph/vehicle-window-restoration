"""训练监控器：按 loss 平台期早停 + 每个"epoch"评一次 + 保留最优检查点。

为什么需要它：
    aprjun_1028 暴露了一个真问题——迭代数固定为 6000 时，真实对从 29 增到 170，
    每对被看的次数从 372 掉到 64，模型处于"欠训练"，但训练曲线看起来早已平。
    固定迭代数无法区分"平了"和"数据还没喂够"，所以改为看**loss 平台期**。

停止判据（预登记，写在 configs/ 的 yaml 里）：
    window = 5 个记录点（log_every）
    连续 patience = 3 个窗口，total 的相对改善 < eps = 0.5%  ->  判为平台期，停
    另外硬上限 --iters，硬下限 --min-iters（防止早期假平台）

每个 epoch = --epoch-iters 次迭代；每次出现新的 ckpt 就调用 eval_progress.py 评一次，
曲线写 runs/<tag>/curve.csv；结束时把分数最高的 ckpt 复制成 ckpt_best.pth。

磁盘保护：ckpt 单个约 1.6 GB，只保留最新 2 个 + 最优 1 个。

用法：
    python train_monitor.py --tag aprjun_long --iters 30000 --min-iters 6000 \
        --epoch-iters 1000 --patience 3 --eps 0.005 -- <finetune_rdnet_s2.py 的其余参数>
"""
import argparse
import csv
import glob
import os
import shutil
import signal
import subprocess
import sys
import time

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PY = os.environ.get("TINTED_PYTHON", sys.executable)
CKPT_MB = 1600


def read_log(run_dir):
    p = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    return rows


def plateau(rows, window=5, patience=3, eps=0.005):
    """连续 patience 个窗口的相对改善 < eps 就返回 True。"""
    if len(rows) < window * (patience + 1):
        return False
    tot = [float(r["total"]) for r in rows]
    flags = []
    for k in range(patience):
        cur = tot[-(k + 1) * window: len(tot) - k * window] if k else tot[-window:]
        prev = tot[-(k + 2) * window:-(k + 1) * window]
        if not cur or not prev:
            return False
        m_cur, m_prev = sum(cur) / len(cur), sum(prev) / len(prev)
        flags.append((m_prev - m_cur) / max(abs(m_prev), 1e-9) < eps)
    return all(flags)


def prune(run_dir, keep=2):
    cks = sorted(glob.glob(os.path.join(run_dir, "ckpt_[0-9]*.pth")),
                 key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    for p in cks[:-keep]:
        try:
            os.remove(p)
        except OSError:
            pass
    return cks[-keep:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--min-iters", type=int, default=6000)
    ap.add_argument("--epoch-iters", type=int, default=1000)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    run_dir = os.path.join(P, "runs", a.tag)
    os.makedirs(run_dir, exist_ok=True)
    rest = [x for x in a.rest if x != "--"]
    if not rest:
        sys.exit("需要在 -- 之后给出 finetune_rdnet_s2.py 的参数")
    if "--out" not in rest:
        rest += ["--out", run_dir]
    if "--iters" not in rest:
        rest += ["--iters", str(a.iters)]
    if "--save-every" not in rest:
        rest += ["--save-every", str(a.epoch_iters)]
    cmd = [PY, "-u", os.path.join(P, "scripts/finetune_rdnet_s2.py")] + rest
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)

    print("[monitor] 预登记判据：window=%d patience=%d eps=%.3f min_iters=%d iters<=%d"
          % (a.window, a.patience, a.eps, a.min_iters, a.iters), flush=True)
    print("[monitor] 命令：" + " ".join(cmd), flush=True)
    if a.dry_run:
        return

    log = open(os.path.join(run_dir, "monitor.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, start_new_session=True)
    reason = "reached_iters"
    evaluated = set()
    try:
        while proc.poll() is None:
            time.sleep(a.poll)
            rows = read_log(run_dir)
            if not rows:
                continue
            it = int(float(rows[-1]["iter"]))
            # 每个 epoch 评一次（有新 ckpt 才评）
            for ck in glob.glob(os.path.join(run_dir, "ckpt_[0-9]*.pth")):
                k = int(os.path.basename(ck).split("_")[1].split(".")[0])
                if k % a.epoch_iters == 0 and k not in evaluated:
                    subprocess.run([PY, os.path.join(P, "scripts/eval_progress.py"),
                                    "--ckpts", ck, "--tag", a.tag, "--device", "cuda:0"],
                                   env=env, stdout=log, stderr=log)
                    evaluated.add(k)
                    prune(run_dir, keep=2)
            if it >= a.min_iters and plateau(rows, a.window, a.patience, a.eps):
                reason = f"plateau_at_{it}"
                print(f"[monitor] 平台期判定于 iter {it}，停止训练", flush=True)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                break
            if it >= a.iters:
                break
    finally:
        log.close()

    # 收尾：评 ckpt_latest，挑综合分最高者存 ckpt_best.pth
    cur = sorted(glob.glob(os.path.join(run_dir, "ckpt_[0-9]*.pth")),
                 key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if cur:
        subprocess.run([PY, os.path.join(P, "scripts/eval_progress.py"),
                        "--ckpts", ",".join(cur[-3:]), "--tag", a.tag + "_final",
                        "--device", "cuda:0"], env=env)
        r = os.path.join(P, "eval", f"progress_{a.tag}_final.csv")
        if os.path.exists(r):
            rows = list(csv.DictReader(open(r, encoding="utf-8")))
            best = max(rows, key=lambda x: float(x["score"]))
            src = [c for c in cur if os.path.basename(c) == best["ckpt"]]
            if src:
                shutil.copy(src[0], os.path.join(run_dir, "ckpt_best.pth"))
                print(f"[monitor] 最优检查点 {best['ckpt']} -> ckpt_best.pth", flush=True)
    with open(os.path.join(run_dir, "plateau_report.yaml"), "w", encoding="utf-8") as f:
        rows = read_log(run_dir)
        f.write(f"tag: {a.tag}\nstop_reason: {reason}\n"
                f"final_iter: {int(float(rows[-1]['iter'])) if rows else -1}\n"
                f"criterion: {{window: {a.window}, patience: {a.patience}, eps: {a.eps}, "
                f"min_iters: {a.min_iters}, max_iters: {a.iters}}}\n")
    print(f"[monitor] 结束：{reason}", flush=True)


if __name__ == "__main__":
    main()
