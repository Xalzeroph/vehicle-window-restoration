#!/usr/bin/env python3
"""把合成数据/权重传到 HF（走 hf-mirror 镜像），可重入：中断后重跑自动续传。

实现：先在 /tmp/hf_stage/{data,model} 用**硬链接**拼出目标目录树（零磁盘开销），
再对整树调 upload_large_folder（它自带断点续传：已传的 sha 会跳过）。

用法:
  python3 share/upload_hf.py --stage data            # 数据（~18.6G / 5.7 万文件）
  python3 share/upload_hf.py --stage model           # 权重（~6G）
  python3 share/upload_hf.py --stage data --dry-run
"""
import argparse, os, shutil, sys, time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
_tok = os.path.expanduser("~/.cache/huggingface/token")
if "HF_TOKEN" not in os.environ and os.path.exists(_tok):
    os.environ["HF_TOKEN"] = open(_tok).read().strip()

from huggingface_hub import HfApi, create_repo, upload_large_folder  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = "/tmp/hf_stage"
DS_REPO = "zmy1234567890/tintedglass-rdt-synth"
MD_REPO = "zmy1234567890/tintedglass-rdnet-synth-ft"

DATA_JOBS = [  # (本地源, 目标树内相对目录)
    ("data/train_sets/train_triplets_rdt_0915", "train/rdt_0915"),
    ("data/test_sets/synth_rrw_v21_0913", "test/synth_v21_0913"),
    ("data/test_sets/synth_rrw_v21_indomain", "test/synth_v21_indomain"),
    ("data/test_sets/synth_rrw_v22_scan", "test/synth_v22_scan"),
    ("data/test_sets/synth_rrw_v23_lowscan", "test/synth_v23_lowscan"),
    ("data/test_sets/real_pair_0917", "test/real_pair_0917"),
    ("data/test_sets/real_pair_apr_0917", "test/real_pair_apr_0917"),
    ("data/test_sets/real_pair_may_0917", "test/real_pair_may_0917"),
    ("data/pools/r_train_0915", "pools"),
    ("data/pools/r_train_0915_meta.csv", "pools/r_train_0915_meta.csv"),
]
MODEL_JOBS = [
    ("runs/ckpt/rdt_plain", "rdt_plain"),
    ("runs/ckpt/rdt_freeze", "rdt_freeze"),
    ("runs/ckpt/rdt_plain_s101", "rdt_plain_s101"),
    ("runs/ckpt/rdt_freeze_s101", "rdt_freeze_s101"),
]


def build_stage(jobs, stage_dir, card):
    """硬链接拼目标树（同分区零开销）；返回 (文件数, 字节数)。"""
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    n_files = n_bytes = 0
    for src, dst in jobs:
        p = os.path.join(REPO, src)
        if not os.path.exists(p):
            print(f"  [跳过] 不存在: {src}")
            continue
        if os.path.isfile(p):
            os.makedirs(os.path.join(stage_dir, os.path.dirname(dst)), exist_ok=True)
            os.link(p, os.path.join(stage_dir, dst))
            n_files += 1; n_bytes += os.path.getsize(p)
            continue
        for root, _, files in os.walk(p):
            rel = os.path.relpath(root, p)
            out = os.path.join(stage_dir, dst) if rel == "." else os.path.join(stage_dir, dst, rel)
            os.makedirs(out, exist_ok=True)
            for f in files:
                s = os.path.join(root, f)
                os.link(s, os.path.join(out, f))
                n_files += 1; n_bytes += os.path.getsize(s)
    shutil.copy(card, os.path.join(stage_dir, "README.md"))
    n_files += 1; n_bytes += os.path.getsize(card)
    return n_files, n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "model"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    api = HfApi()
    print(f"[hf] 登录身份: {api.whoami()['name']}  endpoint={os.environ['HF_ENDPOINT']}")

    if args.stage == "data":
        repo, repo_type, jobs = DS_REPO, "dataset", DATA_JOBS
        card = os.path.join(HERE, "HF_DATASET_CARD.md")
    else:
        repo, repo_type, jobs = MD_REPO, "model", MODEL_JOBS
        card = os.path.join(HERE, "HF_MODEL_CARD.md")

    if args.dry_run:
        for src, dst in jobs:
            p = os.path.join(REPO, src)
            n = sum(len(fs) for _, _, fs in os.walk(p)) if os.path.isdir(p) else 1
            print(f"  {src}  ->  {dst}   ({n} 文件){'' if os.path.exists(p) else '  [缺失!]'}")
        return

    t0 = time.time()
    stage_dir = os.path.join(STAGE, args.stage)
    n, b = build_stage(jobs, stage_dir, card)
    print(f"[hf] 暂存树 {stage_dir}: {n} 文件 / {b/2**30:.2f} GiB（{time.time()-t0:.0f}s）")

    create_repo(repo, repo_type=repo_type, private=False, exist_ok=True)
    print(f"[hf] 开始上传 -> {repo_type} {repo}  (workers={args.workers})", flush=True)
    upload_large_folder(repo_id=repo, repo_type=repo_type, folder_path=stage_dir,
                        num_workers=args.workers)
    print(f"[hf] 完成 -> https://huggingface.co/{repo_type}s/{repo}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
