#!/usr/bin/env bash
# 等待某模型权重下完 → 自动跑 VLM 探针
# 用法: bash scripts/run_vlm_when_ready.sh <model_tag> <repo_dir_name>
set -u
TAG=$1; REPO=$2
ROOT=/home/zmy/data/cache/huggingface/hub/models--${REPO}
export HF_HOME=/home/zmy/data/cache/huggingface HF_HUB_OFFLINE=1
for i in $(seq 1 180); do      # 最多等 3 小时
  INC=$(find "$ROOT/blobs" -name '*.incomplete' 2>/dev/null | wc -l)
  W=$(ls "$ROOT"/snapshots/*/model*.safetensors "$ROOT"/snapshots/*/model*.bin 2>/dev/null | wc -l)
  if [ "$INC" = "0" ] && [ "$W" -ge 1 ]; then
    echo "[ready] $TAG  权重 $W 个文件，无 .incomplete，开始探针 $(date)" 
    CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=6 \
      /DataDisk/miniconda/envs/xreflection/bin/python3 scripts/probe_vlm_hole.py \
      --model "$TAG" --out "eval/vlm_probe_${TAG}.csv" 2>&1 | grep -vE 'Loading weights|it/s\]'
    echo "[done] $TAG $(date)"
    exit 0
  fi
  sleep 60
done
echo "[timeout] $TAG 3 小时未就绪"
