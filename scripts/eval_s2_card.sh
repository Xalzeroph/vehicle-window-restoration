#!/bin/bash
# S2 评测卡：给一个 ckpt 跑 may(dev) 全判据并打印 ①–④ 通过数（训练中每 1k iter 用）。
# 用法: bash scripts/eval_s2_card.sh runs/ckpt/rdt_s2_main_s101/ckpt_latest.pth 1000 0
set -e
CKPT=$1; TAG=$2; GPU=${3:-0}
cd "$(dirname "$0")/.."
PY="conda run -n xreflection python3"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY scripts/infer_rdt_s2.py \
    --ckpt "$CKPT" --sets may --tag $TAG --gpu 0
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY scripts/eval_g2_baseline.py \
    --sets may --extra-arm "s2=runs/pred/may_$TAG" --skip-official --out eval/eval_s2_card_$TAG.csv
$PY scripts/summarize_s2_dev.py "eval/eval_s2_card_$TAG.csv" "$TAG"
