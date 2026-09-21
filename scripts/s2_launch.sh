#!/bin/bash
# S2 六卡启动（GATE-2 已确认 2026-09-20）：主臂×3 seed + B1 对照 + 评测守护×2
# 用法: bash scripts/s2_launch.sh
cd "$(dirname "$0")/.."
PY="conda run -n xreflection python3"
: > /tmp/s2_pids.txt

launch() {  # name gpu cmd...
    local name=$1 gpu=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 setsid nohup $PY "$@" \
        > /tmp/s2_$name.log 2>&1 &
    echo "$! $name gpu$gpu" | tee -a /tmp/s2_pids.txt
}

launch main_s101 1 scripts/finetune_rdnet_s2.py --arm main --seed 101 --gpu 0 --iters 10000 --out runs/ckpt/rdt_s2_main_s101
launch main_s202 2 scripts/finetune_rdnet_s2.py --arm main --seed 202 --gpu 0 --iters 10000 --out runs/ckpt/rdt_s2_main_s202
launch main_s303 3 scripts/finetune_rdnet_s2.py --arm main --seed 303 --gpu 0 --iters 10000 --out runs/ckpt/rdt_s2_main_s303
launch b1_s101   4 scripts/finetune_rdnet_s2.py --arm b1  --seed 101 --gpu 0 --iters 6000  --out runs/ckpt/rdt_s2_b1_s101

watch() {   # gpu dirs logfile
    CUDA_VISIBLE_DEVICES=$3 OMP_NUM_THREADS=2 setsid nohup bash scripts/s2_watch_card.sh "$2" $3 \
        > $4 2>&1 &
    echo "$! watch_g$3 gpu$3" | tee -a /tmp/s2_pids.txt
}
watch g5 "runs/ckpt/rdt_s2_main_s101 runs/ckpt/rdt_s2_main_s202 runs/ckpt/rdt_s2_main_s303" 5 /tmp/s2_card_g5.log
watch g6 "runs/ckpt/rdt_s2_b1_s101" 6 /tmp/s2_card_g6.log
echo "已启动（PID 见 /tmp/s2_pids.txt；训练日志 /tmp/s2_*.log；评测日志 /tmp/s2_card_g*.log）"
