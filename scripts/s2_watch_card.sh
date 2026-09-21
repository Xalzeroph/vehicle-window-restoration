#!/bin/bash
# S2 评测守护：监视一个/多个 ckpt 目录，发现新 ckpt_<iter>.pth 就自动跑评测卡，结果追加到日志。
# 用法: nohup bash scripts/s2_watch_card.sh "runs/ckpt/rdt_s2_main_s101 runs/ckpt/rdt_s2_main_s202" 5 > /tmp/s2_card_g5.log 2>&1 &
DIRS=$1; GPU=${2:-4}
cd "$(dirname "$0")/.."
declare -A SEEN
while true; do
    for DIR in $DIRS; do
        [ -d "$DIR" ] || continue
        TAGNAME=$(basename "$DIR")
        for ck in $(ls -1 "$DIR"/ckpt_[0-9]*.pth 2>/dev/null | sort -t_ -k2 -n); do
            key="$TAGNAME/$(basename "$ck")"
            [ -n "${SEEN[$key]}" ] && continue
            IT=$(basename "$ck" | sed 's/ckpt_\([0-9]*\).pth/\1/')
            echo "=== [$(date +%H:%M:%S)] $key ==="
            bash scripts/eval_s2_card.sh "$ck" "${TAGNAME}_$IT" "$GPU" 2>&1 | tail -5
            SEEN[$key]=1
        done
    done
    sleep 120
done
