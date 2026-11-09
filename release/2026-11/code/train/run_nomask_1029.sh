#!/bin/bash
# aprjun_nomask / mayjun_nomask：关掉 Ω 区域损失，检验"Ω 监督在大 Ω 池上是否有害"
# 与 aprjun_1028 / mayjun15 完全同配方，唯一差别 --lam-om 0
set -u
P=/DataDisk/zmy_code/TintedGlass_Enhance
PY=/DataDisk/miniconda/envs/xreflection/bin/python3
OUT=$P/runs/nomask_1029
LOG=/tmp/nomask
cd "$P" || exit 1
mkdir -p "$OUT" "$LOG"
exec > >(tee -a "$LOG/run.out") 2>&1

COMMON="--arm main --iters 6000 --batch 6 --lr 1e-4 --mona both \
 --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 \
 --lam-grad 5.0 --lam-lap 2.0 --s-jitter 0.0 --save-every 2000 --log-every 200 \
 --seg-iters 2000 --omega-mode real --lam-om 0.0"

free_card () {
  for i in 0 1 2 4 5 6 7; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null | tr -dc '0-9')
    [ -n "$mem" ] && [ "$mem" -lt 2500 ] && { echo "$i"; return; }
  done
  echo ""
}

launch () {
  local tag=$1 seed=$2 pool=$3 domains=$4
  local g; g=$(free_card)
  if [ -z "$g" ]; then echo "[$(date '+%T')] 无空闲卡，等待"; sleep 60; launch "$tag" "$seed" "$pool" "$domains"; return; fi
  CUDA_VISIBLE_DEVICES=$g nohup setsid $PY -u scripts/finetune_rdnet_s2.py $COMMON \
    --seed "$seed" --omega-dir "$pool" --omega-domains "$domains" --assert-under "$pool" \
    --out "$OUT/$tag" --manifest "$OUT/$tag/manifest.json" \
    > "$LOG/$tag.log" 2>&1 < /dev/null &
  echo "[$(date '+%T')] launched $tag seed=$seed domains=$domains GPU=$g lam_om=0"
}

launch aprjun_nomask 101 "$P/data/train_sets/real_pairs_1020" "apr,jun"
sleep 25
launch mayjun_nomask 101 "$P/data/train_sets/real_pairs_1020" "may,jun"
sleep 90

echo "=== 自检 ==="
for t in aprjun_nomask mayjun_nomask; do
  echo "--- $t"
  grep -E "domain 过滤|R 白名单|lam_om|lam-om" "$LOG/$t.log" 2>/dev/null | head -3
  grep -o "it *[0-9]*" "$LOG/$t.log" 2>/dev/null | tail -1
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
