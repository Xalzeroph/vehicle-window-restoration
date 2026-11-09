#!/usr/bin/env bash
# 卡片守护队列：有空闲显存就填我们的训练任务；防崩溃、防 OOM（自动降 batch 重排）
# 状态：/tmp/queue/state.tsv（name<TAB>gpu<TAB>pid<TAB>batch<TAB>outdir）
# 日志：/tmp/queue/q.log
set -u
P=/DataDisk/zmy_code/TintedGlass_Enhance
PY=/DataDisk/miniconda/envs/xreflection/bin/python3
Q=/tmp/queue; mkdir -p $Q; ST=$Q/state.tsv; touch $ST
LOG=$Q/q.log
NEED=6200          # 需要的最小空闲显存 (MiB)
MAXJOB=$(cat $Q/maxjob 2>/dev/null || echo 2)   # 并发臂上限（改 /tmp/queue/maxjob 即时生效）
cd $P || exit 1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }

# ---- 任务表： name|batch|额外参数 -------------------------------------------------
JOBS=(
 "ABL_noLr|4|--omega-domains may,jun --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 --lam-grad 5.0 --lam-lap 2.0 --lam-om 3.0 --lam-r 0"
 "ABL_noRecons|4|--omega-domains may,jun --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 --lam-grad 5.0 --lam-lap 2.0 --lam-om 3.0 --lam-recons 0"
 "ABL_noExcl|4|--omega-domains may,jun --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 --lam-grad 5.0 --lam-lap 2.0 --lam-om 3.0 --lam-excl 0"
 "ABL_noDists|4|--omega-domains may,jun --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 --lam-grad 5.0 --lam-lap 2.0 --lam-om 3.0 --lam-dists 0"
 "ABL_noLap|4|--omega-domains may,jun --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 --lam-grad 5.0 --lam-lap 2.0 --lam-om 3.0 --lam-lap 0"
 "ABL_grad2|4|--omega-domains may,jun --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 --lam-grad 5.0 --lam-lap 2.0 --lam-om 3.0 --lam-grad 2.0"
 "ABL_noHinge|4|--omega-domains may,jun --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 --lam-grad 5.0 --lam-lap 2.0 --lam-om 3.0 --lam-hinge 0"
)

running(){ cut -f1 $ST | grep -qx "$1"; }
launch(){  # name gpu batch extra
  local n=$1 g=$2 b=$3 extra=$4 seed=101
  case "$n" in *s202) seed=202;; *s303) seed=303;; esac
  local out=runs/queue1029/$n
  [ -f $out/ckpt_final.pth ] && { say "skip $n (已完成)"; return 1; }
  # 若同名任务已在别处跑过（arch1029），软链过来避免重复训练
  if [ -d runs/arch1029/$n ] && [ ! -e $out ]; then ln -sfn $P/runs/arch1029/$n $out; fi
  [ -f $out/ckpt_final.pth ] && { say "skip $n (已在 arch1029 完成)"; return 1; }
  CUDA_VISIBLE_DEVICES=$g setsid nohup $PY -u scripts/finetune_rdnet_s2.py \
    --arm main --iters 6000 --mona both --seg-iters 2000 --save-every 6000 --log-every 200 \
    --s-jitter 0.0 --lr 1e-4 --seed $seed --batch $b \
    --omega-dir $P/data/train_sets/real_pairs_1020 $extra \
    --out $out > /tmp/queue/$n.log 2>&1 &
  local pid=$!
  echo -e "$n\t$g\t$pid\t$b\t$out" >> $ST
  say "start $n gpu=$g pid=$pid bs=$b"
  return 0
}

say "=== 守护启动 NEED=$NEED MiB ==="
while :; do
  # 1) 清理已结束/失败的
  tmp=$Q/state.new; : > $tmp
  while IFS=$'\t' read -r n g pid b out; do
    [ -z "${n:-}" ] && continue
    if kill -0 "$pid" 2>/dev/null; then echo -e "$n\t$g\t$pid\t$b\t$out" >> $tmp; continue; fi
    if [ -f "$out/ckpt_final.pth" ]; then say "done $n (bs=$b)"; continue; fi
    if grep -qi "out of memory" /tmp/queue/$n.log 2>/dev/null; then
      nb=$(( b / 2 )); [ $nb -lt 2 ] && nb=2
      say "OOM $n → 重排 bs=$nb"; echo -e "$n\t-\t-\t$nb\t$out" >> $Q/retry.tsv
    else
      say "failed $n（非 OOM，见 /tmp/queue/$n.log）"
    fi
  done < $ST
  mv $tmp $ST
  # 2) 把 retry 里、且不在 running 的塞回去（改变 batch）
  if [ -f $Q/retry.tsv ]; then
    : > $Q/retry.new
    while IFS=$'\t' read -r n g pid nb out; do
      if cut -f1 $ST | grep -qx "$n"; then echo -e "$n\t$g\t$pid\t$nb\t$out" >> $Q/retry.new; fi
    done < $Q/retry.tsv
    mv $Q/retry.new $Q/retry.tsv
  fi
  # 3) 找空闲卡并填任务（并发上限由 /tmp/queue/maxjob 控制）
  MAXJOB=$(cat $Q/maxjob 2>/dev/null || echo 2)
  for g in 0 1 2 3 4 5 6 7; do
    # 并发上限（每张卡前查一次，改 /tmp/queue/maxjob 即时生效）
    if [ "$(wc -l < $ST)" -ge "$MAXJOB" ]; then break; fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    [ -z "${free:-}" ] && continue
    [ "$free" -lt $NEED ] && continue
    # 该卡是否已有我们的任务
    cut -f2 $ST | grep -qx "$g" && continue
    # 取下一个未跑的任务
    for j in "${JOBS[@]}"; do
      n=${j%%|*}; rest=${j#*|}; b=${rest%%|*}; extra=${rest#*|}
      running "$n" && continue
      [ -f runs/queue1029/$n/ckpt_final.pth ] && continue
      # retry 优先
      if [ -f $Q/retry.tsv ] && cut -f1 $Q/retry.tsv | grep -qx "$n"; then
        b=$(awk -F'\t' -v k=$n '$1==k{print $4}' $Q/retry.tsv | head -1)
        awk -F'\t' -v k=$n '$1!=k' $Q/retry.tsv > $Q/r.tmp && mv $Q/r.tmp $Q/retry.tsv
      fi
      launch "$n" "$g" "$b" "$extra" && break
    done
  done
  sleep 180
done
