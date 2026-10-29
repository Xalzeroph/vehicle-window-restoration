#!/usr/bin/env bash
# 合成遮挡三臂自动收尾：等 ckpt_final -> 推理 -> 出表 -> 写摘要
set -u
P=/DataDisk/zmy_code/TintedGlass_Enhance
PY=/DataDisk/miniconda/envs/xreflection/bin/python3
OUT=$P/runs/synoc_1029
LOG=/tmp/synoc/watch.log
cd $P || exit 1
echo "[watch] start $(date '+%F %T')" >> $LOG

for i in $(seq 1 300); do
  ok=1
  for t in A_exact_s101 B_randfix_s101 C_noocc_s101; do
    [ -f $OUT/$t/ckpt_final.pth ] || ok=0
  done
  [ $ok -eq 1 ] && break
  sleep 120
done
echo "[watch] ckpt 就绪 $(date '+%F %T')" >> $LOG

gpu=0
for t in A_exact_s101 B_randfix_s101 C_noocc_s101; do
  if [ ! -d $P/runs/pred/real_pair_1019_synoc_$t ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/eval_arm_1019.py \
      --ckpt $OUT/$t/ckpt_final.pth --name synoc_$t --device cuda:0 \
      > /tmp/synoc/infer_$t.log 2>&1
    echo "[watch] infer $t -> $(grep -o 'PSNR=[0-9.]*' /tmp/synoc/infer_$t.log | tail -1)" >> $LOG
  fi
  gpu=$(( (gpu + 1) % 4 ))
done

cat > /tmp/predmap_synoc.json <<'EOJ'
{
 "identity": "real_pair_1019_identity",
 "rdnet_zs": "real_pair_1019_rdnet_zs",
 "RRW": "real_pair_1019_zhu",
 "aprjun_l3_s101": "real_pair_1019_aprjun_aprjun_s101",
 "synoc_A_exact": "real_pair_1019_synoc_A_exact_s101",
 "synoc_B_randfix": "real_pair_1019_synoc_B_randfix_s101",
 "synoc_C_noocc": "real_pair_1019_synoc_C_noocc_s101"
}
EOJ
$PY scripts/table_final_1019.py --pred-map /tmp/predmap_synoc.json --tag synoc_1029 > /tmp/synoc/table.log 2>&1
tail -1 /tmp/synoc/table.log >> $LOG

$PY - <<'EOF' > /tmp/synoc/RESULT.txt 2>&1
import csv, statistics as st
rows=list(csv.DictReader(open("eval/table_final_1019_synoc_1029.csv",encoding="utf-8")))
per={}
for r in rows: per.setdefault(r["arm"],[]).append(float(r["psnr"]))
print("== 合成遮挡三臂（同配方，只改合成样本的 Ω 监督）==")
for a,v in per.items():
    print(f"  {a:<18} PSNR={st.mean(v):6.2f}")
base=st.mean(per.get("synoc_C_noocc",[0]))
for a in ("synoc_A_exact","synoc_B_randfix"):
    if a in per:
        print(f"  {a} − C(无监督) = {st.mean(per[a])-base:+.2f} dB")
print("\n  判读阈值：本面板 MDE≈0.26 dB；单种子效应 <1 dB 不作结论")
EOF
echo "[watch] done $(date '+%F %T')" >> $LOG
cat /tmp/synoc/RESULT.txt
