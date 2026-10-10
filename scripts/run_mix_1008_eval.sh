#!/usr/bin/env bash
# 混训 6 臂 × 3 域：预测生成 + 统一评测（2026-10-08）
#
# 矩阵（每臂只在其**测试域**上出预测，省时）:
#   b1_apr_may    : 训 apr        → 测 may   现状基线
#   b2_may_apr    : 训 may        → 测 apr   现状基线
#   m3_aprjun_may : 训 apr+jun    → 测 may   ★多域混训
#   m4_mayjun_apr : 训 may+jun    → 测 apr   ★多域混训
#   m5_aprmay_jun : 训 apr+may    → 测 jun
#   m6_aprjun_gm  : 训 apr+jun(gm)→ 测 may   ★色调对照
#
# 注意：infer_rdt_s2.py 内部 os.chdir(PROJ/RDNet)，--ckpt 必须绝对路径。
set -u
R=/DataDisk/zmy_code/TintedGlass_Enhance
cd "$R"
PY=/DataDisk/miniconda/envs/xreflection/bin/python3
export HF_HOME=/home/zmy/data/cache/huggingface HF_HUB_OFFLINE=1
OUT=$R/runs/mix_1008
mkdir -p "$OUT/pred" "$OUT/eval"

# 臂 -> 测试域（tag 即臂名，infer 会写成 runs/pred/<域>_<臂名>）
declare -A DOM=(
  [b1_apr_may]=may
  [b2_may_apr]=apr
  [m3_aprjun_may]=may
  [m4_mayjun_apr]=apr
  [m5_aprmay_jun]=jun
  [m6_aprjun_gm]=may
)

echo "################ 1) 生成预测 ################"
for arm in b1_apr_may b2_may_apr m3_aprjun_may m4_mayjun_apr m5_aprmay_jun m6_aprjun_gm; do
  dom=${DOM[$arm]}
  ck=$R/runs/ckpt/mix_${arm}/ckpt_final.pth
  [ -f "$ck" ] || { echo "  [跳过] $arm 无 ckpt"; continue; }
  echo "--- $arm  测试域=$dom ---"
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 $PY scripts/infer_rdt_s2.py \
      --ckpt "$ck" --sets "$dom" --tag "$arm" --gpu 0 2>&1 | tail -2
done

echo ""
echo "################ 2) 统一评测（--extra-arm 逐臂喂各自测试域）################"
run_eval(){ # arm dom
  local arm=$1 dom=$2
  local pred=$R/runs/pred/${dom}_${arm}
  if [ ! -d "$pred" ]; then echo "  [跳过] $arm: 无预测 $pred"; return; fi
  echo "--- $arm  on $dom ---"
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 $PY scripts/eval_g2_baseline.py --sets "$dom" \
      --extra-arm "${arm}=${pred}" \
      --out "${OUT}/eval/${arm}_on_${dom}.csv" 2>&1 | tail -4
}
for arm in b1_apr_may b2_may_apr m3_aprjun_may m4_mayjun_apr m5_aprmay_jun m6_aprjun_gm; do
  run_eval "$arm" "${DOM[$arm]}"
done

echo ""
echo "################ 完成 ################"
ls -la "$OUT/eval/" 2>/dev/null
echo "DONE_MIX_EVAL"
