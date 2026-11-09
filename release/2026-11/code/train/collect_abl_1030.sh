#!/usr/bin/env bash
# 消融臂收集器（2026-10-30）：把已完成的 ABL_* 臂一次性评测 + 建"留一法贡献表"。
# 用法（服务器）：
#   bash /tmp/abl/collect_abl_1030.sh            # 默认 tag=abl_1030，GPU1
#   GPU=2 bash /tmp/abl/collect_abl_1030.sh abl2 # 换卡 / 换 tag
# 说明：只做机械步骤（评测 + 出表），不做判读；判读规则见
#   docs/2026.10.30_会话交接_架构损失池与消融.md §5/§9
set -u
P=/DataDisk/zmy_code/TintedGlass_Enhance
PY=/DataDisk/miniconda/envs/xreflection/bin/python3
cd "$P" || exit 1
TAG=${1:-abl_1030}
GPU=${GPU:-1}
mkdir -p /tmp/abl

echo "=== 1) 逐臂评测（已评过的跳过）==="
for d in runs/queue1029/ABL_*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  if [ ! -f "$d/ckpt_final.pth" ]; then
    echo "skip $n（无 ckpt_final）"
    continue
  fi
  if [ -d "runs/pred/real_pair_1019_q_$n" ]; then
    echo "skip $n（已评测）"
    continue
  fi
  echo "== eval $n"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u scripts/eval_arm_1019.py \
    --ckpt "$d/ckpt_final.pth" --name "q_$n" --device cuda:0 \
    || echo "!! eval 失败 $n（下次重跑本脚本会补）"
done

echo "=== 2) 生成 predmap（基线 + 参照臂 + 所有已评测 ABL）==="
"$PY" - "$TAG" <<'PY'
import json, os, sys
P = "/DataDisk/zmy_code/TintedGlass_Enhance"
root = os.path.join(P, "runs/pred")
base = {
    "identity": "real_pair_1019_identity",
    "rdnet_zs": "real_pair_1019_rdnet_zs",
    "RRW": "real_pair_1019_zhu",
    "OURS_mayjun_s101": "real_pair_1019_mayjun15_s101",
    "OURS_mayjun_s202": "real_pair_1019_mayjun15_s202",
    "OURS_mayjun_s303": "real_pair_1019_mayjun15_s303",
    "HISLOSS_mayjun_s101": "real_pair_1019_q_O2mayjun_s101",
    "HISLOSS_mayjun_s202": "real_pair_1019_q_O2mayjun_s202",
    "HYB_mayjun_s101": "real_pair_1019_q_HYB_s101",
}
m = {k: v for k, v in base.items() if os.path.isdir(os.path.join(root, v))}
for d in sorted(os.listdir(root)):
    if d.startswith("real_pair_1019_q_ABL_") and os.path.isdir(os.path.join(root, d)):
        m[d.replace("real_pair_1019_", "")] = d
tag = sys.argv[1]
out = "/tmp/predmap_%s.json" % tag
json.dump(m, open(out, "w"), indent=1)
print("arms:", len(m), "->", out)
for k in m:
    print("   ", k)
PY

echo "=== 3) 出表 ==="
"$PY" scripts/table_final_1019.py --pred-map "/tmp/predmap_$TAG.json" --tag "$TAG" 2>&1 | tail -5
echo
echo "=== 表（判读：与 OURS_mayjun_s101=23.07 比，掉越多该项越有用；关掉更高 ⇒ 删该项）==="
cat "eval/table_final_1019_$TAG.md" 2>/dev/null || echo "（表未生成，检查上面报错）"
