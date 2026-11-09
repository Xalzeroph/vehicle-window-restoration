#!/usr/bin/env bash
# 自动收尾：等 randfix 两臂训练完成 → 推理 → 出表 → 写结果摘要
# 由 nohup setsid 启动，跑完即退出。
set -u
P=/DataDisk/zmy_code/TintedGlass_Enhance
PY=/DataDisk/miniconda/envs/xreflection/bin/python3
OUT=$P/runs/randfix_1029
LOG=/tmp/randfix/watch.log
SUM=/tmp/randfix/RESULT.txt
cd $P || exit 1

echo "[watch] start $(date '+%F %T')" >> $LOG

# 1) 等两个 ckpt_final
for i in $(seq 1 240); do      # 最多等 8 小时（120s × 240）
  a=0; b=0
  [ -f $OUT/E1_randfix_s202/ckpt_final.pth ] && a=1
  [ -f $OUT/E1_randfix_s303/ckpt_final.pth ] && b=1
  if [ $a -eq 1 ] && [ $b -eq 1 ]; then break; fi
  sleep 120
done
echo "[watch] ckpt 就绪 $(date '+%F %T')" >> $LOG

# 2) 推理（GPU 0/1）
for pair in "E1_randfix_s202:0" "E1_randfix_s303:1"; do
  tag=${pair%%:*}; gpu=${pair##*:}
  if [ ! -d $P/runs/pred/real_pair_1019_randfix_${tag} ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/eval_arm_1019.py \
      --ckpt $OUT/$tag/ckpt_final.pth --name randfix_${tag} --device cuda:0 \
      > /tmp/randfix/infer_${tag}.log 2>&1
    echo "[watch] infer $tag -> $(grep -o 'PSNR=[0-9.]*' /tmp/randfix/infer_${tag}.log | tail -1)" >> $LOG
  fi
done

# 3) 出表（带关键对照臂）
cat > /tmp/predmap_randfix.json <<'EOJ'
{
 "identity": "real_pair_1019_identity",
 "rdnet_zs": "real_pair_1019_rdnet_zs",
 "RRW_zhu": "real_pair_1019_zhu",
 "random_redraw_s101": "real_pair_1019_om1026_random_s101",
 "random_redraw_s202": "real_pair_1019_om1026_random_s202",
 "random_redraw_s303": "real_pair_1019_om1026_random_s303",
 "real_mask_s101": "real_pair_1019_om1026_real_s101",
 "real_mask_s202": "real_pair_1019_om1026_real_s202",
 "real_mask_s303": "real_pair_1019_om1026_real_s303",
 "noloss_s101": "real_pair_1019_om1026_noloss_s101",
 "E1_randfix_s101": "real_pair_1019_ab_E1_randfix",
 "E1_randfix_s202": "real_pair_1019_randfix_E1_randfix_s202",
 "E1_randfix_s303": "real_pair_1019_randfix_E1_randfix_s303",
 "E2_full_s101": "real_pair_1019_ab_E2_full_om0.6"
}
EOJ
$PY scripts/table_final_1019.py --pred-map /tmp/predmap_randfix.json --tag randfix_1029 > /tmp/randfix/table.log 2>&1
tail -1 /tmp/randfix/table.log >> $LOG

# 口径说明（2026-10-29 发现）：下方 heredoc 里的聚合把**逐场景行**一起平均了（n=15 或 45），
# 得到的不是“臂均值”。正确做法：先按 arm 求场景均值，再跨种子求均值/标准差。
# 本批的正确数字见 2026.10.29_会话交接_口径撤稿与nomask结果.md §3。
# 4) 摘要（固定区域 vs 每步重画）
$PY - <<'EOF' > $SUM 2>&1
import csv, statistics as st
rows = list(csv.DictReader(open("eval/table_final_1019_randfix_1029.csv", encoding="utf-8")))
def vals(key):
    return [float(r["psnr"]) for r in rows if r["arm"].startswith(key)]
groups = {
    "随机盘-每步重画(random)": vals("random_redraw"),
    "随机盘-固定(randomfixed)": vals("E1_randfix"),
    "标注掩码(annotated)": vals("real_mask"),
    "无区域项(noloss)": vals("noloss"),
    "整图相对误差(full)": vals("E2_full"),
}
print("== 固定区域 vs 每步重画（82 对 apr 池，15 场景面板）==")
for k, v in groups.items():
    if v:
        print(f"  {k:<26} n={len(v)}  mean={st.mean(v):6.2f}  sd={st.stdev(v) if len(v)>1 else float('nan'):5.2f}  {['%.2f'%x for x in v]}")
a, b = groups["随机盘-每步重画(random)"], groups["随机盘-固定(randomfixed)"]
if a and b:
    d = st.mean(b) - st.mean(a)
    print(f"\n  固定 − 重画 = {d:+.2f} dB")
    print(f"  判据（预登记）: |{'%.2f'%d}| > 0.5 且固定组 sd < 0.3 -> "
          f"{'可下结论' if abs(d) > 0.5 and (len(b) < 2 or st.stdev(b) < 0.3) else '仅报无显著差异'}")
EOF
echo "[watch] done $(date '+%F %T')" >> $LOG
cat $SUM
