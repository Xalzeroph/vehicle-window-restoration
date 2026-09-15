# ============================================================ #
# TintedGlass 研究实验链（配合 .claude/skills/research-loop 工作流）
#
# 阶段: 建池 → 验收(泄漏+域统计) → 建训练集 → 训练 → 评测 → 汇总
# 每一步 = 一条命令; 共享机守则: GPU 用 CUDA_VISIBLE_DEVICES 显式钉卡
#
# 例子:
#   make check-pool  POOL=data/pools/matched_pool_0915
#   make triplets    TRIPLETS=data/train_sets/train_triplets_match_0915 \
#                    TRIPLET_ARGS="--t-weights 0,0,0 --t-extra-dir data/pools/matched_pool_0915 \
#                                  --t-extra-name matched --t-extra-glob '*.jpg' --t-extra-weight 1"
#   make train       NAME=raw_match ENCODER=raw TRIPLETS=data/train_sets/train_triplets_match_0915 GPU=1
#   make eval-all    NAME=raw_match ENCODER=raw
#   make report
#   make clean-ckpts NAME=raw_match          # 保留 final, 清中间 ckpt（每条省 ~40G）
# ============================================================ #
SHELL := /bin/bash
PY    := /DataDisk/miniconda/envs/xreflection/bin/python3
PROJ  := /DataDisk/zmy_code/TintedGlass_Enhance

POOL     ?= data/pools/matched_pool_0915
TS       ?= /DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/synth_rrw_v21_0913
TS_IND   ?= /DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/synth_rrw_v21_indomain
NAME     ?= raw_match
ENCODER  ?= raw
TRIPLETS ?= data/train_sets/train_triplets_0915
ITERS    ?= 24000
GPU      ?= 1
TRIPLET_ARGS ?=
# tag: NAME 去掉编码器前缀（NAME=raw_match ENCODER=raw → _match）；无前缀则整名做 tag
TAG := _$(patsubst $(ENCODER)_%,%,$(NAME))

.PHONY: help check-pool triplets train eval eval-all report clean-ckpts ps

help:
	@sed -n '2,13p' Makefile | sed 's/^# \{0,1\}//'
	@echo "--- 变量: POOL TS TS_IND NAME ENCODER TRIPLETS ITERS GPU TRIPLET_ARGS ---"

## 训练 T 池验收: 内容级泄漏核查 + 域统计（建池/建集后必跑, 泄漏即停）
check-pool:
	$(PY) scripts/check_t_src_leak.py --t-src $(notdir $(POOL))=$(POOL) --t-glob '*' \
	    --ts-dir data/test_sets/synth_rrw_v21_0913/gt --skip-crop
	$(PY) scripts/domain_stats.py --mode native --src $(POOL)=$(POOL)/*

## 建训练三元组: TRIPLET_ARGS 传池与权重（见文件头例子）
triplets:
	$(PY) scripts/build_train_triplets.py --out $(TRIPLETS) $(TRIPLET_ARGS)
	$(PY) scripts/build_train_triplets.py --out $(TRIPLETS) --check $(TRIPLETS)

## 训练 fac 头（长任务: 用 nohup 跑请配 > /tmp/train_$(NAME).log）
train:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/finetune_fac_head.py \
	    --encoder $(ENCODER) --triplets $(TRIPLETS) \
	    --out runs/ckpt/fac_$(NAME) --iters $(ITERS) --gpu 0

## 评测单个测试集: 产物 eval/eval_iar_$(ENCODER)_$(NAME){,_summary}.csv/json + runs/pred/fac_pred_$(NAME)/
eval:
	$(PY) scripts/eval_iar.py --encoder $(ENCODER) --ckpt runs/ckpt/fac_$(NAME)/ckpt_final.pth \
	    --ts $(TS) --out-root runs/pred/fac_pred_$(NAME) --tag $(TAG) --gpu $(GPU)

## 双测试集评测（v2.1 + indomain）—— 标配, 防"退化过拟合"
eval-all:
	$(PY) scripts/eval_iar.py --encoder $(ENCODER) --ckpt runs/ckpt/fac_$(NAME)/ckpt_final.pth \
	    --ts $(TS) --out-root runs/pred/fac_pred_$(NAME) --tag $(TAG) --gpu $(GPU)
	$(PY) scripts/eval_iar.py --encoder $(ENCODER) --ckpt runs/ckpt/fac_$(NAME)/ckpt_final.pth \
	    --ts $(TS_IND) --out-root runs/pred/fac_pred_indomain_$(NAME) --tag $(TAG)_indomain --gpu $(GPU)

## 汇总所有评测 summary → Markdown 对照表
report:
	$(PY) scripts/summarize_evals.py --glob 'eval/eval_iar_*_summary.json'

## 清中间 ckpt（保留 ckpt_final.pth; 每 500 iter 一个 ~1G, 易吃满盘）
clean-ckpts:
	find runs/ckpt/fac_$(NAME) -name 'ckpt_*.pth' ! -name 'ckpt_final.pth' -delete
	@du -sh runs/ckpt/fac_$(NAME)

## 看当前训练进程
ps:
	@pgrep -af finetune_fac_head.py || echo "(无训练进程)"
