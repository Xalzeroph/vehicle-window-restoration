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
LEVEL_ARGS ?=          # 新档位头: --region-level [--level-reg]
TAG := _$(patsubst $(ENCODER)_%,%,$(NAME))

.PHONY: help check-pool triplets train eval eval-all report clean-ckpts ps

help:
	@echo "Targets: check-pool triplets train eval eval-all report clean-ckpts ps"
	@echo "Variables: POOL TS TS_IND NAME ENCODER TRIPLETS ITERS GPU TRIPLET_ARGS"

check-pool:
	$(PY) scripts/check_t_src_leak.py --t-src $(notdir $(POOL))=$(POOL) --t-glob '*' \
	    --ts-dir data/test_sets/synth_rrw_v21_0913/gt --skip-crop
	$(PY) scripts/domain_stats.py --mode native --src $(POOL)=$(POOL)/*

triplets:
	$(PY) scripts/build_train_triplets.py --out $(TRIPLETS) $(TRIPLET_ARGS)
	$(PY) scripts/build_train_triplets.py --out $(TRIPLETS) --check $(TRIPLETS)

train:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/finetune_fac_head.py \
	    --encoder $(ENCODER) --triplets $(TRIPLETS) \
	    --out runs/ckpt/fac_$(NAME) --iters $(ITERS) --gpu 0 $(LEVEL_ARGS)

eval:
	$(PY) scripts/eval_iar.py --encoder $(ENCODER) --ckpt runs/ckpt/fac_$(NAME)/ckpt_final.pth \
	    --ts $(TS) --out-root runs/pred/fac_pred_$(NAME) --tag $(TAG) --gpu $(GPU) $(LEVEL_ARGS)

eval-all:
	$(PY) scripts/eval_iar.py --encoder $(ENCODER) --ckpt runs/ckpt/fac_$(NAME)/ckpt_final.pth \
	    --ts $(TS) --out-root runs/pred/fac_pred_$(NAME) --tag $(TAG) --gpu $(GPU) $(LEVEL_ARGS)
	$(PY) scripts/eval_iar.py --encoder $(ENCODER) --ckpt runs/ckpt/fac_$(NAME)/ckpt_final.pth \
	    --ts $(TS_IND) --out-root runs/pred/fac_pred_indomain_$(NAME) --tag $(TAG)_indomain --gpu $(GPU) $(LEVEL_ARGS)

report:
	$(PY) scripts/summarize_evals.py --glob 'eval/eval_iar_*_summary.json'

clean-ckpts:
	find runs/ckpt/fac_$(NAME) -name 'ckpt_*.pth' ! -name 'ckpt_final.pth' -delete
	@du -sh runs/ckpt/fac_$(NAME)

ps:
	@pgrep -af finetune_fac_head.py || echo "(无训练进程)"
