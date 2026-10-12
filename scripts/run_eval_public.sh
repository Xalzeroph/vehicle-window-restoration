#!/usr/bin/env bash
# Run LightRR's released checkpoint over the four public SIRR benchmarks.
#
# Protocol notes:
#  - real20 is resized to 420 on the long side (LightRR's own real20_420 setting).
#    This is REQUIRED: CrossChannelAtt is global attention, O((H*W)^2), so a native
#    1114x717 image would need ~25 GB just for the attention matrix.
#  - --if_align then crops to multiples of 32, which is LightRR's default eval path.
set -u
ENVPY=/DataDisk/zmy_code/LightRR_env/lightrr/bin/python
PROJ=/DataDisk/zmy_code/TintedGlass_Enhance
REPO=/home/zmy/mycode/LightRR-CVPR26

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

exec "$ENVPY" "$PROJ/scripts/eval_lightrr_public.py" \
    --repo "$REPO" \
    --ckpt "$REPO/weights/lightrr/lightrr_woNature.pt" \
    --data-root /DataDisk/zmy_code/datasets \
    --out "$PROJ/eval/lightrr_public_1012.csv" \
    --device cuda:0 \
    --resize real20=420
