#!/usr/bin/env bash
# Full public-benchmark matrix for LightRR's released checkpoint, composed to mirror the
# paper's Table 1 so the numbers are directly comparable:
#   Real(20)      = real/real20                     (resized to 420, as the paper does)
#   Objects(200)  = SIR2/SolidObjectDataset/solid200
#   PostCard(199) = SIR2/postcard/postcard199
#   Wild(55)      = SIR2/Wildscene/wild55
# Protocol: --if_align crops to multiples of 32 (LightRR's own default eval path).
set -u
ENVPY=/DataDisk/zmy_code/LightRR_env/lightrr/bin/python
PROJ=/DataDisk/zmy_code/TintedGlass_Enhance
REPO=/home/zmy/mycode/LightRR-CVPR26
DS=/DataDisk/zmy_code/datasets

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

exec "$ENVPY" "$PROJ/scripts/eval_lightrr_public.py" \
    --repo "$REPO" \
    --ckpt "$REPO/weights/lightrr/lightrr_woNature.pt" \
    --data-root "$DS" \
    --out "$PROJ/eval/lightrr_public_full_1012.csv" \
    --device cuda:0 \
    --resize real20=420 \
    --datasets "real20:real/real20,objects:SIR2/SolidObjectDataset/solid200,postcard:SIR2/postcard/postcard199,wild:SIR2/Wildscene/wild55"
