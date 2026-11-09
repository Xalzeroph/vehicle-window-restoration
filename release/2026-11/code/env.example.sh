#!/usr/bin/env bash
# 复现环境变量模板：source 本文件后再跑 code/ 里的脚本。
# 不设这些变量时，代码退回服务器路径（/DataDisk/...），行为与服务器完全一致。

# 1) 本仓根（= 本文件所在目录 code/）
export TINTED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2) 第三方仓库与权重的父目录（RRW 数据集、Zhu/ERRNet/IBCLN/IBCLN/DSRNet 仓库、官方 rdnet.pth 所在处）
#    默认值在代码里是 /DataDisk/zmy_code；本地复现请指到你自己的目录。
export TINTED_THIRD_PARTY="${TINTED_THIRD_PARTY:-$TINTED_ROOT/../third_party}"

# 3) 批量脚本（autoqueue.py / run_dom_v1.py）调用的解释器
export TINTED_PYTHON="${TINTED_PYTHON:-$(command -v python)}"

# 4) 共享机守则：显式钉卡 + 限制线程（本机复现可不管）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "TINTED_ROOT         = $TINTED_ROOT"
echo "TINTED_THIRD_PARTY  = $TINTED_THIRD_PARTY"
echo "TINTED_PYTHON       = $TINTED_PYTHON"
echo "CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"
