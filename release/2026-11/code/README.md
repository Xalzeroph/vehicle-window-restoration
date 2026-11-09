# `code/` —— 贴膜玻璃 SIRR 的可复现代码

> **2026-10-27 拉取建立**：从 `better2:/DataDisk/zmy_code/TintedGlass_Enhance` 取来"复现者要用的全部代码"，
> 按服务器 `omega_repo/`（2026-10-27 建立，commit `a5f0e12`）的目录铺平，并补上它没有收的运行依赖（上游 RDNet fork、数据集加载器、Ω 实现、建池脚本、分析脚本）。
> 服务器仍是**权威源**；这份是可离线、可发布的副本，路径已能用环境变量覆盖（§4）。

---

## 1. 目录

| 目录 | 内容 | 复现者要不要改 |
|---|---|---|
| `omega/` | **唯一权威 Ω 定义**（`omega_of`，遮挡物 footprint） | ✅ 只读 |
| `legacy/` | 旧 Ω 变体（`hole_ratio_otsu` 等，改名 + 警告） | 只读，仅复现旧数字 |
| `train/` | `finetune_rdnet_s2.py`（S2 主训练）、`finetune_rdnet_peft.py`（底座） | ✅ |
| `infer/` | `eval_arm_1019.py`（本方法推理）、`eval_1019_multi.py`（多臂批量）、四个公开基线封装 `{zhu,errnet,ibcln,dsrnet}_infer_1019.py` | ✅ |
| `eval/` | `table_final_1019.py`（基准表：PSNR/SSIM/HF-PSNR/梯度余弦/LapVar + 分区） | ✅ |
| `data_build/` | 建 apr 测试集、建训练池、拆域补 $C_I$ | ✅ |
| `scripts/` | 服务器 `scripts/` 里**复现必需**的 28 个：数据集加载器 `dataset_real_pairs.py`、PEFT 适配器 `rdnet_peft.py`、损失 `gcf_loss.py`、Ω 实现 `build_omega_v2_0920.py`/`eval_g2_baseline.py`、建池 `build_pair_1019.py` 等、评测协议 `eval_protocol.py`、排队 `autoqueue.py`/`run_dom_v1.py`、分析 `agg_*.py`/`detail_metrics_1019.py` | ✅ |
| `RDNet/` | 上游 RDNet（CVPR'25）源码 + **我们的改动**（`models/arch/rdnet_peft.py` 等）；**不含权重与数据** | 只读（改上游需谨慎） |
| `archive/hole_exploration/` | 22 个 `hole_*.py` 探索脚本 | ⛔ 禁止 import |
| `archive/rejected_dino/` | DINO 掩码（判废）说明 | ⛔ |
| `archive/server_scripts_2026_10/` | 本地旧快照 195 个（历史，不是发布内容） | ⛔ |
| `local_scripts/`、`local_tools/` | 本机一次性脚本与检索/标注工具 | 不进发布包 |

## 2. 三条铁律（沿用服务器 `omega_repo`）

1. **一个名字，一种语义。** 语义不同必须改名，不允许同名两份。
2. **Ω 只有一份实现**：`omega/canonical.py::omega_of`；其它定义一律进 `legacy/`，调用方必须在输出里标注用了哪一个。
3. **`archive/` 不参与任何 import。**

## 3. ⚠️ Ω 尚未接线（开源前必须处理的唯一结构性缺口）

`omega/canonical.py` 是权威定义，但**目前没有任何脚本 import 它**——建池脚本仍写着
`from eval_g2_baseline import omega_of`（= 删减版）。两者实测在新池 82 对上 IoU = 1.0000，
但这是"值相等"，不是"单一定义"。

- 现状：`scripts/build_train_real_pairs.py`、`scripts/build_pairs_1019split.py`、`scripts/build_train_omega_0920.py` 用删减版；`eval/table_final_1019.py` 用自己的 `hole_mask`（Otsu on `r=R/T`）。
- 处置选项（审计 D2）：① 把三处 import 改指 `omega.canonical`；② 或明确在 README/论文中声明"训练与评测用两套掩码"。
- **在接线完成前，凡引用 Δ(洞−非洞) 的数字都必须注明用的是哪一套。**

`omega_of` 的语义：**遮挡物 footprint**（反射缺失 + 照度下降，单帧统计上不可分），
**不是**"纯反射缺失"。已知限制 **L1**（只有全局增益 `s=q05(I/T)`，空间增益实测差 26%）
与 **L2**（无 `gT` 判据）请勿"顺手修好"——修之前先读 `../2026.10.26_物理模型_最终版_遮挡阴影.md`。

## 4. 路径：三个环境变量（本地 37 处已参数化）

| 变量 | 含义 | 默认（= 服务器） |
|---|---|---|
| `TINTED_ROOT` | 本仓根（= 本 `code/` 目录） | `/DataDisk/zmy_code/TintedGlass_Enhance` |
| `TINTED_THIRD_PARTY` | 第三方仓库与权重的父目录 | `/DataDisk/zmy_code` |
| `TINTED_PYTHON` | 批量脚本调用的解释器 | 当前 `sys.executable` |

写法统一为 `os.environ.get("TINTED_ROOT", "<服务器默认>")`，**不设变量时行为与服务器完全一致**。
仍带 `/DataDisk` 字样的只剩 3 处 docstring/usage 示例（`infer/eval_1019_multi.py`、`scripts/build_rrw_pool.py`），不影响运行。

## 5. 环境（服务器实测版本）

| 项 | 值 |
|---|---|
| Python | 3.10.20（conda env `xreflection`） |
| torch / torchvision | 2.12.0+cu126 / 0.27.0+cu126（CUDA 12.6） |
| 其它 | cv2 5.0.0、numpy 2.2.6、scikit-image 0.25.2、scipy 1.15.3、Pillow 12.2.0、pyiqa 0.1.16、timm 1.0.29 |
| 卡 | 单张 RTX 3080 10 GB 即可（**FocalNet 主干冻结 + Mona 适配器，可训 65.95 M**）；全量微调需 ≥16 GB |
| 训练配方 | 6000 iters / bs 6 / lr 1e-4 / seed 101；裁剪 224→256→320；30% 真实配对 + 40% 在线合成 + 30% 开源直督；单次 ~3.3 h（共享机） |

安装：`pip install -r requirements.txt`（`requirements.txt` 列出实测版本；RDNet 上游自带的 `RDNet/requirements.txt` 只有 `dominate/visdom`，是给可视化用的）。

## 6. 复现链路

> **已验证 / 未验证**（2026-10-27，在服务器 `/tmp` 用同一份 tar 实测，不污染项目目录）：
> ✅ `omega` 包 import；✅ 四个入口 `--help` 通过（train / build_train_real_pairs / eval_arm_1019 / table_final_1019）；
> ✅ `canonical.omega_of` 与 `real_pairs_1026` 全部 82 对掩码 **IoU = 1.0000**。
> ⛔ **未验证**：任何一步真实训练 / 推理 / 出表（需要数据与权重，属 `OPENSOURCE_TODO.md` 的 B6）。

```bash
export TINTED_ROOT=$(pwd)                 # 本 code/ 目录
export TINTED_THIRD_PARTY=/path/to/third_party   # 见 §7
export TINTED_PYTHON=$(which python)

# S1 数据：原始视频/帧 → 配对测试集（1019 主评测集）
python scripts/build_pair_1019.py --help          # 参数以脚本 argparse 为准

# S2 Ω 掩码 + 训练池（注意 §3：目前走 eval_g2_baseline.omega_of）
python scripts/build_omega_v2_0920.py --help
python data_build/build_train_real_pairs.py --help

# S3 训练（单卡钉卡）
CUDA_VISIBLE_DEVICES=0 python train/finetune_rdnet_s2.py \
    --arm main --seed 101 --gpu 0 --out runs/s2_main --iters 6000 --batch 6

# S4 推理 + 出表（基准表必带 identity 与官方 ckpt）
CUDA_VISIBLE_DEVICES=0 python infer/eval_arm_1019.py \
    --ckpt runs/s2_main/ckpt_final.pth --name s2_main
python eval/table_final_1019.py
```

评测面板纪律（缺了就不算结论）：`real_pair_1019` 15 场景 / 必带 `identity` 与官方冻结 ckpt /
Δ 给配对 95% CI 与 MDE（n=15，sd≈0.36 ⇒ MDE≈0.26 dB）/ never-worse 看逐场景 Δ 的 min。

## 7. 不在此仓的东西（复现者自备）

| 类别 | 具体 | 来源 |
|---|---|---|
| 权重 | `rdnet.pth`（官方）、`cls_model.pth`、`focal.pth`、我们训练的 `runs/**/ckpt_final.pth` | 官方 release / 训练得到 |
| 数据 | `data/test_sets/real_pair_1019`（15 场景）、`real_pair_{apr,may,jun}_*`、`data/train_sets/real_pairs_{1020,1026,1019split}`、`data/raw_1019` | 自采（协议见 `../2026.9.22_自拍配对数据_拍摄规格.md`）；发布计划见 `OPENSOURCE_TODO.md` |
| 第三方 | RRW 反射池（数据集）、Real20/Nature/Wild/Postcard/Objects、以及四个基线仓库（Zhu / ERRNet / IBCLN / DSRNet） | 各自官方仓库 |
| 文档 | 论文与判读档 | 见 `../*.md`、`../output/paper_final.html` |

## 8. 许可与出处（开源前必须核对）

- `RDNet/` 是 **Zhao et al., CVPR'25（RDNet）** 的 fork：上游 LICENSE 未随仓库带过来，**发布前必须回上游确认许可与署名**；本仓对它的改动范围见 `MANIFEST.md` §4。
- `infer/{zhu,errnet,ibcln,dsrnet}_infer_1019.py` 只是**调用封装**，不含对方权重与网络定义；对应实现仍从各自仓库 import。
- 本课题自有的部分（`omega/ legacy/ train/ infer/eval*/ data_build/ scripts/` 与文档）版权归项目作者。
