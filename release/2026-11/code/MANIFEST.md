# MANIFEST —— 服务器 → 本目录（2026-10-27 拉取）

> 服务器根：`better2:/DataDisk/zmy_code/TintedGlass_Enhance`（git 仓库，权威源）
> 本目录：该仓库的**可发布代码子集**，按服务器 `omega_repo/`（2026-10-27，commit `a5f0e12`）的目录铺平，
> 并补入 `omega_repo` 没收纳的运行依赖（上游 RDNet fork、数据集加载器、PEFT、Ω 实现、建池/分析脚本）。

---

## 1. 逐目录映射

| 本目录 | 服务器来源 | 文件数 | 处理 |
|---|---|---|---|
| `omega/` | `omega_repo/omega/` | 2 | 原样（唯一权威 Ω） |
| `legacy/` | `omega_repo/legacy/` | 2 | 原样 |
| `common/` | `omega_repo/common/` | 1 | 原样（说明性空壳） |
| `train/` | `omega_repo/train/` | 2 | 原样 + 路径参数化 |
| `infer/` | `omega_repo/infer/` | 6 | 原样 + 路径参数化 |
| `eval/` | `omega_repo/eval/` | 1 | 原样 + 路径参数化 |
| `data_build/` | `omega_repo/data_build/` | 3 | 原样 + 路径参数化 |
| `scripts/` | `scripts/`（从 236 个里挑出复现必需的 29 个） | 28 | 原样 + 路径参数化 |
| `RDNet/` | `RDNet/`（源码；剔权重、数据集、PDF） | 52 | 原样 |
| `archive/hole_exploration/` | `omega_repo/archive/hole_exploration/` | 22 | 原样（删 1 份重复） |
| `archive/rejected_dino/` | `omega_repo/archive/rejected_dino/` | 1 | 原样 |
| `archive/server_scripts_2026_10/` | **本地旧快照**（非服务器本次拉取） | 195 | 只归档，不属于发布内容 |
| `local_scripts/`、`local_tools/` | 本机 | 80 | 不进发布包 |

**发布树合计 120 个代码/数据文件 + 6 个根文件（README/MANIFEST/DEPRECATED/OPENSOURCE_TODO/requirements.txt/env.example.sh），约 1.1 MB。**

## 2. 本次拉取日志

| 批次 | 内容 | 文件数 |
|---|---|---|
| ① tar | `omega_repo/**`（44）+ `RDNet/**` 源码（47）+ `scripts/` 28 + `Makefile`/`.gitignore` | 124 |
| ② 补拉 | `scripts/autoqueue.py`、`scripts/run_dom_v1.py`（排队脚本，复现三臂/五臂实验要用） | 2 |
| ③ 补拉 | `RDNet/CLAUDE.md`、`RDNet/pretrained/README.md` | 2 |
| ④ 去重删除 | 与 `infer/` 或 `scripts/` 逐字节相同的副本（8 个 archive 重复 + 3 个 `_*_infer.py` 旧副本 + `_zhu_infer_ascii.py` + `scripts/{eval_1019_multi,finetune_rdnet_peft}.py` + `archive/.../eval_g2_baseline.py.trimmed_omega_of`） | −15 |

去重原则：**只删 SHA256 完全一致者**。去重后 `code/` 内只有 3 个空 `__init__.py` 因内容为空而"相同"，其余重复组 = 0。

## 3. 路径参数化（37 处 / 25 个文件，全部通过 `py_compile`）

| 规则 | 处数 |
|---|---|
| `P = "/DataDisk/zmy_code/TintedGlass_Enhance"` → `P = os.environ.get("TINTED_ROOT", "<原值>")` | 20 |
| `PROJ = ...` 同上 | 2 |
| 第三方根（`XR/ERRNET/IBCLN/CKPT/CKPT_DIR/REMO/RRW/OFFICIAL`）→ `os.environ.get("TINTED_THIRD_PARTY", "/DataDisk/zmy_code") + "<后缀>"` | 11 |
| `PY = "/DataDisk/miniconda/envs/xreflection/bin/python3"` → `os.environ.get("TINTED_PYTHON", sys.executable)` | 2 |
| argparse 默认值 / 常量列表元素（RRW GT 路径、ERRNet ckpt、r-pool、preview 输出） | 2 |
| **跨目录 import 补 path**：`train/finetune_rdnet_peft.py`、`data_build/build_train_real_pairs.py` 各加 `sys.path.insert(0, PROJ/"scripts")`（`rdnet_peft` / `eval_protocol` / `eval_g2_baseline` 都在 `scripts/`；服务器布局下同目录所以从未暴露） | 2 |
| **合计** | **39** |

**不设环境变量时行为与服务器完全一致**（默认值仍是服务器路径），所以这份树可以整体回传服务器而不改变行为。

仍含 `/DataDisk` 字样的只剩 3 处，全是 docstring / usage 示例：
`infer/eval_1019_multi.py:8`、`scripts/build_rrw_pool.py:32-33`。

## 4. `RDNet/` 里哪些是我们的改动

按服务器文件 mtime（2026 年被动过的）：

| 文件 | 日期 | 说明 |
|---|---|---|
| `models/arch/RDnet_.py` | 2026-05-19 | 推理路径加 `autocast` fp16（12 GB 卡适配） |
| `models/cls_model_eval_nocls_reg.py` | 2026-05-19 | 训练/推理控制器改动 |
| `models/arch/rdnet_atap.py` | 2026-09-15 | **新增** αTAP（α 头 + 硬减法 + 除法出口） |
| `test_sirs.py` | 2026-09-14 | 评测入口改动 |
| `CLAUDE.md` | 2026-05-19 | 本课题对上游的使用说明 |

其余 44 个源码文件为上游原样。**发布前需与上游 diff 复核一次**（本机未逐行核）。

## 5. 未收录（附理由）

| 未收 | 理由 |
|---|---|
| `Makefile` | 属已删除的 `research-loop` 工作流（P0–P6），命令指向 9 月旧池，照跑会误导 |
| 服务器 `.gitignore` | 与本地仓规则冲突；本目录随主仓走 |
| `pretrained/*.pth`、`RDNet/pretrained/*.pth`、`runs/**` | 权重（数百 MB～GB），走 release 链接 |
| `data/**`、`raw_1019/**`、`_review_1019/**` | 数据集与探针产物 |
| `docs/**`（26 篇）、`fig/**`、`eval/**` | 判读档与结果 CSV，已在 `../` 主仓；论文渲染在 `../output/` |
| `scripts/` 其余约 200 个 | 一次性探针（`_tmp_*`/`_probe_*`/`_launch_*`）、洞探索（已归档 22 个）、历史分析 |

> 另有 4 个只存在于服务器 `scripts/`、`omega_repo/archive/` 未收录的洞探索脚本
> （`check_hole_overlap.py`、`outline_hole_docs.sh`、`probe_hole_temporal.py`、`rh_locator.py`）**未保留**，
> 需要时从服务器取；本地 git 历史里也有（2026-10-27 整理提交之前）。

## 6. 历史本地目录（不属于发布内容）

| 目录 | 内容 |
|---|---|
| `local_scripts/` | 本机一次性脚本 80 个（含 `_from_repo_root/` 64 个，2026-10-27 从 `paper-writing/` 根迁入） |
| `local_tools/` | `_x_grep/_x_show/_x_dump`（查 `_x/*.txt` 论文全文）、`annotate_holes`（交互标注） |
| `archive/server_scripts_2026_10/` | 10 月服务器脚本的本地旧快照 195 个，仅追溯用 |
