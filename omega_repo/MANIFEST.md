# MANIFEST：旧路径 → 新路径

| 新仓路径 | 旧路径 | 处理方式 |
|---|---|---|
| `omega/canonical.py` | `scripts/build_omega_v2_0920.py::omega_of` | **抽取**为独立模块，唯一权威 |
| `omega/__init__.py` | 新增 | 导出 `omega_of, G` |
| `legacy/omega_legacy.py` | `scripts/table_final_1019.py::hole_mask` | 改名 `hole_ratio_otsu` + 警告 |
| `train/finetune_rdnet_s2.py` | `scripts/finetune_rdnet_s2.py` | 原样复制 |
| `train/finetune_rdnet_peft.py` | `scripts/finetune_rdnet_peft.py` | 原样复制 |
| `infer/eval_arm_1019.py` | `scripts/eval_arm_1019.py` | 原样复制 |
| `infer/eval_1019_multi.py` | `scripts/eval_1019_multi.py` | 被 `eval_arm_1019` import |
| `infer/zhu_infer_1019.py` | 同名 | 公开基线：Zhu CVPR'24 |
| `infer/errnet_infer_1019.py` | 同名 | 公开基线：ERRNet |
| `infer/ibcln_infer_1019.py` | 同名 | 公开基线：IBCLN |
| `infer/dsrnet_infer_1019.py` | 同名 | 公开基线：DSRNet |
| `eval/table_final_1019.py` | 同名 | 原样复制（内部仍用旧 Ω，见下） |
| `data_build/build_april_pairs.py` | 同名 | 建 apr GT 测试集 |
| `data_build/build_train_real_pairs.py` | 同名 | 建训练池 |
| `data_build/prep_ablation_data.py` | 同名 | 拆域 + 补 C_I |
| `archive/hole_exploration/hole_*.py` | `scripts/hole*.py`（22 个） | 只读归档 |
| `archive/hole_exploration/eval_g2_baseline.py.trimmed_omega_of` | `scripts/eval_g2_baseline.py` | 删减版 `omega_of`，留证 |

## 尚未完成（step 2，待 D2 决定）

`eval/table_final_1019.py` 内部仍用它自己的 `hole_mask`（对 `r=R/T` 做全局 Otsu），
与 `omega/canonical.py` **不是同一批像素**。

若要统一：把它改为 `from omega.canonical import omega_of`，并**重算 7 个公开方法的
Δ(洞−非洞)**。在决定之前，基准表里的 Δ 一律视为"旧掩码口径"，不得与新口径混用。
