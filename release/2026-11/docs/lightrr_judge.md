# LightRR 微调后 1019 判定（development panel, n=15, uint8 口径）

> 生成于 2026-11-06；判据见 `PREREG_lightrr_eval_1106.md` §5（先于本结果写死）。

## 1. 主表（全图 PSNR，15 场景）

| 臂 | PSNR | SSIM | 与 identity 差 |
|---|---|---|---|
| lightrrft_lr1e4 | 20.221 | 0.7135 | +1.784 |
| lightrrft_lr5e5 | 20.198 | 0.7133 | +1.761 |
| lightrr_official_rerun | 19.175 | 0.6804 | +0.738 |

参照：
- identity: 18.437
- rdnet_zs: 20.053
- oldpool_s101 (base seed101): 21.990
- base 3seed: 22.485
- quiet_s101 (full seed101): 22.709
- full 3seed: 22.987

## 2. 配对 Δ（逐场景配对，n=15；t_14=2.145，95% CI）

| LightRR 臂 | 参照 | Δ PSNR | 95% CI | 胜出场景 | 配对 sd | MDE(α=.05,pow=.8) |
|---|---|---|---|---|---|---|
| lightrrft_lr1e4 | quiet_s101 (full seed101) | -2.489 | [-3.100, -1.878] | 0/15 | 1.103 | 0.798 |
| lightrrft_lr1e4 | full 3seed | -2.766 | [-3.371, -2.162] | 0/15 | 1.091 | 0.789 |
| lightrrft_lr1e4 | oldpool_s101 (base seed101) | -1.770 | [-2.198, -1.341] | 0/15 | 0.774 | 0.560 |
| lightrrft_lr5e5 | quiet_s101 (full seed101) | -2.512 | [-3.091, -1.933] | 0/15 | 1.045 | 0.756 |
| lightrrft_lr5e5 | full 3seed | -2.789 | [-3.368, -2.211] | 0/15 | 1.045 | 0.756 |
| lightrrft_lr5e5 | oldpool_s101 (base seed101) | -1.793 | [-2.211, -1.374] | 0/15 | 0.755 | 0.546 |
| lightrr_official_rerun | quiet_s101 (full seed101) | -3.534 | [-4.156, -2.912] | 0/15 | 1.123 | 0.813 |
| lightrr_official_rerun | full 3seed | -3.812 | [-4.389, -3.235] | 0/15 | 1.041 | 0.753 |
| lightrr_official_rerun | oldpool_s101 (base seed101) | -2.815 | [-3.405, -2.225] | 0/15 | 1.065 | 0.770 |

## 3. 主判据（best-of-2，预登记 §5）

- 两臂 PSNR： lightrrft_lr1e4=20.221；lightrrft_lr5e5=20.198；lightrr_official_rerun=19.175
- **best-of-2 = lightrrft_lr1e4 = 20.221 dB**
- 判定：**P1** —— 动机成立：可写「按官方配方域内微调 100k+ it 后，9.5M LightRR 仍显著低于我们的 RDNet 基线」

## 4. 官方冻结 ckpt 同管线复核

- 本管线 15 场景均值 **19.175**；历史 verify summ = 19.1295（差 +0.046，验收门 ±0.05～0.1）

## 5. 附：与冻结稿数字的对应

- RDNet base 三种子 22.49±0.44；full(quiet) 三种子 22.99±0.34（r11annot 表）
- LightRR 历史 A 组零样本：19.18（论文表）/ 19.13（verify summ）

## 6. 训练语义披露（2026-11-06 04:56 补记）

- **实际训练轮数 = 41 epoch / 102,500 it**（预登记写 40 epoch）。原因：`/tmp/ri_dev/train_ft.py:153`
  主循环 `while engine.epoch <= opt.nEpochs`，而 `engine.py:68` 在每轮训练开始时 `self.epoch += 1`
  ⇒ 边界多跑 1 轮；最终 ckpt 为 `_041_00102500.pt`（`_040_` 不是最终）。
- 两臂对称（lr1e4 / lr5e5 同协议、同样多 1 轮），不影响三档判据；best-of-2 距阈值 22.2 有 ~2 dB 余量，判决对轮数不敏感。
- 同时记于 `/tmp/ri_dev/LIGHTRR_TRAIN_NOTES_1106.md`。

## 7. 指标扩展（`scripts/metrics_ext_1019.py`，全图，n=15）

| 臂 | PSNR | β（残留反射；0=完美，−1=没去） | α | Δaff | Δpow | LapVar 比 |
|---|---|---|---|---|---|---|
| lightrrft_lr1e4 | 20.21 | **−0.489** | 0.990 | 0.43 | 0.08 | 1.12 |
| lightrrft_lr5e5 | 20.19 | **−0.511** | 0.998 | 0.36 | 0.05 | 1.10 |
| lightrr_official_rerun | 19.16 | **−0.859** | 1.084 | 1.45 | 0.33 | 1.58 |

- 解读：域内微调把 β 从 −0.859 抬到 −0.49/−0.51（学会部分去反射），但仍有 ~50% 残留反射系数、且 PSNR 仅 20.2 dB；RDNets 域内微调的 β 为 −0.29~−0.34 —— LightRR 的「学会去反射」程度与 PSNR 都明显不足。
- **不得**把 LightRR 的 A 组零样本 19.18 与 B 组（域内微调）混排；本表微调臂只用于「动机成立」判定，不进论文主表排名。
