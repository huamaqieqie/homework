# JEPA 实验报告

生成日期：2026-07-24

数据目录：`outputs/jepa_viz/experiment_summary_20260724/`

本报告基于已完成的 eval 日志、训练 `metrics.csv`、latent 导出结果和 `tools/jepa_viz` 生成的可视化文件整理。

## 1. PushT：Original / Residual / Factored 对比

| 模型 | 成功率 mean +/- std | Δ vs Original | time/episode |
|---|---:|---:|---:|
| Original 15 | **88.80 +/- 3.03** | 0.00 | 6.42s |
| Residual 15 | **90.00 +/- 3.74** | 1.20 | 7.50s |
| Factored 96-96 15 | **79.60 +/- 4.77** | -9.20 | 5.65s |
| Factored 64-128 15 | **68.40 +/- 8.65** | -20.40 | 7.63s |

![PushT Success Rate Summary](eval_summary_pusht/success_rate_summary.png)

| 指标 | 含义 | 读图方式 |
|---|---|---|
| success rate mean | 多个 seed 的平均成功率 | 越高越好，是主指标。 |
| Δ vs Original | 相对 PushT Original 15 的成功率差值 | 正数表示高于 PushT baseline，负数表示低于 PushT baseline。 |
| std | seed 间波动 | 越小表示越稳定。 |

![PushT Success Rate by Seed](eval_summary_pusht/success_rate_by_seed.png)

| 指标 | 含义 | 读图方式 |
|---|---|---|
| 单点 | 某个 seed 的成功率 | 点越分散，说明 seed 敏感性越强。 |
| 横线 | 该模型的 seed 平均值 | 用于比较模型整体水平。 |

![PushT Time per Episode](eval_summary_pusht/time_per_episode.png)

| 指标 | 含义 | 读图方式 |
|---|---|---|
| time/episode | 平均每个 episode 的评估耗时 | 越低表示 planner/eval 更快。 |
| eval_budget | CEM 每步采样预算 | 本组 PushT 均为 300，因此成功率可横向比较。 |

## 2. Reacher：Original / Residual 对比

| 模型 | 成功率 mean +/- std | Δ vs Reacher Original | time/episode |
|---|---:|---:|---:|
| Original 15 | **62.00 +/- 8.00** | 0.00 | 4.16s |
| Residual 15 | **57.60 +/- 10.14** | -4.40 | 4.12s |

![Reacher Success Rate Summary](eval_summary_reacher/success_rate_summary.png)

| 指标 | 含义 | 读图方式 |
|---|---|---|
| success rate mean | Reacher 多 seed 平均成功率 | 只在 Reacher Original 与 Reacher Residual 之间比较。 |
| Δ vs Reacher Original | 相对 Reacher Original 15 的成功率差值 | 不与 PushT 或 Factored PushT 结果混比。 |
| std | seed 间波动 | 越小表示越稳定。 |

![Reacher Success Rate by Seed](eval_summary_reacher/success_rate_by_seed.png)

| 指标 | 含义 | 读图方式 |
|---|---|---|
| 单点 | 某个 seed 的成功率 | Reacher seed 间波动较大，需要看每个 seed。 |
| 横线 | 该模型的 seed 平均值 | 用于比较 Reacher Original/Residual。 |

![Reacher Time per Episode](eval_summary_reacher/time_per_episode.png)

| 指标 | 含义 | 读图方式 |
|---|---|---|
| time/episode | Reacher 平均每个 episode 的评估耗时 | 本组 Reacher eval_budget 均为 50。 |
| eval_budget | CEM 每步采样预算 | Reacher 使用 50，不能和 PushT budget=300 的耗时直接比较。 |

## 3. PushT Prediction 指标

| 指标 | Original | Residual | 说明 |
|---|---:|---:|---|
| mean cosine | **0.99399** | 0.99254 | z_pred 与 z_target 的平均余弦相似度，越高越接近。 |
| diagonal gap | **0.20243** | 0.16525 | diagonal mean - off-diagonal mean，越大越好。 |
| top-1 horizon matching | **98.70%** | 96.61% | 预测 horizon 匹配到正确 target horizon 的比例。 |
| normal MSE | **0.01070** | 0.01311 | 正常条件下的 prediction MSE，越低越好。 |
| shuffled-action MSE | 0.37885 | **0.28743** | 打乱 action 后的 MSE。 |

### 3.1 Horizon Cosine

| Original | Residual |
|---|---|
| ![Original Horizon Cosine](original_lewm_15/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png) | ![Residual Horizon Cosine](residual_15/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| x 轴 horizon | 第 1/2/3 个 future step | 必须显示多个 horizon；如果只有 1 个点，需要检查导出维度。 |
| y 轴 cosine | cos(z_pred, z_target) | 越高表示预测 latent 越接近 target latent。 |
| action_norm_bin | 按 action 范数分桶 | 不同动作强度下曲线分离越明显，说明 action 条件影响越强。 |

### 3.2 Alignment Heatmap

| Original | Residual |
|---|---|
| ![Original Heatmap](original_lewm_15/prediction_viz/target_pred_alignment_heatmap.png) | ![Residual Heatmap](residual_15/prediction_viz/target_pred_alignment_heatmap.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| diagonal mean | pred horizon 与同 horizon target 的平均相似度 | 越高越好。 |
| off-diagonal mean | pred horizon 与其他 horizon target 的平均相似度 | 越低越好。 |
| diagonal gap | 对角线优势 | 越大说明模型越能区分不同 future step。 |
| top-1 horizon matching | 每个 pred 最相似 target 是否落在正确 horizon | 越高说明 temporal alignment 越清晰。 |

### 3.3 Action Condition Ablation

| Original | Residual |
|---|---|
| ![Original Condition Ablation](original_lewm_15/prediction_viz/condition_ablation.png) | ![Residual Condition Ablation](residual_15/prediction_viz/condition_ablation.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| normal | 使用正常 action condition 的预测结果 | baseline。 |
| condition_removed / zero action | 去掉动作条件后的预测结果 | 如果性能明显变差，说明模型确实利用 action。 |
| condition_shuffled | 打乱动作条件后的预测结果 | 如果变差，说明 action 与样本匹配关系重要。 |
| ablation gap | ablated MSE - normal MSE | gap 越大，action 条件依赖越明显。 |

## 4. PushT Latent 表征诊断

| 指标 | Original z_pred | Residual z_pred | 说明 |
|---|---:|---:|---|
| active dims | 192/192 | 192/192 | std 超过阈值的维度数；当前均未 collapse 到少数维度。 |
| effective rank | **59.59** | 58.32 | 协方差谱的有效秩，越高表示表示分布越分散。 |
| participation ratio | **54.41** | 52.93 | 另一个有效维度估计。 |
| top10 explained var | **0.257** | 0.266 | 前 10 主成分解释方差比例；过高可能表示维度集中。 |
| top50 explained var | **0.912** | 0.921 | 前 50 主成分解释方差比例。 |
| feature std mean | 0.985 | **0.987** | 每维标准差均值，接近 1 较正常。 |
| pairwise cosine mean | 0.00050 | 0.00062 | 样本间 latent 余弦均值，接近 0 表示整体分散。 |
| pairwise cosine q95 | 0.2357 | **0.2302** | 高相似样本分位。 |

### 4.1 Active Dimension Count

| Original | Residual |
|---|---|
| ![Original Active Dims](original_lewm_15/latent_viz/active_dimension_count.png) | ![Residual Active Dims](residual_15/latent_viz/active_dimension_count.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| active / total | 有效维度数 / 总 latent 维度 | 如果 active 很低，说明 latent collapse。 |
| z_context / z_target / z_pred | 三类 latent 分别统计 | 三者都应保持足够维度活跃。 |

### 4.2 Pairwise Cosine Histogram

| Original | Residual |
|---|---|
| ![Original Pairwise Cosine](original_lewm_15/latent_viz/pairwise_cosine_histogram.png) | ![Residual Pairwise Cosine](residual_15/latent_viz/pairwise_cosine_histogram.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| pairwise cosine | batch 内不同样本 latent 的两两余弦相似度 | 集中在 0 附近通常表示样本分布较分散。 |
| q95 | 高相似度尾部 | 如果接近 1 的样本过多，可能存在表征塌缩或样本重复。 |
| density | 归一化频率 | 便于比较不同 latent 类型。 |

### 4.3 Covariance Spectrum

| Original | Residual |
|---|---|
| ![Original Spectrum](original_lewm_15/latent_viz/covariance_eigenvalue_spectrum.png) | ![Residual Spectrum](residual_15/latent_viz/covariance_eigenvalue_spectrum.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| eigenvalue spectrum | latent 协方差矩阵特征值分布 | 如果前几个特征值过大，表示信息集中在少数方向。 |
| cumulative explained variance | 累计解释方差 | 曲线上升越慢，表示维度利用更均匀。 |
| effective rank | 谱分布的有效维度 | 越高通常越不 collapse。 |

### 4.4 Target-Pred Latent Alignment

| Original | Residual |
|---|---|
| ![Original Global Alignment](original_lewm_15/latent_viz/target_pred_latent_alignment_global.png) | ![Residual Global Alignment](residual_15/latent_viz/target_pred_latent_alignment_global.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| target point | PCA 空间中的目标 latent | 与 pred 越近越好。 |
| pred point | PCA 空间中的预测 latent | 与对应 target 的距离反映预测误差。 |
| 灰色连线 | 同一样本 target-pred pair | 线越短表示预测越准。 |

## 5. 训练曲线索引

| 模型 | total loss | val loss | per-horizon MSE | latent std/norm | throughput/lr |
|---|---|---|---|---|---|
| Reacher Original 15 | [total](training/reacher_original_15/total_loss.png) | [val](training/reacher_original_15/val_loss.png) | [horizon](training/reacher_original_15/val_epoch_per_horizon_mse.png) | [std](training/reacher_original_15/latent_std.png) / [norm](training/reacher_original_15/latent_norm.png) | [throughput](training/reacher_original_15/samples_per_sec.png) / [lr](training/reacher_original_15/learning_rate.png) |
| Reacher Residual 15 | [total](training/reacher_residual_15/total_loss.png) | [val](training/reacher_residual_15/val_loss.png) | [horizon](training/reacher_residual_15/val_epoch_per_horizon_mse.png) | [std](training/reacher_residual_15/latent_std.png) / [norm](training/reacher_residual_15/latent_norm.png) | [throughput](training/reacher_residual_15/samples_per_sec.png) / [lr](training/reacher_residual_15/learning_rate.png) |
| Factored 96/96 15 | [total](training/factored_96_96_15/total_loss.png) | [val](training/factored_96_96_15/val_loss.png) | [horizon](training/factored_96_96_15/val_epoch_per_horizon_mse.png) | [std](training/factored_96_96_15/latent_std.png) / [norm](training/factored_96_96_15/latent_norm.png) | [throughput](training/factored_96_96_15/samples_per_sec.png) / [lr](training/factored_96_96_15/learning_rate.png) |
| Factored 64/128 15 | [total](training/factored_64_128_15/total_loss.png) | [val](training/factored_64_128_15/val_loss.png) | [horizon](training/factored_64_128_15/val_epoch_per_horizon_mse.png) | [std](training/factored_64_128_15/latent_std.png) / [norm](training/factored_64_128_15/latent_norm.png) | [throughput](training/factored_64_128_15/samples_per_sec.png) / [lr](training/factored_64_128_15/learning_rate.png) |

| 指标 | 含义 | 读图方式 |
|---|---|---|
| total loss | 训练总 loss | 看整体是否继续下降。 |
| val loss | 验证集 loss | 用于判断泛化和是否过拟合。 |
| per-horizon MSE | 每个 future horizon 的 MSE | 曲线太接近时需检查 horizon 是否正确区分。 |
| latent std / norm | latent 分布尺度 | std 接近 1、norm 稳定通常更健康。 |
| samples_per_sec / learning_rate | 训练效率和学习率变化 | 二者已分图，避免尺度混淆。 |

## 6. 结果说明

- PushT 上 Residual 15 的成功率均值为 90.0%，Original 15 为 88.8%，差距为 1.2 个百分点。
- Reacher 上 Residual 15 的成功率均值为 57.6%，Original 15 为 62.0%，Residual 未带来改善。
- PushT Factored 两个版本均低于 Original/Residual，其中 96/96 明显优于 64/128。
- PushT latent prediction 中，Original 的 normal MSE 更低、diagonal gap 更大，说明 horizon alignment 更清晰。
- 两个 PushT prediction report 都给出 `mean cosine > 0.99` warning；虽然 z_pred 和 z_target 不是完全相同，但仍应在后续分析中持续检查 target leakage 或 latent 读取路径。
