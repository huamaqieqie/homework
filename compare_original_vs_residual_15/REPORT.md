# Original LeWM vs Residual LeWM 15 Epoch 测评报告

## 1. 核心指标总表

| 指标 | Original LeWM | Residual LeWM | 结果倾向 |
|---|---:|---:|---|
| Success rate (`budget=300`) | 88.0% | **92.0%** | Residual 高 4 个百分点 |
| Evaluation time / episode | 8.35s | **4.35s** | Residual 更快 |
| Open-loop MSE mean | **0.01070** | 0.01311 | Original 更低 |
| Mean cosine(z_pred, z_target) | **0.99399** | 0.99254 | Original 略高 |
| Diagonal gap | **0.20243** | 0.16525 | Original 时间对齐更清晰 |
| Top-1 horizon matching accuracy | **98.70%** | 96.65% | Original 更高 |
| Active dims z_pred | 192 / 192 | 192 / 192 | 二者均无明显坍塌 |
| Effective rank z_pred | **59.59** | 58.32 | Original 略高 |
| Zero-action MSE gap | **+0.23029** | +0.17629 | Original 更依赖正常 action |
| Shuffled-action MSE gap | **+0.36815** | +0.27431 | Original 更依赖正常 action |
| Zero-action cosine drop | **-0.12326** | -0.09658 | Original drop 更大 |
| Shuffled-action cosine drop | **-0.19520** | -0.14899 | Original drop 更大 |

说明：当前 success eval 已使用相同设置：`eval.num_eval=50`、`eval_budget=300`、`seed=0`。Residual 成功率更高，但 92% vs 88% 只相差 2 个 episode，仍需要多 seed 验证稳定性。Open-loop prediction、horizon alignment 和 action ablation 上，Original 仍更强。

---

## 2. Success Rate / Planning 指标

| 指标 | Original LeWM | Residual LeWM |
|---|---:|---:|
| Success rate | 88.0% | 92.0% |
| eval episodes | 50 | 50 |
| eval_budget | 300 | 300 |
| 成功数 | 44 / 50 | 46 / 50 |
| evaluation_time | 417.65s | 217.61s |
| evaluation_time / episode | 8.35s | 4.35s |
| 结果文件 | `original_lewm_15/pusht_lewm15_eval_budget300_seed0.txt` | `residual_15/pusht_residual15_eval_seed0.txt` |

说明：这组 success rate 已经是同 `eval_budget=300`、同 `num_eval=50`、同 `seed=0` 的公平比较。Residual 成功率高 4 个百分点，对应多成功 2 个 episode；同时 `evaluation_time / episode` 更低。由于当前只有一个 seed，这个差异还不能视为统计稳定结论。

---

## 3. Open-loop Prediction 指标

| 指标 | Original LeWM | Residual LeWM | 结果倾向 |
|---|---:|---:|---|
| Open-loop MSE mean | **0.01070** | 0.01311 | Original 更低 |
| Horizon-1 MSE | **0.01091** | 0.01348 | Original 更低 |
| Horizon-2 MSE | **0.01052** | 0.01230 | Original 更低 |
| Horizon-3 MSE | **0.01067** | 0.01356 | Original 更低 |
| Horizon-1 cosine | **0.99392** | 0.99245 | Original 更高 |
| Horizon-2 cosine | **0.99410** | 0.99299 | Original 更高 |
| Horizon-3 cosine | **0.99395** | 0.99218 | Original 更高 |
| Median cosine | **0.99715** | 0.99600 | Original 更高 |
| q95 cosine | 0.99930 | **0.99937** | 接近 |
| z_pred 与 z_target exact equal | false | false | 无直接泄漏证据 |
| z_pred 与 z_target allclose | false | false | 无直接泄漏证据 |

说明：修正 horizon label 后，三个 horizon 都能正确显示。Original 在每个 horizon 的 MSE / cosine 都优于 Residual。但在同 budget 的环境测评中 Residual 成功率更高，说明 open-loop latent prediction MSE 与最终 planning success 并不完全一致。

---

## 4. Prediction Alignment 可视化

### 4.1 Target-Pred Alignment Heatmap

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/target_pred_alignment_heatmap.png) | ![](residual_15/jepa_eval/prediction_viz/target_pred_alignment_heatmap.png) |

| 指标 | Original LeWM | Residual LeWM | 含义 |
|---|---:|---:|---|
| diagonal mean | **0.99399** | 0.99254 | 正确 horizon 的 pred-target cosine |
| off-diagonal mean | **0.79156** | 0.82729 | 错位 horizon 的相似度，越低越好 |
| diagonal gap | **0.20243** | 0.16525 | diagonal 与 off-diagonal 的差距，越大表示时间对齐越清晰 |
| top-1 horizon matching accuracy | **98.70%** | 96.65% | 预测 horizon 最匹配正确 target horizon 的比例 |

说明：二者都有明显对角线，说明都学到了 horizon 对齐。Original 的 off-diagonal 更低、diagonal gap 更大，时间区分更清晰。Residual 的 off-diagonal 更高，说明不同 horizon 的表示更相近，可能更偏向平滑预测。

### 4.2 Target-Pred Cosine vs Horizon

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png) | ![](residual_15/jepa_eval/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png) |

| Horizon | Original mean cosine | Residual mean cosine |
|---:|---:|---:|
| 1 | **0.99392** | 0.99245 |
| 2 | **0.99410** | 0.99299 |
| 3 | **0.99395** | 0.99218 |

说明：horizon 轴已修正，现在 CSV 中显示 `1, 2, 3`。Original 在三个 horizon 上均略高。

---

## 5. Action-conditioned / Ablation 分析

### 5.1 Normal vs Zero / Shuffled Action

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/condition_ablation.png) | ![](residual_15/jepa_eval/prediction_viz/condition_ablation.png) |

| 模型 | Condition | MSE | Cosine |
|---|---|---:|---:|
| Original | normal | **0.01070** | **0.99399** |
| Original | zero action | 0.24099 | 0.87073 |
| Original | shuffled action | 0.37885 | 0.79879 |
| Residual | normal | 0.01311 | 0.99254 |
| Residual | zero action | **0.18940** | **0.89596** |
| Residual | shuffled action | **0.28743** | **0.84355** |

| Gap 指标 | Original LeWM | Residual LeWM | 解释 |
|---|---:|---:|---|
| zero-action MSE gap | **+0.23029** | +0.17629 | 去掉动作后 MSE 上升幅度 |
| shuffled-action MSE gap | **+0.36815** | +0.27431 | 打乱动作后 MSE 上升幅度 |
| zero-action cosine drop | **-0.12326** | -0.09658 | 去掉动作后 cosine 下降幅度 |
| shuffled-action cosine drop | **-0.19520** | -0.14899 | 打乱动作后 cosine 下降幅度 |

说明：两个模型都明显使用了 action，因为 zero/shuffled action 会显著恶化预测。Original 的 gap 更大，说明它对正确 action 更敏感；Residual 即使动作被 zero/shuffled，退化幅度较小，可能更依赖当前 latent 的 identity / smoothness。

### 5.2 Action Norm vs Prediction

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/action_norm_vs_cosine_scatter.png) | ![](residual_15/jepa_eval/prediction_viz/action_norm_vs_cosine_scatter.png) |
| ![](original_lewm_15/jepa_eval/prediction_viz/action_norm_bin_vs_mse_boxplot.png) | ![](residual_15/jepa_eval/prediction_viz/action_norm_bin_vs_mse_boxplot.png) |
| ![](original_lewm_15/jepa_eval/prediction_viz/action_norm_bin_vs_cosine_boxplot.png) | ![](residual_15/jepa_eval/prediction_viz/action_norm_bin_vs_cosine_boxplot.png) |

说明：这组图用于观察动作幅度是否影响预测误差。结合 ablation，两个模型都不是完全忽略 action。

### 5.3 Delta-z PCA by Action Norm

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/delta_z_pca_by_action_norm.png) | ![](residual_15/jepa_eval/latent_viz/delta_z_pca_by_action_norm.png) |

说明：该图观察 `z_pred - z_context` 是否随 action norm 呈现结构变化。Residual 理论上更强调 delta_z，因此这张图对 Residual 尤其重要。

---

## 6. Latent Collapse / Diversity 可视化

### 6.1 Active Dimension Count

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/active_dimension_count.png) | ![](residual_15/jepa_eval/latent_viz/active_dimension_count.png) |

| 指标 | Original z_pred | Residual z_pred |
|---|---:|---:|
| active dims | 192 / 192 | 192 / 192 |
| effective rank | **59.59** | 58.32 |
| participation ratio | **54.41** | 52.93 |
| top10 explained variance ratio | **0.2570** | 0.2657 |
| top50 explained variance ratio | **0.9115** | 0.9214 |

说明：两个模型都没有 active dimension collapse。Residual 的 effective rank 略低、top-k explained variance 略高，表示能量略更集中，但差距不大。

### 6.2 Pairwise Cosine Histogram

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/pairwise_cosine_histogram.png) | ![](residual_15/jepa_eval/latent_viz/pairwise_cosine_histogram.png) |

| 指标 | Original z_pred | Residual z_pred |
|---|---:|---:|
| pairwise cosine mean | 0.00050 | 0.00062 |
| pairwise cosine median | -0.02300 | -0.01585 |
| pairwise cosine std | 0.13177 | 0.13362 |
| pairwise cosine q95 | 0.23568 | **0.23022** |

说明：两个模型的 pairwise cosine 都集中在 0 附近，没有大面积接近 1，说明 batch 内 latent 仍有区分度。Residual 与 Original 在 diversity 上差异较小。

### 6.3 Covariance Spectrum / Explained Variance

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/covariance_eigenvalue_spectrum.png) | ![](residual_15/jepa_eval/latent_viz/covariance_eigenvalue_spectrum.png) |
| ![](original_lewm_15/jepa_eval/latent_viz/cumulative_explained_variance.png) | ![](residual_15/jepa_eval/latent_viz/cumulative_explained_variance.png) |

说明：Residual 的 top10/top50 explained variance 略高，effective rank 略低，说明 Residual 表示略更集中。当前差距不足以说明坍塌。

---

## 7. Target-Pred Latent Alignment

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/target_pred_latent_alignment_global.png) | ![](residual_15/jepa_eval/latent_viz/target_pred_latent_alignment_global.png) |
| ![](original_lewm_15/jepa_eval/latent_viz/target_pred_latent_alignment.png) | ![](residual_15/jepa_eval/latent_viz/target_pred_latent_alignment.png) |

说明：全局 target-pred PCA alignment 直观看预测点和目标点在同一个 PCA 空间中的偏差。结合 MSE 表格看，Original 的点对整体更接近；Residual open-loop 更差，但同 budget 环境测评的 success rate 更高。

---

## 8. 主要结论

| 结论项 | 判断 |
|---|---|
| Success rate | 同 `eval_budget=300` 下，Residual 为 92%，Original 为 88%，Residual 高 4 个百分点 |
| Evaluation time | 同 50 episodes 下，Residual 总时间 217.61s，Original 总时间 417.65s |
| Open-loop MSE | Original 更低，说明原版 latent prediction 更贴近 target |
| Prediction cosine | Original 在三个 horizon 上均略高 |
| Horizon alignment | Original 的 diagonal gap 和 top-1 matching 更高，时间对齐更清楚 |
| Action usage | 两者都使用 action；Original 的 shuffled/zero action gap 更大 |
| Latent collapse | 二者均无明显 collapse，active dims 均为 192/192 |
| Latent diversity | 二者 pairwise cosine 都健康，差距较小 |

综合判断：**在当前单 seed、公平 `eval_budget=300` 的环境测评中，Residual LeWM 的 success rate 高于 Original LeWM，并且 evaluation time 更低。** 但在 open-loop latent prediction、horizon alignment 和 action sensitivity 上，Original 更好。这个结果说明 Residual 的收益更可能来自更平滑或更适合 planner 的 rollout，而不是更精确的 one-step / short-horizon latent prediction。由于 success 差距只有 2 个 episode，后续必须补多 seed 才能判断是否稳定。

