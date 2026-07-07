# Original LeWM vs Residual LeWM 15 Epoch 测评报告

## 1. 实验对象

| 项目 | Original LeWM | Residual LeWM |
|---|---:|---:|
| Checkpoint | `lewm_15/weights_epoch_15.pt` | `lewm_residual_15/weights_epoch_15.pt` |
| Latent 样本数 | 1024 | 1024 |
| z_context shape | `[1024, 3, 192]` | `[1024, 3, 192]` |
| z_target shape | `[1024, 3, 192]` | `[1024, 3, 192]` |
| z_pred shape | `[1024, 3, 192]` | `[1024, 3, 192]` |
| detected horizons | 3 | 3 |
| latent dim | 192 | 192 |

---

## 2. 核心指标总表

| 指标 | Original LeWM | Residual LeWM | 结果倾向 |
|---|---:|---:|---|
| Success rate | 88.0% | **92.0%** | Residual 更高 |
| Open-loop MSE mean | **0.01070** | 0.01311 | Original 更低 |
| Horizon-1 MSE | **0.01091** | 0.01348 | Original 更低 |
| Horizon-2 MSE | **0.01052** | 0.01230 | Original 更低 |
| Horizon-3 MSE | **0.01067** | 0.01356 | Original 更低 |
| Mean cosine(z_pred, z_target) | **0.99399** | 0.99254 | Original 略高 |
| Median cosine | **0.99715** | 0.99600 | Original 略高 |
| Alignment diagonal mean | **0.99399** | 0.99254 | Original 略高 |
| Alignment off-diagonal mean | **0.79156** | 0.82729 | Original 更低 |
| Diagonal gap | **0.20243** | 0.16525 | Original 时间对齐更清晰 |
| Top-1 horizon matching accuracy | **98.70%** | 96.61% | Original 更高 |
| Active dims z_pred | 192 / 192 | 192 / 192 | 二者均无明显坍塌 |
| Effective rank z_pred | **59.59** | 58.32 | Original 略高 |
| Pairwise cosine mean z_pred | 0.00050 | 0.00062 | 接近 |
| Pairwise cosine std z_pred | 0.13177 | 0.13362 | 接近 |
| Pairwise cosine q95 z_pred | 0.23568 | **0.23022** | Residual 略低 |

---

## 3. Success Rate / Planning 指标

| 指标 | Original LeWM | Residual LeWM |
|---|---:|---:|
| Success rate | 88.0% | **92.0%** |
| eval episodes | 50 | 50 |
| 成功数 | 44 / 50 | **46 / 50** |
| CEM solve time 记录条数 | 4 | 12 |
| 日志中 CEM solve time 总和 | 143.45s | 112.48s |
| 日志中 CEM solve time 均值 | 35.86s | 9.37s |
| 日志文件 | `logs/pusht_lewm15_eval_seed0.log` | `logs/pusht_residual15_eval_seed0.log` |

说明：这里的 CEM solve time 只来自日志中实际打印出的 `CEM solve time` 行，两个模型的记录条数不同，因此不能作为严格 planning time 对比。正式 planning time 应优先使用 `eval.py` 写出的 `evaluation_time / eval.num_eval`。Success rate 是当前更可靠的最终任务指标。Residual 在 50 episode 上高出 4 个百分点。

---

## 4. Open-loop Prediction 指标

| 指标 | Original LeWM | Residual LeWM |
|---|---:|---:|
| Open-loop MSE mean | **0.01070** | 0.01311 |
| Horizon-1 MSE | **0.01091** | 0.01348 |
| Horizon-2 MSE | **0.01052** | 0.01230 |
| Horizon-3 MSE | **0.01067** | 0.01356 |
| Mean cosine | **0.99399** | 0.99254 |
| Median cosine | **0.99715** | 0.99600 |
| q95 cosine | 0.99930 | **0.99937** |
| z_pred 与 z_target exact equal | false | false |
| z_pred 与 z_target allclose | false | false |
| max abs diff | 1.94976 | 2.10156 |

说明：Original 在 open-loop MSE 和 mean cosine 上更优，说明它的 latent one-step / short-horizon 预测更接近 target。Residual 虽然 open-loop 指标略差，但最终 success rate 更高，说明 planning 结果不完全由 open-loop MSE 决定。

---

## 5. Prediction Alignment 可视化

### 5.1 Target-Pred Alignment Heatmap

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/target_pred_alignment_heatmap.png) | ![](residual_15/jepa_eval/prediction_viz/target_pred_alignment_heatmap.png) |

| 指标 | Original LeWM | Residual LeWM | 含义 |
|---|---:|---:|---|
| diagonal mean | **0.99399** | 0.99254 | 正确 horizon 的 pred-target cosine |
| off-diagonal mean | **0.79156** | 0.82729 | 错位 horizon 的相似度，越低越好 |
| diagonal gap | **0.20243** | 0.16525 | diagonal 与 off-diagonal 的差距，越大表示时间对齐越清晰 |
| top-1 horizon matching accuracy | **98.70%** | 96.61% | 预测 horizon 最匹配正确 target horizon 的比例 |

说明：二者都有明显对角线，说明都学到了 horizon 对齐。Original 的 off-diagonal 更低、diagonal gap 更大，因此时间区分更清晰。Residual 的 off-diagonal 更高，表示不同 horizon latent 更相似，可能更接近平滑/identity 风格预测。

### 5.2 Target-Pred Cosine vs Horizon

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png) | ![](residual_15/jepa_eval/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png) |

注意：旧版图中曾出现 CSV 的 `horizon` 列全为 `1` 的问题，这是因为导出文件中的 `future_horizon_index` 存储的是固定预测 offset。已修正绘图脚本：当检测到多个 horizon 但标签全相同时，会跳过该字段，改用 `target_time_index` 或默认 `1..H`。同步新数据后需要重新运行 prediction 可视化并更新本节图片。

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

说明：两个模型都没有 active dimension collapse。Residual 的 effective rank 和 participation ratio 略低，top-k explained variance 略高，说明 Residual 的表示能量略更集中，但差距不大。

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

说明：两个模型的 pairwise cosine 都集中在 0 附近，没有大面积接近 1，说明 batch 内 latent 仍有较好区分度。Residual 与 Original 在 diversity 上差异较小。

### 6.3 Covariance Spectrum / Explained Variance

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/covariance_eigenvalue_spectrum.png) | ![](residual_15/jepa_eval/latent_viz/covariance_eigenvalue_spectrum.png) |
| ![](original_lewm_15/jepa_eval/latent_viz/cumulative_explained_variance.png) | ![](residual_15/jepa_eval/latent_viz/cumulative_explained_variance.png) |

说明：Residual 的 top10/top50 explained variance 略高，effective rank 略低，说明 Residual 表示略更集中。当前差距不大，不能单独说明坍塌。

---

## 7. Action-conditioned 分析

### 7.1 Action Norm vs Cosine

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/action_norm_vs_cosine_scatter.png) | ![](residual_15/jepa_eval/prediction_viz/action_norm_vs_cosine_scatter.png) |

### 7.2 Action Norm Bin vs MSE / Cosine

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/prediction_viz/action_norm_bin_vs_mse_boxplot.png) | ![](residual_15/jepa_eval/prediction_viz/action_norm_bin_vs_mse_boxplot.png) |
| ![](original_lewm_15/jepa_eval/prediction_viz/action_norm_bin_vs_cosine_boxplot.png) | ![](residual_15/jepa_eval/prediction_viz/action_norm_bin_vs_cosine_boxplot.png) |

说明：这组图用于判断动作幅度是否影响预测误差。旧数据没有导出 shuffled/zero action 的 ablation latent，因此旧报告无法直接计算 action shuffle gap。当前导出脚本已新增 `--export-action-ablations`，同步新数据后会额外生成 `z_pred_action_shuffled` 和 `z_pred_action_zero`，用于计算 condition ablation。

### 7.3 Delta-z PCA by Action Norm

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/delta_z_pca_by_action_norm.png) | ![](residual_15/jepa_eval/latent_viz/delta_z_pca_by_action_norm.png) |

说明：该图观察 `z_pred - z_context` 是否随 action norm 呈现结构变化。Residual 模式理论上更强调 delta_z，因此这张图对 residual 尤其重要。

---

## 8. Target-Pred Latent Alignment

| Original LeWM | Residual LeWM |
|---|---|
| ![](original_lewm_15/jepa_eval/latent_viz/target_pred_latent_alignment_global.png) | ![](residual_15/jepa_eval/latent_viz/target_pred_latent_alignment_global.png) |
| ![](original_lewm_15/jepa_eval/latent_viz/target_pred_latent_alignment.png) | ![](residual_15/jepa_eval/latent_viz/target_pred_latent_alignment.png) |

说明：全局 target-pred PCA alignment 直观看预测点和目标点在同一个 PCA 空间中的偏差。结合 MSE 表格看，Original 的点对整体更接近；Residual 虽然 open-loop 更差，但在环境成功率上更高。

---

## 9. 主要结论

| 结论项 | 判断 |
|---|---|
| Success rate | Residual 15 epoch 为 92%，Original 15 epoch 为 88%，Residual 更好 |
| Open-loop MSE | Original 更低，说明原版 latent prediction 更贴近 target |
| Prediction cosine | Original 略高 |
| Horizon alignment | Original 的 diagonal gap 和 top-1 matching 更高，时间对齐更清楚 |
| Latent collapse | 二者均无明显 collapse，active dims 均为 192/192 |
| Latent diversity | 二者 pairwise cosine 都健康，差距较小 |
| Action-conditioned 证据 | 当前缺少 shuffled/zero action ablation，不能严格判断 action 使用强度 |

综合判断：**Residual LeWM 在最终任务成功率上优于 Original LeWM，但在 open-loop latent prediction 指标上不如 Original。** 这说明 residual 改动可能让 planner 使用的 latent rollout 更适合任务执行，但它没有直接改善短 horizon 的 MSE/cosine。后续需要通过 action ablation、更多 seed、以及完整 planning time 统计确认该提升是否稳定。

---

## 10. 后续建议

1. 补跑至少 3 个 seed 的 success eval，避免单 seed 偶然性。
2. 使用新版 `visualize_prediction.py` 重新生成 horizon 曲线，确认 x 轴为 `1, 2, 3`。
3. 使用新版 `export_latents.py --export-action-ablations` 重新导出 latent，计算 shuffled / zero action 的 condition ablation。
4. 正式 planning time 使用 `evaluation_time / eval.num_eval`，不要用不完整 CEM 日志行替代。
5. 报告中同时保留 success rate 和 open-loop MSE，避免只根据一个指标判断模型。
6. 如果 residual 多 seed 仍提升 success rate，可以继续推进 Factored Latent；否则先分析 residual 是否过度 identity。
