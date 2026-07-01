# JEPA Viz Guide

本文档统一说明 `tools/jepa_viz/` 生成的训练曲线、latent 导出文件、latent 可视化结果和 prediction 能力可视化结果。  
运行命令集中放在 `COMMANDS.md`。

默认输出目录：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/
```

## 1. Training Plots

训练曲线默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/training/
```

这些图由 `tools/jepa_viz/plot_training_curves.py` 从 `metrics.csv`、`jsonl` 或文本日志中生成。

### train_total_loss.png / val_loss.png / total_loss*.png

总损失曲线会拆成训练和验证图，同时保留合并图、log-scale 图和后期 zoom-in 图。

常见指标：

```text
fit/loss_total
fit/loss
validate/loss_total
validate/loss
```

当前 LeWM 训练里的总损失大致是：

```text
loss_total = pred_loss + sigreg_weight * sigreg_loss
```

输出：

```text
train_total_loss.png
val_loss.png
total_loss.png
total_loss_log.png
total_loss_zoom_step_ge_10000.png
```

怎么看：

- 正常情况：整体下降，允许有波动。
- train 和 val 都下降：训练正常。
- train 下降但 val 不降或上升：可能过拟合。
- train 和 val 都不降：优先检查数据、action 对齐、学习率和 loss。
- 突然爆炸：可能是学习率过大、batch 异常、数据异常值或 mixed precision 不稳定。

如果 `fit/loss_total` 和 `fit/loss` 完全重复，脚本只保留一个，并在 `training_plot_summary.json` 中说明。best val loss 对应 step 也会写入 summary。

### l1_mse_loss.png

未来 latent 预测误差曲线。

常见指标：

```text
fit/loss_future_l1
fit/loss_mse
validate/loss_future_l1
validate/loss_mse
```

含义：

- `loss_mse`：预测 latent 和目标 latent 的均方误差。
- `loss_future_l1`：预测 latent 和目标 latent 的平均绝对误差。

怎么看：

- MSE 对大误差更敏感，L1 更稳定。
- MSE 很高但 L1 不高：说明少数样本或维度误差很大。
- L1/MSE 都长期不降：future latent 预测没有学好。

### cosine_loss.png

预测 latent 和目标 latent 的方向误差。

常见指标：

```text
fit/loss_future_cos
validate/loss_future_cos
```

定义：

```text
loss_future_cos = 1 - cosine_similarity(z_pred, z_target)
```

怎么看：

- 越接近 0，说明预测 latent 方向越接近目标 latent。
- MSE 下降但 cosine loss 不下降：可能只学到尺度变化，没有学到方向结构。
- cosine loss 下降但 MSE 不下降：方向对了，但 latent norm 或尺度仍不匹配。

### train_per_horizon_mse.png / val_epoch_per_horizon_mse.png / val_step_per_horizon_mse.png

不同 future step 的预测误差。

常见指标：

```text
fit/loss_mse_horizon_01
fit/loss_mse_horizon_02
validate/loss_mse_horizon_01
```

输出：

```text
train_per_horizon_mse.png
train_per_horizon_mse_zoom_step_ge_10000.png
val_epoch_per_horizon_mse.png
val_epoch_per_horizon_mse_zoom_step_ge_10000.png
val_step_per_horizon_mse.png
val_step_per_horizon_mse_zoom_step_ge_10000.png
```

怎么看：

- 越远的 horizon 通常越难，loss 可能越高。
- 某个 horizon 异常高：可能是 target slicing 或 action/observation 时间对齐有问题。
- 当前配置如果 `wm.num_preds=1`，通常只会有 `horizon_01`。
- 每个 horizon 的 final、best、last_1k mean 会写入 `training_plot_summary.json`。
- 如果多个 horizon 曲线过于接近，summary 中会输出 warning，提醒检查 horizon 维度是否正确区分。

### latent_std.png

latent 表征的整体标准差。若日志里有 `z_context`、`z_pred`、`z_target` 分开的指标，会在同一张图里分别显示。

常见指标：

```text
fit/latent_std
validate/latent_std
```

怎么看：

- 太接近 0：latent 可能塌缩。
- 持续变大：latent 可能发散。
- 稳定在合理范围：通常更健康。
- 图中会画 `y=1.0` 参考线。
- 图例会标出 final value 和 last_1k mean。
- `training_plot_summary.json` 会记录 final、best、last_1k mean/std。

### active_dimensions.png

活跃 latent 维度数量。

常见指标：

```text
fit/active_dim_count
validate/active_dim_count
```

定义：

```text
active_dim_count = number of latent dimensions whose std > threshold
```

默认阈值是 `1e-2`。

怎么看：

- 数值太低：大量 latent 维度没有变化，可能表征塌缩。
- 数值逐渐上升并稳定：模型在使用更多 latent 维度。
- 数值剧烈震荡：训练不稳定或 batch 分布差异大。

### latent_norm.png

latent 向量范数。

常见指标：

```text
fit/latent_norm
validate/latent_norm
```

怎么看：

- norm 过小：latent 可能接近 0。
- norm 持续变大：表征尺度可能发散。
- norm 稳定：latent 尺度通常较健康。
- 若日志能识别 `latent_dim` / `hidden_size` / `embed_dim`，图中会画 `sqrt(latent_dim)` 参考线。
- 图例会标出 final value 和 last_1k mean。
- 如果日志里有 `z_context`、`z_pred`、`z_target` 分开的 norm，会分开画。

### pairwise_cosine_mean.png / pairwise_cosine_std.png

batch 内 latent 两两 cosine similarity 的统计曲线。

常见指标：

```text
fit/pairwise_cos_mean
fit/pairwise_cos_std
validate/pairwise_cos_mean
validate/pairwise_cos_std
```

怎么看：

- mean 接近 1：很多样本 latent 方向几乎一样，可能塌缩。
- mean 接近 0 且 std 合理：latent 分布更分散。
- std 很低：样本间相似度过于一致，可能缺少区分度。
- mean 和 std 分成两张图，避免尺度混在一起。
- 最后 1000 step 的 mean/std 会写入 `training_plot_summary.json`。

### pairwise_cosine_histogram.png

batch 内 pairwise cosine similarity 的直方图。

常见指标：

```text
fit/pairwise_cos_hist_bin_00
...
fit/pairwise_cos_hist_bin_19
```

怎么看：

- 大量集中在 1 附近：表征可能塌缩。
- 分布较宽：latent 更有区分度。
- 大量集中在 -1 附近：可能出现强反向结构，需要结合 loss 判断。
- 直方图 y 轴使用 fraction。
- 图中会标出 mean、median、q05、q95。

### samples_per_sec.png / learning_rate.png

吞吐量和学习率分开画，避免 learning rate 被吞吐量尺度压扁。

常见指标：

```text
fit/samples_per_sec
fit/learning_rate_0
hparams/lr_default_0
```

怎么看：

- 吞吐量突然下降：可能是 dataloader、IO、GPU 利用率或 checkpoint 保存导致。
- 学习率应该符合 scheduler 预期。
- loss 异常时，先对照学习率是否处在 warmup 或 decay 阶段。
- `samples_per_sec.png` 同时显示 raw 曲线和 moving average。
- `learning_rate.png` 单独使用 log y 轴。
- 如果吞吐量出现明显跳变，`training_plot_summary.json` 会输出 warning 和对应 step。

### per_dataset_loss.png

不同数据源的 loss 曲线。

常见指标：

```text
fit/per_dataset_loss/<source>
validate/per_dataset_loss/<source>
```

只有 batch 里存在下面字段之一时才会生成：

```text
dataset
dataset_id
source
source_id
```

怎么看：

- 某个 source loss 明显更高：该数据源更难，或数据分布/预处理不同。
- 单一 HDF5 数据集通常不会生成这张图。

### batch_source_counts.png

每个 batch 中不同数据源的样本数量。

常见指标：

```text
fit/batch_source_counts/<source>
validate/batch_source_counts/<source>
```

用途：

- 检查多数据源训练时 batch 是否均衡。
- 某个 source 长期为 0：dataloader 没有采到该数据源。

### 建议观察顺序

```text
total_loss.png
train_total_loss.png
val_loss.png
total_loss_log.png
total_loss_zoom_step_ge_10000.png
l1_mse_loss.png
cosine_loss.png
latent_std.png
active_dimensions.png
latent_norm.png
pairwise_cosine_mean.png
pairwise_cosine_std.png
pairwise_cosine_histogram.png
train_per_horizon_mse.png
val_epoch_per_horizon_mse.png
val_step_per_horizon_mse.png
samples_per_sec.png
learning_rate.png
training_plot_summary.json
```

如果 loss 正常下降但 latent 图异常，说明模型可能在用不健康的 latent 表征完成预测。  
如果 latent 图正常但 loss 不下降，优先检查 action 条件、target slicing、数据预处理和 learning rate。

## 2. Latent Export

latent 导出默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/latents/
```

主要文件：

```text
latents.npz
metadata.jsonl
summary.json
```

### latents.npz / latents.pt

核心字段：

```text
z_context
z_target
z_pred
z_all
action_condition
future_horizon_index
context_time_index
target_time_index
```

如果 batch 中存在这些字段，也会尽量保存：

```text
action
state
proprio
observation
dataset / dataset_id / source / source_id
step_idx / episode_idx / ep_idx
task_id / object_id / category_id / label
success / is_success
mask_pos / mask_position
target_block_pos / target_block_position
```

### metadata.jsonl

每行对应一个 sample，通常包含：

```text
sample_id
export_index
split
dataset/source 信息，如果 batch 里有
frame/time/episode/task/object/category/success/mask 信息，如果 batch 里有
```

### summary.json

记录导出摘要，包括 checkpoint、输出路径、样本数量、latent shape、`history_size`、`num_preds`、missing/unexpected keys 等。

### Sanity Check

导出过程中会检查：

- `z_context`、`z_target`、`z_pred` 是否存在。
- latent 是否包含 NaN/Inf。
- `z_pred` 和 `z_target` batch size 是否一致。
- `z_pred` 和 `z_target` shape 是否一致。
- `metadata.jsonl` 行数是否等于 latent sample 数。

如果没有真实 checkpoint，需要先提供以下其中之一：

```text
Lightning .ckpt 文件
save_pretrained 生成的 .pt/.pth 文件
包含上述文件的 checkpoint 目录
```

## 3. Latent Visualization

latent 可视化默认读取：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/latents/
```

默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/latent_viz/
```

### PCA Scatter

每个 latent 类型各生成一张 PCA 2D scatter，`z_context`、`z_target`、`z_pred` 使用同一个 PCA basis，并使用统一的 xlim / ylim：

```text
pca_z_context_shared_by_<color>.png
pca_z_target_shared_by_<color>.png
pca_z_pred_shared_by_<color>.png
```

脚本会把 `[N, T, D]` 的 latent 展平成 `N*T` 个点。

支持通过 `--color-by` 选择颜色来源，例如：

```text
source
dataset
task
action
action_norm
action_0
action_1
object
success
time_index
horizon
prediction_error
prediction_cosine
step_idx
episode_idx
```

如果 `--color-by action`，并且 latent 文件中有 `action`，颜色使用 action vector 的 L2 norm。

### UMAP Scatter

每个 latent 类型各生成一张 UMAP 2D scatter，并且三类 latent 使用同一个 UMAP fit：

```text
umap_z_context_shared_by_<color>.png
umap_z_target_shared_by_<color>.png
umap_z_pred_shared_by_<color>.png
```

如果服务器环境有 `umap-learn`，会使用 UMAP。  
如果没有，不会重复生成 fallback PCA；此时只保留 shared PCA 图。

### Target-Pred Latent Alignment

输出：

```text
target_pred_latent_alignment.png
target_pred_latent_alignment_global.png
```

这组图用于观察每个样本的 target latent 和 predicted latent 是否在同一个 PCA 空间中对齐。  
如果每个样本只有 target 和 pred 两个点，图不会再命名为 trajectory。

`target_pred_latent_alignment.png` 是若干样本的子图，每个子图包含：

- target 点。
- pred 点。
- target 到 pred 的灰色连线。
- 统一的 xlim / ylim，避免不同子图自动缩放造成视觉误导。
- 标题中的 `sample_id`、`cosine similarity` 和 `MSE`。
- 样本按 MSE 选出 best / median / worst，而不是只取前几个 export index。

`target_pred_latent_alignment_global.png` 会把所有样本的 target 和 pred 画在同一个 PCA 空间里，并用灰色线连接对应 pair。

如果 metadata 或 latent 数组中存在 horizon 信息，还会按 horizon 分组输出：

```text
target_pred_latent_alignment_<horizon_key>_<horizon>.png
target_pred_latent_alignment_global_<horizon_key>_<horizon>.png
```

支持识别的 horizon 字段包括：

```text
horizon
future_horizon
future_horizon_index
target_horizon
pred_horizon
horizon_idx
```

### Nearest Neighbor Retrieval

输出：

```text
nearest_neighbors.html
```

默认使用 `z_target` 做 query latent。每个 query 显示 top-k 最近样本的 metadata。  
如果 metadata 中有 `image_path` / `frame_path`，HTML 会显示图片；否则显示 sample id、episode、step、task、object、success 等标签。

### Collapse Diagnostics

输出：

```text
covariance_eigenvalue_spectrum.png
cumulative_explained_variance.png
pairwise_cosine_histogram.png
pairwise_cosine_same_vs_different_<label>_<latent>.png
feature_std_distribution.png
active_dimension_count.png
collapse_diagnostics_summary.json
```

用途：

- `covariance_eigenvalue_spectrum.png`：对比 `z_context`、`z_target`、`z_pred` 的协方差特征值谱。
- `cumulative_explained_variance.png`：累计 explained variance，用来观察 top-k 维度覆盖率。
- `pairwise_cosine_histogram.png`：对比三类 latent 的样本间方向相似度，支持 density y 轴，并标出 mean / median / q95。
- `pairwise_cosine_same_vs_different_<label>_<latent>.png`：如果 metadata 有 task / action / episode / source，可对比 same-label 与 different-label。
- `feature_std_distribution.png`：三类 latent 的 feature std overlay，并标出 min / median / max std 以及 active threshold。
- `active_dimension_count.png`：三类 latent 的 active / total 维度数。
- `collapse_diagnostics_summary.json`：包含 effective rank、participation ratio、top-10 / top-50 / top-100 explained variance ratio。

active dimension 支持两种阈值：

```text
--active-threshold               absolute threshold，例如 1e-2
--active-relative-threshold      relative threshold，按 max feature std 的比例计算
```

实际使用阈值为 absolute threshold 和 relative threshold 中较大的一个。

### Action-Conditioned Analysis

输出：

```text
delta_z_pca_by_action_norm.png
delta_z_pca_by_action_<component>.png
action_condition_ablation.png
action_condition_ablation_summary.json
```

`delta_z = z_pred - z_context`，用于观察 action-conditioned JEPA 的 latent 变化方向。

- `delta_z_pca_by_action_norm.png`：按 action norm 着色。
- `delta_z_pca_by_action_<component>.png`：按 action 各分量着色。
- `action_condition_ablation.png`：如果 latent 文件中存在 shuffled/zero action prediction，对比 normal / shuffled / zero 的 MSE 和 cosine。

### 常见输出清单

```text
pca_z_context_shared_by_*.png
pca_z_target_shared_by_*.png
pca_z_pred_shared_by_*.png
umap_z_context_shared_by_*.png
umap_z_target_shared_by_*.png
umap_z_pred_shared_by_*.png
target_pred_latent_alignment.png
target_pred_latent_alignment_global.png
delta_z_pca_by_action_norm.png
delta_z_pca_by_action_<component>.png
action_condition_ablation.png
nearest_neighbors.html
covariance_eigenvalue_spectrum.png
cumulative_explained_variance.png
pairwise_cosine_histogram.png
pairwise_cosine_same_vs_different_<label>_<latent>.png
feature_std_distribution.png
active_dimension_count.png
collapse_diagnostics_summary.json
visualization_summary.json
```

## 4. Prediction Visualization

prediction 能力可视化默认读取：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/latents/
```

默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/prediction_viz/
```

输入 latent 至少需要：

```text
z_context
z_target
z_pred
metadata.jsonl
```

### target_pred_cosine_vs_horizon_by_*.png

预测 latent 与目标 latent 在不同 future horizon 上的 cosine similarity。

横轴：

```text
future horizon
```

纵轴：

```text
cos(z_pred, z_target)
```

shape 处理规则：

```text
[N, D]       -> 1 个 horizon
[N, H, D]    -> 按 H 分别计算 cos(z_pred[:, h], z_target[:, h])
[N, T, H, D] -> T 作为 token/frame 维度，H 作为 horizon 维度，先对 token 聚合，再按 horizon 画图
```

支持通过 `--group-by` 按下面字段分组：

```text
source
dataset
task
action
object
success
```

如果 `--group-by action`，脚本会优先读取 latent 文件里的 `action` 数组，并按 action norm 做分桶。

多 horizon 时，每条曲线显示 mean ± std，或者通过 `--interval ci95` 显示 95% CI。  
只有一个 horizon 时，脚本会输出：

```text
target_pred_cosine_boxplot_by_*.png
```

而不是画只有一个点的 line plot。

怎么看：

- 越接近 1，说明预测 latent 和目标 latent 方向越一致。
- 越远 horizon 明显下降：长期预测更难，属于常见现象。
- 某个分组显著更低：该数据源、任务或动作区间更难，或者条件信息对齐有问题。

同时会输出对应的 csv：

```text
target_pred_cosine_vs_horizon_by_*.csv
```

### target_pred_alignment_heatmap.png

预测 horizon 与目标 horizon 的 pairwise cosine heatmap。  
如果 latent 是 `[N, T, H, D]`，脚本会先对 token/frame 维度聚合，heatmap 只表达 horizon 对齐关系。

行：

```text
pred horizon
```

列：

```text
target horizon
```

颜色：

```text
cosine similarity
```

怎么看：

- 对角线明显：第 k 个 prediction 更接近第 k 个 target，时间对应关系比较清楚。
- 非对角区域也很亮：模型可能混淆 future step，或者多个 target latent 很相似。
- 整体都偏低：prediction latent 与 target latent 对齐较差。

同时会输出矩阵：

```text
target_pred_alignment_heatmap.csv
target_pred_alignment_heatmap_metrics.json
```

metrics 文件包含：

```text
diagonal_mean
off_diagonal_mean
diagonal_gap
top1_horizon_matching_accuracy
```

其中 `diagonal_gap = diagonal_mean - off_diagonal_mean`。如果 diagonal 很高但 off-diagonal 也很高，这个值会更直观地暴露“整体都很像”的问题。

### action-conditioned prediction plots

如果 latent 中有 `action` 或 `action_condition`，会额外输出：

```text
action_norm_vs_cosine_scatter.png
action_norm_bin_vs_cosine_boxplot.png
action_norm_bin_vs_mse_boxplot.png
<action_component>_bin_vs_cosine_boxplot.png
<action_component>_bin_vs_mse_boxplot.png
```

如果 metadata 中有 `action_x` / `action_y` / `action_z` / `gripper`，会优先使用这些字段分组；否则会从 action vector 的前几个分量中读取。

### Shape and leakage checks

每次运行都会在 `prediction_report.md` / `prediction_report.json` 里记录：

```text
z_context shape
z_pred shape
z_target shape
detected num_horizons
detected latent_dim
```

同时会检查：

```text
z_pred 和 z_target 是否完全相同
z_pred 和 z_target 是否 allclose
mean cosine 是否 > 0.99
```

如果 mean cosine > 0.99，会输出 warning，提示检查 target leakage 或 tensor 读取错误。

### rollout_drift_curve.png

多步 rollout error 曲线。只有 latent 文件中包含 rollout prediction 和 rollout target 数组时才会生成。

脚本会寻找这些 prediction 字段：

```text
z_rollout_pred
rollout_pred
z_pred_rollout
multi_step_z_pred
```

以及这些 target 字段：

```text
z_rollout_target
rollout_target
z_target_rollout
multi_step_z_target
```

如果当前导出的 latent 没有这些字段，脚本会跳过，并在 `prediction_report.md` 中说明原因。

### condition_ablation.png

条件消融对比图。只有 latent 文件中已经导出不同 condition 设置下的 prediction 时才会生成。

支持字段包括：

```text
z_pred_condition_removed
z_pred_no_condition
z_pred_without_condition
z_pred_condition_shuffled
z_pred_shuffled_condition
z_pred_condition_replaced
z_pred_replaced_condition
```

图中会对比：

```text
normal condition
condition removed
condition shuffled
condition replaced
```

指标包括：

```text
prediction MSE
prediction cosine similarity
```

如果只有普通 `z_pred`，脚本会跳过这张图。当前工具不会为了画图重新运行模型做 ablation；它只分析已经导出的 latent。

### goal_distance.png

goal distance 图。只有 latent 文件中包含 goal latent 时才会生成。

支持字段：

```text
z_goal
goal_latent
goal_emb
```

图中比较：

```text
d(z_current, z_goal)
d(z_pred_after_action, z_goal)
```

默认用 `z_context` 的最后一个 token 作为 current latent，用 `z_pred` 的最后一个 token 作为 action 后预测 latent。

如果只有 goal image、没有 goal latent，脚本会跳过这张图，并在报告中说明需要先导出 goal latent。

### prediction_report.md / prediction_report.json

每次运行都会生成报告：

```text
prediction_report.md
prediction_report.json
```

报告会列出：

- 已生成的图和 csv。
- 因缺少 rollout / ablation / goal latent 而跳过的项目。
- 输入 latent 文件路径。
- latent 数组 shape。
