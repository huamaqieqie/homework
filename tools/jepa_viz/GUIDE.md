# JEPA Viz Guide

本文档统一说明 `tools/jepa_viz/` 生成的训练曲线、latent 导出文件和 latent 可视化结果。  
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

### total_loss.png

总损失曲线，包含训练损失；如果日志里有验证指标，也会同时画验证损失。

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

怎么看：

- 正常情况：整体下降，允许有波动。
- train 和 val 都下降：训练正常。
- train 下降但 val 不降或上升：可能过拟合。
- train 和 val 都不降：优先检查数据、action 对齐、学习率和 loss。
- 突然爆炸：可能是学习率过大、batch 异常、数据异常值或 mixed precision 不稳定。

说明：单独的 `train_val_loss.png` 已合并到 `total_loss.png`，避免重复图。

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

### per_horizon_loss.png

不同 future step 的预测误差。

常见指标：

```text
fit/loss_mse_horizon_01
fit/loss_mse_horizon_02
validate/loss_mse_horizon_01
```

怎么看：

- 越远的 horizon 通常越难，loss 可能越高。
- 某个 horizon 异常高：可能是 target slicing 或 action/observation 时间对齐有问题。
- 当前配置如果 `wm.num_preds=1`，通常只会有 `horizon_01`。

### latent_std.png

latent 表征的整体标准差。

常见指标：

```text
fit/latent_std
validate/latent_std
```

怎么看：

- 太接近 0：latent 可能塌缩。
- 持续变大：latent 可能发散。
- 稳定在合理范围：通常更健康。

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

### pairwise_cosine_stats.png

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

### throughput_lr.png

吞吐量和学习率曲线。

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
l1_mse_loss.png
cosine_loss.png
latent_std.png
active_dimensions.png
latent_norm.png
pairwise_cosine_stats.png
pairwise_cosine_histogram.png
throughput_lr.png
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

每个 latent 类型各生成一张 PCA 2D scatter：

```text
pca_z_context_by_<color>.png
pca_z_target_by_<color>.png
pca_z_pred_by_<color>.png
```

脚本会把 `[N, T, D]` 的 latent 展平成 `N*T` 个点。

支持通过 `--color-by` 选择颜色来源，例如：

```text
source
dataset
task
action
object
success
time_index
step_idx
episode_idx
```

如果 `--color-by action`，并且 latent 文件中有 `action`，颜色使用 action vector 的 L2 norm。

### UMAP Scatter

每个 latent 类型各生成一张 UMAP 2D scatter：

```text
umap_z_context_by_<color>.png
umap_z_target_by_<color>.png
umap_z_pred_by_<color>.png
```

如果服务器环境有 `umap-learn`，会使用 UMAP。  
如果没有，会自动 fallback 到 PCA，并输出：

```text
umap_fallback_pca_z_context_by_<color>.png
umap_fallback_pca_z_target_by_<color>.png
umap_fallback_pca_z_pred_by_<color>.png
```

### Latent Trajectory

输出：

```text
latent_trajectory.png
```

脚本优先按下面字段寻找 episode / sequence 分组：

```text
episode_idx
ep_idx
video_id
sequence_id
sample_id
```

然后按下面字段排序：

```text
step_idx
time_idx
frame_idx
export_index
```

图中会显示：

```text
target latent trajectory
pred latent trajectory
goal latent，如果 latents 中存在 z_goal / goal_latent / goal_emb
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
pairwise_cosine_histogram.png
feature_std_distribution.png
active_dimension_count.png
collapse_diagnostics_summary.json
```

用途：

- `covariance_eigenvalue_spectrum.png`：看 latent 是否只集中在少数主方向。
- `pairwise_cosine_histogram.png`：看样本间方向是否过度相似。
- `feature_std_distribution.png`：看各维度是否有足够变化。
- `active_dimension_count.png`：统计 std 大于 `1e-2` 的维度数量。

### 常见输出清单

```text
pca_z_context_by_*.png
pca_z_target_by_*.png
pca_z_pred_by_*.png
umap_z_context_by_*.png
umap_z_target_by_*.png
umap_z_pred_by_*.png
latent_trajectory.png
nearest_neighbors.html
covariance_eigenvalue_spectrum.png
pairwise_cosine_histogram.png
feature_std_distribution.png
active_dimension_count.png
collapse_diagnostics_summary.json
visualization_summary.json
```
