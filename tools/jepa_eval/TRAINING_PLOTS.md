# JEPA Training Plot Guide

本文档解释 `outputs/jepa_eval/training/` 下训练监控图片的含义。  
这些图由 `tools/jepa_eval/plot_training_curves.py` 从 `metrics.csv` 或日志中生成。

## total_loss.png

总损失曲线，包含训练损失；如果日志里有验证指标，也会同时画验证损失。

对应指标通常是：

```text
fit/loss_total
fit/loss
validate/loss_total
validate/loss
```

在当前 LeWM 训练里，总损失大致是：

```text
loss_total = pred_loss + sigreg_weight * sigreg_loss
```

怎么看：

- 正常情况：整体下降，允许有波动。
- train 和 val 都下降：训练正常。
- train 下降但 val 不降或上升：可能过拟合。
- 如果长期不下降：学习率、数据预处理、模型容量或 loss 权重可能有问题。
- 如果突然爆炸：可能是学习率过大、batch 异常、数据里有异常值，或 mixed precision 不稳定。

说明：

- 为了减少重复图，单独的 `train_val_loss.png` 已合并到 `total_loss.png`。
- train 和 val 都不降：模型还没学到有效预测，优先检查数据、action 对齐、学习率和 loss。
- val 大幅抖动：验证集太小、batch 分布不稳定，或某些 episode 边界样本影响较大。

## l1_mse_loss.png

未来 latent 预测误差曲线。

对应指标：

```text
fit/loss_future_l1
fit/loss_mse
validate/loss_future_l1
validate/loss_mse
```

含义：

- `loss_mse`：预测 latent 和目标 latent 的均方误差。
- `loss_future_l1`：预测 latent 和目标 latent 的平均绝对误差。
- `pred_loss`：训练代码里实际参与总 loss 的原始预测项；当前实现通常和 `loss_mse` 数值相同或非常接近，因此不再单独画，避免曲线重复。

怎么看：

- MSE 对大误差更敏感，L1 更稳定。
- 如果 MSE 很高但 L1 不高：说明少数样本或维度误差很大。
- 如果 L1/MSE 都长期不降：future latent 预测没有学好。

## cosine_loss.png

预测 latent 和目标 latent 的方向误差。

对应指标：

```text
fit/loss_future_cos
validate/loss_future_cos
```

当前定义：

```text
loss_future_cos = 1 - cosine_similarity(z_pred, z_target)
```

怎么看：

- 越接近 0，说明预测 latent 方向越接近目标 latent。
- 如果 MSE 下降但 cosine loss 不下降：模型可能只学到尺度变化，没有学到方向结构。
- 如果 cosine loss 下降但 MSE 不下降：方向对了，但 latent norm 或尺度仍不匹配。

## per_horizon_loss.png

不同 future step 的预测误差。

对应指标：

```text
fit/loss_mse_horizon_01
fit/loss_mse_horizon_02
...
validate/loss_mse_horizon_01
...
```

含义：

模型如果预测多个未来步，这张图显示每个未来 horizon 的 MSE。

怎么看：

- 正常情况：越远的 horizon 通常越难，loss 可能越高。
- 如果某个 horizon 异常高：可能是 target slicing 或 action/observation 时间对齐有问题。
- 当前配置如果 `wm.num_preds=1`，通常只会有 `horizon_01`。

## latent_std.png

latent 表征的整体标准差。

对应指标：

```text
fit/latent_std
validate/latent_std
```

含义：

统计 batch 内所有 latent 维度的标准差，用于观察表征是否塌缩或爆炸。

怎么看：

- 太接近 0：latent 可能塌缩，所有样本表征很像。
- 持续变得很大：latent 可能发散。
- 稳定在一个合理范围：通常更健康。

## active_dimensions.png

活跃 latent 维度数量。

对应指标：

```text
fit/active_dim_count
validate/active_dim_count
```

当前定义：

```text
active_dim_count = number of latent dimensions whose std > threshold
```

默认阈值为：

```text
1e-2
```

怎么看：

- 数值太低：大量 latent 维度没有变化，可能表征塌缩。
- 数值逐渐上升并稳定：模型在使用更多 latent 维度。
- 数值剧烈震荡：训练不稳定或 batch 分布差异大。

## latent_norm.png

latent 向量范数。

对应指标：

```text
fit/latent_norm
validate/latent_norm
```

含义：

统计 latent 向量的平均 L2 norm。

怎么看：

- norm 过小：可能 latent 被压到接近 0。
- norm 持续变大：可能表征尺度发散。
- norm 稳定：通常说明 latent 尺度健康。

## pairwise_cosine_stats.png

batch 内 latent 两两 cosine similarity 的统计曲线。

对应指标：

```text
fit/pairwise_cos_mean
fit/pairwise_cos_std
validate/pairwise_cos_mean
validate/pairwise_cos_std
```

含义：

随机取 batch 内 latent，计算样本之间两两 cosine similarity。

怎么看：

- mean 过高，接近 1：很多样本 latent 方向几乎一样，可能塌缩。
- mean 接近 0 且 std 合理：latent 分布更分散。
- std 很低：样本间相似度过于一致，可能缺少区分度。

## pairwise_cosine_histogram.png

batch 内 pairwise cosine similarity 的直方图。

对应指标：

```text
fit/pairwise_cos_hist_bin_00
...
fit/pairwise_cos_hist_bin_19
```

含义：

显示样本之间 latent 方向相似度的分布。

怎么看：

- 大量集中在 1 附近：表征可能塌缩。
- 分布较宽：latent 更有区分度。
- 大量集中在 -1 附近：表征可能出现强反向结构，需要结合 loss 判断是否异常。

## throughput_lr.png

吞吐量和学习率曲线。

对应指标：

```text
fit/samples_per_sec
fit/learning_rate_0
hparams/lr_default_0
```

含义：

- `samples_per_sec`：训练吞吐量。
- `learning_rate_0` / `hparams/lr_default_0`：当前学习率。

怎么看：

- 吞吐量突然下降：可能是 dataloader、IO、GPU 利用率或 checkpoint 保存导致。
- 学习率应该符合 scheduler 预期。
- 如果 loss 异常，先对照学习率是否处在 warmup 或 decay 阶段。

## per_dataset_loss.png

不同数据源的 loss 曲线。

对应指标：

```text
fit/per_dataset_loss/<source>
validate/per_dataset_loss/<source>
```

只有当 batch 里存在以下字段之一时才会生成：

```text
dataset
dataset_id
source
source_id
```

怎么看：

- 某个 source loss 明显更高：该数据源更难，或数据分布/预处理不同。
- 如果当前训练只用单一 HDF5 数据集，可能不会生成这张图。

## batch_source_counts.png

每个 batch 中不同数据源的样本数量。

对应指标：

```text
fit/batch_source_counts/<source>
validate/batch_source_counts/<source>
```

用途：

- 检查多数据源训练时 batch 是否均衡。
- 如果某个 source 长期为 0，说明 dataloader 没有采到该数据源。

## 建议观察顺序

建议按这个顺序看：

1. `total_loss.png`
2. `l1_mse_loss.png`
3. `cosine_loss.png`
4. `latent_std.png`
5. `active_dimensions.png`
6. `latent_norm.png`
7. `pairwise_cosine_stats.png`
8. `pairwise_cosine_histogram.png`
9. `throughput_lr.png`

如果 loss 正常下降，但 latent 图异常，说明模型可能在用不健康的 latent 表征完成预测。  
如果 latent 图正常，但 loss 不下降，优先检查 action 条件、target slicing、数据预处理和 learning rate。
