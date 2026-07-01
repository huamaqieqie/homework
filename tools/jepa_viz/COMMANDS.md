# JEPA Viz Commands

本文档是通用命令模板，适用于任何 JEPA / V-JEPA / action-conditioned JEPA 项目。

核心约定只有两个：

1. 训练曲线脚本需要一个训练日志文件，例如 `metrics.csv`、`jsonl` 或文本日志。
2. latent 可视化脚本需要一个 latent 目录，里面有 `latents.npz` 或 `latents.pt`，以及可选的 `metadata.jsonl`。

默认输出目录在工具目录下，并按日期时间自动分目录：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/
```

你可以通过修改下面这些变量，把工具用于任意模型、任意仓库、任意输出位置。

## 0. 通用路径变量

先设置你的项目路径和工具路径：

```bash
export PROJECT_ROOT=<你的项目根目录>
cd $PROJECT_ROOT

# 如果 jepa_viz 在当前项目里，通常这样写：
export JEPA_VIZ_DIR=$PROJECT_ROOT/tools/jepa_viz

# 如果 jepa_viz 被复制到别的地方，就改成实际路径：
# export JEPA_VIZ_DIR=/path/to/tools/jepa_viz
```

设置统一输出根目录。建议显式设置一个时间戳目录，便于区分不同运行：

```bash
export JEPA_VIZ_RUN_NAME=$(date +%Y%m%d_%H%M%S)
export JEPA_VIZ_OUTPUT_ROOT=$JEPA_VIZ_DIR/output/$JEPA_VIZ_RUN_NAME
```

如果你不想把结果写进代码目录，可以改成任意路径：

```bash
export JEPA_VIZ_OUTPUT_ROOT=<你的输出目录>/$JEPA_VIZ_RUN_NAME
```

例如：

```bash
export JEPA_VIZ_OUTPUT_ROOT=$PROJECT_ROOT/outputs/jepa_viz/$JEPA_VIZ_RUN_NAME
```

## 1. 训练曲线可视化

准备输入日志路径：

```bash
export TRAIN_LOG=<训练日志或metrics.csv路径>
export TRAINING_OUT=$JEPA_VIZ_OUTPUT_ROOT/training
```

一次性生成曲线：

```bash
MODE=plot \
TRAIN_LOG=$TRAIN_LOG \
TRAINING_OUT=$TRAINING_OUT \
bash $JEPA_VIZ_DIR/run_jepa_viz_template.sh
```

训练过程中持续刷新曲线：

```bash
MODE=watch \
TRAIN_LOG=$TRAIN_LOG \
TRAINING_OUT=$TRAINING_OUT \
WATCH_INTERVAL=60 \
bash $JEPA_VIZ_DIR/run_jepa_viz_template.sh
```

直接调用脚本：

```bash
python $JEPA_VIZ_DIR/plot_training_curves.py \
  --log $TRAIN_LOG \
  --out $TRAINING_OUT
```

输出目录：

```text
$TRAINING_OUT
```

## 2. Latent 可视化

如果你的模型已经能导出 latent，只需要准备一个目录：

```bash
export LATENT_DIR=<包含latents.npz或latents.pt的目录>
export LATENT_VIZ_OUT=$JEPA_VIZ_OUTPUT_ROOT/latent_viz
```

该目录建议包含：

```text
latents.npz 或 latents.pt
metadata.jsonl
```

其中 latent 文件至少包含：

```text
z_context
z_target
z_pred
```

运行可视化：

```bash
MODE=visualize_latents \
LATENT_DIR=$LATENT_DIR \
LATENT_VIZ_OUT=$LATENT_VIZ_OUT \
LATENT_COLOR_BY=action \
LATENT_MAX_POINTS=5000 \
LATENT_ALIGNMENT_COUNT=4 \
LATENT_ACTIVE_THRESHOLD=1e-2 \
LATENT_ACTIVE_RELATIVE_THRESHOLD=0.0 \
LATENT_PAIRWISE_DENSITY=1 \
LATENT_MAX_ACTION_COMPONENTS=8 \
LATENT_NN_QUERIES=8 \
LATENT_TOP_K=5 \
bash $JEPA_VIZ_DIR/run_jepa_viz_template.sh
```

最小测试版本：

```bash
MODE=visualize_latents \
LATENT_DIR=$LATENT_DIR \
LATENT_VIZ_OUT=$JEPA_VIZ_OUTPUT_ROOT/latent_viz_test \
LATENT_COLOR_BY=action \
LATENT_MAX_POINTS=512 \
LATENT_ALIGNMENT_COUNT=4 \
LATENT_ACTIVE_THRESHOLD=1e-2 \
LATENT_ACTIVE_RELATIVE_THRESHOLD=0.0 \
LATENT_PAIRWISE_DENSITY=1 \
LATENT_MAX_ACTION_COMPONENTS=8 \
LATENT_NN_QUERIES=2 \
LATENT_TOP_K=3 \
bash $JEPA_VIZ_DIR/run_jepa_viz_template.sh
```

直接调用脚本：

```bash
python $JEPA_VIZ_DIR/visualize_latents.py \
  --latent-dir $LATENT_DIR \
  --out $LATENT_VIZ_OUT \
  --color-by action \
  --alignment-count 4 \
  --active-threshold 1e-2 \
  --active-relative-threshold 0.0 \
  --pairwise-density \
  --max-action-components 8
```

常用 `--color-by`：

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

## 3. 可选：用当前仓库的导出适配器导出 latent

`export_latents.py` 是当前仓库提供的导出适配器。  
它适合“训练配置可以实例化模型和 dataloader”的项目。不同项目只需要替换这些路径和 override：

```bash
export CONFIG_PATH=<训练配置路径>
export CKPT_PATH=<checkpoint路径>
export DATASET_NAME=<数据集路径或名字>
export LATENT_OUT=$JEPA_VIZ_OUTPUT_ROOT/latents
```

通过模板运行：

```bash
MODE=export_latents \
LATENT_CONFIG=$CONFIG_PATH \
LATENT_CHECKPOINT=$CKPT_PATH \
LATENT_SPLIT=val \
LATENT_MAX_SAMPLES=1024 \
LATENT_OUT=$LATENT_OUT \
BATCH_SIZE=128 \
bash $JEPA_VIZ_DIR/run_jepa_viz_template.sh
```

如果你的项目需要 Hydra override，可以直接调用脚本，并把 override 放在最后：

```bash
python $JEPA_VIZ_DIR/export_latents.py \
  --config $CONFIG_PATH \
  --checkpoint $CKPT_PATH \
  --split val \
  --max-samples 1024 \
  --out $LATENT_OUT \
  --dataset $DATASET_NAME \
  <hydra_override_1> \
  <hydra_override_2>
```

示例：

```bash
python $JEPA_VIZ_DIR/export_latents.py \
  --config $CONFIG_PATH \
  --checkpoint $CKPT_PATH \
  --split val \
  --max-samples 1024 \
  --out $LATENT_OUT \
  --dataset $DATASET_NAME \
  <hydra_override>
```

如果你的模型不是当前导出适配器支持的结构，推荐自己导出标准格式：

```text
latents.npz
metadata.jsonl
```

然后直接使用第 2 节的 `visualize_latents.py`。

## 4. Prediction 能力可视化

准备输入 latent 目录：

```bash
export LATENT_DIR=<包含latents.npz或latents.pt的目录>
export PREDICTION_VIZ_OUT=$JEPA_VIZ_OUTPUT_ROOT/prediction_viz
```

运行 prediction 可视化：

```bash
MODE=visualize_prediction \
LATENT_DIR=$LATENT_DIR \
PREDICTION_VIZ_OUT=$PREDICTION_VIZ_OUT \
PREDICTION_GROUP_BY=action \
PREDICTION_MAX_GROUPS=8 \
PREDICTION_ACTION_BINS=4 \
PREDICTION_INTERVAL=std \
PREDICTION_HEATMAP_VMIN_QUANTILE=0.05 \
bash $JEPA_VIZ_DIR/run_jepa_viz_template.sh
```

直接调用脚本：

```bash
python $JEPA_VIZ_DIR/visualize_prediction.py \
  --latent-dir $LATENT_DIR \
  --out $PREDICTION_VIZ_OUT \
  --group-by action \
  --max-groups 8 \
  --action-bins 4 \
  --interval std \
  --heatmap-vmin-quantile 0.05
```

输出内容：

```text
target_pred_cosine_vs_horizon_by_*.png
target_pred_cosine_boxplot_by_*.png，如果只有一个 horizon
target_pred_cosine_vs_horizon_by_*.csv
target_pred_alignment_heatmap.png
target_pred_alignment_heatmap.csv
target_pred_alignment_heatmap_metrics.json
action_norm_vs_cosine_scatter.png
action_norm_bin_vs_cosine_boxplot.png
action_norm_bin_vs_mse_boxplot.png
<action_component>_bin_vs_cosine_boxplot.png
<action_component>_bin_vs_mse_boxplot.png
prediction_report.md
prediction_report.json
```

如果 latent 文件中没有 rollout、ablation 或 goal latent 对应数组，脚本会跳过这些图，并在 `prediction_report.md` 中说明原因。

## 5. 一步导出并可视化

适用于当前导出适配器可以直接工作的项目：

```bash
export CONFIG_PATH=<训练配置路径>
export CKPT_PATH=<checkpoint路径>
export LATENT_OUT=$JEPA_VIZ_OUTPUT_ROOT/latents
export LATENT_VIZ_OUT=$JEPA_VIZ_OUTPUT_ROOT/latent_viz
export PREDICTION_VIZ_OUT=$JEPA_VIZ_OUTPUT_ROOT/prediction_viz

MODE=all \
LATENT_CONFIG=$CONFIG_PATH \
LATENT_CHECKPOINT=$CKPT_PATH \
LATENT_SPLIT=val \
LATENT_MAX_SAMPLES=1024 \
LATENT_OUT=$LATENT_OUT \
LATENT_DIR=$LATENT_OUT \
LATENT_VIZ_OUT=$LATENT_VIZ_OUT \
PREDICTION_VIZ_OUT=$PREDICTION_VIZ_OUT \
LATENT_COLOR_BY=action \
LATENT_ALIGNMENT_COUNT=4 \
LATENT_ACTIVE_THRESHOLD=1e-2 \
LATENT_ACTIVE_RELATIVE_THRESHOLD=0.0 \
LATENT_PAIRWISE_DENSITY=1 \
LATENT_MAX_ACTION_COMPONENTS=8 \
PREDICTION_GROUP_BY=action \
PREDICTION_ACTION_BINS=4 \
PREDICTION_INTERVAL=std \
PREDICTION_HEATMAP_VMIN_QUANTILE=0.05 \
BATCH_SIZE=128 \
bash $JEPA_VIZ_DIR/run_jepa_viz_template.sh
```

## 6. 常见输出

训练曲线：

```text
$JEPA_VIZ_OUTPUT_ROOT/training/
  train_total_loss.png
  val_loss.png
  total_loss.png
  total_loss_log.png
  total_loss_zoom_step_ge_10000.png
  l1_mse_loss.png
  cosine_loss.png
  latent_std.png
  latent_norm.png
  active_dimensions.png
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

latent 文件：

```text
$JEPA_VIZ_OUTPUT_ROOT/latents/
  latents.npz
  metadata.jsonl
  summary.json
```

latent 可视化：

```text
$JEPA_VIZ_OUTPUT_ROOT/latent_viz/
  pca_z_context_shared_by_*.png
  pca_z_target_shared_by_*.png
  pca_z_pred_shared_by_*.png
  umap_z_context_shared_by_*.png，如果安装了 umap-learn
  umap_z_target_shared_by_*.png，如果安装了 umap-learn
  umap_z_pred_shared_by_*.png，如果安装了 umap-learn
  target_pred_latent_alignment.png
  target_pred_latent_alignment_global.png
  target_pred_latent_alignment_<horizon_key>_<horizon>.png，如果 metadata 中有 horizon
  target_pred_latent_alignment_global_<horizon_key>_<horizon>.png，如果 metadata 中有 horizon
  delta_z_pca_by_action_norm.png
  delta_z_pca_by_action_<component>.png
  action_condition_ablation.png，如果 latent 中有 shuffled/zero action prediction
  nearest_neighbors.html
  covariance_eigenvalue_spectrum.png
  cumulative_explained_variance.png
  pairwise_cosine_histogram.png
  pairwise_cosine_same_vs_different_<label>_<latent>.png，如果 metadata 中有可用标签
  feature_std_distribution.png
  active_dimension_count.png
  visualization_summary.json
```

prediction 可视化：

```text
$JEPA_VIZ_OUTPUT_ROOT/prediction_viz/
  target_pred_cosine_vs_horizon_by_*.png
  target_pred_cosine_boxplot_by_*.png，如果只有一个 horizon
  target_pred_cosine_vs_horizon_by_*.csv
  target_pred_alignment_heatmap.png
  target_pred_alignment_heatmap.csv
  target_pred_alignment_heatmap_metrics.json
  action_norm_vs_cosine_scatter.png
  action_norm_bin_vs_cosine_boxplot.png
  action_norm_bin_vs_mse_boxplot.png
  <action_component>_bin_vs_cosine_boxplot.png
  <action_component>_bin_vs_mse_boxplot.png
  rollout_drift_curve.png，如果导出的 latent 支持 rollout
  condition_ablation.png，如果导出的 latent 支持 condition ablation
  goal_distance.png，如果导出的 latent 包含 goal latent
  prediction_report.md
  prediction_report.json
```
