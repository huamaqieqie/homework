# JEPA Viz Commands

本文档只记录 `tools/jepa_viz/` 这一套工具的常用命令。  
默认输出目录是：

```bash
$SERVER_REPO/tools/jepa_viz/output
```

服务器仓库路径：

```bash
export SERVER_REPO=/data1/Johnny/challenge/wrf/homework
cd $SERVER_REPO
source env_cache.sh
```

## 1. 训练曲线可视化

训练时终端日志会出现类似：

```text
[RegistryLogger] run_id=... run_dir=... metrics.csv will live here
```

把 `RUN_DIR` 替换成日志里的真实路径：

```bash
export RUN_DIR=<训练日志里的run_dir>

MODE=plot \
TRAIN_LOG=$RUN_DIR/metrics.csv \
TRAINING_OUT=$SERVER_REPO/tools/jepa_viz/output/training \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

训练过程中持续刷新曲线：

```bash
export RUN_DIR=<训练日志里的run_dir>

MODE=watch \
TRAIN_LOG=$RUN_DIR/metrics.csv \
TRAINING_OUT=$SERVER_REPO/tools/jepa_viz/output/training \
WATCH_INTERVAL=60 \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

也可以直接调用脚本：

```bash
python tools/jepa_viz/plot_training_curves.py \
  --log $RUN_DIR/metrics.csv \
  --out $SERVER_REPO/tools/jepa_viz/output/training
```

## 2. 导出 Latents

把 `CKPT_PATH` 替换成真实 checkpoint：

```bash
export CKPT_PATH=/data1/Johnny/challenge/wrf/homework/outputs/stable-wm/checkpoints/lewm_15/weights_epoch_15.pt
```

导出验证集 latent：

```bash
MODE=export_latents \
DATA_CONFIG=pusht \
DATASET_NAME=/manifoldai-training/johnny/challenge/lewm-pusht/datasets/pusht_expert_train.h5 \
LATENT_CHECKPOINT=$CKPT_PATH \
LATENT_SPLIT=val \
LATENT_MAX_SAMPLES=1024 \
LATENT_OUT=$SERVER_REPO/tools/jepa_viz/output/latents \
BATCH_SIZE=128 \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

最小导出测试：

```bash
MODE=export_latents \
DATA_CONFIG=pusht \
DATASET_NAME=/manifoldai-training/johnny/challenge/lewm-pusht/datasets/pusht_expert_train.h5 \
LATENT_CHECKPOINT=$CKPT_PATH \
LATENT_SPLIT=val \
LATENT_MAX_SAMPLES=16 \
LATENT_OUT=$SERVER_REPO/tools/jepa_viz/output/latents_test \
BATCH_SIZE=8 \
NUM_WORKERS=0 \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

直接调用脚本：

```bash
python tools/jepa_viz/export_latents.py \
  --config $SERVER_REPO/le-wm/config/train/lewm.yaml \
  --checkpoint $CKPT_PATH \
  --split val \
  --max-samples 1024 \
  --out $SERVER_REPO/tools/jepa_viz/output/latents \
  --dataset /manifoldai-training/johnny/challenge/lewm-pusht/datasets/pusht_expert_train.h5 \
  data=pusht
```

## 3. Latent 可视化

使用默认 latent 输出目录：

```bash
MODE=visualize_latents \
LATENT_DIR=$SERVER_REPO/tools/jepa_viz/output/latents \
LATENT_VIZ_OUT=$SERVER_REPO/tools/jepa_viz/output/latent_viz \
LATENT_COLOR_BY=action \
LATENT_MAX_POINTS=5000 \
LATENT_NN_QUERIES=8 \
LATENT_TOP_K=5 \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

最小可视化测试：

```bash
MODE=visualize_latents \
LATENT_DIR=$SERVER_REPO/tools/jepa_viz/output/latents \
LATENT_VIZ_OUT=$SERVER_REPO/tools/jepa_viz/output/latent_viz_test \
LATENT_COLOR_BY=action \
LATENT_MAX_POINTS=512 \
LATENT_NN_QUERIES=2 \
LATENT_TOP_K=3 \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

直接调用脚本：

```bash
python tools/jepa_viz/visualize_latents.py \
  --latent-dir $SERVER_REPO/tools/jepa_viz/output/latents \
  --out $SERVER_REPO/tools/jepa_viz/output/latent_viz \
  --color-by action
```

## 4. 一步导出并可视化

```bash
MODE=all \
DATA_CONFIG=pusht \
DATASET_NAME=/manifoldai-training/johnny/challenge/lewm-pusht/datasets/pusht_expert_train.h5 \
LATENT_CHECKPOINT=$CKPT_PATH \
LATENT_SPLIT=val \
LATENT_MAX_SAMPLES=1024 \
LATENT_OUT=$SERVER_REPO/tools/jepa_viz/output/latents \
LATENT_DIR=$SERVER_REPO/tools/jepa_viz/output/latents \
LATENT_VIZ_OUT=$SERVER_REPO/tools/jepa_viz/output/latent_viz \
LATENT_COLOR_BY=action \
BATCH_SIZE=128 \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

## 5. 换输出根目录

如果不想写到 `tools/jepa_viz/output`，可以指定新的输出根目录：

```bash
JEPA_VIZ_OUTPUT_ROOT=$SERVER_REPO/outputs/my_jepa_viz \
MODE=visualize_latents \
LATENT_DIR=$SERVER_REPO/outputs/my_jepa_viz/latents \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

## 6. 常见输出

训练曲线：

```text
tools/jepa_viz/output/training/
```

latent 文件：

```text
tools/jepa_viz/output/latents/
  latents.npz
  metadata.jsonl
  summary.json
```

latent 可视化：

```text
tools/jepa_viz/output/latent_viz/
  pca_z_context_by_*.png
  pca_z_target_by_*.png
  pca_z_pred_by_*.png
  umap_z_context_by_*.png 或 umap_fallback_pca_z_context_by_*.png
  latent_trajectory.png
  nearest_neighbors.html
  covariance_eigenvalue_spectrum.png
  pairwise_cosine_histogram.png
  feature_std_distribution.png
  active_dimension_count.png
  visualization_summary.json
```
