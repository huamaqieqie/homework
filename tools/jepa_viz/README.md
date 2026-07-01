# JEPA Visualization Toolkit

这个目录是统一的 JEPA 评估与可视化工具入口。  
它尽量只假设一件事：latent 文件遵循下面的通用格式。

```text
latents.npz 或 latents.pt
metadata.jsonl
```

`latents.npz` / `latents.pt` 至少包含：

```text
z_context
z_target
z_pred
```

可选包含：

```text
action
action_condition
dataset / source
task / object / category / success
step_idx / episode_idx / time_index
pixels / image_path / frame_path
```

默认输出目录统一放在：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/
```

模板脚本默认也会把 matplotlib/cache/tmp 放到这个目录下，避免写到 `/root`。

你也可以通过命令行参数或环境变量指定任意输入/输出地址。

## 目录结构

```text
tools/jepa_viz/
  README.md
  COMMANDS.md
  GUIDE.md
  plot_training_curves.py
  export_latents.py
  visualize_latents.py
  visualize_prediction.py
  run_jepa_viz_template.sh
  output/
    <YYYYMMDD_HHMMSS>/
      training/
      latents/
      latent_viz/
      prediction_viz/
```

说明：

- `COMMANDS.md`：服务器端常用命令集合。
- `GUIDE.md`：训练曲线、latent 导出文件、latent 可视化结果的统一解释文档。
- `plot_training_curves.py`：从 `csv/jsonl/txt` 训练日志画训练曲线。
- `export_latents.py`：LeWM 当前项目的 latent 导出适配器。
- `visualize_latents.py`：通用 latent 可视化，只依赖 `latents.npz/.pt + metadata.jsonl`，其他 JEPA 模型也可以使用。
- `visualize_prediction.py`：prediction 能力可视化，分析 `z_pred` 与 `z_target` 的 horizon、alignment、ablation、goal distance。
- `run_jepa_viz_template.sh`：统一命令模板。

后续只维护 `tools/jepa_viz/` 这一套入口。

## 训练曲线

```bash
python tools/jepa_viz/plot_training_curves.py \
  --log <metrics.csv 或日志文件> \
  --out $JEPA_VIZ_OUTPUT_ROOT/training
```

不传 `--out` 时默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/training
```

## Latent 导出

当前 `export_latents.py` 是这个 LeWM 仓库的导出适配器。其他 JEPA 项目可以直接生成同样格式的 `latents.npz + metadata.jsonl`，然后跳过这一步使用 `visualize_latents.py`。

```bash
python tools/jepa_viz/export_latents.py \
  --config <config_path> \
  --checkpoint <checkpoint_path> \
  --split val \
  --max-samples 1024 \
  --out $JEPA_VIZ_OUTPUT_ROOT/latents \
  --dataset <dataset_path_or_name> \
  <hydra_override>
```

不传 `--out` 时默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/latents
```

## Latent 可视化

```bash
python tools/jepa_viz/visualize_latents.py \
  --latent-dir $JEPA_VIZ_OUTPUT_ROOT/latents \
  --out $JEPA_VIZ_OUTPUT_ROOT/latent_viz \
  --color-by action
```

不传参数时默认读取：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/latents
```

默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/latent_viz
```

## Prediction 可视化

```bash
python tools/jepa_viz/visualize_prediction.py \
  --latent-dir $JEPA_VIZ_OUTPUT_ROOT/latents \
  --out $JEPA_VIZ_OUTPUT_ROOT/prediction_viz \
  --group-by action
```

默认输出到：

```text
tools/jepa_viz/output/<YYYYMMDD_HHMMSS>/prediction_viz
```

## 统一模板

```bash
MODE=visualize_latents \
LATENT_DIR=<latent输入目录> \
LATENT_VIZ_OUT=<可视化输出目录> \
LATENT_COLOR_BY=action \
bash tools/jepa_viz/run_jepa_viz_template.sh
```

可用模式：

```text
plot
watch
export_latents
visualize_latents
visualize_prediction
all
```

默认输出根目录：

```bash
JEPA_VIZ_OUTPUT_ROOT=tools/jepa_viz/output/<YYYYMMDD_HHMMSS>
```

可以覆盖为任意位置：

```bash
JEPA_VIZ_RUN_NAME=$(date +%Y%m%d_%H%M%S) \
JEPA_VIZ_OUTPUT_ROOT=/data1/Johnny/challenge/wrf/homework/outputs/my_jepa_viz/$JEPA_VIZ_RUN_NAME \
bash tools/jepa_viz/run_jepa_viz_template.sh
```
