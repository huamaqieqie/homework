# JEPA Latent Export

`tools/jepa_eval/export_latents.py` 用于从训练配置和 checkpoint 中导出 JEPA 评估需要的 latent，不参与训练流程。

## 输出文件

默认输出目录：

```text
outputs/jepa_eval/latents/
```

主要文件：

```text
latents.npz
metadata.jsonl
summary.json
```

`latents.npz` 包含：

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

如果 batch 中存在这些字段，也会尽量保存为数组：

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

`metadata.jsonl` 每行对应一个 sample，包含：

```text
sample_id
export_index
split
dataset/source 信息，如果 batch 里有
frame/time/episode/task/object/category/success/mask 信息，如果 batch 里有
```

## Sanity Check

导出过程中会检查：

- `z_context`、`z_target`、`z_pred` 是否存在。
- latent 是否包含 NaN/Inf。
- `z_pred` 和 `z_target` batch size 是否一致。
- `z_pred` 和 `z_target` shape 是否一致。
- `metadata.jsonl` 行数是否等于 latent sample 数。

## 服务器命令

直接运行脚本：

```bash
export SERVER_REPO=/data1/Johnny/challenge/wrf/homework
cd $SERVER_REPO
source env_cache.sh

python tools/jepa_eval/export_latents.py \
  --config $SERVER_REPO/le-wm/config/train/lewm.yaml \
  --checkpoint <你的checkpoint路径> \
  --split val \
  --max-samples 1024 \
  --out $SERVER_REPO/outputs/jepa_eval/latents \
  --dataset /manifoldai-training/johnny/challenge/lewm-pusht/datasets/pusht_expert_train.h5 \
  data=pusht
```

通过模板运行：

```bash
export SERVER_REPO=/data1/Johnny/challenge/wrf/homework
cd $SERVER_REPO
source env_cache.sh

MODE=export_latents \
DATA_CONFIG=pusht \
DATASET_NAME=/manifoldai-training/johnny/challenge/lewm-pusht/datasets/pusht_expert_train.h5 \
LATENT_CHECKPOINT=<你的checkpoint路径> \
LATENT_SPLIT=val \
LATENT_MAX_SAMPLES=1024 \
LATENT_OUT=$SERVER_REPO/outputs/jepa_eval/latents \
BATCH_SIZE=128 \
bash tools/jepa_eval/run_train_eval_template.sh
```

如果还没有真实 checkpoint，需要先在服务器训练完成后提供以下其中之一：

```text
Lightning .ckpt 文件
save_pretrained 生成的 .pt/.pth 文件
包含上述文件的 checkpoint 目录
```
