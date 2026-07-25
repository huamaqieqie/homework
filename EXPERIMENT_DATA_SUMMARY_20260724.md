# 实验数据整理 2026-07-24

数据来源：服务器 `/data1/Johnny/challenge/wrf/homework`。

本报告只整理已经存在的训练/测评文件，不包含新启动的训练或测评。

## 1. 环境成功率总表

所有结果均为 `num_eval=50` 的多 seed 测评。PushT 使用 `eval_budget=300`，Reacher 使用 `eval_budget=50`。

| 实验组 | 数据集 | checkpoint / policy | seeds | success rate | mean ± std | min / max | 平均 time / episode |
|---|---|---|---|---:|---:|---:|---:|
| Original LeWM 15 | PushT | `lewm_15/weights_epoch_15.pt` | 0,1,2,3,4 | 88, 88, 86, 94, 88 | **88.8 ± 3.03** | 86 / 94 | 6.42s |
| Residual LeWM 15 | PushT | `lewm_residual_15/weights_epoch_15.pt` | 0,1,2,3,4 | 94, 88, 86, 88, 94 | **90.0 ± 3.74** | 86 / 94 | 7.50s |
| Factored LeWM 15, 96/96 | PushT | `factored_pusht_15_seed3072_restart_20260718/weights_epoch_15.pt` | 0,1,2,3,4 | 82, 80, 74, 86, 76 | **79.6 ± 4.77** | 74 / 86 | 5.54s |
| Factored LeWM 15, 64/128 | PushT | `checkpoints/factored64_128_pusht_15_seed3072/weights_epoch_15.pt` | 0,1,2,3,4 | 70, 80, 66, 70, 56 | **68.4 ± 8.65** | 56 / 80 | 7.63s |
| Original LeWM 15 | Reacher | `lewm_reacher_15/weights_epoch_15.pt` | 0,1,2,3,4 | 72, 58, 52, 60, 68 | **62.0 ± 8.00** | 52 / 72 | 4.15s |
| Residual LeWM 15 | Reacher | `lewm_residual_reacher_15/weights_epoch_15.pt` | 0,1,2,3,4 | 72, 54, 48, 50, 64 | **57.6 ± 10.14** | 48 / 72 | 3.93s |

## 2. PushT 结论

| 对比 | 结果 |
|---|---|
| Residual vs Original | Residual 平均成功率 90.0%，Original 88.8%，Residual 高 1.2 个百分点；差距很小，需要更多 seed 才能判断稳定性。 |
| Factored 96/96 vs Original | Factored 96/96 平均 79.6%，比 Original 低 9.2 个百分点。 |
| Factored 96/96 vs Residual | Factored 96/96 比 Residual 低 10.4 个百分点。 |
| Factored 64/128 | 平均 68.4%，明显低于 Original、Residual 和 Factored 96/96。 |
| 时间 | Factored 96/96 平均 time/episode 最低，为 5.54s；但成功率下降明显。 |

当前 PushT 最强成功率是 Residual 15 epoch，但只比 Original 高 1.2 个百分点。Factored 两个版本均未超过 baseline，尤其 64/128 版本退化明显。

## 3. Reacher 结论

| 对比 | 结果 |
|---|---|
| Original 15 | 平均成功率 62.0%，seed 间波动较大。 |
| Residual 15 | 平均成功率 57.6%，低于 Original 4.4 个百分点。 |
| Residual vs Original | Residual 没有改善 Reacher，且 std 更高。 |
| 与论文结果 | 论文 Reacher LeWM success rate 约 86%；当前复现的 Original/Residual 都明显偏低。 |

Reacher 当前主要问题不是 residual 是否有效，而是整体复现结果低于官方论文水平。需要先验证官方 checkpoint 或检查 dataset/eval 环境是否完全对齐。

## 4. PushT Latent / Prediction 指标

这些指标来自：

- `compare_original_vs_residual_15/original_lewm_15/jepa_eval/prediction_viz/prediction_report.json`
- `compare_original_vs_residual_15/residual_15/jepa_eval/prediction_viz/prediction_report.json`

| 指标 | Original LeWM 15 | Residual LeWM 15 | 结果 |
|---|---:|---:|---|
| latent shape | `[N,H,D]` | `[N,H,D]` | 一致 |
| detected horizons | 3 | 3 | 一致 |
| latent dim | 192 | 192 | 一致 |
| mean cosine(z_pred,z_target) | **0.99399** | 0.99254 | Original 略高 |
| median cosine | **0.99715** | 0.99600 | Original 略高 |
| q95 cosine | 0.99930 | **0.99937** | 接近 |
| diagonal mean | **0.99399** | 0.99254 | Original 更高 |
| off-diagonal mean | **0.79156** | 0.82729 | Original 更低 |
| diagonal gap | **0.20243** | 0.16525 | Original horizon 区分更清晰 |
| top-1 horizon matching accuracy | **98.70%** | 96.65% | Original 更高 |
| normal MSE | **0.01070** | 0.01311 | Original 更低 |
| zero-action MSE | 0.24099 | **0.18940** | Residual 在去掉 action 后更低 |
| shuffled-action MSE | 0.37885 | **0.28743** | Residual 在打乱 action 后更低 |

解释：Original 的 open-loop latent prediction 和 horizon alignment 更好；Residual 的环境成功率略高，但 latent prediction 指标不占优。Residual 对 action ablation 的退化幅度更小，说明它可能更依赖当前 latent 的 identity/smoothness，而不是更强的 action-conditioned prediction。

## 5. Factored 实验配置

| 实验组 | predictor_mode | static_dim | dynamic_dim | epoch | batch size | 数据集 |
|---|---|---:|---:|---:|---:|---|
| `factored_pusht_15_seed3072_restart_20260718` | factored | 96 | 96 | 15 | 128 | PushT expert train |
| `factored64_128_pusht_15_seed3072` | factored | 64 | 128 | 15 | 128 | PushT expert train |

两个 Factored 版本均使用 `lr=5e-5`、`weight_decay=1e-3`、`embed_dim=192`。从现有结果看，96/96 明显优于 64/128，但仍低于 Original/Residual baseline。

## 6. 训练与 checkpoint 状态

| 模型 | checkpoint 状态 |
|---|---|
| `lewm_15` | 1-15 epoch 权重存在 |
| `lewm_residual_15` | 1-15 epoch 权重存在 |
| `lewm_reacher_15` | 1-15 epoch 权重存在 |
| `lewm_residual_reacher_15` | 1-15 epoch 权重存在 |
| `lewm_reacher_30` | 目前只看到 1-10 epoch 权重，未看到完整 30 epoch |
| `lewm_residual_reacher_30` | 目前只看到 1-9 epoch 权重，未看到完整 30 epoch |
| `factored_pusht_15_seed3072_restart_20260718` | 1-15 epoch 权重存在 |
| `factored64_128_pusht_15_seed3072` | 1-15 epoch 权重存在 |

Reacher 30 epoch 的两个训练曾被 kill，目前服务器已有部分 checkpoint，但还没有对应 30 epoch eval 结果。

## 7. 官方 Reacher checkpoint 状态

服务器上已经存在：

- `outputs/stable-wm/hf_reacher/lewm-reacher/config.json`
- `outputs/stable-wm/hf_reacher/lewm-reacher/weights.pt`
- `outputs/stable-wm/checkpoints/reacher/lewm_object.ckpt`

但最近一次官方 eval 仍失败，原因是 `policy=reacher/lewm` 走到了 HuggingFace cache 风格目录：

```text
outputs/stable-wm/checkpoints/models--reacher--lewm
```

并报错：

```text
No .pt file found in outputs/stable-wm/checkpoints/models--reacher--lewm
```

因此官方 Reacher 86% baseline 还没有在当前服务器 eval 环境中成功复现。后续需要把官方 checkpoint 放到 `stable_worldmodel` 当前 `_resolve_hf` 逻辑期待的位置，或改用本地绝对 checkpoint 路径/正确 policy 格式。

