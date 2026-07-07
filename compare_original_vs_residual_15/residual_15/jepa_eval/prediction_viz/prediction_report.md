# JEPA Prediction Visualization Report

Latents: `/data1/Johnny/challenge/wrf/homework/outputs/jepa_eval/residual_15/latents/latents.npz`

## Shape Check
- `z_context_shape`: `[1024, 3, 192]`
- `z_pred_shape`: `[1024, 3, 192]`
- `z_target_shape`: `[1024, 3, 192]`
- `z_pred_layout`: `{'layout': '[N,H,D]', 'token_axis': None, 'horizon_axis': 1}`
- `z_target_layout`: `{'layout': '[N,H,D]', 'token_axis': None, 'horizon_axis': 1}`
- `canonical_prediction_shape`: `[1024, 1, 3, 192]`
- `detected_num_tokens`: `1`
- `detected_num_horizons`: `3`
- `detected_latent_dim`: `192`

## Warnings
- mean cosine > 0.99; check target leakage or tensor reading errors.

## Alignment Heatmap Metrics
- `diagonal_mean`: `0.9925407767295837`
- `off_diagonal_mean`: `0.827293872833252`
- `diagonal_gap`: `0.1652469038963318`
- `top1_horizon_matching_accuracy`: `0.9661458333333334`

## Generated
- `target_pred_cosine_vs_horizon_by_action_norm_bin.csv`
- `target_pred_cosine_vs_horizon_by_action_norm_bin.png`
- `target_pred_alignment_heatmap.png`
- `target_pred_alignment_heatmap.csv`
- `target_pred_alignment_heatmap_metrics.json`
- `action_norm_vs_cosine_scatter.png`
- `action_norm_bin_vs_cosine_boxplot.png`
- `action_norm_bin_vs_mse_boxplot.png`
- `action_0_bin_vs_cosine_boxplot.png`
- `action_0_bin_vs_mse_boxplot.png`
- `action_1_bin_vs_cosine_boxplot.png`
- `action_1_bin_vs_mse_boxplot.png`
- `action_2_bin_vs_cosine_boxplot.png`
- `action_2_bin_vs_mse_boxplot.png`
- `action_3_bin_vs_cosine_boxplot.png`
- `action_3_bin_vs_mse_boxplot.png`

## Skipped
- `rollout_drift`: No rollout prediction/target arrays found. Expected one of ('z_rollout_pred', 'rollout_pred', 'z_pred_rollout', 'multi_step_z_pred') and one of ('z_rollout_target', 'rollout_target', 'z_target_rollout', 'multi_step_z_target').
- `condition_ablation`: Only normal z_pred is available. Export ablation prediction arrays such as z_pred_condition_removed, z_pred_condition_shuffled, or z_pred_condition_replaced to enable this plot.
- `goal_distance`: No goal latent array found. Expected z_goal, goal_latent, or goal_emb. Goal images alone cannot be used unless their latents are exported.
