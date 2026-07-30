# Residual LeWM

Residual LeWM keeps the original LeWM encoder, action encoder, SIGReg loss,
dataset, and planner flow. It only changes how the predictor output is
interpreted.

## Formula

Direct predictor baseline:

```text
z_hat_next = predictor(z_t, action_t)
```

Residual predictor:

```text
delta_z = predictor(z_t, action_t)
z_hat_next = z_t + residual_scale * delta_z
```

The training target stays unchanged. The prediction loss is still computed
between `z_hat_next` and the target latent.

## How to Enable

The default config keeps the original direct baseline:

```yaml
predictor_mode: direct
residual_scale: 1.0
```

Enable residual mode with a Hydra override:

```bash
python train.py model.predictor_mode=residual model.residual_scale=1.0
```

Keep the original LeWM baseline:

```bash
python train.py model.predictor_mode=direct
```

## Metrics to Watch

Residual LeWM adds lightweight training metrics:

- `delta_norm`: mean norm of the effective predicted latent delta.
- `target_delta_norm`: mean norm of `z_target - z_context`.
- `delta_cos`: cosine similarity between predicted and target deltas.
- `pred_latent_mse`: MSE between predicted latent and target latent.
- `predictor_mode_is_residual`: `1.0` for residual mode, `0.0` for direct mode.

`delta_norm` is computed from the effective displacement
`z_hat_next - z_t`, including `residual_scale`. `raw_delta_norm` records the
unscaled predictor output so residual runs can diagnose whether the scale or
the predictor itself is responsible for the step size.

## Autoregressive Multi-step Training

Setting only `wm.num_preds=3` uses the legacy shifted-target objective. It does
not feed an earlier prediction into the next prediction.

Enable true recursive three-step training with:

```bash
python train.py \
  data=dmc \
  model.predictor_mode=residual \
  model.residual_scale=1.0 \
  wm.history_size=3 \
  wm.num_preds=3 \
  wm.autoregressive_rollout=true \
  loss.rollout.discount=0.5
```

At rollout step 1, the predictor uses the three encoded context latents. At
steps 2 and 3, the oldest latent is dropped and the previous prediction is
appended to the context. The corresponding future action is appended at each
step.

The prediction objective is a normalized discounted mean:

```text
(L_step1 + 0.5 * L_step2 + 0.25 * L_step3) / 1.75
```

Normalization keeps the prediction-loss scale comparable with one-step
training. Metrics named `pred_loss_horizon_01`, `02`, and `03` expose each
unweighted horizon loss separately.

## Scope of This Version

This first version does not implement:

- Factored Latent.
- Feature-space Flow Matching.
- Action contrastive loss.
- AdaLN architecture changes.
- Dataset or planner objective changes.
