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

For residual mode, `delta_norm` is computed from the raw predictor output. For
direct mode, it is computed from the effective displacement `z_hat_next - z_t`
so the same dashboard can compare both modes.

## Scope of This Version

This first version does not implement:

- Factored Latent.
- Feature-space Flow Matching.
- Action contrastive loss.
- AdaLN architecture changes.
- Dataset or planner objective changes.
