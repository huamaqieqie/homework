# Factored LeWM

## Scope

This directory implements the first Factored Latent version described in
`LeWM后续方案与可视化.docx`. It preserves the original encoder, action-conditioned
autoregressive predictor, planner interface, and full-latent MSE objective while
splitting the latent representation into static and dynamic branches.

## Model

For a latent `z_t = [s_t, d_t]`:

```text
s_hat_{t+1} = s_t
delta_d = predictor(d_t, action_t)
d_hat_{t+1} = d_t + residual_scale * delta_d
z_hat_{t+1} = concat(s_hat_{t+1}, d_hat_{t+1})
```

`JEPA.predict()` owns this behavior, so training, offline latent export, and CEM
rollout all use the same factored transition rule.

The predictor exposes separate latent-input and action-condition dimensions.
This keeps the original 192-dimensional action embedding while reducing only
the predicted state branch to `dynamic_dim`, avoiding an action-capacity change
as an experimental confound.

The default dimensions are:

```yaml
wm:
  embed_dim: 192
  static_dim: 96
  dynamic_dim: 96
```

Training rejects a configuration unless `static_dim + dynamic_dim == embed_dim`.
The model also validates the actual latent width before splitting it.

## Loss

The implemented objective is:

```text
L = prediction_mse
    + lambda_dynamic * SIGReg(dynamic_latent)
    + alpha_static * MSE(s_{t+1}, s_t)
    + beta_static_variance * static_variance_loss
    + gamma_decorrelation * static_dynamic_decorrelation
```

The first-version defaults match the document's recommendation: full-latent
prediction MSE, dynamic-only SIGReg, and static consistency are enabled. The two
fallback regularizers for branch degeneration are implemented but disabled by
default:

```yaml
loss:
  sigreg:
    weight: 0.09
  static_consistency:
    weight: 0.1
  static_variance:
    weight: 0.0
    target_std: 1.0
    eps: 0.0001
  static_dynamic_decorrelation:
    weight: 0.0
```

- `static_variance_loss` is a per-feature standard-deviation hinge. Enable it if
  the static branch collapses.
- `static_dynamic_decorrelation_loss` penalizes squared normalized
  cross-correlation between the branches. Enable it if they learn duplicated
  information.

Change one fallback term at a time so its effect can be attributed cleanly.

## Training metrics

In addition to the existing LeWM metrics, training logs:

- `static_consistency_mse`
- `static_copy_mse`
- `static_target_delta_norm`
- `dynamic_delta_norm`
- `dynamic_effective_delta_norm`
- `dynamic_target_delta_norm`
- `dynamic_delta_cos`
- `dynamic_to_static_delta_ratio`
- `static_latent_std` and `dynamic_latent_std`
- `static_active_dim_count` and `dynamic_active_dim_count`
- `static_dynamic_decorrelation`

`static_copy_mse` should be exactly zero up to numerical precision because the
static prediction is copied from context. It is logged as an invariant check,
not as evidence that the learned static branch is semantically correct.

## Latent export

`model(..., return_latents=True)` retains the original keys and additionally
returns:

```text
z_context_static    z_context_dynamic
z_target_static     z_target_dynamic
z_pred_static       z_pred_dynamic
```

This makes branch-specific collapse, consistency, action sensitivity, PCA, and
effective-rank analysis possible without changing the training path.

## Run on the server only

Do not run these commands in the local development checkout. After syncing the
folder to the server:

```bash
export SERVER_REPO=/data1/Johnny/challenge/wrf/homework
export CONFIG_PATH=$SERVER_REPO/le-wm-factored/config/train/lewm.yaml
export CKPT_PATH=<your_checkpoint_path>
export LOG_PATH=<your_log_path>
export OUT_DIR=$SERVER_REPO/tools/jepa_viz/output

cd "$SERVER_REPO/le-wm-factored"
python train.py data=pusht
```

To resume from an explicit server checkpoint:

```bash
python train.py resume_ckpt_path="$CKPT_PATH"
```

Example overrides:

```bash
# Change the split while preserving a 192-dimensional full latent.
python train.py wm.static_dim=64 wm.dynamic_dim=128

# Add static variance protection after observing static collapse.
python train.py loss.static_variance.weight=0.01

# Add branch decorrelation after observing duplicated information.
python train.py loss.static_dynamic_decorrelation.weight=0.01
```

For planning evaluation, use the saved policy path relative to the server's
`STABLEWM_HOME`, without the `_object.ckpt` suffix:

```bash
python eval.py --config-name=pusht.yaml policy=<relative_policy_path>
```

Start with the default 96/96 split and disabled fallback regularizers. Compare
against Original and Residual LeWM using the same dataset, seed, training budget,
planning horizon, CEM sample count, and evaluation budget.
