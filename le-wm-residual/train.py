import os
from functools import partial
from pathlib import Path

from local_paths import configure_output_paths

configure_output_paths()

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import (
    JEPATrainStatsCallback,
    SaveCkptCallback,
    get_column_normalizer,
    get_img_preprocessor,
)


def _sanitize_metric_key(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "_")


def _get_eval_metric_cfg(cfg, key, default):
    eval_cfg = cfg.get("eval_metrics", {})
    if hasattr(eval_cfg, "get"):
        return eval_cfg.get(key, default)
    return default


def _autoregressive_rollout(model, emb, act_emb, history_size, num_preds):
    """Predict future latents recursively from the last context state."""
    required_steps = history_size + num_preds
    if emb.size(1) < required_steps:
        raise ValueError(
            "Autoregressive rollout requires at least history_size + num_preds "
            f"latent steps, got {emb.size(1)} < {required_steps}."
        )
    if act_emb.size(1) < required_steps - 1:
        raise ValueError(
            "Autoregressive rollout requires one action per predicted transition, "
            f"got {act_emb.size(1)} action steps for {required_steps} latent steps."
        )

    rollout_emb = emb[:, :history_size]
    rollout_act = act_emb[:, :history_size]
    predictions = []
    raw_predictions = []
    prediction_bases = []

    for step in range(num_preds):
        emb_context = rollout_emb[:, -history_size:]
        act_context = rollout_act[:, -history_size:]
        pred_emb, pred_raw = model.predict(
            emb_context,
            act_context,
            return_raw=True,
        )

        prediction_bases.append(emb_context[:, -1:])
        predictions.append(pred_emb[:, -1:])
        raw_predictions.append(pred_raw[:, -1:])
        rollout_emb = torch.cat([rollout_emb, pred_emb[:, -1:]], dim=1)

        if step + 1 < num_preds:
            next_action_idx = history_size + step
            rollout_act = torch.cat(
                [rollout_act, act_emb[:, next_action_idx : next_action_idx + 1]],
                dim=1,
            )

    return (
        torch.cat(predictions, dim=1),
        torch.cat(raw_predictions, dim=1),
        torch.cat(prediction_bases, dim=1),
    )


def _discounted_rollout_loss(pred_emb, tgt_emb, discount):
    if not 0.0 < discount <= 1.0:
        raise ValueError(f"loss.rollout.discount must be in (0, 1], got {discount}.")

    per_horizon = (pred_emb - tgt_emb).pow(2).mean(dim=(0, 2))
    weights = discount ** torch.arange(
        pred_emb.size(1),
        device=pred_emb.device,
        dtype=pred_emb.dtype,
    )
    loss = (per_horizon * weights).sum() / weights.sum()
    return loss, per_horizon


def _log_jepa_viz_metrics(module, batch, stage, emb, tgt_emb, pred_emb, loss_total, cfg):
    with torch.no_grad():
        diff = pred_emb.detach() - tgt_emb.detach()
        mse = diff.pow(2).mean()
        l1 = diff.abs().mean()
        cos = F.cosine_similarity(pred_emb.detach(), tgt_emb.detach(), dim=-1)
        cos_loss = 1.0 - cos.mean()

        flat_emb = emb.detach().reshape(-1, emb.size(-1)).float()
        latent_std_per_dim = flat_emb.std(dim=0, unbiased=False)
        latent_norm = flat_emb.norm(dim=-1)
        active_threshold = float(_get_eval_metric_cfg(cfg, "active_dim_std_threshold", 1e-2))
        active_dims = (latent_std_per_dim > active_threshold).float().sum()

        metrics = {
            f"{stage}/loss_total": loss_total.detach(),
            f"{stage}/loss_mse": mse,
            f"{stage}/loss_future_l1": l1,
            f"{stage}/loss_future_cos": cos_loss,
            f"{stage}/latent_mean": flat_emb.mean(),
            f"{stage}/latent_std": flat_emb.std(unbiased=False),
            f"{stage}/latent_norm": latent_norm.mean(),
            f"{stage}/active_dim_count": active_dims,
        }

        horizon_mse = diff.pow(2).mean(dim=(0, 2))
        for horizon_idx, value in enumerate(horizon_mse, start=1):
            metrics[f"{stage}/loss_mse_horizon_{horizon_idx:02d}"] = value

        if flat_emb.size(0) > 1:
            max_pairwise_samples = int(_get_eval_metric_cfg(cfg, "max_pairwise_samples", 512))
            pairwise_emb = flat_emb[:max_pairwise_samples]
            normed = F.normalize(pairwise_emb, dim=-1)
            pairwise = normed @ normed.T
            mask = ~torch.eye(pairwise.size(0), dtype=torch.bool, device=pairwise.device)
            pairwise = pairwise[mask]
            metrics[f"{stage}/pairwise_cos_mean"] = pairwise.mean()
            metrics[f"{stage}/pairwise_cos_std"] = pairwise.std(unbiased=False)
            metrics[f"{stage}/pairwise_cos_min"] = pairwise.min()
            metrics[f"{stage}/pairwise_cos_max"] = pairwise.max()
            hist = torch.histc(pairwise.float(), bins=20, min=-1.0, max=1.0)
            hist = hist / hist.sum().clamp_min(1.0)
            for bin_idx, value in enumerate(hist):
                metrics[f"{stage}/pairwise_cos_hist_bin_{bin_idx:02d}"] = value

        source_key = next((key for key in ("dataset", "dataset_id", "source", "source_id") if key in batch), None)
        if source_key is not None:
            per_sample_mse = diff.pow(2).mean(dim=tuple(range(1, diff.ndim)))
            sources = batch[source_key]
            if torch.is_tensor(sources) and sources.ndim > 0 and sources.size(0) == per_sample_mse.size(0):
                for source in torch.unique(sources.detach().cpu()):
                    source_mask = sources == source.to(sources.device)
                    source_name = _sanitize_metric_key(source.item())
                    metrics[f"{stage}/batch_source_counts/{source_name}"] = source_mask.float().sum()
                    metrics[f"{stage}/per_dataset_loss/{source_name}"] = per_sample_mse[source_mask].mean()

        module.log_dict(metrics, on_step=True, sync_dist=True)


def _log_residual_predictor_metrics(module, stage, base_emb, tgt_emb, pred_emb, pred_raw):
    with torch.no_grad():
        target_delta = tgt_emb.detach() - base_emb.detach()
        pred_delta = pred_emb.detach() - base_emb.detach()
        is_residual = getattr(module.model, "predictor_mode", "direct") == "residual"
        raw_delta = pred_raw.detach()

        metrics = {
            f"{stage}/pred_latent_mse": (pred_emb.detach() - tgt_emb.detach()).pow(2).mean(),
            f"{stage}/delta_norm": pred_delta.norm(dim=-1).mean(),
            f"{stage}/raw_delta_norm": raw_delta.norm(dim=-1).mean(),
            f"{stage}/target_delta_norm": target_delta.norm(dim=-1).mean(),
            f"{stage}/delta_cos": F.cosine_similarity(
                pred_delta.reshape(-1, pred_delta.size(-1)),
                target_delta.reshape(-1, target_delta.size(-1)),
                dim=-1,
            ).mean(),
            f"{stage}/predictor_mode_is_residual": torch.tensor(
                float(is_residual),
                device=pred_emb.device,
            ),
        }

        module.log_dict(metrics, on_step=True, sync_dist=True)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    autoregressive_rollout = bool(cfg.wm.get("autoregressive_rollout", False))

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    if autoregressive_rollout:
        tgt_emb = emb[:, ctx_len : ctx_len + n_preds]
        pred_emb, pred_raw, prediction_bases = _autoregressive_rollout(
            self.model,
            emb,
            act_emb,
            history_size=ctx_len,
            num_preds=n_preds,
        )
        rollout_cfg = cfg.loss.get("rollout", {})
        discount = float(rollout_cfg.get("discount", 1.0))
        output["pred_loss"], horizon_losses = _discounted_rollout_loss(
            pred_emb,
            tgt_emb,
            discount,
        )
        for horizon_idx, horizon_loss in enumerate(horizon_losses, start=1):
            output[f"pred_loss_horizon_{horizon_idx:02d}"] = horizon_loss
    else:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]  # label
        pred_emb, pred_raw = self.model.predict(
            ctx_emb,
            ctx_act,
            return_raw=True,
        )
        prediction_bases = ctx_emb
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # LeWM loss
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    _log_jepa_viz_metrics(self, batch, stage, emb, tgt_emb, pred_emb, output["loss"], cfg)
    _log_residual_predictor_metrics(
        self,
        stage,
        prediction_bases,
        tgt_emb,
        pred_emb,
        pred_raw,
    )
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    if cfg.wm.history_size < 1:
        raise ValueError(f"wm.history_size must be >= 1, got {cfg.wm.history_size}.")
    if cfg.wm.num_preds < 1:
        raise ValueError(f"wm.num_preds must be >= 1, got {cfg.wm.num_preds}.")

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )
    train_stats_callback = JEPATrainStatsCallback()

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, train_stats_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    resume_ckpt_path = cfg.get("resume_ckpt_path")
    if resume_ckpt_path:
        ckpt_path = Path(resume_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_ckpt_path does not exist: {ckpt_path}")
    else:
        ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
