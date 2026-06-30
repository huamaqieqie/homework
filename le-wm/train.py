import os
from functools import partial
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_output_root = Path(os.environ.get("LEWM_OUTPUT_ROOT", _repo_root / "outputs"))
os.environ.setdefault("XDG_CACHE_HOME", str(_output_root / ".cache"))
os.environ.setdefault("XDG_CONFIG_HOME", str(_output_root / ".config"))
os.environ.setdefault("XDG_DATA_HOME", str(_output_root / ".local"))
os.environ.setdefault("STABLEWM_HOME", str(_output_root / "stable-wm"))

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


def _log_jepa_eval_metrics(module, batch, stage, emb, tgt_emb, pred_emb, loss_total, cfg):
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


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    _log_jepa_eval_metrics(self, batch, stage, emb, tgt_emb, pred_emb, output["loss"], cfg)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
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
