#!/usr/bin/env python
"""Export JEPA latents for offline evaluation.

This script is intentionally separate from the training entrypoint. It rebuilds
the dataset/model from a training config, loads a checkpoint, runs inference on
the requested split, and writes latents plus sample metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
LEWM_DIR = REPO_ROOT / "le-wm"
TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TOOL_DIR / "output" / "latents"
if str(LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(LEWM_DIR))

from local_paths import configure_output_paths  # noqa: E402

configure_output_paths()

import hydra  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402


OPTIONAL_METADATA_KEYS = (
    "dataset",
    "dataset_id",
    "source",
    "source_id",
    "frame_idx",
    "frame_index",
    "time_idx",
    "time_index",
    "timestep",
    "step_idx",
    "episode_idx",
    "ep_idx",
    "text",
    "instruction",
    "task_instruction",
    "task",
    "task_id",
    "object",
    "object_id",
    "category",
    "category_id",
    "label",
    "success",
    "is_success",
    "mask_pos",
    "mask_position",
    "target_block_pos",
    "target_block_position",
)

OPTIONAL_ARRAY_KEYS = (
    "action",
    "state",
    "proprio",
    "observation",
    "dataset",
    "dataset_id",
    "source",
    "source_id",
    "step_idx",
    "episode_idx",
    "ep_idx",
    "task_id",
    "object_id",
    "category_id",
    "label",
    "success",
    "is_success",
    "mask_pos",
    "mask_position",
    "target_block_pos",
    "target_block_position",
)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, max_samples=None):
        self.dataset = dataset
        self.indices = list(indices)
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_id = int(self.indices[idx])
        sample = self.dataset[sample_id]
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["_jepa_viz_sample_id"] = sample_id
            return sample
        return {"sample": sample, "_jepa_viz_sample_id": sample_id}


def parse_args():
    parser = argparse.ArgumentParser(description="Export JEPA latents from a checkpoint.")
    parser.add_argument("--config", required=True, help="Training config file, e.g. le-wm/config/train/lewm.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or directory.")
    parser.add_argument("--split", default="val", choices=("train", "val", "all"), help="Dataset split to export.")
    parser.add_argument("--max-samples", type=int, default=1024, help="Maximum samples to export.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--format", choices=("npz", "pt"), default="npz", help="Latent tensor file format.")
    parser.add_argument("--dataset", default=None, help="Optional override for data.dataset.name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional export batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    args, overrides = parser.parse_known_args()
    return args, overrides


def register_resolvers():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def load_config(config_path, overrides):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    register_resolvers()
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.stem, overrides=overrides)
    return cfg


def build_dataset_and_split(cfg, split, max_samples):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg,
    )

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    dataset.transform = spt.data.transforms.Compose(*transforms)

    if split == "all":
        indices = list(range(len(dataset)))
    else:
        rnd_gen = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=rnd_gen,
        )
        selected = train_set if split == "train" else val_set
        indices = getattr(selected, "indices", None)
        if indices is None:
            indices = list(range(len(selected)))
            dataset = selected

    return IndexedDataset(dataset, indices, max_samples=max_samples)


def build_loader(cfg, dataset, args):
    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    loader_kwargs["shuffle"] = False
    loader_kwargs["drop_last"] = False

    if args.batch_size is not None:
        loader_kwargs["batch_size"] = args.batch_size
    if args.num_workers is not None:
        loader_kwargs["num_workers"] = args.num_workers

    if int(loader_kwargs.get("num_workers", 0)) == 0:
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)

    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    if "action" in moved and torch.is_tensor(moved["action"]):
        moved["action"] = torch.nan_to_num(moved["action"], 0.0)
    return moved


def is_tensor_state_dict(value):
    return isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values())


def resolve_checkpoint_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = []
    for pattern in ("*.ckpt", "*.pt", "*.pth", "checkpoints/*.ckpt", "checkpoints/*.pt", "checkpoints/*.pth"):
        candidates.extend(path.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt/.pt/.pth files found under checkpoint directory: {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def checkpoint_state_dicts(payload):
    if is_tensor_state_dict(payload):
        yield payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            value = payload.get(key)
            if is_tensor_state_dict(value):
                yield value


def strip_prefix(state_dict, prefix):
    stripped = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not stripped:
        return None
    return stripped


def load_checkpoint(model, checkpoint, device, strict=False):
    checkpoint = resolve_checkpoint_path(checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    model_keys = set(model.state_dict().keys())
    candidates = []

    for state_dict in checkpoint_state_dicts(payload):
        variants = [state_dict]
        for prefix in ("model.", "module.model.", "pl_module.model.", "_orig_mod.", "module."):
            stripped = strip_prefix(state_dict, prefix)
            if stripped is not None:
                variants.append(stripped)

        for variant in variants:
            match_count = len(model_keys.intersection(variant.keys()))
            if match_count:
                candidates.append((match_count, variant))

    if not candidates:
        raise ValueError(f"Could not find JEPA model weights in checkpoint: {checkpoint}")

    _, state_dict = max(candidates, key=lambda item: item[0])
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, incompatible


def make_jsonable(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    return value


def value_at_batch_index(value, idx):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        return value[idx]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value
        return value[idx]
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def metadata_for_batch(batch, split, start_index):
    sample_ids = batch.get("_jepa_viz_sample_id")
    batch_size = None
    if torch.is_tensor(sample_ids):
        batch_size = int(sample_ids.numel())
    else:
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                batch_size = int(value.size(0))
                break

    rows = []
    for local_idx in range(batch_size or 0):
        sample_id = value_at_batch_index(sample_ids, local_idx) if sample_ids is not None else start_index + local_idx
        row = {
            "sample_id": make_jsonable(sample_id),
            "export_index": start_index + local_idx,
            "split": split,
        }
        for key in OPTIONAL_METADATA_KEYS:
            if key in batch:
                row[key] = make_jsonable(value_at_batch_index(batch[key], local_idx))
        rows.append(row)
    return rows


def append_optional_arrays(storage, batch):
    for key in OPTIONAL_ARRAY_KEYS:
        if key in batch and torch.is_tensor(batch[key]):
            storage.setdefault(key, []).append(tensor_to_numpy(batch[key]))


def validate_batch(latents):
    required = ("z_context", "z_target", "z_pred")
    for key in required:
        if key not in latents:
            raise ValueError(f"Missing latent output: {key}")
        tensor = latents[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f"{key} is not a tensor: {type(tensor)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains NaN or Inf")

    if latents["z_pred"].size(0) != latents["z_target"].size(0):
        raise ValueError("z_pred and z_target batch sizes do not match")
    if latents["z_pred"].shape != latents["z_target"].shape:
        raise ValueError(
            f"z_pred shape {tuple(latents['z_pred'].shape)} does not match "
            f"z_target shape {tuple(latents['z_target'].shape)}"
        )


def concatenate(items):
    return np.concatenate(items, axis=0) if items else None


def save_outputs(args, cfg, arrays, metadata, checkpoint_path, incompatible):
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    z_context = concatenate(arrays["z_context"])
    z_target = concatenate(arrays["z_target"])
    z_pred = concatenate(arrays["z_pred"])
    z_all = concatenate(arrays["z_all"])
    action_condition = concatenate(arrays["action_condition"])

    if len(metadata) != z_context.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata)}) != latent rows ({z_context.shape[0]})")

    pred_steps = z_pred.shape[1]
    context_steps = z_context.shape[1]
    future_horizon_index = np.full((z_context.shape[0], pred_steps), int(cfg.wm.num_preds), dtype=np.int64)
    context_time_index = np.broadcast_to(np.arange(context_steps, dtype=np.int64), (z_context.shape[0], context_steps))
    target_time_index = np.broadcast_to(
        np.arange(int(cfg.wm.num_preds), int(cfg.wm.num_preds) + pred_steps, dtype=np.int64),
        (z_context.shape[0], pred_steps),
    )

    payload = {
        "z_context": z_context,
        "z_target": z_target,
        "z_pred": z_pred,
        "z_all": z_all,
        "action_condition": action_condition,
        "future_horizon_index": future_horizon_index,
        "context_time_index": context_time_index,
        "target_time_index": target_time_index,
    }

    for key, values in arrays["optional"].items():
        payload[key] = concatenate(values)

    latent_path = out_dir / f"latents.{args.format}"
    if args.format == "npz":
        np.savez_compressed(latent_path, **payload)
    else:
        torch.save({key: torch.from_numpy(value) for key, value in payload.items()}, latent_path)

    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": str(checkpoint_path),
        "latent_path": str(latent_path),
        "metadata_path": str(metadata_path),
        "num_samples": int(z_context.shape[0]),
        "z_context_shape": list(z_context.shape),
        "z_target_shape": list(z_target.shape),
        "z_pred_shape": list(z_pred.shape),
        "history_size": int(cfg.wm.history_size),
        "num_preds": int(cfg.wm.num_preds),
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    args, overrides = parse_args()
    if args.dataset:
        overrides.append(f"data.dataset.name={args.dataset}")

    cfg = load_config(args.config, overrides)
    dataset = build_dataset_and_split(cfg, args.split, args.max_samples)
    loader = build_loader(cfg, dataset, args)

    model = hydra.utils.instantiate(cfg.model)
    device = torch.device(args.device)
    model = model.to(device)
    checkpoint_path, incompatible = load_checkpoint(model, args.checkpoint, device, strict=args.strict)
    model.eval()
    model.requires_grad_(False)

    arrays = {
        "z_context": [],
        "z_target": [],
        "z_pred": [],
        "z_all": [],
        "action_condition": [],
        "optional": {},
    }
    metadata = []

    with torch.inference_mode():
        for batch in loader:
            start_index = len(metadata)
            rows = metadata_for_batch(batch, args.split, start_index)
            batch = move_to_device(batch, device)
            latents = model(
                batch,
                return_latents=True,
                history_size=int(cfg.wm.history_size),
                num_preds=int(cfg.wm.num_preds),
            )
            validate_batch(latents)

            for key in ("z_context", "z_target", "z_pred", "z_all", "action_condition"):
                arrays[key].append(tensor_to_numpy(latents[key]))
            append_optional_arrays(arrays["optional"], batch)
            metadata.extend(rows)

    if not metadata:
        raise ValueError("No samples were exported. Check --split, --max-samples, and dataset length.")

    summary = save_outputs(args, cfg, arrays, metadata, checkpoint_path, incompatible)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
