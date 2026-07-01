#!/usr/bin/env python
"""Visualize exported JEPA latents.

The script consumes the files written by tools/jepa_viz/export_latents.py:

  latent-dir/
    latents.npz or latents.pt
    metadata.jsonl

It writes PCA/UMAP scatter plots, target-pred alignment plots, nearest-neighbor HTML, and
collapse diagnostics to tools/jepa_viz/output/latent_viz by default.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from datetime import datetime
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
RUN_NAME = os.environ.get("JEPA_VIZ_RUN_NAME") or datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("JEPA_VIZ_OUTPUT_ROOT", TOOL_DIR / "output" / RUN_NAME))
DEFAULT_LATENT_DIR = DEFAULT_OUTPUT_ROOT / "latents"
DEFAULT_VIZ_DIR = DEFAULT_OUTPUT_ROOT / "latent_viz"

os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_OUTPUT_ROOT / ".cache"))
os.environ.setdefault("XDG_CONFIG_HOME", str(DEFAULT_OUTPUT_ROOT / ".config"))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TMPDIR", str(DEFAULT_OUTPUT_ROOT / "tmp"))
for key in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR", "TMPDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LATENT_KEYS = ("z_context", "z_target", "z_pred")
GROUP_KEYS = ("episode_idx", "ep_idx", "video_id", "video", "sequence_id", "sequence", "sample_id")
TIME_KEYS = ("step_idx", "time_idx", "time_index", "frame_idx", "frame_index", "timestep", "export_index")
HORIZON_KEYS = ("horizon", "future_horizon", "future_horizon_index", "target_horizon", "pred_horizon", "horizon_idx")
IMAGE_KEYS = ("image_path", "frame_path", "pixels_path", "image", "pixels")
ACTION_KEYS = ("action", "action_condition")
LABEL_GROUP_CANDIDATES = (
    ("task", ("task", "task_id", "instruction", "task_instruction")),
    ("action", ACTION_KEYS),
    ("episode", ("episode_idx", "ep_idx", "video_id", "sequence_id")),
    ("source", ("source", "source_id", "dataset", "dataset_id")),
)
ACTION_ABLATION_PRED_KEYS = {
    "normal": ("z_pred",),
    "shuffled": ("z_pred_action_shuffled", "z_pred_condition_shuffled", "z_pred_shuffled_action", "z_pred_shuffled_condition"),
    "zero": ("z_pred_action_zero", "z_pred_zero_action", "z_pred_condition_removed", "z_pred_no_condition", "z_pred_without_condition"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize exported JEPA latents.")
    parser.add_argument(
        "--latent-dir",
        default=str(DEFAULT_LATENT_DIR),
        help="Directory containing latents.npz/.pt and metadata.jsonl.",
    )
    parser.add_argument("--out", default=str(DEFAULT_VIZ_DIR), help="Output directory for visualizations.")
    parser.add_argument(
        "--color-by",
        default="source",
        help="Metadata or array key for coloring: source, dataset, task, action, object, success, time_index, ...",
    )
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum points per scatter plot.")
    parser.add_argument("--max-diagnostic-points", type=int, default=3000, help="Maximum points for diagnostics.")
    parser.add_argument("--active-threshold", type=float, default=1e-2, help="Absolute feature std threshold for active dimensions.")
    parser.add_argument(
        "--active-relative-threshold",
        type=float,
        default=0.0,
        help="Relative active threshold as a fraction of the max feature std for each latent.",
    )
    parser.add_argument(
        "--pairwise-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use density on pairwise cosine histograms.",
    )
    parser.add_argument(
        "--alignment-count",
        "--trajectory-count",
        dest="trajectory_count",
        type=int,
        default=4,
        help="Number of target-pred alignment panels to draw.",
    )
    parser.add_argument("--trajectory-latent-step", default="last", choices=("first", "last", "mean"))
    parser.add_argument("--nn-latent", default="z_target", choices=LATENT_KEYS, help="Latent space for retrieval.")
    parser.add_argument("--nn-queries", type=int, default=8, help="Number of query samples for retrieval.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest neighbors per query.")
    parser.add_argument(
        "--max-action-components",
        type=int,
        default=8,
        help="Maximum action components to plot for delta_z PCA. Use 0 or negative to plot all components.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_latents(latent_dir):
    latent_dir = Path(latent_dir).expanduser().resolve()
    npz_path = latent_dir / "latents.npz"
    pt_path = latent_dir / "latents.pt"

    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}
        latent_path = npz_path
    elif pt_path.exists():
        import torch

        payload = torch.load(pt_path, map_location="cpu")
        arrays = {
            key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            for key, value in payload.items()
        }
        latent_path = pt_path
    else:
        raise FileNotFoundError(f"No latents.npz or latents.pt found in {latent_dir}")

    missing = [key for key in LATENT_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"Missing latent arrays in {latent_path}: {missing}")

    metadata_path = latent_dir / "metadata.jsonl"
    metadata = []
    if metadata_path.exists():
        with metadata_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))

    sample_count = arrays["z_target"].shape[0]
    if not metadata:
        metadata = [{"sample_id": idx, "export_index": idx} for idx in range(sample_count)]
    if len(metadata) != sample_count:
        raise ValueError(f"metadata rows ({len(metadata)}) != latent samples ({sample_count})")

    return arrays, metadata, latent_path


def ensure_dir(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_latent(array):
    array = np.asarray(array)
    if array.ndim == 2:
        return array.astype(np.float32), np.zeros(array.shape[0], dtype=np.int64)
    if array.ndim == 3:
        n_samples, n_steps, dim = array.shape
        step_index = np.broadcast_to(np.arange(n_steps), (n_samples, n_steps)).reshape(-1)
        return array.reshape(n_samples * n_steps, dim).astype(np.float32), step_index.astype(np.int64)
    raise ValueError(f"Expected latent shape [N,D] or [N,T,D], got {array.shape}")


def sample_points(points, labels, max_points, seed):
    if points.shape[0] <= max_points:
        return points, labels
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(points.shape[0], size=max_points, replace=False))
    labels = {key: value[keep] for key, value in labels.items()}
    return points[keep], labels


def metadata_values(metadata, key):
    aliases = {
        "dataset/source": ("source", "source_id", "dataset", "dataset_id"),
        "source": ("source", "source_id", "dataset", "dataset_id"),
        "dataset": ("dataset", "dataset_id", "source", "source_id"),
        "task": ("task", "task_id", "instruction", "task_instruction"),
        "object": ("object", "object_id", "category", "category_id"),
        "success": ("success", "is_success", "label"),
        "time": TIME_KEYS,
        "time_index": TIME_KEYS,
        "episode": ("episode_idx", "ep_idx", "video_id", "sequence_id"),
        "horizon": HORIZON_KEYS,
    }
    keys = aliases.get(key, (key,))
    for candidate in keys:
        values = [row.get(candidate) for row in metadata]
        if any(value is not None for value in values):
            return np.asarray(values, dtype=object), candidate
    return np.asarray([None for _ in metadata], dtype=object), key


def array_time_values(arrays, latent_key):
    if latent_key == "z_context" and "context_time_index" in arrays:
        return np.asarray(arrays["context_time_index"]).reshape(-1), "context_time_index"
    if latent_key in ("z_target", "z_pred") and "target_time_index" in arrays:
        return np.asarray(arrays["target_time_index"]).reshape(-1), "target_time_index"
    return None, None


def repeated_sample_values(values, n_steps):
    values = np.asarray(values, dtype=object)
    if n_steps == 1:
        return values
    return np.repeat(values, n_steps)


def vector_norm(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return np.asarray([values.item()])
    if values.ndim == 1:
        return values
    return np.linalg.norm(values.reshape(values.shape[0], -1), axis=1)


def first_array(arrays, keys):
    for key in keys:
        if key in arrays:
            return key, np.asarray(arrays[key])
    return None, None


def action_array(arrays):
    return first_array(arrays, ACTION_KEYS)


def action_component_index(color_by):
    prefixes = ("action_", "action_component_", "action_dim_")
    for prefix in prefixes:
        if color_by.startswith(prefix):
            try:
                return int(color_by[len(prefix) :])
            except ValueError:
                return None
    if color_by.startswith("action[") and color_by.endswith("]"):
        try:
            return int(color_by[len("action[") : -1])
        except ValueError:
            return None
    return None


def action_values_for_steps(arrays, n_samples, n_steps, mode="norm", component=None):
    action_key, action = action_array(arrays)
    if action is None or action.shape[0] != n_samples:
        return None, None

    flat = action.reshape(action.shape[0], -1) if action.ndim <= 2 else action.reshape(action.shape[0], action.shape[1], -1)
    if action.ndim >= 3 and action.shape[1] == n_steps:
        if mode == "component":
            if component is None or component >= flat.shape[-1]:
                return None, None
            values = flat[:, :, component].reshape(n_samples * n_steps)
            return values, f"{action_key}_{component}"
        return vector_norm(flat.reshape(n_samples * n_steps, flat.shape[-1])), f"{action_key}_norm"

    sample_flat = action.reshape(n_samples, -1)
    if mode == "component":
        if component is None or component >= sample_flat.shape[1]:
            return None, None
        return repeated_sample_values(sample_flat[:, component], n_steps), f"{action_key}_{component}"
    return repeated_sample_values(vector_norm(sample_flat), n_steps), f"{action_key}_norm"


def color_values(arrays, metadata, latent_key, color_by):
    array = np.asarray(arrays[latent_key])
    n_samples = array.shape[0]
    n_steps = array.shape[1] if array.ndim == 3 else 1

    if color_by in ("time", "time_index", "frame", "frame_index"):
        values, source = array_time_values(arrays, latent_key)
        if values is not None:
            return values, source

    if color_by in ("action", "action_norm"):
        values, source = action_values_for_steps(arrays, n_samples, n_steps, mode="norm")
        if values is not None:
            return values, source

    component = action_component_index(color_by)
    if component is not None:
        values, source = action_values_for_steps(arrays, n_samples, n_steps, mode="component", component=component)
        if values is not None:
            return values, source

    if color_by in ("horizon", "future_horizon", "future_horizon_index"):
        values, source = horizon_values(arrays, metadata, "last")
        if values is not None:
            return repeated_sample_values(values, n_steps), source

    if color_by in ("prediction_error", "pred_error", "mse", "cosine_error", "prediction_cosine"):
        z_target = sample_level_latent(arrays["z_target"], "last")
        z_pred = sample_level_latent(arrays["z_pred"], "last")
        cosine, mse = target_pred_metrics(z_target, z_pred)
        values = cosine if color_by == "prediction_cosine" else mse
        source = "prediction_cosine" if color_by == "prediction_cosine" else "prediction_mse"
        return repeated_sample_values(values, n_steps), source

    if color_by in arrays:
        values = np.asarray(arrays[color_by])
        if values.shape[0] == n_samples and values.ndim >= 2 and values.shape[1] == n_steps:
            return values.reshape(n_samples * n_steps, *values.shape[2:]).reshape(n_samples * n_steps, -1)[:, 0], color_by
        if values.shape[0] == n_samples:
            sample_values = vector_norm(values) if values.ndim > 1 else values
            return repeated_sample_values(sample_values, n_steps), color_by

    values, source = metadata_values(metadata, color_by)
    return repeated_sample_values(values, n_steps), source


def numeric_or_categorical(values):
    values = np.asarray(values, dtype=object)
    numeric = []
    for value in values:
        if value is None:
            numeric.append(np.nan)
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).reshape(-1)
            numeric.append(float(np.linalg.norm(arr)))
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return False, values
    numeric = np.asarray(numeric, dtype=float)
    if np.isnan(numeric).all():
        return False, values
    return True, numeric


def plot_scatter(embedding, values, title, out_path, color_label, xlim=None, ylim=None):
    is_numeric, color = numeric_or_categorical(values)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    if is_numeric:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=10, alpha=0.72, cmap="viridis", linewidths=0)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)
    else:
        labels = np.asarray(["<missing>" if value is None else str(value) for value in color], dtype=object)
        unique = np.unique(labels)
        if len(unique) > 20:
            labels = np.asarray([str(value)[:32] for value in labels], dtype=object)
            unique = np.unique(labels)
        palette = plt.get_cmap("tab20", max(len(unique), 1))
        for idx, label in enumerate(unique):
            mask = labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.72, color=palette(idx), label=label, linewidths=0)
        ax.legend(loc="best", fontsize=7, frameon=False, markerscale=1.8)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fit_shared_pca(points, seed):
    points = np.asarray(points, dtype=np.float32)
    try:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(points), "PCA"
    except ImportError:
        centered = points - points.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T, "PCA"


def fit_shared_umap(points, seed):
    try:
        import umap
    except ImportError:
        return None, None

    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(np.asarray(points, dtype=np.float32)), "UMAP"


def shared_projection_inputs(arrays, metadata, args):
    per_key = {}
    combined = []
    start = 0
    for latent_key in LATENT_KEYS:
        points, step_index = flatten_latent(arrays[latent_key])
        colors, color_source = color_values(arrays, metadata, latent_key, args.color_by)
        labels = {"color": np.asarray(colors), "step": step_index}
        points, labels = sample_points(points, labels, args.max_points, args.seed)
        end = start + points.shape[0]
        per_key[latent_key] = {
            "points": points,
            "labels": labels,
            "slice": slice(start, end),
            "color_source": color_source,
        }
        combined.append(points)
        start = end
    return np.concatenate(combined, axis=0), per_key


def plot_shared_projection_scatter(arrays, metadata, method, args, out_dir):
    combined, per_key = shared_projection_inputs(arrays, metadata, args)
    if method == "pca":
        embedding, used_method = fit_shared_pca(combined, args.seed)
        prefix = "pca"
    elif method == "umap":
        embedding, used_method = fit_shared_umap(combined, args.seed)
        if embedding is None:
            return []
        prefix = "umap"
    else:
        raise ValueError(f"Unknown projection method: {method}")

    xlim, ylim = axis_limits(embedding)
    written = []
    for latent_key, info in per_key.items():
        latent_embedding = embedding[info["slice"]]
        color_source = info["color_source"]
        filename = f"{prefix}_{latent_key}_shared_by_{safe_name(color_source)}.png"
        plot_scatter(
            latent_embedding,
            info["labels"]["color"],
            f"{used_method} {latent_key} by {color_source} (shared basis)",
            out_dir / filename,
            color_source,
            xlim=xlim,
            ylim=ylim,
        )
        written.append(filename)
    return written


def sample_level_latent(array, step_mode):
    array = np.asarray(array)
    if array.ndim == 2:
        return array.astype(np.float32)
    if step_mode == "first":
        return array[:, 0].astype(np.float32)
    if step_mode == "last":
        return array[:, -1].astype(np.float32)
    return array.mean(axis=1).astype(np.float32)


def first_present_key(metadata, keys):
    for key in keys:
        if any(key in row and row[key] is not None for row in metadata):
            return key
    return None


def get_row_value(row, key, default=None):
    value = row.get(key, default)
    if isinstance(value, list):
        return tuple(value)
    return value


def grouped_indices(metadata):
    group_key = first_present_key(metadata, GROUP_KEYS)
    time_key = first_present_key(metadata, TIME_KEYS)
    groups = {}
    for idx, row in enumerate(metadata):
        group_value = get_row_value(row, group_key, idx) if group_key else idx
        groups.setdefault(group_value, []).append(idx)
    return group_key or "sample_id", time_key or "export_index", groups


def sort_group(indices, metadata, time_key):
    def sort_value(idx):
        value = metadata[idx].get(time_key, idx)
        if isinstance(value, list):
            return value[0] if value else idx
        try:
            return float(value)
        except (TypeError, ValueError):
            return idx

    return sorted(indices, key=sort_value)


def axis_limits(*point_sets):
    points = np.concatenate([np.asarray(points) for points in point_sets if points is not None and len(points)], axis=0)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def target_pred_metrics(z_target, z_pred):
    target_norm = np.linalg.norm(z_target, axis=1)
    pred_norm = np.linalg.norm(z_pred, axis=1)
    cosine = (z_target * z_pred).sum(axis=1) / np.maximum(target_norm * pred_norm, 1e-12)
    mse = ((z_pred - z_target) ** 2).mean(axis=1)
    return cosine, mse


def sample_id(metadata, idx):
    row = metadata[idx]
    for key in ("sample_id", "export_index", "episode_idx", "ep_idx", "frame_idx", "step_idx"):
        if key in row and row[key] is not None:
            return row[key]
    return idx


def select_step_value(values, step_mode):
    values = np.asarray(values)
    if values.ndim == 0:
        return values.item()
    if values.ndim == 1:
        if step_mode == "first":
            return values[0]
        if step_mode == "last":
            return values[-1]
        return values.mean() if np.issubdtype(values.dtype, np.number) else values[len(values) // 2]
    return select_step_value(values.reshape(-1), step_mode)


def horizon_values(arrays, metadata, step_mode):
    key = first_present_key(metadata, HORIZON_KEYS)
    if key:
        values = []
        for row in metadata:
            value = row.get(key)
            if isinstance(value, list):
                value = select_step_value(value, step_mode)
            values.append(value)
        return np.asarray(values, dtype=object), key

    for key in HORIZON_KEYS:
        if key not in arrays:
            continue
        values = np.asarray(arrays[key])
        if values.shape[0] == len(metadata):
            if values.ndim == 1:
                return values.astype(object), key
            return np.asarray([select_step_value(value, step_mode) for value in values], dtype=object), key
        if values.ndim == 1:
            return np.asarray([select_step_value(values, step_mode) for _ in metadata], dtype=object), key
    return None, None


def format_metric(value):
    return f"{float(value):.4f}"


def plot_alignment_panels(target_2d, pred_2d, indices, metadata, cosine, mse, args, out_dir, filename, subtitle=None):
    if not indices:
        return None

    xlim, ylim = axis_limits(target_2d, pred_2d)
    cols = 2
    rows = int(math.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 5.0), squeeze=False)

    for ax, idx in zip(axes.reshape(-1), indices):
        ax.plot(
            [target_2d[idx, 0], pred_2d[idx, 0]],
            [target_2d[idx, 1], pred_2d[idx, 1]],
            color="0.65",
            linewidth=1.4,
            zorder=1,
        )
        ax.scatter(target_2d[idx, 0], target_2d[idx, 1], s=48, label="target", color="#4c78a8", zorder=2)
        ax.scatter(pred_2d[idx, 0], pred_2d[idx, 1], s=48, label="pred", color="#f58518", zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"sample_id={sample_id(metadata, idx)} | cos={format_metric(cosine[idx])} | mse={format_metric(mse[idx])}")
        ax.set_xlabel("PCA dim 1")
        ax.set_ylabel("PCA dim 2")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes.reshape(-1)[len(indices) :]:
        ax.axis("off")

    title = f"Target-Pred Latent Alignment ({args.trajectory_latent_step} step)"
    if subtitle:
        title += f" - {subtitle}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_alignment_global(target_2d, pred_2d, indices, cosine, mse, out_dir, filename, subtitle=None):
    if not indices:
        return None

    xlim, ylim = axis_limits(target_2d, pred_2d)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for idx in indices:
        ax.plot(
            [target_2d[idx, 0], pred_2d[idx, 0]],
            [target_2d[idx, 1], pred_2d[idx, 1]],
            color="0.7",
            linewidth=0.75,
            alpha=0.55,
            zorder=1,
        )
    ax.scatter(target_2d[indices, 0], target_2d[indices, 1], s=16, label="target", color="#4c78a8", alpha=0.82, zorder=2)
    ax.scatter(pred_2d[indices, 0], pred_2d[indices, 1], s=16, label="pred", color="#f58518", alpha=0.82, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    title = "Global Target-Pred Latent Alignment"
    if subtitle:
        title += f" - {subtitle}"
    title += f"\nmean cos={format_metric(cosine[indices].mean())}, mean mse={format_metric(mse[indices].mean())}"
    ax.set_title(title)
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def alignment_panel_indices(mse, count):
    count = max(int(count), 1)
    order = np.argsort(mse)
    anchors = [order[0], order[len(order) // 2], order[-1]]
    if count <= 3:
        return list(dict.fromkeys(int(idx) for idx in anchors[:count]))

    quantiles = np.linspace(0, 1, count)
    selected = [order[int(round(q * (len(order) - 1)))] for q in quantiles]
    selected = [anchors[0], anchors[1], anchors[2], *selected]
    return list(dict.fromkeys(int(idx) for idx in selected))[:count]


def plot_target_pred_alignment(arrays, metadata, args, out_dir):
    z_target = sample_level_latent(arrays["z_target"], args.trajectory_latent_step)
    z_pred = sample_level_latent(arrays["z_pred"], args.trajectory_latent_step)
    combined = np.concatenate([z_target, z_pred], axis=0)

    projected, _ = fit_shared_pca(combined, args.seed)
    target_2d = projected[: len(z_target)]
    pred_2d = projected[len(z_target) : len(z_target) + len(z_pred)]
    cosine, mse = target_pred_metrics(z_target, z_pred)

    written = []
    all_indices = list(range(len(metadata)))
    panel_indices = alignment_panel_indices(mse, min(args.trajectory_count, len(all_indices)))

    filename = plot_alignment_panels(
        target_2d,
        pred_2d,
        panel_indices,
        metadata,
        cosine,
        mse,
        args,
        out_dir,
        "target_pred_latent_alignment.png",
    )
    if filename:
        written.append(filename)

    filename = plot_alignment_global(
        target_2d,
        pred_2d,
        all_indices,
        cosine,
        mse,
        out_dir,
        "target_pred_latent_alignment_global.png",
    )
    if filename:
        written.append(filename)

    horizons, horizon_key = horizon_values(arrays, metadata, args.trajectory_latent_step)
    if horizons is not None:
        labels = np.asarray(["<missing>" if value is None else str(value) for value in horizons], dtype=object)
        for horizon in np.unique(labels):
            horizon_indices = np.flatnonzero(labels == horizon).astype(int).tolist()
            safe_horizon = safe_name(horizon)
            subtitle = f"{horizon_key}={horizon}"
            filename = plot_alignment_global(
                target_2d,
                pred_2d,
                horizon_indices,
                cosine,
                mse,
                out_dir,
                f"target_pred_latent_alignment_global_{safe_name(horizon_key)}_{safe_horizon}.png",
                subtitle=subtitle,
            )
            if filename:
                written.append(filename)

            horizon_mse = mse[horizon_indices]
            local_indices = alignment_panel_indices(horizon_mse, min(args.trajectory_count, len(horizon_indices)))
            panel_indices = [horizon_indices[idx] for idx in local_indices]
            filename = plot_alignment_panels(
                target_2d,
                pred_2d,
                panel_indices,
                metadata,
                cosine,
                mse,
                args,
                out_dir,
                f"target_pred_latent_alignment_{safe_name(horizon_key)}_{safe_horizon}.png",
                subtitle=subtitle,
            )
            if filename:
                written.append(filename)

    return written


def action_sample_matrix(arrays):
    _, action = action_array(arrays)
    if action is None:
        return None
    return action.reshape(action.shape[0], -1).astype(np.float32)


def plot_delta_z_action_analysis(arrays, metadata, args, out_dir):
    z_context = sample_level_latent(arrays["z_context"], args.trajectory_latent_step)
    z_pred = sample_level_latent(arrays["z_pred"], args.trajectory_latent_step)
    if z_context.shape != z_pred.shape:
        return []

    delta_z = z_pred - z_context
    if delta_z.shape[0] < 2:
        return []
    embedding, _ = fit_shared_pca(delta_z, args.seed)
    xlim, ylim = axis_limits(embedding)
    written = []

    action = action_sample_matrix(arrays)
    if action is None or action.shape[0] != delta_z.shape[0]:
        values, source = metadata_values(metadata, args.color_by)
        filename = f"delta_z_pca_by_{safe_name(source)}.png"
        plot_scatter(
            embedding,
            values,
            f"PCA delta_z by {source}",
            out_dir / filename,
            source,
            xlim=xlim,
            ylim=ylim,
        )
        return [filename]

    action_norm = vector_norm(action)
    filename = "delta_z_pca_by_action_norm.png"
    plot_scatter(
        embedding,
        action_norm,
        "PCA delta_z by action norm",
        out_dir / filename,
        "action_norm",
        xlim=xlim,
        ylim=ylim,
    )
    written.append(filename)

    component_count = action.shape[1]
    if args.max_action_components > 0:
        component_count = min(component_count, args.max_action_components)
    for component in range(component_count):
        filename = f"delta_z_pca_by_action_{component}.png"
        plot_scatter(
            embedding,
            action[:, component],
            f"PCA delta_z by action[{component}]",
            out_dir / filename,
            f"action_{component}",
            xlim=xlim,
            ylim=ylim,
        )
        written.append(filename)
    return written


def plot_action_ablation(arrays, args, out_dir):
    variants = {}
    for name, keys in ACTION_ABLATION_PRED_KEYS.items():
        key, value = first_array(arrays, keys)
        if value is not None:
            variants[name] = (key, value)
    if len(variants) <= 1:
        return []

    z_target = sample_level_latent(arrays["z_target"], args.trajectory_latent_step)
    names = []
    mse_values = []
    cosine_values = []
    for name, (_, pred) in variants.items():
        pred = sample_level_latent(pred, args.trajectory_latent_step)
        if pred.shape != z_target.shape:
            continue
        cosine, mse = target_pred_metrics(z_target, pred)
        names.append(name)
        mse_values.append(float(mse.mean()))
        cosine_values.append(float(cosine.mean()))

    if len(names) <= 1:
        return []

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axes[0].bar(x, mse_values, color="#e15759")
    axes[0].set_title("Prediction MSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.18)

    axes[1].bar(x, cosine_values, color="#4c78a8")
    axes[1].set_title("Prediction Cosine")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("Action Condition Ablation")
    fig.tight_layout()
    filename = "action_condition_ablation.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)

    summary = {
        name: {"mse": mse, "cosine": cosine}
        for name, mse, cosine in zip(names, mse_values, cosine_values)
    }
    summary_name = "action_condition_ablation_summary.json"
    with (out_dir / summary_name).open("w") as f:
        json.dump(summary, f, indent=2)
    return [filename, summary_name]


def pairwise_cosine_values(points, max_points, seed, labels=None):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(points.shape[0], size=max_points, replace=False))
        points = points[keep]
        labels = None if labels is None else np.asarray(labels, dtype=object)[keep]
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-12)
    sim = points @ points.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    values = sim[mask]
    if labels is None:
        return values, None, None
    label_eq = labels[:, None] == labels[None, :]
    same = values[label_eq[mask]]
    different = values[~label_eq[mask]]
    return values, same, different


def diagnostic_points(arrays, latent_key, args):
    points = sample_level_latent(arrays[latent_key], args.trajectory_latent_step)
    if points.shape[0] > args.max_diagnostic_points:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(points.shape[0], size=args.max_diagnostic_points, replace=False))
        points = points[keep]
    return points.astype(np.float32)


def explained_variance(points):
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    total = float(np.maximum(eigvals.sum(), 1e-12))
    ratio = eigvals / total
    return eigvals, ratio


def effective_rank(evr):
    evr = np.asarray(evr, dtype=np.float64)
    evr = evr[evr > 0]
    if evr.size == 0:
        return 0.0
    entropy = -(evr * np.log(evr)).sum()
    return float(np.exp(entropy))


def participation_ratio(eigvals):
    eigvals = np.asarray(eigvals, dtype=np.float64)
    denom = np.square(eigvals).sum()
    if denom <= 0:
        return 0.0
    return float(np.square(eigvals.sum()) / denom)


def active_threshold(feature_std, args):
    relative = float(args.active_relative_threshold) * float(feature_std.max() if feature_std.size else 0.0)
    return max(float(args.active_threshold), relative)


def cumulative_top(evr, k):
    if evr.size == 0:
        return 0.0
    return float(evr[: min(k, evr.size)].sum())


def pairwise_label_values(arrays, metadata, n_samples):
    for label_name, keys in LABEL_GROUP_CANDIDATES:
        if label_name == "action":
            action_key, action = action_array(arrays)
            if action is None or action.shape[0] != n_samples:
                continue
            norms = vector_norm(action.reshape(action.shape[0], -1))
            edges = np.unique(np.quantile(norms, np.linspace(0, 1, 5)))
            if len(edges) <= 2:
                labels = np.asarray([f"action_norm={value:.3g}" for value in norms], dtype=object)
            else:
                bins = np.digitize(norms, edges[1:-1], right=True)
                labels = np.asarray([f"action_q{bin_idx}" for bin_idx in bins], dtype=object)
            return labels, f"{action_key}_norm_bin"

        key = first_present_key(metadata, keys)
        if key:
            values = ["<missing>" if row.get(key) is None else str(row.get(key)) for row in metadata]
            if len(set(values)) > 1:
                return np.asarray(values, dtype=object), key
    return None, None


def plot_collapse_diagnostics(arrays, metadata, args, out_dir):
    diagnostics = {}
    written = []
    palette = {"z_context": "#4c78a8", "z_target": "#59a14f", "z_pred": "#f58518"}

    for latent_key in LATENT_KEYS:
        points = diagnostic_points(arrays, latent_key, args)
        eigvals, evr = explained_variance(points)
        feature_std = points.std(axis=0)
        threshold = active_threshold(feature_std, args)
        active_dims = int((feature_std > threshold).sum())
        pairwise, _, _ = pairwise_cosine_values(points, args.max_diagnostic_points, args.seed)
        diagnostics[latent_key] = {
            "points": points,
            "eigvals": eigvals,
            "evr": evr,
            "feature_std": feature_std,
            "active_threshold": threshold,
            "active_dims": active_dims,
            "pairwise": pairwise,
            "summary": {
                "num_points": int(points.shape[0]),
                "latent_dim": int(points.shape[1]),
                "active_dim_threshold_absolute": float(args.active_threshold),
                "active_dim_threshold_relative": float(args.active_relative_threshold),
                "active_dim_threshold_used": float(threshold),
                "active_dim_count": active_dims,
                "active_dim_total": int(points.shape[1]),
                "effective_rank": effective_rank(evr),
                "participation_ratio": participation_ratio(eigvals),
                "top10_explained_variance_ratio": cumulative_top(evr, 10),
                "top50_explained_variance_ratio": cumulative_top(evr, 50),
                "top100_explained_variance_ratio": cumulative_top(evr, 100),
                "feature_std_mean": float(feature_std.mean()),
                "feature_std_min": float(feature_std.min()),
                "feature_std_median": float(np.median(feature_std)),
                "feature_std_max": float(feature_std.max()),
                "pairwise_cosine_mean": float(pairwise.mean()) if pairwise.size else None,
                "pairwise_cosine_median": float(np.median(pairwise)) if pairwise.size else None,
                "pairwise_cosine_q95": float(np.quantile(pairwise, 0.95)) if pairwise.size else None,
                "pairwise_cosine_std": float(pairwise.std()) if pairwise.size else None,
                "largest_eigenvalue": float(eigvals[0]) if eigvals.size else None,
            },
        }

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for latent_key in LATENT_KEYS:
        eigvals = diagnostics[latent_key]["eigvals"]
        ax.plot(np.arange(1, len(eigvals) + 1), eigvals, lw=1.8, label=latent_key, color=palette[latent_key])
    ax.set_yscale("log")
    ax.set_title("Covariance Eigenvalue Spectrum")
    ax.set_xlabel("dimension rank")
    ax.set_ylabel("eigenvalue")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    filename = "covariance_eigenvalue_spectrum.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for latent_key in LATENT_KEYS:
        evr = diagnostics[latent_key]["evr"]
        ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), lw=1.8, label=latent_key, color=palette[latent_key])
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlabel("dimension rank")
    ax.set_ylabel("cumulative explained variance ratio")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    filename = "cumulative_explained_variance.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for latent_key in LATENT_KEYS:
        values = diagnostics[latent_key]["pairwise"]
        ax.hist(
            values,
            bins=60,
            range=(-1, 1),
            density=args.pairwise_density,
            alpha=0.38,
            label=latent_key,
            color=palette[latent_key],
        )
        if values.size:
            ax.axvline(values.mean(), color=palette[latent_key], linewidth=1.2, linestyle="-")
            ax.axvline(np.median(values), color=palette[latent_key], linewidth=1.2, linestyle="--")
            ax.axvline(np.quantile(values, 0.95), color=palette[latent_key], linewidth=1.2, linestyle=":")
    ax.set_title("Pairwise Cosine Histogram")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density" if args.pairwise_density else "count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    filename = "pairwise_cosine_histogram.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    labels, label_source = pairwise_label_values(arrays, metadata, len(metadata))
    if labels is not None:
        for latent_key in LATENT_KEYS:
            points = sample_level_latent(arrays[latent_key], args.trajectory_latent_step).astype(np.float32)
            _, same, different = pairwise_cosine_values(points, args.max_diagnostic_points, args.seed, labels=labels)
            if same is None or different is None or same.size == 0 or different.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.hist(same, bins=60, range=(-1, 1), density=args.pairwise_density, alpha=0.55, label=f"same {label_source}", color="#4c78a8")
            ax.hist(
                different,
                bins=60,
                range=(-1, 1),
                density=args.pairwise_density,
                alpha=0.55,
                label=f"different {label_source}",
                color="#e15759",
            )
            ax.set_title(f"Pairwise Cosine Same vs Different {label_source} ({latent_key})")
            ax.set_xlabel("cosine similarity")
            ax.set_ylabel("density" if args.pairwise_density else "count")
            ax.legend(frameon=False)
            ax.grid(alpha=0.18)
            fig.tight_layout()
            filename = f"pairwise_cosine_same_vs_different_{safe_name(label_source)}_{latent_key}.png"
            fig.savefig(out_dir / filename, dpi=160)
            plt.close(fig)
            written.append(filename)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for latent_key in LATENT_KEYS:
        feature_std = diagnostics[latent_key]["feature_std"]
        ax.hist(feature_std, bins=60, alpha=0.34, label=latent_key, color=palette[latent_key])
        stats = (feature_std.min(), np.median(feature_std), feature_std.max())
        for value, linestyle in zip(stats, (":", "--", "-")):
            ax.axvline(value, color=palette[latent_key], linestyle=linestyle, linewidth=1.0)
        ax.axvline(diagnostics[latent_key]["active_threshold"], color=palette[latent_key], linestyle="-.", linewidth=1.0)
    ax.set_title("Feature Std Distribution")
    ax.set_xlabel("per-dimension std")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    filename = "feature_std_distribution.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    active = [diagnostics[key]["active_dims"] for key in LATENT_KEYS]
    total = [diagnostics[key]["points"].shape[1] for key in LATENT_KEYS]
    inactive = [dim - active_dim for dim, active_dim in zip(total, active)]
    x = np.arange(len(LATENT_KEYS))
    ax.bar(x, active, color="#59a14f", label="active")
    ax.bar(x, inactive, bottom=active, color="#e15759", label="inactive")
    for idx, (active_dim, dim) in enumerate(zip(active, total)):
        ax.text(idx, dim, f"{active_dim}/{dim}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(LATENT_KEYS)
    ax.set_title("Active Dimension Count")
    ax.set_ylabel("dimensions")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    filename = "active_dimension_count.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    summary = {latent_key: diagnostics[latent_key]["summary"] for latent_key in LATENT_KEYS}
    if labels is not None:
        summary["same_different_label_source"] = label_source
    filename = "collapse_diagnostics_summary.json"
    with (out_dir / filename).open("w") as f:
        json.dump(summary, f, indent=2)
    written.append(filename)

    return written


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "_")


def metadata_snippet(row):
    keys = [
        "sample_id",
        "episode_idx",
        "ep_idx",
        "step_idx",
        "source",
        "dataset",
        "task",
        "task_id",
        "object",
        "object_id",
        "category",
        "category_id",
        "success",
        "is_success",
    ]
    parts = []
    for key in keys:
        if key in row:
            parts.append(f"{html.escape(key)}={html.escape(str(row[key]))}")
    return "<br>".join(parts) if parts else html.escape(json.dumps(row, ensure_ascii=False)[:400])


def image_html(row, arrays, idx, max_width=120):
    for key in IMAGE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return f'<img src="{html.escape(value)}" style="max-width:{max_width}px;max-height:{max_width}px">'
    for key in ("pixels", "image"):
        if key in arrays:
            # Keep the HTML self-contained without dumping image arrays. The exported
            # metadata remains useful even when original frames are unavailable.
            return f"<div>array:{html.escape(key)}[{idx}]</div>"
    return "<div class='placeholder'>no image</div>"


def topk_neighbors(points, query_indices, top_k):
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=top_k + 1, metric="euclidean")
        nn.fit(points)
        distances, indices = nn.kneighbors(points[query_indices])
        return distances[:, 1:], indices[:, 1:]
    except ImportError:
        diff = points[query_indices, None, :] - points[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        for row_idx, query_idx in enumerate(query_indices):
            distances[row_idx, query_idx] = np.inf
        indices = np.argsort(distances, axis=1)[:, :top_k]
        distances = np.take_along_axis(distances, indices, axis=1)
        return distances, indices


def write_retrieval_html(arrays, metadata, args, out_dir):
    points = sample_level_latent(arrays[args.nn_latent], args.trajectory_latent_step)
    n_samples = points.shape[0]
    if n_samples < 2:
        return None
    rng = np.random.default_rng(args.seed)
    query_count = min(args.nn_queries, n_samples)
    query_indices = np.sort(rng.choice(n_samples, size=query_count, replace=False))
    top_k = min(args.top_k, n_samples - 1)
    distances, neighbor_indices = topk_neighbors(points.astype(np.float32), query_indices, top_k)

    rows = []
    for q_pos, q_idx in enumerate(query_indices):
        cells = [
            "<td class='query'>"
            f"<div class='role'>query</div>{image_html(metadata[q_idx], arrays, q_idx)}"
            f"<div class='meta'>{metadata_snippet(metadata[q_idx])}</div>"
            "</td>"
        ]
        for rank, idx in enumerate(neighbor_indices[q_pos], start=1):
            cells.append(
                "<td>"
                f"<div class='role'>top {rank}, dist={distances[q_pos, rank - 1]:.4f}</div>"
                f"{image_html(metadata[idx], arrays, int(idx))}"
                f"<div class='meta'>{metadata_snippet(metadata[idx])}</div>"
                "</td>"
            )
        rows.append("<tr>" + "\n".join(cells) + "</tr>")

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>JEPA latent nearest neighbors</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; min-width: 150px; }}
    .query {{ background: #f7f7f7; }}
    .role {{ font-weight: 700; margin-bottom: 6px; }}
    .meta {{ font-size: 12px; color: #333; line-height: 1.35; margin-top: 6px; }}
    .placeholder {{ width: 120px; height: 80px; display: grid; place-items: center; background: #eee; color: #777; }}
  </style>
</head>
<body>
  <h1>JEPA latent nearest neighbors</h1>
  <p>latent={html.escape(args.nn_latent)}, step={html.escape(args.trajectory_latent_step)}, top_k={top_k}</p>
  <table>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    filename = "nearest_neighbors.html"
    with (out_dir / filename).open("w") as f:
        f.write(doc)
    return filename


def write_summary(out_dir, latent_path, arrays, metadata, written, color_by):
    summary = {
        "latent_path": str(latent_path),
        "num_samples": len(metadata),
        "arrays": {key: list(value.shape) for key, value in arrays.items() if hasattr(value, "shape")},
        "color_by": color_by,
        "files": written,
    }
    with (out_dir / "visualization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    written.append("visualization_summary.json")


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out)
    arrays, metadata, latent_path = load_latents(args.latent_dir)
    written = []

    written.extend(plot_shared_projection_scatter(arrays, metadata, "pca", args, out_dir))
    written.extend(plot_shared_projection_scatter(arrays, metadata, "umap", args, out_dir))

    written.extend(plot_target_pred_alignment(arrays, metadata, args, out_dir))
    written.extend(plot_delta_z_action_analysis(arrays, metadata, args, out_dir))
    written.extend(plot_action_ablation(arrays, args, out_dir))

    retrieval = write_retrieval_html(arrays, metadata, args, out_dir)
    if retrieval:
        written.append(retrieval)

    written.extend(plot_collapse_diagnostics(arrays, metadata, args, out_dir))
    write_summary(out_dir, latent_path, arrays, metadata, written, args.color_by)

    print(json.dumps({"out": str(out_dir), "files": written}, indent=2))


if __name__ == "__main__":
    main()
