#!/usr/bin/env python
"""Visualize exported JEPA latents.

The script consumes the files written by tools/jepa_viz/export_latents.py:

  latent-dir/
    latents.npz or latents.pt
    metadata.jsonl

It writes PCA/UMAP scatter plots, trajectory plots, nearest-neighbor HTML, and
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
IMAGE_KEYS = ("image_path", "frame_path", "pixels_path", "image", "pixels")


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
    parser.add_argument("--trajectory-count", type=int, default=4, help="Number of episode/sequence trajectories to draw.")
    parser.add_argument("--trajectory-latent-step", default="last", choices=("first", "last", "mean"))
    parser.add_argument("--nn-latent", default="z_target", choices=LATENT_KEYS, help="Latent space for retrieval.")
    parser.add_argument("--nn-queries", type=int, default=8, help="Number of query samples for retrieval.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest neighbors per query.")
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


def color_values(arrays, metadata, latent_key, color_by):
    array = np.asarray(arrays[latent_key])
    n_samples = array.shape[0]
    n_steps = array.shape[1] if array.ndim == 3 else 1

    if color_by in ("time", "time_index", "frame", "frame_index"):
        values, source = array_time_values(arrays, latent_key)
        if values is not None:
            return values, source

    if color_by == "action" and "action" in arrays:
        action = np.asarray(arrays["action"])
        if action.shape[0] == n_samples and action.ndim >= 3 and action.shape[1] == n_steps:
            return vector_norm(action.reshape(n_samples * n_steps, *action.shape[2:])), "action_norm"
        if action.shape[0] == n_samples:
            return repeated_sample_values(vector_norm(action), n_steps), "action_norm"

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


def fit_transform_2d(points, method, seed):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        raise ValueError("Need at least two latent points for 2D projection.")

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=seed)
            return reducer.fit_transform(points), "UMAP"
        except ImportError:
            method = "pca"

    if method == "pca":
        try:
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2, random_state=seed)
            return reducer.fit_transform(points), "PCA"
        except ImportError:
            centered = points - points.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            return centered @ vt[:2].T, "PCA"

    raise ValueError(f"Unknown projection method: {method}")


def plot_scatter(embedding, values, title, out_path, color_label):
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
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_projection_scatter(arrays, metadata, latent_key, method, args, out_dir):
    points, step_index = flatten_latent(arrays[latent_key])
    colors, color_source = color_values(arrays, metadata, latent_key, args.color_by)
    labels = {"color": np.asarray(colors), "step": step_index}
    points, labels = sample_points(points, labels, args.max_points, args.seed)
    embedding, used_method = fit_transform_2d(points, method, args.seed)
    prefix = "umap_fallback_pca" if method == "umap" and used_method == "PCA" else used_method.lower()
    filename = f"{prefix}_{latent_key}_by_{safe_name(color_source)}.png"
    plot_scatter(
        embedding,
        labels["color"],
        f"{used_method} {latent_key} by {color_source}",
        out_dir / filename,
        color_source,
    )
    return filename


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


def plot_trajectories(arrays, metadata, args, out_dir):
    z_target = sample_level_latent(arrays["z_target"], args.trajectory_latent_step)
    z_pred = sample_level_latent(arrays["z_pred"], args.trajectory_latent_step)
    combined = np.concatenate([z_target, z_pred], axis=0)

    group_key, time_key, groups = grouped_indices(metadata)
    selected = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[: args.trajectory_count]
    if not selected:
        return None

    goal_latent = None
    for key in ("z_goal", "goal_latent", "goal_emb"):
        if key in arrays:
            goal_latent = sample_level_latent(arrays[key], args.trajectory_latent_step)
            break

    projection_input = np.concatenate([combined, goal_latent], axis=0) if goal_latent is not None else combined
    projected, _ = fit_transform_2d(projection_input, "pca", args.seed)
    target_2d = projected[: len(z_target)]
    pred_2d = projected[len(z_target) : len(z_target) + len(z_pred)]
    goal_2d = projected[len(combined) :] if goal_latent is not None else None

    cols = 2
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 5.0), squeeze=False)

    for ax, (group_value, indices) in zip(axes.reshape(-1), selected):
        indices = sort_group(indices, metadata, time_key)
        ax.plot(target_2d[indices, 0], target_2d[indices, 1], "-o", ms=3, lw=1.5, label="target")
        ax.plot(pred_2d[indices, 0], pred_2d[indices, 1], "-o", ms=3, lw=1.5, label="pred")
        if goal_2d is not None:
            ax.scatter(goal_2d[indices[-1], 0], goal_2d[indices[-1], 1], marker="*", s=120, label="goal", color="black")
        ax.set_title(f"{group_key}={group_value}")
        ax.set_xlabel("PCA dim 1")
        ax.set_ylabel("PCA dim 2")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes.reshape(-1)[len(selected) :]:
        ax.axis("off")

    fig.suptitle(f"Latent Trajectories ({args.trajectory_latent_step} step, sorted by {time_key})")
    fig.tight_layout()
    filename = "latent_trajectory.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def pairwise_cosine_values(points, max_points, seed):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        points = points[np.sort(rng.choice(points.shape[0], size=max_points, replace=False))]
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-12)
    sim = points @ points.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return sim[mask]


def plot_collapse_diagnostics(arrays, args, out_dir):
    points = sample_level_latent(arrays["z_target"], args.trajectory_latent_step)
    if points.shape[0] > args.max_diagnostic_points:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(points.shape[0], size=args.max_diagnostic_points, replace=False))
        points = points[keep]

    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    feature_std = points.std(axis=0)
    active_dims = int((feature_std > 1e-2).sum())
    pairwise = pairwise_cosine_values(points, args.max_diagnostic_points, args.seed)

    written = []

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(np.arange(1, len(eigvals) + 1), eigvals, lw=1.8)
    ax.set_yscale("log")
    ax.set_title("Covariance Eigenvalue Spectrum")
    ax.set_xlabel("dimension rank")
    ax.set_ylabel("eigenvalue")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    filename = "covariance_eigenvalue_spectrum.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(pairwise, bins=60, range=(-1, 1), color="#4c78a8", alpha=0.85)
    ax.set_title("Pairwise Cosine Histogram")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    filename = "pairwise_cosine_histogram.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(feature_std, bins=60, color="#59a14f", alpha=0.85)
    ax.axvline(1e-2, color="black", linestyle="--", linewidth=1.2, label="active threshold=1e-2")
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

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.bar(["active", "inactive"], [active_dims, points.shape[1] - active_dims], color=["#59a14f", "#e15759"])
    ax.set_title("Active Dimension Count")
    ax.set_ylabel("dimensions")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    filename = "active_dimension_count.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    written.append(filename)

    summary = {
        "latent_key": "z_target",
        "num_points": int(points.shape[0]),
        "latent_dim": int(points.shape[1]),
        "active_dim_threshold": 1e-2,
        "active_dim_count": active_dims,
        "feature_std_mean": float(feature_std.mean()),
        "feature_std_min": float(feature_std.min()),
        "feature_std_max": float(feature_std.max()),
        "pairwise_cosine_mean": float(pairwise.mean()) if pairwise.size else None,
        "pairwise_cosine_std": float(pairwise.std()) if pairwise.size else None,
        "largest_eigenvalue": float(eigvals[0]) if eigvals.size else None,
    }
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

    for latent_key in LATENT_KEYS:
        written.append(plot_projection_scatter(arrays, metadata, latent_key, "pca", args, out_dir))
        written.append(plot_projection_scatter(arrays, metadata, latent_key, "umap", args, out_dir))

    trajectory = plot_trajectories(arrays, metadata, args, out_dir)
    if trajectory:
        written.append(trajectory)

    retrieval = write_retrieval_html(arrays, metadata, args, out_dir)
    if retrieval:
        written.append(retrieval)

    written.extend(plot_collapse_diagnostics(arrays, args, out_dir))
    write_summary(out_dir, latent_path, arrays, metadata, written, args.color_by)

    print(json.dumps({"out": str(out_dir), "files": written}, indent=2))


if __name__ == "__main__":
    main()
