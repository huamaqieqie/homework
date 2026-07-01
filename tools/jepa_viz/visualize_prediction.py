#!/usr/bin/env python
"""Visualize JEPA prediction quality from exported latents."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
RUN_NAME = os.environ.get("JEPA_VIZ_RUN_NAME") or datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("JEPA_VIZ_OUTPUT_ROOT", TOOL_DIR / "output" / RUN_NAME))
DEFAULT_LATENT_DIR = DEFAULT_OUTPUT_ROOT / "latents"
DEFAULT_PREDICTION_DIR = DEFAULT_OUTPUT_ROOT / "prediction_viz"

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


GROUP_ALIASES = {
    "source": ("source", "source_id", "dataset", "dataset_id"),
    "dataset": ("dataset", "dataset_id", "source", "source_id"),
    "task": ("task", "task_id", "instruction", "task_instruction"),
    "object": ("object", "object_id", "category", "category_id"),
    "success": ("success", "is_success", "label"),
    "action": ("action", "action_condition"),
    "action_x": ("action_x",),
    "action_y": ("action_y",),
    "action_z": ("action_z",),
    "gripper": ("gripper", "action_gripper"),
}

ACTION_COMPONENT_KEYS = ("action_x", "action_y", "action_z", "gripper", "action_gripper")

ROLL_OUT_PRED_KEYS = ("z_rollout_pred", "rollout_pred", "z_pred_rollout", "multi_step_z_pred")
ROLL_OUT_TARGET_KEYS = ("z_rollout_target", "rollout_target", "z_target_rollout", "multi_step_z_target")
GOAL_KEYS = ("z_goal", "goal_latent", "goal_emb")
ABLATION_KEYS = {
    "condition_removed": ("z_pred_condition_removed", "z_pred_no_condition", "z_pred_without_condition"),
    "condition_shuffled": ("z_pred_condition_shuffled", "z_pred_shuffled_condition"),
    "condition_replaced": ("z_pred_condition_replaced", "z_pred_replaced_condition"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize JEPA prediction quality from exported latents.")
    parser.add_argument("--latent-dir", default=str(DEFAULT_LATENT_DIR), help="Directory with latents.npz/.pt and metadata.jsonl.")
    parser.add_argument("--out", default=str(DEFAULT_PREDICTION_DIR), help="Output directory for prediction visualizations.")
    parser.add_argument("--group-by", default="source", help="Group cosine-vs-horizon by source/dataset/task/action/etc.")
    parser.add_argument("--max-groups", type=int, default=8, help="Maximum groups to draw.")
    parser.add_argument("--action-bins", type=int, default=4, help="Number of quantile bins for continuous action grouping.")
    parser.add_argument("--interval", default="std", choices=("std", "ci95"), help="Error band for cosine-vs-horizon curves.")
    parser.add_argument("--heatmap-vmin-quantile", type=float, default=0.05, help="Lower quantile for heatmap color scale.")
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

    for key in ("z_pred", "z_target", "z_context"):
        if key not in arrays:
            raise KeyError(f"Missing required latent array: {key}")

    metadata_path = latent_dir / "metadata.jsonl"
    metadata = []
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = [json.loads(line) for line in f if line.strip()]

    n_samples = arrays["z_pred"].shape[0]
    if not metadata:
        metadata = [{"sample_id": idx, "export_index": idx} for idx in range(n_samples)]
    if len(metadata) != n_samples:
        raise ValueError(f"metadata rows ({len(metadata)}) != latent samples ({n_samples})")

    return arrays, metadata, latent_path


def ensure_3d(array, name):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array[:, None, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"{name} must have shape [N,D] or [N,T,D], got {array.shape}")


def as_token_horizon(array, name):
    """Return [N, tokens, horizons, D] without flattening horizon."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array[:, None, None, :], {"layout": "[N,D]", "token_axis": None, "horizon_axis": None}
    if array.ndim == 3:
        return array[:, None, :, :], {"layout": "[N,H,D]", "token_axis": None, "horizon_axis": 1}
    if array.ndim == 4:
        return array, {"layout": "[N,T,H,D]", "token_axis": 1, "horizon_axis": 2}
    raise ValueError(f"{name} must have shape [N,D], [N,H,D], or [N,T,H,D], got {array.shape}")


def prediction_tensors(arrays):
    pred, pred_layout = as_token_horizon(arrays["z_pred"], "z_pred")
    target, target_layout = as_token_horizon(arrays["z_target"], "z_target")
    if pred.shape != target.shape:
        raise ValueError(f"z_pred canonical shape {pred.shape} != z_target canonical shape {target.shape}")
    return pred, target, pred_layout, target_layout


def shape_report(arrays):
    pred, target, pred_layout, target_layout = prediction_tensors(arrays)
    context = np.asarray(arrays["z_context"])
    report = {
        "z_context_shape": list(context.shape),
        "z_pred_shape": list(np.asarray(arrays["z_pred"]).shape),
        "z_target_shape": list(np.asarray(arrays["z_target"]).shape),
        "z_pred_layout": pred_layout,
        "z_target_layout": target_layout,
        "canonical_prediction_shape": list(pred.shape),
        "detected_num_tokens": int(pred.shape[1]),
        "detected_num_horizons": int(pred.shape[2]),
        "detected_latent_dim": int(pred.shape[3]),
    }
    return report


def normalize(array, axis=-1):
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def cosine_same_step(pred, target):
    pred, pred_layout = as_token_horizon(pred, "z_pred")
    target, target_layout = as_token_horizon(target, "z_target")
    if pred.shape != target.shape:
        raise ValueError(f"z_pred shape {pred.shape} != z_target shape {target.shape}")
    return (normalize(pred) * normalize(target)).sum(axis=-1).mean(axis=1)


def mse_same_step(pred, target):
    pred, pred_layout = as_token_horizon(pred, "z_pred")
    target, target_layout = as_token_horizon(target, "z_target")
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {target.shape}")
    return ((pred - target) ** 2).mean(axis=(1, 3))


def sample_vector(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[:, -1, :]
    if array.ndim == 4:
        return array[:, -1, -1, :]
    raise ValueError(f"Expected [N,D], [N,T,D], or [N,T,H,D], got {array.shape}")


def alignment_heatmap_values(pred, target):
    pred = normalize(pred)
    target = normalize(target)
    per_sample = np.einsum("nthd,ntkd->nhk", pred, target) / max(pred.shape[1], 1)
    return per_sample.mean(axis=0), per_sample


def horizon_values(arrays, n_steps):
    for key in ("future_horizon_index", "target_horizon", "pred_horizon", "horizon", "target_time_index"):
        if key in arrays:
            values = np.asarray(arrays[key])
            if values.ndim == 3 and values.shape[2] == n_steps:
                return values[0, 0].astype(int)
            if values.ndim == 2 and values.shape[0] == arrays["z_pred"].shape[0] and values.shape[1] == n_steps:
                return values[0].astype(int)
            if values.ndim == 2 and values.shape[1] == n_steps:
                return values[0].astype(int)
            if values.ndim == 1 and values.shape[0] == n_steps:
                return values.astype(int)
    return np.arange(1, n_steps + 1)


def metadata_group(metadata, group_by):
    aliases = GROUP_ALIASES.get(group_by, (group_by,))
    for key in aliases:
        values = [row.get(key) for row in metadata]
        if any(value is not None for value in values):
            return np.asarray(["<missing>" if value is None else str(value) for value in values], dtype=object), key
    return np.asarray(["all" for _ in metadata], dtype=object), "all"


def action_group(arrays, n_steps, n_bins):
    action_key, action = find_first(arrays, ("action", "action_condition"))
    if action is None:
        return None, None

    action = np.asarray(action)
    if action.shape[0] != arrays["z_pred"].shape[0]:
        return None, None

    if action.ndim >= 3 and action.shape[1] == n_steps:
        norms = np.linalg.norm(action.reshape(action.shape[0], action.shape[1], -1), axis=-1).mean(axis=1)
    else:
        norms = np.linalg.norm(action.reshape(action.shape[0], -1), axis=-1)

    edges = np.quantile(norms, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.asarray([f"action_norm={norm:.3g}" for norm in norms], dtype=object), f"{action_key}_norm"

    bins = np.digitize(norms, edges[1:-1], right=True)
    labels = []
    for bin_idx in bins:
        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        labels.append(f"action_norm[{lo:.3g},{hi:.3g}]")
    return np.asarray(labels, dtype=object), f"{action_key}_norm_bin"


def group_values(arrays, metadata, group_by, n_steps, action_bins):
    if group_by == "action":
        labels, source = action_group(arrays, n_steps, action_bins)
        if labels is not None:
            return labels, source
    return metadata_group(metadata, group_by)


def top_groups(labels, max_groups):
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    keep = unique[order[:max_groups]]
    return keep


def plot_cosine_vs_horizon(arrays, metadata, args, out_dir, report):
    cosine = cosine_same_step(arrays["z_pred"], arrays["z_target"])
    horizons = horizon_values(arrays, cosine.shape[1])
    labels, group_source = group_values(arrays, metadata, args.group_by, cosine.shape[1], args.action_bins)
    groups = top_groups(labels, args.max_groups)

    csv_name = f"target_pred_cosine_vs_horizon_by_{safe_name(group_source)}.csv"
    with (out_dir / csv_name).open("w") as f:
        f.write("group,horizon,mean_cosine,std_cosine,ci95_cosine,count\n")
        for horizon_idx, horizon in enumerate(horizons):
            values = cosine[:, horizon_idx]
            std = values.std()
            ci95 = 1.96 * std / np.sqrt(max(values.shape[0], 1))
            f.write(f"all,{horizon},{values.mean()},{std},{ci95},{values.shape[0]}\n")
        for group in groups:
            mask = labels == group
            if mask.sum() == 0:
                continue
            for horizon_idx, horizon in enumerate(horizons):
                values = cosine[mask, horizon_idx]
                std = values.std()
                ci95 = 1.96 * std / np.sqrt(max(values.shape[0], 1))
                f.write(f"{quote_csv(group)},{horizon},{values.mean()},{std},{ci95},{int(mask.sum())}\n")
    report["generated"].append(csv_name)

    if cosine.shape[1] == 1:
        plot_groups = ["all", *list(groups)]
        data = [cosine[:, 0]]
        tick_labels = [f"all\nn={cosine.shape[0]}"]
        for group in groups:
            mask = labels == group
            if mask.sum() == 0:
                continue
            data.append(cosine[mask, 0])
            tick_labels.append(f"{group}\nn={int(mask.sum())}")

        fig, ax = plt.subplots(figsize=(max(7.5, len(data) * 1.2), 5.0))
        ax.boxplot(data, labels=tick_labels, showmeans=True)
        ax.set_title(f"Target-Pred Cosine by {group_source} (single detected horizon)")
        ax.set_xlabel(group_source)
        ax.set_ylabel("cos(z_pred, z_target)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout()
        filename = f"target_pred_cosine_boxplot_by_{safe_name(group_source)}.png"
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)
        report["generated"].append(filename)
        report["cosine_vs_horizon"] = {
            "mode": "boxplot_single_horizon",
            "group_source": group_source,
            "detected_horizons": [int(h) for h in horizons],
        }
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    def draw_curve(mask, label, linewidth):
        values = cosine[mask]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        band = std if args.interval == "std" else 1.96 * std / np.sqrt(max(values.shape[0], 1))
        line = ax.plot(horizons, mean, marker="o", linewidth=linewidth, label=f"{label} (n={values.shape[0]})")[0]
        ax.fill_between(horizons, mean - band, mean + band, color=line.get_color(), alpha=0.16, linewidth=0)

    draw_curve(np.ones(cosine.shape[0], dtype=bool), "all", 2.2)
    for group in groups:
        mask = labels == group
        if mask.sum() == 0:
            continue
        draw_curve(mask, str(group), 1.4)

    ax.set_title(f"Target-Pred Cosine vs Horizon by {group_source}")
    ax.set_xlabel("future horizon")
    ax.set_ylabel("cos(z_pred[:, h], z_target[:, h])")
    ax.set_xticks(horizons)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    filename = f"target_pred_cosine_vs_horizon_by_{safe_name(group_source)}.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)
    report["cosine_vs_horizon"] = {
        "mode": f"line_mean_{args.interval}",
        "group_source": group_source,
        "detected_horizons": [int(h) for h in horizons],
    }


def plot_alignment_heatmap(arrays, args, out_dir, report):
    z_pred, z_target, pred_layout, target_layout = prediction_tensors(arrays)
    heatmap, per_sample = alignment_heatmap_values(z_pred, z_target)
    num_horizons = heatmap.shape[0]
    diag = np.diag(heatmap)
    off_mask = ~np.eye(num_horizons, dtype=bool)
    off_values = heatmap[off_mask]
    diagonal_mean = float(diag.mean()) if diag.size else None
    off_diagonal_mean = float(off_values.mean()) if off_values.size else None
    diagonal_gap = None if off_diagonal_mean is None else float(diagonal_mean - off_diagonal_mean)
    top1 = per_sample.argmax(axis=2)
    target_index = np.broadcast_to(np.arange(num_horizons), top1.shape)
    top1_accuracy = float((top1 == target_index).mean())
    vmin = float(np.quantile(heatmap, args.heatmap_vmin_quantile)) if heatmap.size else -1.0
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(heatmap, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("Target-Pred Alignment Heatmap")
    ax.set_xlabel("target horizon")
    ax.set_ylabel("pred horizon")
    ax.set_xticks(np.arange(num_horizons))
    ax.set_yticks(np.arange(num_horizons))
    ax.set_xticklabels(np.arange(1, num_horizons + 1))
    ax.set_yticklabels(np.arange(1, num_horizons + 1))
    for row in range(num_horizons):
        for col in range(num_horizons):
            ax.text(col, row, f"{heatmap[row, col]:.3f}", ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout()
    filename = "target_pred_alignment_heatmap.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)

    matrix_name = "target_pred_alignment_heatmap.csv"
    np.savetxt(out_dir / matrix_name, heatmap, delimiter=",")
    report["generated"].append(matrix_name)

    metrics_name = "target_pred_alignment_heatmap_metrics.json"
    metrics = {
        "axis": "horizon",
        "num_horizons": int(num_horizons),
        "num_tokens_aggregated": int(z_pred.shape[1]),
        "diagonal_mean": diagonal_mean,
        "off_diagonal_mean": off_diagonal_mean,
        "diagonal_gap": diagonal_gap,
        "top1_horizon_matching_accuracy": top1_accuracy,
        "vmin_quantile": float(args.heatmap_vmin_quantile),
        "vmin": vmin,
        "vmax": vmax,
        "z_pred_layout": pred_layout,
        "z_target_layout": target_layout,
    }
    with (out_dir / metrics_name).open("w") as f:
        json.dump(metrics, f, indent=2)
    report["generated"].append(metrics_name)
    report["alignment_heatmap"] = metrics


def find_first(arrays, keys):
    for key in keys:
        if key in arrays:
            return key, arrays[key]
    return None, None


def add_prediction_checks(arrays, report):
    shapes = shape_report(arrays)
    report["shape_check"] = shapes
    warnings = report.setdefault("warnings", [])
    pred, target, _, _ = prediction_tensors(arrays)
    cosine = cosine_same_step(arrays["z_pred"], arrays["z_target"])
    mean_cosine = float(cosine.mean())
    max_abs_diff = float(np.max(np.abs(pred - target)))
    exact_equal = bool(np.array_equal(pred, target))
    allclose_equal = bool(np.allclose(pred, target))
    checks = {
        "z_pred_z_target_exact_equal": exact_equal,
        "z_pred_z_target_allclose": allclose_equal,
        "z_pred_z_target_max_abs_diff": max_abs_diff,
        "mean_cosine": mean_cosine,
        "median_cosine": float(np.median(cosine)),
        "q95_cosine": float(np.quantile(cosine, 0.95)),
    }
    if exact_equal or allclose_equal:
        warnings.append("z_pred and z_target are identical/allclose; check tensor reading or target leakage.")
    if mean_cosine > 0.99:
        warnings.append("mean cosine > 0.99; check target leakage or tensor reading errors.")
    report["prediction_checks"] = checks
    return shapes


def action_matrix(arrays):
    key, action = find_first(arrays, ("action", "action_condition"))
    if action is None:
        return None, None
    action = np.asarray(action, dtype=np.float32)
    return key, action.reshape(action.shape[0], -1)


def quantile_bin_labels(values, n_bins, prefix):
    values = np.asarray(values, dtype=np.float32)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        return np.asarray([f"{prefix}={value:.3g}" for value in values], dtype=object)
    bins = np.digitize(values, edges[1:-1], right=True)
    return np.asarray([f"{prefix}[{edges[idx]:.3g},{edges[idx + 1]:.3g}]" for idx in bins], dtype=object)


def boxplot_by_labels(values, labels, title, ylabel, out_path, max_groups):
    labels = np.asarray(labels, dtype=object)
    groups = top_groups(labels, max_groups)
    data = []
    tick_labels = []
    for group in groups:
        mask = labels == group
        if mask.sum() == 0:
            continue
        data.append(values[mask])
        tick_labels.append(f"{group}\nn={int(mask.sum())}")
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, len(data) * 1.2), 5.0))
    ax.boxplot(data, labels=tick_labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path.name


def metadata_component_values(metadata, key):
    values = [row.get(key) for row in metadata]
    if not any(value is not None for value in values):
        return None
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return None
    return np.asarray(numeric, dtype=np.float32)


def plot_action_conditioned_analysis(arrays, metadata, args, out_dir, report):
    cosine = cosine_same_step(arrays["z_pred"], arrays["z_target"]).mean(axis=1)
    mse = mse_same_step(arrays["z_pred"], arrays["z_target"]).mean(axis=1)
    action_key, action = action_matrix(arrays)
    if action is None or action.shape[0] != cosine.shape[0]:
        report["skipped"]["action_conditioned_analysis"] = "No action/action_condition array with matching sample count found."
        return

    action_norm = np.linalg.norm(action, axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(action_norm, cosine, c=mse, s=14, alpha=0.72, cmap="viridis", linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("prediction MSE")
    ax.set_title("Action Norm vs Target-Pred Cosine")
    ax.set_xlabel(f"{action_key} norm")
    ax.set_ylabel("mean cosine over horizons")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    filename = "action_norm_vs_cosine_scatter.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)

    labels = quantile_bin_labels(action_norm, args.action_bins, "action_norm")
    for values, metric_name, ylabel in (
        (cosine, "cosine", "mean cosine over horizons"),
        (mse, "mse", "mean MSE over horizons"),
    ):
        filename = f"action_norm_bin_vs_{metric_name}_boxplot.png"
        written = boxplot_by_labels(
            values,
            labels,
            f"Action Norm Bin vs {metric_name.upper()}",
            ylabel,
            out_dir / filename,
            args.max_groups,
        )
        if written:
            report["generated"].append(written)

    component_summary = {}
    for comp_idx, comp_name in enumerate(("action_x", "action_y", "action_z", "gripper")):
        component = metadata_component_values(metadata, comp_name)
        source = comp_name
        if component is None and comp_idx < action.shape[1]:
            component = action[:, comp_idx]
            source = f"{action_key}_{comp_idx}"
        if component is None or component.shape[0] != cosine.shape[0]:
            continue
        comp_labels = quantile_bin_labels(component, args.action_bins, source)
        for values, metric_name, ylabel in (
            (cosine, "cosine", "mean cosine over horizons"),
            (mse, "mse", "mean MSE over horizons"),
        ):
            filename = f"{safe_name(source)}_bin_vs_{metric_name}_boxplot.png"
            written = boxplot_by_labels(
                values,
                comp_labels,
                f"{source} Bin vs {metric_name.upper()}",
                ylabel,
                out_dir / filename,
                args.max_groups,
            )
            if written:
                report["generated"].append(written)
        component_summary[source] = {
            "min": float(np.min(component)),
            "max": float(np.max(component)),
            "mean": float(np.mean(component)),
        }

    report["action_conditioned_analysis"] = {
        "action_key": action_key,
        "action_norm_min": float(action_norm.min()),
        "action_norm_max": float(action_norm.max()),
        "action_norm_mean": float(action_norm.mean()),
        "component_summary": component_summary,
    }


def plot_rollout_drift(arrays, out_dir, report):
    pred_key, rollout_pred = find_first(arrays, ROLL_OUT_PRED_KEYS)
    target_key, rollout_target = find_first(arrays, ROLL_OUT_TARGET_KEYS)
    if rollout_pred is None or rollout_target is None:
        report["skipped"]["rollout_drift"] = (
            "No rollout prediction/target arrays found. Expected one of "
            f"{ROLL_OUT_PRED_KEYS} and one of {ROLL_OUT_TARGET_KEYS}."
        )
        return

    cosine = cosine_same_step(rollout_pred, rollout_target)
    mse = mse_same_step(rollout_pred, rollout_target)
    steps = np.arange(1, cosine.shape[1] + 1)

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.plot(steps, mse.mean(axis=0), marker="o", color="#e15759", label="MSE")
    ax1.set_xlabel("rollout step")
    ax1.set_ylabel("MSE", color="#e15759")
    ax1.tick_params(axis="y", labelcolor="#e15759")
    ax1.grid(alpha=0.18)

    ax2 = ax1.twinx()
    ax2.plot(steps, cosine.mean(axis=0), marker="o", color="#4c78a8", label="cosine")
    ax2.set_ylabel("cosine", color="#4c78a8")
    ax2.tick_params(axis="y", labelcolor="#4c78a8")
    ax2.set_ylim(-1.05, 1.05)
    fig.suptitle(f"Rollout Drift ({pred_key} vs {target_key})")
    fig.tight_layout()
    filename = "rollout_drift_curve.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)


def plot_condition_ablation(arrays, out_dir, report):
    z_target = arrays["z_target"]
    variants = {"normal": arrays["z_pred"]}
    for name, keys in ABLATION_KEYS.items():
        key, value = find_first(arrays, keys)
        if value is not None:
            variants[name] = value

    if len(variants) == 1:
        report["skipped"]["condition_ablation"] = (
            "Only normal z_pred is available. Export ablation prediction arrays such as "
            "z_pred_condition_removed, z_pred_condition_shuffled, or z_pred_condition_replaced to enable this plot."
        )
        return

    names = []
    mse_values = []
    cos_values = []
    for name, pred in variants.items():
        names.append(name)
        mse_values.append(float(mse_same_step(pred, z_target).mean()))
        cos_values.append(float(cosine_same_step(pred, z_target).mean()))

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axes[0].bar(x, mse_values, color="#e15759")
    axes[0].set_title("Prediction MSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.18)

    axes[1].bar(x, cos_values, color="#4c78a8")
    axes[1].set_title("Prediction Cosine")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("Condition Ablation")
    fig.tight_layout()
    filename = "condition_ablation.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)

    report["condition_ablation"] = {
        name: {"mse": mse, "cosine": cos}
        for name, mse, cos in zip(names, mse_values, cos_values)
    }


def plot_goal_distance(arrays, out_dir, report):
    goal_key, goal = find_first(arrays, GOAL_KEYS)
    if goal is None:
        report["skipped"]["goal_distance"] = (
            "No goal latent array found. Expected z_goal, goal_latent, or goal_emb. "
            "Goal images alone cannot be used unless their latents are exported."
        )
        return

    z_context = sample_vector(arrays["z_context"])
    z_pred = sample_vector(arrays["z_pred"])
    goal = np.asarray(goal, dtype=np.float32)
    if goal.ndim == 3:
        goal = goal[:, -1, :]
    elif goal.ndim == 4:
        goal = goal[:, -1, -1, :]
    elif goal.ndim != 2:
        report["skipped"]["goal_distance"] = f"Goal latent {goal_key} has unsupported shape {goal.shape}."
        return

    if goal.shape[0] != z_context.shape[0]:
        report["skipped"]["goal_distance"] = f"Goal latent sample count {goal.shape[0]} != z_context sample count {z_context.shape[0]}."
        return

    d_current = np.linalg.norm(z_context - goal, axis=-1)
    d_pred = np.linalg.norm(z_pred - goal, axis=-1)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.boxplot([d_current, d_pred], labels=["current-goal", "pred-goal"], showmeans=True)
    ax.set_title(f"Goal Distance ({goal_key})")
    ax.set_ylabel("L2 distance")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    filename = "goal_distance.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)

    report["goal_distance"] = {
        "goal_key": goal_key,
        "current_mean": float(d_current.mean()),
        "pred_after_action_mean": float(d_pred.mean()),
    }


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "_")


def quote_csv(value):
    text = str(value)
    if any(char in text for char in [",", "\"", "\n"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def write_report(out_dir, latent_path, arrays, report):
    report["latent_path"] = str(latent_path)
    report["arrays"] = {key: list(value.shape) for key, value in arrays.items() if hasattr(value, "shape")}

    with (out_dir / "prediction_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    lines = ["# JEPA Prediction Visualization Report", ""]
    lines.append(f"Latents: `{latent_path}`")
    lines.append("")
    if "shape_check" in report:
        lines.append("## Shape Check")
        for key, value in report["shape_check"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    if report.get("warnings"):
        lines.append("## Warnings")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    if "alignment_heatmap" in report:
        lines.append("## Alignment Heatmap Metrics")
        for key in ("diagonal_mean", "off_diagonal_mean", "diagonal_gap", "top1_horizon_matching_accuracy"):
            lines.append(f"- `{key}`: `{report['alignment_heatmap'].get(key)}`")
        lines.append("")
    lines.append("## Generated")
    if report["generated"]:
        for filename in report["generated"]:
            lines.append(f"- `{filename}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Skipped")
    if report["skipped"]:
        for name, reason in report["skipped"].items():
            lines.append(f"- `{name}`: {reason}")
    else:
        lines.append("- None")
    lines.append("")
    (out_dir / "prediction_report.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata, latent_path = load_latents(args.latent_dir)

    report = {"generated": [], "skipped": {}, "warnings": []}
    shapes = add_prediction_checks(arrays, report)
    plot_cosine_vs_horizon(arrays, metadata, args, out_dir, report)
    plot_alignment_heatmap(arrays, args, out_dir, report)
    plot_action_conditioned_analysis(arrays, metadata, args, out_dir, report)
    plot_rollout_drift(arrays, out_dir, report)
    plot_condition_ablation(arrays, out_dir, report)
    plot_goal_distance(arrays, out_dir, report)
    write_report(out_dir, latent_path, arrays, report)

    print(
        json.dumps(
            {
                "out": str(out_dir),
                "shape_check": shapes,
                "warnings": report["warnings"],
                "generated": report["generated"],
                "skipped": report["skipped"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
