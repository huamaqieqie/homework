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
}

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


def normalize(array, axis=-1):
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def cosine_same_step(pred, target):
    pred = ensure_3d(pred, "z_pred")
    target = ensure_3d(target, "z_target")
    if pred.shape != target.shape:
        raise ValueError(f"z_pred shape {pred.shape} != z_target shape {target.shape}")
    return (normalize(pred) * normalize(target)).sum(axis=-1)


def mse_same_step(pred, target):
    pred = ensure_3d(pred, "z_pred")
    target = ensure_3d(target, "z_target")
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {target.shape}")
    return ((pred - target) ** 2).mean(axis=-1)


def horizon_values(arrays, n_steps):
    for key in ("future_horizon_index", "target_time_index"):
        if key in arrays:
            values = np.asarray(arrays[key])
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

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(horizons, cosine.mean(axis=0), marker="o", linewidth=2.2, label="all")

    for group in groups:
        mask = labels == group
        if mask.sum() == 0:
            continue
        ax.plot(horizons, cosine[mask].mean(axis=0), marker="o", linewidth=1.4, label=str(group))

    ax.set_title(f"Target-Pred Cosine vs Horizon by {group_source}")
    ax.set_xlabel("future horizon")
    ax.set_ylabel("cos(z_pred, z_target)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    filename = f"target_pred_cosine_vs_horizon_by_{safe_name(group_source)}.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)

    csv_name = f"target_pred_cosine_vs_horizon_by_{safe_name(group_source)}.csv"
    with (out_dir / csv_name).open("w") as f:
        f.write("group,horizon,mean_cosine,count\n")
        for horizon_idx, horizon in enumerate(horizons):
            f.write(f"all,{horizon},{cosine[:, horizon_idx].mean()},{cosine.shape[0]}\n")
        for group in groups:
            mask = labels == group
            for horizon_idx, horizon in enumerate(horizons):
                f.write(f"{quote_csv(group)},{horizon},{cosine[mask, horizon_idx].mean()},{int(mask.sum())}\n")
    report["generated"].append(csv_name)


def plot_alignment_heatmap(arrays, out_dir, report):
    z_pred = ensure_3d(arrays["z_pred"], "z_pred")
    z_target = ensure_3d(arrays["z_target"], "z_target")
    pred_norm = normalize(z_pred)
    target_norm = normalize(z_target)
    heatmap = np.einsum("npd,ntd->npt", pred_norm, target_norm).mean(axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(heatmap, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("Target-Pred Alignment Heatmap")
    ax.set_xlabel("target horizon / token")
    ax.set_ylabel("pred horizon / token")
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_yticks(np.arange(heatmap.shape[0]))
    ax.set_xticklabels(np.arange(1, heatmap.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, heatmap.shape[0] + 1))
    fig.tight_layout()
    filename = "target_pred_alignment_heatmap.png"
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)
    report["generated"].append(filename)

    matrix_name = "target_pred_alignment_heatmap.csv"
    np.savetxt(out_dir / matrix_name, heatmap, delimiter=",")
    report["generated"].append(matrix_name)


def find_first(arrays, keys):
    for key in keys:
        if key in arrays:
            return key, arrays[key]
    return None, None


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

    z_context = ensure_3d(arrays["z_context"], "z_context")
    z_pred = ensure_3d(arrays["z_pred"], "z_pred")
    goal = np.asarray(goal, dtype=np.float32)
    if goal.ndim == 3:
        goal = goal[:, -1, :]
    elif goal.ndim != 2:
        report["skipped"]["goal_distance"] = f"Goal latent {goal_key} has unsupported shape {goal.shape}."
        return

    z_current = z_context[:, -1, :]
    z_after = z_pred[:, -1, :]
    if goal.shape[0] != z_current.shape[0]:
        report["skipped"]["goal_distance"] = f"Goal latent sample count {goal.shape[0]} != z_context sample count {z_current.shape[0]}."
        return

    d_current = np.linalg.norm(z_current - goal, axis=-1)
    d_pred = np.linalg.norm(z_after - goal, axis=-1)

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

    report = {"generated": [], "skipped": {}}
    plot_cosine_vs_horizon(arrays, metadata, args, out_dir, report)
    plot_alignment_heatmap(arrays, out_dir, report)
    plot_rollout_drift(arrays, out_dir, report)
    plot_condition_ablation(arrays, out_dir, report)
    plot_goal_distance(arrays, out_dir, report)
    write_report(out_dir, latent_path, arrays, report)

    print(json.dumps({"out": str(out_dir), "generated": report["generated"], "skipped": report["skipped"]}, indent=2))


if __name__ == "__main__":
    main()
