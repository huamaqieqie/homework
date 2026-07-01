#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


LAST_WINDOW = 1000
ZOOM_STEP = 10000

TOOL_DIR = Path(__file__).resolve().parent
RUN_NAME = os.environ.get("JEPA_VIZ_RUN_NAME") or datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("JEPA_VIZ_OUTPUT_ROOT", TOOL_DIR / "output" / RUN_NAME))
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "training"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TABLE_RE = re.compile(r"\|\s*([^|]+?)\s*\|\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\|")


def configure_output_paths(out_dir):
    output_root = Path(os.environ.get("LEWM_OUTPUT_ROOT", Path(out_dir).resolve().parents[1]))
    paths = {
        "XDG_CACHE_HOME": output_root / ".cache",
        "XDG_CONFIG_HOME": output_root / ".config",
        "XDG_DATA_HOME": output_root / ".local",
        "MPLCONFIGDIR": output_root / ".cache" / "matplotlib",
        "TMPDIR": output_root / "tmp",
    }

    if os.environ.get("LEWM_RESPECT_EXTERNAL_CACHE", "0") == "1":
        for key, path in paths.items():
            os.environ.setdefault(key, str(path))
    else:
        for key, path in paths.items():
            os.environ[key] = str(path)

    for key in paths:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def to_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            if {"last", "min", "max", "count"}.intersection(value):
                if "last" in value:
                    flat[name] = value["last"]
            else:
                flat.update(flatten_dict(value, name))
        else:
            flat[name] = value
    return flat


def read_csv_log(path):
    series = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            step = to_float(row.get("step"))
            x = step if step is not None else row_idx
            for key, value in row.items():
                y = to_float(value)
                if y is not None:
                    series[key].append((x, y))
    return series


def read_jsonl_log(path):
    series = defaultdict(list)
    with path.open() as handle:
        for row_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                add_json_record(series, data, row_idx)
    return series


def add_json_record(series, data, row_idx):
    flat = flatten_dict(data)
    step = to_float(flat.get("step") or flat.get("global_step"))
    x = step if step is not None else row_idx

    metrics = data.get("metrics") if isinstance(data, dict) else None
    if isinstance(metrics, dict):
        flat.update(flatten_dict(metrics))

    for key, value in flat.items():
        if key.startswith("metrics/"):
            key = key[len("metrics/"):]
        y = to_float(value)
        if y is not None:
            series[key].append((x, y))


def read_json_log(path):
    series = defaultdict(list)
    try:
        with path.open() as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return read_jsonl_log(path)

    if isinstance(data, list):
        for row_idx, item in enumerate(data):
            if isinstance(item, dict):
                add_json_record(series, item, row_idx)
    elif isinstance(data, dict):
        add_json_record(series, data, 0)

    return series


def read_txt_log(path):
    series = defaultdict(list)
    counts = defaultdict(int)
    with path.open(errors="replace") as handle:
        for line in handle:
            line = strip_ansi(line)
            match = TABLE_RE.search(line)
            if not match:
                continue
            key = match.group(1).strip()
            y = to_float(match.group(2))
            if y is None:
                continue
            counts[key] += 1
            series[key].append((counts[key], y))
    return series


def read_log(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_log(path)
    if suffix == ".jsonl":
        return read_jsonl_log(path)
    if suffix == ".json":
        return read_json_log(path)
    return read_txt_log(path)


def first_existing(series, names):
    for name in names:
        if name in series and series[name]:
            return name
    return None


def series_arrays(values):
    xs, ys = zip(*values)
    return list(xs), list(ys)


def clean_label(name):
    return name.replace("fit/", "train/").replace("validate/", "val/")


def key_matches(key, include_any, exclude_any=()):
    lower = key.lower()
    return any(token in lower for token in include_any) and not any(token in lower for token in exclude_any)


def matching_keys(series, include_any, exclude_any=()):
    return sorted(key for key in series if series[key] and key_matches(key, include_any, exclude_any))


def last_window_values(values, window=LAST_WINDOW):
    if not values:
        return []
    max_step = values[-1][0]
    cutoff = max_step - window
    selected = [y for x, y in values if x >= cutoff]
    return selected or [y for _, y in values[-min(len(values), window) :]]


def metric_stats(values):
    if not values:
        return {}
    ys = [y for _, y in values]
    last_values = last_window_values(values)
    best = min(ys)
    best_idx = ys.index(best)
    return {
        "final_step": values[-1][0],
        "final": ys[-1],
        "best": best,
        "best_step": values[best_idx][0],
        "last_1k_mean": sum(last_values) / len(last_values),
        "last_1k_std": float_std(last_values),
    }


def float_std(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def moving_average(values, window=50):
    if not values:
        return []
    averaged = []
    running = []
    for x, y in values:
        running.append(y)
        if len(running) > window:
            running.pop(0)
        averaged.append((x, sum(running) / len(running)))
    return averaged


def detect_latent_dim(series):
    candidates = [
        "latent_dim",
        "model/latent_dim",
        "hparams/latent_dim",
        "hparams/model/latent_dim",
        "hidden_size",
        "model/hidden_size",
        "hparams/hidden_size",
        "hparams/model/hidden_size",
        "embed_dim",
        "embedding_dim",
    ]
    for key in series:
        lower = key.lower()
        if any(candidate in lower for candidate in candidates) and series[key]:
            value = series[key][-1][1]
            if value > 0:
                return int(round(value)), key
    return None, None


def infer_latent_kind(key):
    lower = key.lower()
    for kind in ("z_context", "context", "z_pred", "pred", "z_target", "target"):
        if kind in lower:
            return kind.replace("z_", "")
    return None


def latent_metric_keys(series, metric_name):
    metric = metric_name.lower()
    if metric == "latent_norm":
        keys = [
            key
            for key in series
            if series[key]
            and (("latent_norm" in key.lower()) or ("latent" in key.lower() and "norm" in key.lower()))
            and not any(token in key.lower() for token in ("grad", "weight", "param"))
        ]
    elif metric == "latent_std":
        keys = [
            key
            for key in series
            if series[key]
            and (("latent_std" in key.lower()) or ("latent" in key.lower() and "std" in key.lower()))
            and not any(token in key.lower() for token in ("grad", "weight", "param"))
        ]
    else:
        keys = matching_keys(series, (metric,), ())
    preferred = []
    generic = []
    for key in keys:
        lower = key.lower()
        if any(kind in lower for kind in ("z_context", "context", "z_pred", "pred", "z_target", "target")):
            preferred.append(key)
        elif metric in lower:
            generic.append(key)
    return preferred or generic


def plot_lines_enhanced(series, names, out_path, title, ylabel, yscale=None, zoom_start=None, ref_lines=(), annotations=True):
    selected = [(name, series[name]) for name in names if name in series and series[name]]
    if not selected:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for name, values in selected:
        filtered = [(x, y) for x, y in values if zoom_start is None or x >= zoom_start]
        if yscale == "log":
            filtered = [(x, y) for x, y in filtered if y > 0]
        if not filtered:
            continue
        xs, ys = series_arrays(filtered)
        label = clean_label(name)
        if annotations:
            stats = metric_stats(filtered)
            if stats:
                label += f" final={stats['final']:.4g}, last1k={stats['last_1k_mean']:.4g}"
        ax.plot(xs, ys, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    for y_value, label, style in ref_lines:
        ax.axhline(y_value, linestyle=style, linewidth=1.2, color="black", alpha=0.6, label=label)
    if yscale:
        ax.set_yscale(yscale)
    if zoom_start is not None:
        ax.set_title(f"{title} (step >= {zoom_start})")
    else:
        ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def plot_lines(series, names, out_path, title, ylabel):
    selected = [(name, series[name]) for name in names if name in series and series[name]]
    if not selected:
        return False

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for name, values in selected:
        xs, ys = zip(*values)
        plt.plot(xs, ys, label=name)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def plot_prefix(series, prefixes, out_path, title, ylabel):
    names = []
    for key in sorted(series):
        if any(key.startswith(prefix) for prefix in prefixes):
            names.append(key)
    return plot_lines(series, names, out_path, title, ylabel)


def plot_pairwise_hist(series, out_path):
    bin_keys = []
    for prefix in ("fit/pairwise_cos_hist_bin_", "train/pairwise_cos_hist_bin_", "validate/pairwise_cos_hist_bin_"):
        bin_keys = sorted(key for key in series if key.startswith(prefix) and series[key])
        if bin_keys:
            break

    if not bin_keys:
        return False

    import matplotlib.pyplot as plt

    values = [series[key][-1][1] for key in bin_keys]
    edges = [(-1.0 + i * 0.1) for i in range(len(values) + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(values))]

    plt.figure(figsize=(9, 5))
    plt.bar(centers, values, width=0.09, align="center")
    plt.title("Pairwise Cosine Histogram")
    plt.xlabel("pairwise cosine")
    plt.ylabel("fraction")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def plot_total_loss(series, out_dir, written, summary):
    train_candidates = ["fit/loss_total", "fit/loss", "train/loss_total", "train/loss"]
    val_candidates = ["validate/loss_total", "validate/loss", "validate/loss_epoch", "val/loss_total", "val/loss"]
    train_names = [name for name in train_candidates if name in series and series[name]]
    val_names = [name for name in val_candidates if name in series and series[name]]

    if "fit/loss_total" in train_names and "fit/loss" in train_names:
        total = series["fit/loss_total"]
        plain = series["fit/loss"]
        if len(total) == len(plain) and all(abs(a[1] - b[1]) < 1e-12 for a, b in zip(total, plain)):
            train_names.remove("fit/loss")
            summary["warnings"].append("fit/loss_total and fit/loss are identical; plotted fit/loss_total only.")

    if train_names and plot_lines_enhanced(series, train_names, out_dir / "train_total_loss.png", "Train Total Loss", "loss"):
        written.append("train_total_loss.png")
    if val_names and plot_lines_enhanced(series, val_names, out_dir / "val_loss.png", "Validation Loss", "loss"):
        written.append("val_loss.png")

    combined = train_names + val_names
    if combined:
        if plot_lines_enhanced(series, combined, out_dir / "total_loss.png", "Total Loss (Train / Val)", "loss"):
            written.append("total_loss.png")
        if plot_lines_enhanced(series, combined, out_dir / "total_loss_log.png", "Total Loss (Log Scale)", "loss", yscale="log"):
            written.append("total_loss_log.png")
        if plot_lines_enhanced(series, combined, out_dir / "total_loss_zoom_step_ge_10000.png", "Total Loss Zoom", "loss", zoom_start=ZOOM_STEP):
            written.append("total_loss_zoom_step_ge_10000.png")

    val_best = {}
    for name in val_names:
        stats = metric_stats(series[name])
        if stats:
            val_best[name] = {"best": stats["best"], "best_step": stats["best_step"]}
    if val_best:
        summary["total_loss"] = {"best_val": val_best}


def plot_latent_metric(series, out_dir, written, summary, metric_name, filename, title, ylabel, ref_lines):
    keys = latent_metric_keys(series, metric_name)
    if not keys:
        return
    if plot_lines_enhanced(series, keys, out_dir / filename, title, ylabel, ref_lines=ref_lines):
        written.append(filename)
    summary[filename] = {key: metric_stats(series[key]) for key in keys}


def plot_latent_norm_std(series, out_dir, written, summary):
    latent_dim, latent_dim_key = detect_latent_dim(series)
    norm_refs = []
    if latent_dim:
        norm_refs.append((math.sqrt(latent_dim), f"sqrt(latent_dim={latent_dim}) from {latent_dim_key}", "--"))
    plot_latent_metric(series, out_dir, written, summary, "latent_norm", "latent_norm.png", "Latent Norm", "norm", norm_refs)
    plot_latent_metric(series, out_dir, written, summary, "latent_std", "latent_std.png", "Latent Std", "std", [(1.0, "target std=1.0", "--")])


def pairwise_stat_keys(series, stat):
    return matching_keys(series, (f"pairwise_cos_{stat}",), ())


def plot_pairwise_stats(series, out_dir, written, summary):
    mean_keys = pairwise_stat_keys(series, "mean")
    std_keys = pairwise_stat_keys(series, "std")
    if mean_keys and plot_lines_enhanced(series, mean_keys, out_dir / "pairwise_cosine_mean.png", "Pairwise Cosine Mean", "cosine"):
        written.append("pairwise_cosine_mean.png")
    if std_keys and plot_lines_enhanced(series, std_keys, out_dir / "pairwise_cosine_std.png", "Pairwise Cosine Std", "std"):
        written.append("pairwise_cosine_std.png")

    summary["pairwise_cosine"] = {
        "mean_last_1k": {key: metric_stats(series[key]) for key in mean_keys},
        "std_last_1k": {key: metric_stats(series[key]) for key in std_keys},
    }

    if plot_pairwise_hist_enhanced(series, out_dir / "pairwise_cosine_histogram.png", summary):
        written.append("pairwise_cosine_histogram.png")


def plot_pairwise_hist_enhanced(series, out_path, summary):
    bin_keys = []
    source_prefix = None
    for prefix in ("fit/pairwise_cos_hist_bin_", "train/pairwise_cos_hist_bin_", "validate/pairwise_cos_hist_bin_"):
        bin_keys = sorted(key for key in series if key.startswith(prefix) and series[key])
        if bin_keys:
            source_prefix = prefix
            break

    if not bin_keys:
        return False

    import matplotlib.pyplot as plt

    values = [max(series[key][-1][1], 0.0) for key in bin_keys]
    total = sum(values)
    fractions = [value / total for value in values] if total > 0 else values
    edges = [(-1.0 + i * (2.0 / len(values))) for i in range(len(values) + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(values))]
    width = edges[1] - edges[0] if len(edges) > 1 else 0.1

    def quantile_from_hist(q):
        cumulative = 0.0
        for center, fraction in zip(centers, fractions):
            cumulative += fraction
            if cumulative >= q:
                return center
        return centers[-1]

    hist_mean = sum(center * fraction for center, fraction in zip(centers, fractions))
    median = quantile_from_hist(0.5)
    q05 = quantile_from_hist(0.05)
    q95 = quantile_from_hist(0.95)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(centers, fractions, width=width * 0.9, align="center")
    for value, label, style in ((hist_mean, "mean", "-"), (median, "median", "--"), (q05, "q05", ":"), (q95, "q95", ":")):
        ax.axvline(value, linestyle=style, linewidth=1.2, label=f"{label}={value:.3f}")
    ax.set_title("Pairwise Cosine Histogram")
    ax.set_xlabel("pairwise cosine")
    ax.set_ylabel("fraction")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    summary["pairwise_cosine_histogram"] = {
        "source_prefix": source_prefix,
        "mean": hist_mean,
        "median": median,
        "q05": q05,
        "q95": q95,
    }
    return True


def horizon_index(key):
    match = re.search(r"horizon[_/-]?(\d+)", key.lower())
    return int(match.group(1)) if match else None


def horizon_split(key):
    lower = key.lower()
    if lower.startswith(("fit/", "train/")):
        return "train"
    if "validate_epoch" in lower or "val_epoch" in lower or lower.endswith("_epoch") or "/epoch" in lower:
        return "val_epoch"
    if lower.startswith(("validate/", "val/", "validate_step/", "val_step/")) or "validate_step" in lower or "val_step" in lower:
        return "val_step"
    return None


def horizon_keys_by_split(series):
    splits = {"train": [], "val_epoch": [], "val_step": []}
    for key in series:
        lower = key.lower()
        if "loss_mse_horizon" not in lower and "mse_horizon" not in lower:
            continue
        split = horizon_split(key)
        if split:
            splits[split].append(key)
    return {split: sorted(keys, key=lambda key: (horizon_index(key) or 999, key)) for split, keys in splits.items()}


def plot_per_horizon(series, out_dir, written, summary):
    splits = horizon_keys_by_split(series)
    filenames = {
        "train": "train_per_horizon_mse.png",
        "val_epoch": "val_epoch_per_horizon_mse.png",
        "val_step": "val_step_per_horizon_mse.png",
    }
    summary["per_horizon_mse"] = {}
    for split, keys in splits.items():
        if not keys:
            continue
        filename = filenames[split]
        if plot_lines_enhanced(series, keys, out_dir / filename, f"{split.replace('_', ' ').title()} Per-Horizon MSE", "MSE"):
            written.append(filename)
        zoom_name = filename.replace(".png", "_zoom_step_ge_10000.png")
        if plot_lines_enhanced(series, keys, out_dir / zoom_name, f"{split.replace('_', ' ').title()} Per-Horizon MSE Zoom", "MSE", zoom_start=ZOOM_STEP):
            written.append(zoom_name)
        summary["per_horizon_mse"][split] = {key: metric_stats(series[key]) for key in keys}
        check_horizon_similarity(series, keys, split, summary)


def check_horizon_similarity(series, keys, split, summary):
    if len(keys) < 2:
        return
    means = []
    for key in keys:
        stats = metric_stats(series[key])
        if stats:
            means.append(stats["last_1k_mean"])
    if len(means) < 2:
        return
    spread = max(means) - min(means)
    scale = max(abs(value) for value in means) or 1.0
    if spread / scale < 0.01:
        summary["warnings"].append(
            f"{split} per-horizon MSE curves are very close in the last 1k steps; check whether horizon dimension is correctly separated."
        )


def plot_throughput_lr(series, out_dir, written, summary):
    throughput_keys = matching_keys(series, ("samples_per_sec",), ())
    lr_keys = matching_keys(series, ("learning_rate", "lr_default"), ())
    if throughput_keys:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for key in throughput_keys:
            values = series[key]
            xs, ys = series_arrays(values)
            ax.plot(xs, ys, alpha=0.35, label=f"{clean_label(key)} raw")
            ma = moving_average(values)
            ma_x, ma_y = series_arrays(ma)
            ax.plot(ma_x, ma_y, linewidth=2.0, label=f"{clean_label(key)} moving avg")
            detect_throughput_jumps(values, key, summary)
        ax.set_title("Samples Per Second")
        ax.set_xlabel("step")
        ax.set_ylabel("samples/sec")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        filename = "samples_per_sec.png"
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)
        written.append(filename)

    if lr_keys and plot_lines_enhanced(series, lr_keys, out_dir / "learning_rate.png", "Learning Rate", "learning rate", yscale="log"):
        written.append("learning_rate.png")


def detect_throughput_jumps(values, key, summary):
    if len(values) < 10:
        return
    ys = [y for _, y in values]
    jumps = []
    for idx in range(1, len(values)):
        prev = ys[idx - 1]
        curr = ys[idx]
        if prev <= 0:
            continue
        ratio = curr / prev
        if ratio > 2.5 or ratio < 0.4:
            jumps.append({"step": values[idx][0], "prev": prev, "current": curr, "ratio": ratio})
    if jumps:
        summary["warnings"].append(f"Throughput jump detected for {key}; see training_plot_summary.json.")
        summary.setdefault("throughput_jumps", {})[key] = jumps[:20]


def write_training_summary(out_dir, summary, written):
    summary["files"] = written
    with (out_dir / "training_plot_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    written.append("training_plot_summary.json")


def write_plots(log_path, out_dir):
    series = read_log(log_path)
    written = []
    summary = {"log_path": str(log_path), "warnings": []}

    plot_total_loss(series, out_dir, written, summary)

    plots = [
        (
            "l1_mse_loss.png",
            "L1 / MSE Loss",
            "loss",
            ["fit/loss_future_l1", "fit/loss_mse", "validate/loss_future_l1", "validate/loss_mse"],
        ),
        (
            "cosine_loss.png",
            "Cosine Loss",
            "1 - cosine",
            ["fit/loss_future_cos", "validate/loss_future_cos"],
        ),
        (
            "active_dimensions.png",
            "Active Dimensions",
            "count",
            ["fit/active_dim_count", "validate/active_dim_count"],
        ),
    ]

    for filename, title, ylabel, names in plots:
        if plot_lines(series, names, out_dir / filename, title, ylabel):
            written.append(filename)

    prefix_plots = [
        ("per_dataset_loss.png", ["fit/per_dataset_loss/", "validate/per_dataset_loss/"], "Per-Dataset Loss", "loss"),
        ("batch_source_counts.png", ["fit/batch_source_counts/", "validate/batch_source_counts/"], "Batch Source Counts", "count"),
    ]
    for filename, prefixes, title, ylabel in prefix_plots:
        if plot_prefix(series, prefixes, out_dir / filename, title, ylabel):
            written.append(filename)

    plot_latent_norm_std(series, out_dir, written, summary)
    plot_pairwise_stats(series, out_dir, written, summary)
    plot_per_horizon(series, out_dir, written, summary)
    plot_throughput_lr(series, out_dir, written, summary)
    write_training_summary(out_dir, summary, written)

    return written


def main():
    parser = argparse.ArgumentParser(description="Plot JEPA training metrics from CSV, JSONL, or TXT logs.")
    parser.add_argument("--log", required=True, help="Path to a training log file.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for PNG figures.")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh plots while training is running.")
    parser.add_argument("--interval", type=float, default=60.0, help="Refresh interval in seconds for --watch.")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_output_paths(out_dir)

    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install matplotlib and rerun.") from exc
    matplotlib.use("Agg")

    while True:
        written = write_plots(log_path, out_dir)
        if written:
            print(f"Wrote {len(written)} figure(s) to {out_dir}:")
            for filename in written:
                print(f"  {filename}")
        else:
            print("No supported JEPA metrics found in the log.")

        if not args.watch:
            break

        print(f"Watching {log_path}; refreshing in {args.interval:g}s. Press Ctrl+C to stop.")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
