#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path


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


def write_plots(log_path, out_dir):
    series = read_log(log_path)
    written = []

    plots = [
        (
            "total_loss.png",
            "Total Loss",
            "loss",
            ["fit/loss_total", "fit/loss", "train/loss_total", "train/loss"],
        ),
        (
            "l1_mse_loss.png",
            "L1 / MSE Loss",
            "loss",
            ["fit/loss_future_l1", "fit/loss_mse", "fit/pred_loss", "validate/loss_future_l1", "validate/loss_mse", "validate/pred_loss"],
        ),
        (
            "cosine_loss.png",
            "Cosine Loss",
            "1 - cosine",
            ["fit/loss_future_cos", "validate/loss_future_cos"],
        ),
        (
            "train_val_loss.png",
            "Train / Val Loss",
            "loss",
            ["fit/loss_total", "fit/loss", "validate/loss_total", "validate/loss", "validate/loss_epoch"],
        ),
        (
            "latent_std.png",
            "Latent Std",
            "std",
            ["fit/latent_std", "validate/latent_std"],
        ),
        (
            "active_dimensions.png",
            "Active Dimensions",
            "count",
            ["fit/active_dim_count", "validate/active_dim_count"],
        ),
        (
            "latent_norm.png",
            "Latent Norm",
            "norm",
            ["fit/latent_norm", "validate/latent_norm"],
        ),
        (
            "pairwise_cosine_stats.png",
            "Pairwise Cosine Stats",
            "cosine",
            ["fit/pairwise_cos_mean", "fit/pairwise_cos_std", "validate/pairwise_cos_mean", "validate/pairwise_cos_std"],
        ),
        (
            "throughput_lr.png",
            "Throughput / Learning Rate",
            "value",
            ["fit/samples_per_sec", "fit/learning_rate_0", "hparams/lr_default_0"],
        ),
    ]

    for filename, title, ylabel, names in plots:
        if plot_lines(series, names, out_dir / filename, title, ylabel):
            written.append(filename)

    prefix_plots = [
        ("per_dataset_loss.png", ["fit/per_dataset_loss/", "validate/per_dataset_loss/"], "Per-Dataset Loss", "loss"),
        ("per_horizon_loss.png", ["fit/loss_mse_horizon_", "validate/loss_mse_horizon_"], "Per-Horizon MSE", "loss"),
        ("batch_source_counts.png", ["fit/batch_source_counts/", "validate/batch_source_counts/"], "Batch Source Counts", "count"),
    ]
    for filename, prefixes, title, ylabel in prefix_plots:
        if plot_prefix(series, prefixes, out_dir / filename, title, ylabel):
            written.append(filename)

    if plot_pairwise_hist(series, out_dir / "pairwise_cosine_histogram.png"):
        written.append("pairwise_cosine_histogram.png")

    return written


def main():
    parser = argparse.ArgumentParser(description="Plot JEPA training metrics from CSV, JSONL, or TXT logs.")
    parser.add_argument("--log", required=True, help="Path to a training log file.")
    parser.add_argument("--out", default="outputs/jepa_eval/training", help="Output directory for PNG figures.")
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
