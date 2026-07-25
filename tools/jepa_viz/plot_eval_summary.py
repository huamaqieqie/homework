#!/usr/bin/env python3
"""Summarize JEPA planner/eval text logs into comparison plots.

The tool is intentionally format-light: pass one or more groups as

    --group "Model name=path/or/glob/*.txt"

Each group can point at eval output .txt files or tee .log files. The parser
looks for common fields written by LeWM/stable_worldmodel eval scripts:
success_rate, evaluation_time, eval.num_eval, eval_budget, seed, policy, and
optional CEM solve time lines.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
RUN_NAME = os.environ.get("JEPA_VIZ_RUN_NAME") or datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("JEPA_VIZ_OUTPUT_ROOT", TOOL_DIR / "output" / RUN_NAME))
DEFAULT_OUT = DEFAULT_OUTPUT_ROOT / "eval_summary"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_OUTPUT_ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT_ROOT / ".cache" / "matplotlib"))
for key in ("XDG_CACHE_HOME", "MPLCONFIGDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class EvalRow:
    group: str
    path: str
    success_rate: float | None = None
    evaluation_time_sec: float | None = None
    num_eval: int | None = None
    eval_budget: int | None = None
    seed: int | None = None
    policy: str | None = None
    cem_solve_time_mean_sec: float | None = None
    cem_solve_time_count: int = 0

    @property
    def time_per_episode_sec(self) -> float | None:
        if self.evaluation_time_sec is None or not self.num_eval:
            return None
        return self.evaluation_time_sec / self.num_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grouped JEPA eval summary from text logs.")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=GLOB",
        help="Evaluation group. Can be repeated. Example: 'Residual=outputs/*seed*.txt'",
    )
    parser.add_argument("--root", default=".", help="Root used to resolve relative globs.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory.")
    parser.add_argument("--title", default="JEPA Evaluation Summary", help="Figure title prefix.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def maybe_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def maybe_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def first_match(patterns: list[str], text: str, flags: int = re.MULTILINE) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1)
    return None


def parse_eval_file(path: Path, group: str) -> EvalRow:
    text = path.read_text(errors="replace")
    success = first_match(
        [
            rf"'success_rate'\s*:\s*({FLOAT})",
            rf'"success_rate"\s*:\s*({FLOAT})',
            rf"\bsuccess_rate\s*[:=]\s*({FLOAT})",
            rf"\bSuccess rate\s*[:=]\s*({FLOAT})",
        ],
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    eval_time = first_match(
        [
            rf"\bevaluation_time\s*:\s*({FLOAT})\s*seconds",
            rf"\bevaluation_time_sec\s*[:=]\s*({FLOAT})",
            rf"\beval(?:uation)? time\s*[:=]\s*({FLOAT})",
        ],
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    num_eval = first_match([r"^\s*num_eval\s*:\s*(\d+)", r"^\s*eval\.num_eval\s*[:=]\s*(\d+)"], text)
    eval_budget = first_match([r"^\s*eval_budget\s*:\s*(\d+)", r"^\s*eval\.eval_budget\s*[:=]\s*(\d+)"], text)
    seed = first_match([r"^\s*seed\s*:\s*(-?\d+)", r"seed(?:=|_)(-?\d+)"], text)
    if seed is None:
        seed = first_match([r"seed(-?\d+)"], path.name, re.IGNORECASE)
    policy = first_match([r"^\s*policy\s*:\s*(.+?)\s*$"], text)

    cem_times = [float(value) for value in re.findall(rf"CEM solve time\s*:\s*({FLOAT})\s*seconds", text, re.IGNORECASE)]
    cem_mean = sum(cem_times) / len(cem_times) if cem_times else None

    return EvalRow(
        group=group,
        path=str(path),
        success_rate=maybe_float(success),
        evaluation_time_sec=maybe_float(eval_time),
        num_eval=maybe_int(num_eval),
        eval_budget=maybe_int(eval_budget),
        seed=maybe_int(seed),
        policy=policy,
        cem_solve_time_mean_sec=cem_mean,
        cem_solve_time_count=len(cem_times),
    )


def expand_group_specs(specs: list[str], root: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    if not specs:
        raise SystemExit("At least one --group NAME=GLOB is required.")
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --group {spec!r}; expected NAME=GLOB.")
        name, pattern = spec.split("=", 1)
        name = name.strip()
        pattern = pattern.strip()
        resolved = pattern if Path(pattern).is_absolute() else str(root / pattern)
        paths = [Path(p) for p in sorted(glob.glob(resolved))]
        if not paths:
            print(f"WARNING: no files matched group {name!r}: {resolved}")
            continue
        for path in paths:
            rows.append(parse_eval_file(path, name))
    if not rows:
        raise SystemExit("No eval files were parsed.")
    return rows


def finite(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    m = mean(values)
    assert m is not None
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize(rows: list[EvalRow]) -> list[dict[str, object]]:
    by_group: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_group[row.group].append(row)

    summary = []
    for group, group_rows in by_group.items():
        successes = finite([row.success_rate for row in group_rows])
        eval_times = finite([row.evaluation_time_sec for row in group_rows])
        time_per_ep = finite([row.time_per_episode_sec for row in group_rows])
        cem_times = finite([row.cem_solve_time_mean_sec for row in group_rows])
        summary.append(
            {
                "group": group,
                "n_files": len(group_rows),
                "n_success": len(successes),
                "success_rate_mean": mean(successes),
                "success_rate_std": std(successes),
                "success_rate_min": min(successes) if successes else None,
                "success_rate_max": max(successes) if successes else None,
                "evaluation_time_mean_sec": mean(eval_times),
                "time_per_episode_mean_sec": mean(time_per_ep),
                "cem_solve_time_mean_sec": mean(cem_times),
                "eval_budget_values": sorted({row.eval_budget for row in group_rows if row.eval_budget is not None}),
                "num_eval_values": sorted({row.num_eval for row in group_rows if row.num_eval is not None}),
                "policies": sorted({row.policy for row in group_rows if row.policy}),
            }
        )
    return summary


def write_csvs(rows: list[EvalRow], summary: list[dict[str, object]], out: Path) -> None:
    row_fields = [
        "group",
        "path",
        "success_rate",
        "evaluation_time_sec",
        "time_per_episode_sec",
        "num_eval",
        "eval_budget",
        "seed",
        "policy",
        "cem_solve_time_mean_sec",
        "cem_solve_time_count",
    ]
    with (out / "eval_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for row in rows:
            payload = {field: getattr(row, field) for field in row_fields if field != "time_per_episode_sec"}
            payload["time_per_episode_sec"] = row.time_per_episode_sec
            writer.writerow(payload)

    summary_fields = [
        "group",
        "n_files",
        "n_success",
        "success_rate_mean",
        "success_rate_std",
        "success_rate_min",
        "success_rate_max",
        "evaluation_time_mean_sec",
        "time_per_episode_mean_sec",
        "cem_solve_time_mean_sec",
        "eval_budget_values",
        "num_eval_values",
        "policies",
    ]
    with (out / "eval_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for item in summary:
            writer.writerow(item)
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def group_order(summary: list[dict[str, object]]) -> list[str]:
    return [str(item["group"]) for item in summary]


def plot_bar(summary: list[dict[str, object]], out: Path, title: str, dpi: int) -> None:
    groups = group_order(summary)
    means = [item["success_rate_mean"] for item in summary]
    errors = [item["success_rate_std"] for item in summary]
    labels = [f"{g}\n(n={item['n_success']})" for g, item in zip(groups, summary)]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(groups)), 5.5))
    xs = list(range(len(groups)))
    ax.bar(xs, [m if m is not None else 0 for m in means], yerr=[e if e is not None else 0 for e in errors], capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{title}: success rate")
    ax.grid(axis="y", alpha=0.25)
    for x, m, e in zip(xs, means, errors):
        if m is not None:
            ax.text(x, min(102, m + (e or 0) + 2), f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "success_rate_summary.png", dpi=dpi)
    plt.close(fig)


def plot_by_seed(rows: list[EvalRow], summary: list[dict[str, object]], out: Path, title: str, dpi: int) -> None:
    groups = group_order(summary)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(groups)), 5.5))
    fallback_offsets = defaultdict(int)
    for idx, group in enumerate(groups):
        group_rows = [row for row in rows if row.group == group and row.success_rate is not None]
        xs = []
        ys = []
        for row in group_rows:
            if row.seed is None:
                fallback_offsets[group] += 1
                x = idx + 0.05 * fallback_offsets[group]
            else:
                x = idx + (row.seed - 2) * 0.04
            xs.append(x)
            ys.append(row.success_rate)
        ax.scatter(xs, ys, s=45, label=group)
        if ys:
            ax.plot([idx - 0.18, idx + 0.18], [sum(ys) / len(ys)] * 2, linewidth=2)
    ax.set_xticks(list(range(len(groups))))
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{title}: per-seed success rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "success_rate_by_seed.png", dpi=dpi)
    plt.close(fig)


def plot_runtime(summary: list[dict[str, object]], out: Path, title: str, dpi: int) -> None:
    runtime_keys = [
        ("time_per_episode_mean_sec", "time_per_episode.png", "time per episode (s)"),
        ("evaluation_time_mean_sec", "evaluation_time.png", "evaluation time (s)"),
        ("cem_solve_time_mean_sec", "cem_solve_time.png", "CEM solve time (s)"),
    ]
    groups = group_order(summary)
    xs = list(range(len(groups)))
    for key, filename, ylabel in runtime_keys:
        values = [item.get(key) for item in summary]
        if not any(v is not None for v in values):
            continue
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(groups)), 5.0))
        ax.bar(xs, [float(v) if v is not None else 0.0 for v in values])
        ax.set_xticks(xs)
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {ylabel}")
        ax.grid(axis="y", alpha=0.25)
        for x, value in zip(xs, values):
            if value is not None:
                ax.text(x, float(value), f"{float(value):.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=dpi)
        plt.close(fig)


def write_report(summary: list[dict[str, object]], out: Path, title: str) -> None:
    lines = [f"# {title}", "", "## Summary", ""]
    lines.append("| group | n | success mean ± std | min | max | time/episode | eval budget | num eval |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for item in summary:
        success_mean = item["success_rate_mean"]
        success_std = item["success_rate_std"]
        success = "NA" if success_mean is None else f"{success_mean:.2f} ± {(success_std or 0):.2f}"
        time_per_ep = item["time_per_episode_mean_sec"]
        lines.append(
            "| {group} | {n} | {success} | {minv} | {maxv} | {time_ep} | {budget} | {num_eval} |".format(
                group=item["group"],
                n=item["n_success"],
                success=success,
                minv="NA" if item["success_rate_min"] is None else f"{item['success_rate_min']:.2f}",
                maxv="NA" if item["success_rate_max"] is None else f"{item['success_rate_max']:.2f}",
                time_ep="NA" if time_per_ep is None else f"{time_per_ep:.2f}s",
                budget=", ".join(map(str, item["eval_budget_values"])) or "NA",
                num_eval=", ".join(map(str, item["num_eval_values"])) or "NA",
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Success Rate Summary](success_rate_summary.png)",
            "",
            "![Success Rate by Seed](success_rate_by_seed.png)",
            "",
            "![Time per Episode](time_per_episode.png)",
            "",
        ]
    )
    if (out / "cem_solve_time.png").exists():
        lines.extend(["![CEM Solve Time](cem_solve_time.png)", ""])
    (out / "eval_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = expand_group_specs(args.group, Path(args.root).expanduser().resolve())
    summary = summarize(rows)
    write_csvs(rows, summary, out)
    plot_bar(summary, out, args.title, args.dpi)
    plot_by_seed(rows, summary, out, args.title, args.dpi)
    plot_runtime(summary, out, args.title, args.dpi)
    write_report(summary, out, args.title)

    print(json.dumps({"out": str(out), "num_files": len(rows), "groups": group_order(summary)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
