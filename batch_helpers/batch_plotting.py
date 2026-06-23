#!/usr/bin/env python3

"""Plotting helpers"""

from collections import Counter
from collections import defaultdict
import math
from pathlib import Path
import statistics
import random
from typing import Callable, Dict, List, Optional, Tuple

from batch_helpers.batch_parsing_models import TestResult


def _mean_or_nan(values: List[Optional[float]]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return float("nan")
    return statistics.mean(valid)


def _combo_label(result: TestResult) -> str:
    if result.combo_factors:
        parts = [f"{key}={result.combo_factors[key]}" for key in sorted(result.combo_factors)]
        return f"combo {result.combo_id}: " + ", ".join(parts)
    if result.combo_id is not None:
        return f"combo {result.combo_id}"
    return "unassigned"


def _plot_metric_with_average(
    plt,
    run_ids: List[int],
    values_optional: List[Optional[float]],
    label: str,
    plot_path: Path,
    *,
    average_values_optional: Optional[List[Optional[float]]] = None,
    average_label_prefix: str = "Average",
    line_label: Optional[str] = None,
    threshold: Optional[Tuple[float, str, str]] = None,
    highlight_points: Optional[Tuple[List[int], List[float], str, str]] = None,
) -> None:
    values = [v if v is not None else math.nan for v in values_optional]
    average_source = average_values_optional if average_values_optional is not None else values_optional
    average = _mean_or_nan(average_source)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(run_ids, values, marker="o", linestyle="-", linewidth=1.8, label=line_label)

    if highlight_points is not None:
        x_vals, y_vals, point_label, point_color = highlight_points
        if x_vals and y_vals:
            ax.scatter(x_vals, y_vals, color=point_color, marker="x", s=60, label=point_label)

    if threshold is not None:
        threshold_value, threshold_label, threshold_color = threshold
        ax.axhline(threshold_value, color=threshold_color, linestyle="-", linewidth=1.2, label=threshold_label)

    if math.isfinite(average):
        ax.axhline(average, linestyle="--", linewidth=1.4, label=f"{average_label_prefix} = {average:.2f}")

    if line_label or highlight_points is not None or threshold is not None or math.isfinite(average):
        ax.legend()

    ax.set_title(f"{label} Across Survival Tests")
    ax.set_xlabel("Test Run")
    ax.set_ylabel(label)
    ax.set_xticks(run_ids)
    # force origin at zero for metrics like energy
    if 'Energy' in label:
        try:
            ax.set_ylim(bottom=0, top=4000)
        except Exception:
            pass
    ax.grid(True, linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_metric_by_combo(
    plt,
    results: List[TestResult],
    label: str,
    plot_path: Path,
    value_getter: Callable[[TestResult], Optional[float]],
    *,
    y_label: Optional[str] = None,
    threshold: Optional[Tuple[float, str, str]] = None,
) -> None:
    grouped: Dict[int, List[TestResult]] = defaultdict(list)
    for result in results:
        combo_id = result.combo_id if result.combo_id is not None else -1
        grouped[combo_id].append(result)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")

    for index, combo_id in enumerate(sorted(grouped.keys())):
        combo_results = sorted(
            grouped[combo_id],
            key=lambda item: ((item.block_id or 0), item.run_id),
        )
        x_values = [result.block_id if result.block_id is not None else result.run_id for result in combo_results]
        y_values = [value_getter(result) if value_getter(result) is not None else math.nan for result in combo_results]

        color = cmap(index % cmap.N)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            color=color,
            label=_combo_label(combo_results[0]),
        )

    if threshold is not None:
        threshold_value, threshold_label, threshold_color = threshold
        ax.axhline(threshold_value, color=threshold_color, linestyle="-", linewidth=1.2, label=threshold_label)

    ax.set_title(f"{label} Across Combinations and Blocks")
    ax.set_xlabel("Block")
    ax.set_ylabel(y_label or label)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(fontsize="small", ncol=2)
    # Force zero-origin for energy plots
    if 'Energy' in label:
        try:
            ax.set_ylim(bottom=0, top=4000)
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _percentile(sorted_vals: List[float], percent: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _bootstrap_median_ci(values: List[float], n_resamples: int = 1000, rng_seed: Optional[int] = None) -> Tuple[float, float, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(rng_seed)
    medians = []
    m = len(vals)
    for _ in range(n_resamples):
        sample = [rng.choice(vals) for __ in range(m)]
        medians.append(statistics.median(sample))
    medians.sort()
    lower = _percentile(medians, 0.025)
    upper = _percentile(medians, 0.975)
    median_est = statistics.median(vals)
    return (median_est, lower, upper)


def _plot_metric_with_median_ci(
    plt,
    run_ids: List[int],
    values_optional: List[Optional[float]],
    label: str,
    plot_path: Path,
    *,
    y_label: Optional[str] = None,
    n_bootstrap: int = 1000,
    rng_seed: Optional[int] = None,
) -> None:
    values = [v if v is not None else math.nan for v in values_optional]
    valid_pairs = [(rid, v) for rid, v in zip(run_ids, values) if v is not None and not math.isnan(v)]
    x_all = [rid for rid, _ in valid_pairs]
    y_all = [v for _, v in valid_pairs]

    median_est, lower_ci, upper_ci = _bootstrap_median_ci(y_all, n_resamples=n_bootstrap, rng_seed=rng_seed)

    fig, ax = plt.subplots(figsize=(10, 5))
    # plot per-run points
    if x_all:
        ax.scatter(x_all, y_all, color="tab:blue", alpha=0.8, label="Per-run")

    # median and CI
    if not math.isnan(median_est):
        ax.axhline(median_est, linestyle="--", color="black", linewidth=1.6, label=f"Median = {median_est:.2f}")
        ax.fill_between(
            [min(run_ids) - 0.5, max(run_ids) + 0.5],
            [lower_ci, lower_ci],
            [upper_ci, upper_ci],
            color="gray",
            alpha=0.2,
            label=f"95% CI ({lower_ci:.2f}, {upper_ci:.2f})",
        )

    if line_label := None:
        pass

    ax.set_title(f"{label} Across Survival Tests")
    ax.set_xlabel("Test Run")
    ax.set_ylabel(y_label or label)
    ax.set_xticks(run_ids)
    if 'Energy' in label:
        try:
            ax.set_ylim(bottom=0, top=4000)
        except Exception:
            pass
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_front_clearance_with_breaches(plt, results: List[TestResult], run_ids: List[int], output_dir: Path) -> None:
    threshold = next(
        (r.min_front_clearance for r in results if r.min_front_clearance is not None),
        None,
    )

    breach_x_values: List[int] = []
    breach_y_values: List[float] = []
    for result in results:
        breach_values = result.distance_breach_values_m or []
        breach_x_values.extend([result.run_id] * len(breach_values))
        breach_y_values.extend(breach_values)

    _plot_metric_with_average(
        plt,
        run_ids,
        [r.min_front_clearance for r in results],
        "Minimum Front Clearance (m)",
        output_dir / "min_front_clearance_plot.png",
        average_label_prefix="Average minimum",
        line_label="Run minimum clearance",
        #threshold=(threshold, f"Threshold = {threshold:.2f} m", "red") if threshold is not None else None,
        #highlight_points=(breach_x_values, breach_y_values, "Distance breach points", "orange") if breach_x_values else None,
    )


def _plot_lane_marking_totals(plt, results: List[TestResult], output_dir: Path) -> None:
    marking_counts: Counter[str] = Counter()
    for result in results:
        marking_counts.update(result.lane_mark_counts)

    fig, ax = plt.subplots(figsize=(10, 5))
    if marking_counts:
        labels = sorted(marking_counts.keys())
        values = [marking_counts[label] for label in labels]
        ax.bar(labels, values)
        ax.set_ylabel("Count")
    else:
        labels = []
        ax.text(0.5, 0.5, "No lane invasion markings detected", ha="center", va="center")
        ax.set_yticks([])

    ax.set_title("Total Lane Invasions by Marking Types")
    ax.set_xlabel("Lane Marking Type")
    if labels:
        ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    plot_path = output_dir / "lane_invasion_marking_type_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_lane_marking_by_run(plt, results: List[TestResult], run_ids: List[int], output_dir: Path) -> None:
    marking_types = sorted({mark for result in results for mark in result.lane_mark_counts.keys()})
    fig, ax = plt.subplots(figsize=(12, 6))
    if marking_types:
        cmap = plt.get_cmap("tab20")
        for i, marking_type in enumerate(marking_types):
            color = cmap(i % cmap.N)
            values = [result.lane_mark_counts.get(marking_type, 0) for result in results]
            avg_value = statistics.mean(values) if values else 0.0
            ax.plot(run_ids, values, marker="o", linestyle="-", linewidth=1.6, color=color, label=marking_type)
            ax.axhline(
                avg_value,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                color=color,
                label=f"{marking_type} avg = {avg_value:.2f}",
            )
        ax.set_ylabel("Count")
        ax.legend(title="Lane Mark", fontsize="small", ncol=2)
    else:
        ax.text(0.5, 0.5, "No lane invasion markings detected", ha="center", va="center")
        ax.set_yticks([])

    ax.set_title("Lane Invasion Marking Types Across Survival Tests")
    ax.set_xlabel("Test Run")
    ax.set_xticks(run_ids)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    plot_path = output_dir / "lane_invasion_marking_type_by_run_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_energy_range(plt, results: List[TestResult], energy_dir: Path) -> None:
    cpu_values = [r.cpu_energy_j for r in results if r.cpu_energy_j is not None]
    gpu_values = [r.gpu_energy_j for r in results if r.gpu_energy_j is not None]
    total_values = [
        (r.cpu_energy_j + r.gpu_energy_j)
        for r in results
        if r.cpu_energy_j is not None and r.gpu_energy_j is not None
    ]

    ranges = []
    labels = []

    for label, values in (("CPU", cpu_values), ("GPU", gpu_values), ("Total", total_values)):
        if not values:
            continue
        ranges.append((min(values), max(values)))
        labels.append(label)

    plot_path = energy_dir / "energy_min_max_plot.png"
    fig, ax = plt.subplots(figsize=(9, 5))

    if not ranges:
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Energy Consumption Range Across Tests")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        return

    x_positions = list(range(len(labels)))
    for x_pos, low_high in zip(x_positions, ranges):
        low, high = low_high
        ax.vlines(x_pos, low, high, color="tab:blue", linewidth=3)
        ax.scatter([x_pos, x_pos], [low, high], color="tab:blue", s=45)
        ax.text(x_pos, high, f"{high:.1f}", ha="center", va="bottom", fontsize="small")
        ax.text(x_pos, low, f"{low:.1f}", ha="center", va="top", fontsize="small")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy Consumption Range Across Tests")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def create_plots(results: List[TestResult], output_dir: Path, energy_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    run_ids = [r.run_id for r in results]

    collision_values: List[Optional[int]] = [
        (r.collisions if (r.status != "FAIL" and r.collisions is not None) else None)
        for r in results
    ]
    collision_average_values: List[Optional[int]] = [r.collisions for r in results if r.status != "FAIL"]
    failed_points = [
        (r.run_id, float(r.collisions))
        for r in results
        if r.status == "FAIL" and r.collisions is not None
    ]

    _plot_metric_with_average(
        plt,
        run_ids,
        collision_values,
        "Collisions",
        output_dir / "collisions_plot.png",
        average_values_optional=collision_average_values,
        average_label_prefix="Average (excluding FAIL)",
        line_label="Collisions",
        threshold=(5.0, "Failure threshold = 5", "red"),
        highlight_points=(
            [run_id for run_id, _ in failed_points],
            [collisions for _, collisions in failed_points],
            "FAIL run",
            "red",
        ),
    )
    _plot_metric_with_average(
        plt,
        run_ids,
        [r.lane_invasions for r in results],
        "Lane Invasions",
        output_dir / "lane_invasions_plot.png",
    )
    _plot_front_clearance_with_breaches(plt, results, run_ids, output_dir)

    breach_counts = [float(len(r.distance_breach_values_m or [])) for r in results]
    _plot_metric_with_average(
        plt,
        run_ids,
        breach_counts,
        "Distance Breaches",
        output_dir / "distance_breaches_plot.png",
        line_label="Number of breaches",
    )

    distance_threshold = next(
        (
            r.min_required_distance_traveled_m
            for r in results
            if r.min_required_distance_traveled_m is not None and r.min_required_distance_traveled_m > 0
        ),
        None,
    )
    
    _plot_metric_with_average(
        plt,
        run_ids,
        [r.distance_traveled_m for r in results],
        "Ego Distance Traveled (m)",
        output_dir / "distance_traveled_plot.png",
        line_label="Distance traveled",
        threshold=(distance_threshold, f"Threshold = {distance_threshold:.2f} m", "red") if distance_threshold is not None else None,
    )

    _plot_metric_with_median_ci(
        plt,
        run_ids,
        [r.gpu_energy_j for r in results],
        "GPU Energy (J)",
        energy_dir / "gpu_energy_plot.png",
        y_label="GPU Energy (J)",
    )
   
    _plot_metric_with_median_ci(
        plt,
        run_ids,
        [r.cpu_energy_j for r in results],
        "CPU Energy (J)",
        energy_dir / "cpu_energy_plot.png",
        y_label="CPU Energy (J)",
    )

    _plot_energy_range(plt, results, energy_dir)
    _plot_lane_marking_totals(plt, results, output_dir)
    _plot_lane_marking_by_run(plt, results, run_ids, output_dir)


def create_factor_plots(results: List[TestResult], output_dir: Path, energy_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    _plot_metric_by_combo(
        plt,
        results,
        "Collisions",
        output_dir / "collisions_plot.png",
        lambda r: float(r.collisions) if r.collisions is not None else None,
        threshold=(5.0, "Failure threshold = 5", "red"),
    )
    _plot_metric_by_combo(
        plt,
        results,
        "Lane Invasions",
        output_dir / "lane_invasions_plot.png",
        lambda r: float(r.lane_invasions) if r.lane_invasions is not None else None,
    )
    _plot_metric_by_combo(
        plt,
        results,
        "Minimum Front Clearance (m)",
        output_dir / "min_front_clearance_plot.png",
        lambda r: float(r.min_front_clearance) if r.min_front_clearance is not None else None,
        y_label="Minimum Front Clearance (m)",
    )
    _plot_metric_by_combo(
        plt,
        results,
        "Distance Breaches",
        output_dir / "distance_breaches_plot.png",
        lambda r: float(len(r.distance_breach_values_m or [])),
    )
    _plot_metric_by_combo(
        plt,
        results,
        "Ego Distance Traveled (m)",
        output_dir / "distance_traveled_plot.png",
        lambda r: float(r.distance_traveled_m) if r.distance_traveled_m is not None else None,
        y_label="Ego Distance Traveled (m)",
    )
    _plot_metric_by_combo(
        plt,
        results,
        "CPU Energy (J)",
        energy_dir / "cpu_energy_plot.png",
        lambda r: float(r.cpu_energy_j) if r.cpu_energy_j is not None else None,
        y_label="CPU Energy (J)",
    )
    _plot_metric_by_combo(
        plt,
        results,
        "GPU Energy (J)",
        energy_dir / "gpu_energy_plot.png",
        lambda r: float(r.gpu_energy_j) if r.gpu_energy_j is not None else None,
        y_label="GPU Energy (J)",
    )

    _plot_energy_range(plt, results, energy_dir)
    _plot_lane_marking_totals(plt, results, output_dir)
    _plot_lane_marking_by_run(plt, results, [r.block_id if r.block_id is not None else r.run_id for r in results], output_dir)
