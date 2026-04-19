#!/usr/bin/env python3

"""Plotting helpers"""

from collections import Counter
import math
from pathlib import Path
import statistics
from typing import List, Optional, Tuple

from batch_parsing_models import TestResult


def _mean_or_nan(values: List[Optional[float]]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return float("nan")
    return statistics.mean(valid)


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
    ax.grid(True, linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


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
    _plot_metric_with_average(
        plt,
        run_ids,
        [r.distance_breaches for r in results],
        "Distance Breaches",
        output_dir / "distance_breaches_plot.png",
    )
    _plot_metric_with_average(
        plt,
        run_ids,
        [r.min_observed_front_rear_distance for r in results],
        "Minimum Vehicle Gap Distance (Front/Rear)",
        output_dir / "min_front_rear_distance_plot.png",
    )
    _plot_metric_with_average(
        plt,
        run_ids,
        [r.cpu_energy_j for r in results],
        "CPU Energy (J)",
        energy_dir / "cpu_energy_plot.png",
        line_label="CPU energy",
    )
    _plot_metric_with_average(
        plt,
        run_ids,
        [r.gpu_energy_j for r in results],
        "GPU Energy (J)",
        energy_dir / "gpu_energy_plot.png",
        line_label="GPU energy",
    )
    _plot_energy_range(plt, results, energy_dir)
    _plot_lane_marking_totals(plt, results, output_dir)
    _plot_lane_marking_by_run(plt, results, run_ids, output_dir)
