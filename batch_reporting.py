#!/usr/bin/env python3

"""Reporting and artifact generation helpers"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
from typing import List, Optional

from batch_parsing_models import TestResult


def save_aggregate_files(results: List[TestResult], output_dir: Path) -> None:
    json_path = output_dir / "results.json"
    log_path = output_dir / "combined.log"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("=== Survival Batch Combined Log ===\n\n")
        for r in results:
            f.write(f"--- Run {r.run_id} (seed={r.seed}) ---\n")
            f.write(f"command: {' '.join(r.command)}\n")
            f.write(f"return_code: {r.return_code}\n")
            f.write(f"elapsed_seconds: {r.elapsed_seconds:.3f}\n")
            f.write(f"status: {r.status}\n")
            f.write(f"collisions: {r.collisions}\n")
            f.write(f"lane_invasions: {r.lane_invasions}\n")
            f.write(f"distance_breaches: {r.distance_breaches}\n")
            f.write(f"distance_traveled_m: {r.distance_traveled_m}\n")
            f.write(f"min_required_distance_traveled_m: {r.min_required_distance_traveled_m}\n")
            f.write(f"cpu_energy_before_uj: {r.cpu_energy_before_uj}\n")
            f.write(f"cpu_energy_after_uj: {r.cpu_energy_after_uj}\n")
            f.write(f"cpu_energy_j: {r.cpu_energy_j}\n")
            f.write(f"cpu_energy_uj: {r.cpu_energy_uj}\n")
            f.write(f"cpu_energy_error: {r.cpu_energy_error}\n")
            f.write(f"cpu_energy_parse_ok: {r.cpu_energy_parse_ok}\n")
            f.write(f"gpu_energy_j: {r.gpu_energy_j}\n")
            f.write(f"gpu_average_power_w: {r.gpu_average_power_w}\n")
            f.write(f"gpu_sample_interval_seconds: {r.gpu_sample_interval_seconds}\n")
            f.write(f"gpu_sample_count: {r.gpu_sample_count}\n")
            f.write(f"gpu_energy_error: {r.gpu_energy_error}\n")
            f.write(f"gpu_energy_parse_ok: {r.gpu_energy_parse_ok}\n")
            f.write(f"parse_ok: {r.parse_ok}\n")
            f.write("stdout:\n")
            f.write(r.stdout_text + "\n")
            if r.stderr_text.strip():
                f.write("stderr:\n")
                f.write(r.stderr_text + "\n")
            f.write("\n")


def mean_or_nan(values: List[Optional[float]]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return float("nan")
    return statistics.mean(valid)


def save_cpu_energy_table_markdown(results: List[TestResult], energy_dir: Path) -> Path:
    md_path = energy_dir / "cpu_energy_table.md"

    lines = [
        "# CPU Energy Breakdown Table",
        "",
        "| Run | E_start (uJ) | E_end (uJ) | DeltaE (J) | Time (s) | P_avg (W) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        if r.cpu_energy_j is None:
            lines.append(f"| {r.run_id} | n/a | n/a | n/a | {r.elapsed_seconds:.2f} | n/a |")
            continue

        energy_start = f"{int(r.cpu_energy_before_uj):,}" if r.cpu_energy_before_uj is not None else "n/a"
        energy_end = f"{int(r.cpu_energy_after_uj):,}" if r.cpu_energy_after_uj is not None else "n/a"
        avg_power = (r.cpu_energy_j / r.elapsed_seconds) if r.cpu_energy_j is not None and r.elapsed_seconds > 0 else None
        power_str = f"{avg_power:.3f}" if avg_power is not None else "n/a"
        lines.append(
            f"| {r.run_id} | {energy_start} | {energy_end} | {r.cpu_energy_j:.3f} | {r.elapsed_seconds:.2f} | {power_str} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def save_gpu_energy_table_markdown(results: List[TestResult], energy_dir: Path) -> Path:
    md_path = energy_dir / "gpu_energy_table.md"

    lines = [
        "# GPU Energy Breakdown Table",
        "",
        "| Run | DeltaE (J) | Avg Power (W) | Samples | Sample Interval (s) | Time (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        if r.gpu_energy_j is None:
            lines.append(f"| {r.run_id} | n/a | n/a | n/a | n/a | {r.elapsed_seconds:.2f} |")
            continue

        avg_power = f"{r.gpu_average_power_w:.3f}" if r.gpu_average_power_w is not None else "n/a"
        sample_count = str(r.gpu_sample_count) if r.gpu_sample_count is not None else "n/a"
        sample_interval = f"{r.gpu_sample_interval_seconds:.3f}" if r.gpu_sample_interval_seconds is not None else "n/a"
        lines.append(
            f"| {r.run_id} | {r.gpu_energy_j:.3f} | {avg_power} | {sample_count} | {sample_interval} | {r.elapsed_seconds:.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def print_summary(
    results: List[TestResult],
    output_dir: Path,
    cpu_energy_table_path: Optional[Path] = None,
    gpu_energy_table_path: Optional[Path] = None,
) -> None:
    collisions_avg = mean_or_nan([r.collisions for r in results])
    lane_avg = mean_or_nan([r.lane_invasions for r in results])
    distance_avg = mean_or_nan([r.distance_breaches for r in results])
    traveled_avg = mean_or_nan([r.distance_traveled_m for r in results])
    cpu_energy_avg = mean_or_nan([r.cpu_energy_j for r in results])
    gpu_energy_avg = mean_or_nan([r.gpu_energy_j for r in results])

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    print("\n=== Batch Summary ===")
    print(f"runs: {len(results)}")
    print(f"PASS: {pass_count} | FAIL: {fail_count}")
    print(f"average collisions: {collisions_avg:.2f}" if math.isfinite(collisions_avg) else "average collisions: n/a")
    print(f"average lane invasions: {lane_avg:.2f}" if math.isfinite(lane_avg) else "average lane invasions: n/a")
    print(f"average distance breaches: {distance_avg:.2f}" if math.isfinite(distance_avg) else "average distance breaches: n/a")
    print(f"average distance traveled: {traveled_avg:.2f} m" if math.isfinite(traveled_avg) else "average distance traveled: n/a")
    print(f"average cpu energy: {cpu_energy_avg:.2f} J" if math.isfinite(cpu_energy_avg) else "average cpu energy: n/a")
    print(f"average gpu energy: {gpu_energy_avg:.2f} J" if math.isfinite(gpu_energy_avg) else "average gpu energy: n/a")
    print(f"results folder: {output_dir}")
    if cpu_energy_table_path is not None:
        print(f"cpu energy table: {cpu_energy_table_path}")
    if gpu_energy_table_path is not None:
        print(f"gpu energy table: {gpu_energy_table_path}")
