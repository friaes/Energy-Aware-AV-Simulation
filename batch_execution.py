#!/usr/bin/env python3

"""Execution and dispatch helpers"""

from pathlib import Path
import subprocess
import time
from typing import List

from batch_parsing_models import (
    TestResult,
    parse_cpu_energy_output,
    parse_gpu_energy_output,
    parse_test_output,
)


def run_single_test(
    run_id: int,
    seed: int,
    python_exe: str,
    test_script: Path,
    forwarded_args: List[str],
    host: str,
    rpc_port: int,
    tm_port: int,
    cpu_energy_script: Path,
    gpu_energy_script: Path,
    gpu_sample_interval: float,
    gpu_log_csv_path: Path,
) -> TestResult:
    command = [
        python_exe,
        str(cpu_energy_script),
        "--",
        python_exe,
        str(gpu_energy_script),
        "--interval",
        str(gpu_sample_interval),
        "--log-csv",
        str(gpu_log_csv_path),
        "--",
        python_exe,
        str(test_script),
        "--host",
        host,
        "--port",
        str(rpc_port),
        "--tm-port",
        str(tm_port),
        "--seed",
        str(seed),
        "--no-progress",
        *forwarded_args,
    ]

    start = time.monotonic()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - start

    stdout_text = process.stdout.strip()
    stderr_text = process.stderr.strip()
    (
        status,
        collisions,
        lane_invasions,
        distance_breaches,
        distance_breach_values_m,
        min_observed_front_rear_distance,
        front_rear_min_distance,
        distance_traveled_m,
        min_required_distance_traveled_m,
        lane_mark_counts,
        parse_ok,
    ) = parse_test_output(stdout_text)
    cpu_energy_j, cpu_energy_uj, cpu_energy_before_uj, cpu_energy_after_uj, cpu_energy_error, cpu_energy_parse_ok = parse_cpu_energy_output(stdout_text)
    gpu_energy_j, gpu_average_power_w, gpu_sample_interval_seconds, gpu_sample_count, gpu_energy_error, gpu_energy_parse_ok = parse_gpu_energy_output(stdout_text)

    return TestResult(
        run_id=run_id,
        seed=seed,
        command=command,
        return_code=process.returncode,
        elapsed_seconds=elapsed,
        status=status,
        collisions=collisions,
        lane_invasions=lane_invasions,
        distance_breaches=distance_breaches,
        distance_breach_values_m=distance_breach_values_m,
        min_observed_front_rear_distance=min_observed_front_rear_distance,
        front_rear_min_distance=front_rear_min_distance,
        distance_traveled_m=distance_traveled_m,
        min_required_distance_traveled_m=min_required_distance_traveled_m,
        cpu_energy_j=cpu_energy_j,
        cpu_energy_uj=cpu_energy_uj,
        cpu_energy_before_uj=cpu_energy_before_uj,
        cpu_energy_after_uj=cpu_energy_after_uj,
        cpu_energy_error=cpu_energy_error,
        cpu_energy_parse_ok=cpu_energy_parse_ok,
        gpu_energy_j=gpu_energy_j,
        gpu_average_power_w=gpu_average_power_w,
        gpu_sample_interval_seconds=gpu_sample_interval_seconds,
        gpu_sample_count=gpu_sample_count,
        gpu_energy_error=gpu_energy_error,
        gpu_energy_parse_ok=gpu_energy_parse_ok,
        lane_mark_counts=lane_mark_counts,
        parse_ok=parse_ok,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def worker_run_tests(
    slot,
    run_specs: List[tuple[int, int]],
    python_exe: str,
    test_script: Path,
    forwarded_args: List[str],
    cpu_energy_script: Path,
    gpu_energy_script: Path,
    gpu_sample_interval: float,
    gpu_logs_dir: Path,
) -> List[TestResult]:
    local_results: List[TestResult] = []
    for run_id, seed in run_specs:
        result = run_single_test(
            run_id=run_id,
            seed=seed,
            python_exe=python_exe,
            test_script=test_script,
            forwarded_args=forwarded_args,
            host=slot.host,
            rpc_port=slot.rpc_port,
            tm_port=slot.tm_port,
            cpu_energy_script=cpu_energy_script,
            gpu_energy_script=gpu_energy_script,
            gpu_sample_interval=gpu_sample_interval,
            gpu_log_csv_path=gpu_logs_dir / f"gpu_log_run_{run_id}.csv",
        )
        local_results.append(result)
        print(
            f"[batch] run {result.run_id} on server {slot.slot_id} complete: "
            f"status={result.status} collisions={result.collisions} "
            f"lane={result.lane_invasions} distance={result.distance_breaches}",
            flush=True,
        )
    return local_results


def worker_run_warmups(
    slot,
    warmup_specs: List[tuple[int, int]],
    python_exe: str,
    test_script: Path,
    forwarded_args: List[str],
    cpu_energy_script: Path,
    gpu_energy_script: Path,
    gpu_sample_interval: float,
    gpu_logs_dir: Path,
    cycle_id: int,
) -> None:
    for warmup_index, seed in warmup_specs:
        run_single_test(
            run_id=warmup_index,
            seed=seed,
            python_exe=python_exe,
            test_script=test_script,
            forwarded_args=forwarded_args,
            host=slot.host,
            rpc_port=slot.rpc_port,
            tm_port=slot.tm_port,
            cpu_energy_script=cpu_energy_script,
            gpu_energy_script=gpu_energy_script,
            gpu_sample_interval=gpu_sample_interval,
            gpu_log_csv_path=(
                gpu_logs_dir / f"gpu_log_warmup_cycle_{cycle_id}_server_{slot.slot_id}_run_{warmup_index}.csv"
            ),
        )
        print(
            f"[warmup] server {slot.slot_id} warmup {warmup_index}/{len(warmup_specs)} complete",
            flush=True,
        )
