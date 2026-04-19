#!/usr/bin/env python3
import argparse
import csv
import datetime
import json
import subprocess
import sys
import time
from typing import List, Optional, Tuple


GPU_ENERGY_JSON_PREFIX = "GPU_ENERGY_JSON:"


def get_gpu_power_w() -> Tuple[Optional[float], Optional[str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return None, f"failed to run nvidia-smi: {exc}"

    if result.returncode != 0:
        err = (result.stderr or "").strip() or (result.stdout or "").strip()
        return None, f"nvidia-smi failed (exit {result.returncode}): {err}"

    line = (result.stdout or "").strip().splitlines()
    if not line:
        return None, "nvidia-smi returned no power data"

    value_text = line[0].strip()
    if value_text.upper() == "N/A":
        return None, "nvidia-smi reported power.draw as N/A"

    try:
        return float(value_text), None
    except ValueError:
        return None, f"unable to parse power value from nvidia-smi output: {value_text!r}"


def integrate_energy_j(samples: List[Tuple[float, float]], end_time: float) -> Optional[float]:
    if not samples:
        return None

    energy_j = 0.0
    for idx in range(len(samples) - 1):
        t0, p0 = samples[idx]
        t1, _ = samples[idx + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        energy_j += p0 * dt

    last_sample_time, last_power_w = samples[-1]
    tail_dt = end_time - last_sample_time
    if tail_dt > 0:
        energy_j += last_power_w * tail_dt

    return energy_j


def _emit_payload(payload: dict) -> None:
    print(f"{GPU_ENERGY_JSON_PREFIX} {json.dumps(payload, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and estimate GPU energy using nvidia-smi."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.90,
        help="Sampling interval in seconds (default: 0.90).",
    )
    parser.add_argument(
        "--log-csv",
        default="gpu_log.csv",
        help="CSV path to store samples.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after a standalone -- separator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        print("usage: measure_gpu_energy.py [--interval S] [--log-csv PATH] -- <command> [args...]", file=sys.stderr)
        return 2

    samples: List[Tuple[float, float]] = []
    sampler_first_error: Optional[str] = None
    csv_file = None
    csv_writer = None

    if args.log_csv:
        try:
            csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["timestamp", "power_W"])
        except OSError as exc:
            print(f"unable to open CSV log file {args.log_csv}: {exc}", file=sys.stderr)
            return 2

    start = time.monotonic()

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        elapsed_seconds = time.monotonic() - start
        _emit_payload(
            {
                "command": command,
                "sample_interval_seconds": args.interval,
                "elapsed_seconds": elapsed_seconds,
                "sample_count": len(samples),
                "average_power_w": None,
                "energy_j": None,
                "error": f"failed to start child process: {exc}",
                "exit_code": 127,
            }
        )
        if csv_file is not None:
            csv_file.close()
        return 127

    while process.poll() is None:
        ts = time.monotonic()
        power_w, error = get_gpu_power_w()
        if error is not None:
            if sampler_first_error is None:
                sampler_first_error = error
        elif power_w is not None:
            samples.append((ts, power_w))
            if csv_writer is not None:
                csv_writer.writerow([datetime.datetime.now().isoformat(), power_w])

        time.sleep(args.interval)

    stdout_text, stderr_text = process.communicate()
    end_time = time.monotonic()
    elapsed_seconds = end_time - start

    if csv_file is not None:
        csv_file.close()

    if stdout_text:
        sys.stdout.write(stdout_text)
        if not stdout_text.endswith("\n"):
            sys.stdout.write("\n")
    if stderr_text:
        sys.stderr.write(stderr_text)
        if not stderr_text.endswith("\n"):
            sys.stderr.write("\n")

    energy_j = integrate_energy_j(samples, end_time)
    average_power_w = (energy_j / elapsed_seconds) if (energy_j is not None and elapsed_seconds > 0) else None

    error_text: Optional[str] = None
    if not samples and sampler_first_error is not None:
        error_text = sampler_first_error

    _emit_payload(
        {
            "command": command,
            "sample_interval_seconds": args.interval,
            "elapsed_seconds": elapsed_seconds,
            "sample_count": len(samples),
            "average_power_w": average_power_w,
            "energy_j": energy_j,
            "error": error_text,
            "exit_code": process.returncode,
        }
    )
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
