#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


INTEL_RAPL_DOMAIN = Path("/sys/class/powercap/intel-rapl:0")
ENERGY_JSON_PREFIX = "CPU_ENERGY_JSON:"


def _read_int_file(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def _capture_energy_snapshot() -> Tuple[Optional[int], Optional[int], Optional[str]]:
    energy_path = INTEL_RAPL_DOMAIN / "energy_uj"
    max_range_path = INTEL_RAPL_DOMAIN / "max_energy_range_uj"

    if not energy_path.exists() or not max_range_path.exists():
        return None, None, f"Intel RAPL energy counter not found: {energy_path}"

    energy_uj = _read_int_file(energy_path)

    max_energy_range_uj = _read_int_file(max_range_path)
    return energy_uj, max_energy_range_uj, None


def _compute_energy(
    before_uj: Optional[int],
    after_uj: Optional[int],
    max_range_uj: Optional[int],
) -> Tuple[Optional[int], Optional[str]]:
    if before_uj is None or after_uj is None:
        return None, "missing energy snapshot"

    delta = after_uj - before_uj
    if delta >= 0:
        return delta, None

    return delta + max_range_uj, None


def _emit_payload(payload: dict) -> None:
    print(f"{ENERGY_JSON_PREFIX} {json.dumps(payload, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and measure CPU package energy with RAPL counters."
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
        print("usage: measure_cpu_energy.py -- <command> [args...]", file=sys.stderr)
        return 2

    before_energy_uj, max_range_uj, before_error = _capture_energy_snapshot()
    start = time.monotonic()

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        elapsed_seconds = time.monotonic() - start
        _emit_payload(
            {
                "command": command,
                "elapsed_seconds": elapsed_seconds,
                "energy_j": None,
                "energy_uj": None,
                "error": f"failed to start child process: {exc}",
                "exit_code": 127,
            }
        )
        return 127

    elapsed_seconds = time.monotonic() - start

    if process.stdout:
        sys.stdout.write(process.stdout)
        if not process.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if process.stderr:
        sys.stderr.write(process.stderr)
        if not process.stderr.endswith("\n"):
            sys.stderr.write("\n")

    after_energy_uj, _, after_error = _capture_energy_snapshot()

    energy_uj: Optional[int] = None
    energy_error = before_error or after_error

    if energy_error is None:
        energy_uj, energy_error = _compute_energy(before_energy_uj, after_energy_uj, max_range_uj)

    _emit_payload(
        {
            "command": command,
            "elapsed_seconds": elapsed_seconds,
            "energy_before_uj": before_energy_uj,
            "energy_after_uj": after_energy_uj,
            "energy_j": (energy_uj / 1_000_000.0) if energy_uj is not None else None,
            "energy_uj": energy_uj,
            "error": energy_error,
            "exit_code": process.returncode,
        }
    )
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())