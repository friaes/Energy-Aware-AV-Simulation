#!/usr/bin/env python3

"""Run multiple CARLA survival tests across several local CARLA servers.

This script launches several local CARLA servers (CarlaUE4.sh) on different
ports, dispatches survival_test.py runs in parallel, and aggregates results
with logs and plots for:
- collisions per run (+ average)
- lane invasions per run (+ average)
- distance breaches per run (+ average)
- CPU package energy per run (+ average)
- GPU energy per run (+ average)
"""

import argparse
import json
from dataclasses import replace
import csv
import itertools
import os
import random
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from batch_helpers.batch_execution import worker_run_tests, worker_run_warmups
from batch_helpers.batch_parsing_models import ServerSlot, TestResult
from batch_helpers.batch_plotting import create_factor_plots, create_plots
from batch_helpers.batch_reporting import (
    print_summary,
    save_aggregate_files,
    save_cpu_energy_table_markdown,
    save_gpu_energy_table_markdown,
)

CARLA_COMMAND = ["-quality-level=Epic", "-nosound"]


def wait_for_tcp(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            time.sleep(0.5)
        finally:
            sock.close()
    return False


def wait_for_tcp_close(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            return True
        finally:
            sock.close()
        time.sleep(0.5)
    return False


def wait_for_carla_world(host: str, port: int, timeout_seconds: float, python_exe: str) -> Tuple[bool, str]:
    deadline = time.monotonic() + timeout_seconds
    probe_code = (
        "import carla,sys; "
        "client=carla.Client(sys.argv[1], int(sys.argv[2])); "
        "client.set_timeout(2.0); "
        "client.get_world(); "
        "print('ready')"
    )

    last_error = ""
    while time.monotonic() < deadline:
        probe = subprocess.run(
            [python_exe, "-c", probe_code, host, str(port)],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return True, ""
        stderr_text = (probe.stderr or "").strip()
        stdout_text = (probe.stdout or "").strip()
        last_error = stderr_text if stderr_text else stdout_text
        time.sleep(1.0)

    return False, last_error


def start_server(
    slot_id: int,
    host: str,
    rpc_port: int,
    tm_port: int,
    carla_script: Path,
    output_dir: Path,
    carla_extra_args: List[str],
    startup_timeout: float,
) -> ServerSlot:
    log_path = output_dir / f"carla_server_{slot_id}.log"
    command = [
        str(carla_script),
        f"-carla-rpc-port={rpc_port}",
        *CARLA_COMMAND,
        *carla_extra_args,
    ]
    try:
        print(f"[servers] launch command: {' '.join(command)}", flush=True)
    except Exception:
        pass

    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_file.close()

    if not wait_for_tcp(host, rpc_port, startup_timeout):
        stop_server(process)
        raise RuntimeError(
            f"CARLA server {slot_id} did not expose RPC port {rpc_port} within {startup_timeout}s. "
            f"See log: {log_path}"
        )

    return ServerSlot(
        slot_id=slot_id,
        host=host,
        rpc_port=rpc_port,
        tm_port=tm_port,
        process=process,
        log_path=log_path,
    )


def tail_server_log(log_path: Optional[Path], lines: int = 30) -> str:
    if not log_path:
        return "<no log file>"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "<unable to read server log>"
    parts = text.splitlines()
    return "\n".join(parts[-lines:]) if parts else "<empty server log>"


def stop_server(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return


def _sanitize_path_component(value: str) -> str:
    cleaned = []
    for character in value:
        if character.isalnum() or character in ("-", "_", "."):
            cleaned.append(character)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._")
    return result or "value"


def _combo_dir_name(combo_id: int, combo_values: Dict[str, object]) -> str:
    parts = [f"combo_{combo_id:02d}"]
    for key in sorted(combo_values):
        parts.append(f"{_sanitize_path_component(str(key))}={_sanitize_path_component(str(combo_values[key]))}")
    return "__".join(parts)


def _combo_forwarded_args(combo_values: Dict[str, object]) -> List[str]:
    forwarded: List[str] = []
    for key in sorted(combo_values):
        forwarded.append(f"--{key.replace('_', '-')}")
        forwarded.append(str(combo_values[key]))
    return forwarded


def _parse_factor_spec(spec: str) -> Optional[tuple[str, List[str]]]:
    if "=" not in spec:
        return None
    name, values_text = spec.split("=", 1)
    levels = [value for value in values_text.split(",") if value != ""]
    if not name.strip() or not levels:
        return None
    return name.strip(), levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple survival_test.py instances by starting one/several local CARLA server(s) and aggregating results"
    )
    # Batch configuration parameters
    parser.add_argument("--runs", type=int, default=5, help="Number of survival test runs (used in non-factor mode)")
    parser.add_argument("--servers", type=int, default=1, help="Number of CARLA servers to launch")
    parser.add_argument("--output-dir", default="out", help="Output folder for logs and plots (default: out)")
    # Server connection and startup parameters
    parser.add_argument("--host", default="127.0.0.1", help="Host used by survival_test.py and startup checks")
    parser.add_argument("--rpc-base-port", type=int, default=2000, help="Base CARLA RPC port for server 1")
    parser.add_argument("--rpc-port-step", type=int, default=100, help="RPC port step between servers (default: 100)")
    parser.add_argument("--tm-base-port", type=int, default=8000, help="Base Traffic Manager port for server 1")
    parser.add_argument("--tm-port-step", type=int, default=100, help="TM port step between servers (default: 100)")
    # Startup timing parameters
    parser.add_argument("--server-startup-stagger", type=float, default=2.0, help="Seconds delay between server launches")
    parser.add_argument("--server-start-timeout", type=float, default=120.0, help="Seconds to wait for each server RPC port (default: 120)")
    parser.add_argument("--server-world-ready-timeout", type=float, default=120.0, help="Seconds to wait for each server to answer client.get_world()")
    # Warm-up run and server restart parameters 
    parser.add_argument("--warmup-runs", type=int, default=3, help="Fallback warm-up runs per server when phase-specific options are not set (excluded from results)")
    parser.add_argument("--initial-warmup-runs", type=int, default=None, help="Warm-up runs per server before the first measured cycle (excluded from results)")
    parser.add_argument("--restart-warmup-runs", type=int, default=None, help="Warm-up runs per server after each server restart cycle (excluded from results)")
    parser.add_argument("--server-restart-every-runs", type=int, default=0, help="Restart all CARLA servers every N runs (0 disables restarts)")
    # CARLA launch configuration parameters
    parser.add_argument("--carla-script", default="~/Carla/CarlaUE4.sh", help="Path to CarlaUE4.sh used to launch servers")
    parser.add_argument("--carla-extra-args", default="", help="Extra args appended to CarlaUE4.sh command")
    parser.add_argument("--keep-servers", action="store_true", help="Do not stop CARLA server processes on exit")
    # Path and script configuration parameters
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used to run survival_test.py")
    parser.add_argument("--test-script", default="survival_test.py", help="Path to survival test script")
    parser.add_argument("--cpu-energy-script", default="measure_cpu_energy.py", help="Path to the CPU energy wrapper script")
    parser.add_argument("--gpu-energy-script", default="measure_gpu_energy.py", help="Path to the GPU energy wrapper script")
    
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0, help="GPU power sampling interval in seconds")
    parser.add_argument("--base-seed", type=int, default=500, help="Base seed; run i uses base_seed + i")
    parser.add_argument("--schedule-seed", type=int, default=None, help="Seed used to deterministically shuffle combos per block (defaults to base-seed)")
    # Forwarded arguments to survival_test.py
    parser.add_argument("--test-args", nargs=argparse.REMAINDER, help="Arguments forwarded to survival_test.py after --test-args")
    # Factorial experiment parameters
    parser.add_argument("--factor", action="append", help="Experiment factor in form name=level1,level2 (can be repeated)")
    parser.add_argument("--blocks", type=int, default=1, help="Number of randomized blocks to run (each block contains all factor combinations shuffled)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.servers <= 0:
        print("--servers must be > 0", file=sys.stderr)
        return 2
    if args.server_restart_every_runs < 0:
        print("--server-restart-every-runs must be >= 0", file=sys.stderr)
        return 2
    if args.warmup_runs < 0:
        print("--warmup-runs must be >= 0", file=sys.stderr)
        return 2
    if args.initial_warmup_runs is not None and args.initial_warmup_runs < 0:
        print("--initial-warmup-runs must be >= 0", file=sys.stderr)
        return 2
    if args.restart_warmup_runs is not None and args.restart_warmup_runs < 0:
        print("--restart-warmup-runs must be >= 0", file=sys.stderr)
        return 2

    this_file_dir = Path(__file__).resolve().parent
    test_script = Path(args.test_script)
    if not test_script.is_absolute():
        test_script = (this_file_dir / test_script).resolve()

    cpu_energy_script = Path(args.cpu_energy_script)
    if not cpu_energy_script.is_absolute():
        cpu_energy_script = (this_file_dir / cpu_energy_script).resolve()

    gpu_energy_script = Path(args.gpu_energy_script)
    if not gpu_energy_script.is_absolute():
        gpu_energy_script = (this_file_dir / gpu_energy_script).resolve()

    if not test_script.exists():
        print(f"test script not found: {test_script}", file=sys.stderr)
        return 2

    if not cpu_energy_script.exists():
        print(f"CPU energy script not found: {cpu_energy_script}", file=sys.stderr)
        return 2

    if not gpu_energy_script.exists():
        print(f"GPU energy script not found: {gpu_energy_script}", file=sys.stderr)
        return 2

    carla_script = Path(args.carla_script).expanduser().resolve()
    if not carla_script.exists():
        print(f"CARLA script not found: {carla_script}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    energy_dir = output_dir / "energy"
    gpu_logs_dir = energy_dir / "gpu_logs"

    # Server logs are written during startup, so ensure folder exists first.
    output_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)
    gpu_logs_dir.mkdir(parents=True, exist_ok=True)

    factor_mode = bool(args.factor)

    # Forward any extra args after --test-args to the test script
    forwarded_args = list(args.test_args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    explicit_output_dir = any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in forwarded_args)
    if not factor_mode and not explicit_output_dir:
        forwarded_args = forwarded_args + ["--output-dir", str(output_dir)]

    carla_extra_args = args.carla_extra_args.split() if args.carla_extra_args else []

    results: List[TestResult] = []
    started_servers: List[ServerSlot] = []

    restart_every = args.server_restart_every_runs
    if args.keep_servers and restart_every > 0:
        print(
            "[servers] warning: --keep-servers is incompatible with --server-restart-every-runs; disabling periodic restarts",
            file=sys.stderr,
            flush=True,
        )
        restart_every = 0

    # If experimental factors are provided, build a blocked-factorial schedule.
    # In this mode, total measured runs are derived from blocks x combinations.
    schedule_map: Dict[int, dict] = {}
    schedule_rows = []
    combo_dirs: Dict[int, Path] = {}
    effective_runs = args.runs
    combo_specs: List[dict] = []

    if factor_mode:
        if args.blocks <= 0:
            print("--blocks must be > 0 when --factor is used", file=sys.stderr)
            return 2

        factors: List[tuple[str, List[str]]] = []
        for spec in args.factor:
            parsed = _parse_factor_spec(spec)
            if parsed is None:
                print(f"ignoring malformed factor spec: {spec}", file=sys.stderr)
                continue
            factors.append(parsed)

        combos = []
        if factors:
            names = [n for n, _ in factors]
            level_lists = [levels for _, levels in factors]
            for combo_idx, combo in enumerate(itertools.product(*level_lists), start=1):
                combo_values = dict(zip(names, combo))
                combos.append((combo_idx, combo_values))

        if not combos:
            print("no valid factor combinations were built from --factor", file=sys.stderr)
            return 2

        derived_runs = args.blocks * len(combos)
        if args.runs != derived_runs:
            print(
                f"[schedule] info: blocked mode uses blocks*combinations = {derived_runs} runs; ignoring --runs={args.runs}",
                flush=True,
            )
        effective_runs = derived_runs

        for combo_id, combo_values in combos:
            combo_dir = output_dir / _combo_dir_name(combo_id, combo_values)
            combo_dirs[combo_id] = combo_dir
            combo_specs.append({"combo_id": combo_id, "combo_values": combo_values, "dir": combo_dir})
            combo_dir.mkdir(parents=True, exist_ok=True)

    if effective_runs <= 0:
        print("run count must be > 0", file=sys.stderr)
        return 2

    run_specs_all: List[tuple[int, int]] = []
    for i in range(effective_runs):
        run_id = i + 1
        seed = args.base_seed + i if args.base_seed is not None else random.randint(1, 1_000_000)
        run_specs_all.append((run_id, seed))

    if factor_mode:
        # Build schedule by randomized complete blocks.
        seed_base = args.schedule_seed if args.schedule_seed is not None else (args.base_seed or 12345)
        run_index = 0
        for block_id in range(1, args.blocks + 1):
            rng = random.Random(seed_base + block_id)
            block_combos = combos.copy()
            rng.shuffle(block_combos)
            order_in_block = 0
            for combo_id, combo_values in block_combos:
                order_in_block += 1
                run_index += 1
                run_id, _ = run_specs_all[run_index - 1]
                combo_dir = combo_dirs[combo_id]
                combo_forwarded_args = _combo_forwarded_args(combo_values)
                if not explicit_output_dir:
                    combo_forwarded_args = combo_forwarded_args + ["--output-dir", str(combo_dir)]
                schedule_map[run_id] = {
                    "block_id": block_id,
                    "combo_id": combo_id,
                    "combo_values": combo_values,
                    "forwarded_args": combo_forwarded_args,
                    "order_in_block": order_in_block,
                    "combo_dir": combo_dir,
                }
                schedule_rows.append((run_id, block_id, combo_id, order_in_block, combo_values))

        # Write schedule CSV for reproducibility
        try:
            schedule_path = output_dir / "schedule.csv"
            with schedule_path.open("w", encoding="utf-8", newline="") as csvf:
                fieldnames = ["run_id", "block_id", "combo_id", "order_in_block"]
                if schedule_rows:
                    fieldnames.extend(sorted(schedule_rows[0][4].keys()))
                writer = csv.DictWriter(csvf, fieldnames=fieldnames)
                writer.writeheader()
                for run_id, block_id, combo_id, order_in_block, combo_values in schedule_rows:
                    row = dict(combo_values)
                    row.update({"run_id": run_id, "block_id": block_id, "combo_id": combo_id, "order_in_block": order_in_block})
                    writer.writerow(row)
            print(f"[schedule] written {len(schedule_rows)} scheduled runs to {schedule_path}")
        except Exception as exc:
            print(f"[schedule] warning: failed to write schedule CSV: {exc}", file=sys.stderr)

    initial_warmup_runs = args.initial_warmup_runs if args.initial_warmup_runs is not None else args.warmup_runs
    restart_warmup_runs = args.restart_warmup_runs if args.restart_warmup_runs is not None else args.warmup_runs

    chunk_size = restart_every if restart_every > 0 else effective_runs
    
    try:
        for chunk_start in range(0, len(run_specs_all), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(run_specs_all))
            chunk_run_specs = run_specs_all[chunk_start:chunk_end]
            started_servers = []
            servers: List[ServerSlot] = []

            if restart_every > 0:
                chunk_id = (chunk_start // chunk_size) + 1
                print(
                    f"[servers] restart cycle {chunk_id}: preparing runs {chunk_run_specs[0][0]}..{chunk_run_specs[-1][0]}",
                    flush=True,
                )

            # Start CARLA servers and verify world readiness for this chunk
            print("[servers] starting CARLA servers...", flush=True)
            for idx in range(args.servers):
                slot_id = idx + 1
                rpc_port = args.rpc_base_port + idx * args.rpc_port_step
                tm_port = args.tm_base_port + idx * args.tm_port_step

                if idx > 0 and args.server_startup_stagger > 0:
                    print(f"[servers] waiting {args.server_startup_stagger}s before launching server {slot_id}", flush=True)
                    time.sleep(args.server_startup_stagger)

                print(f"[servers] launching server {slot_id} host={args.host} rpc={rpc_port} tm={tm_port}", flush=True)
                server = start_server(
                    slot_id=slot_id,
                    host=args.host,
                    rpc_port=rpc_port,
                    tm_port=tm_port,
                    carla_script=carla_script,
                    output_dir=output_dir,
                    carla_extra_args=carla_extra_args,
                    startup_timeout=args.server_start_timeout,
                )
                started_servers.append(server)

                world_ready, world_err = wait_for_carla_world(
                    host=args.host,
                    port=rpc_port,
                    timeout_seconds=args.server_world_ready_timeout,
                    python_exe=args.python_exe,
                )

                if world_ready and server.process is not None and server.process.poll() is None:
                    servers.append(server)
                    print(f"[servers] server {slot_id} world-ready on {args.host}:{rpc_port}", flush=True)
                else:
                    stop_server(server.process)
                    reason = world_err or "server process exited before world readiness"
                    print(
                        f"[servers] warning: server {slot_id} not usable at {args.host}:{rpc_port} - {reason}",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(f"[servers] log tail ({server.log_path}):\n{tail_server_log(server.log_path)}", file=sys.stderr, flush=True)

            if not servers:
                print(
                    f"ERROR: no CARLA servers are reachable at {args.host}:{args.rpc_base_port} ",
                    file=sys.stderr,
                )
                return 1

            print(f"[servers] found {len(servers)} reachable servers, starting test dispatch...\n", flush=True)

            is_first_cycle = chunk_start == 0
            cycle_warmup_runs = initial_warmup_runs if is_first_cycle else restart_warmup_runs

            if cycle_warmup_runs > 0:
                cycle_label = "initial" if is_first_cycle else "restart"
                print(
                    f"[warmup] running {cycle_warmup_runs} {cycle_label} warm-up run(s) per server (excluded from reports)",
                    flush=True,
                )
                with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                    warmup_futures = []
                    for server in servers:
                        warmup_seed_base = (args.base_seed + effective_runs) if args.base_seed is not None else None
                        warmup_specs = [
                            (
                                warmup_idx + 1,
                                (
                                    warmup_seed_base + chunk_start + (server.slot_id * 10_000) + warmup_idx
                                    if warmup_seed_base is not None
                                    else random.randint(1, 1_000_000)
                                ),
                            )
                            for warmup_idx in range(cycle_warmup_runs)
                        ]
                        warmup_futures.append(
                            executor.submit(
                                worker_run_warmups,
                                server,
                                warmup_specs,
                                args.python_exe,
                                test_script,
                                forwarded_args,
                                cpu_energy_script,
                                gpu_energy_script,
                                args.gpu_sample_interval,
                                gpu_logs_dir,
                                (chunk_start // chunk_size) + 1,
                            )
                        )
                    for future in as_completed(warmup_futures):
                        future.result()

            run_specs_per_server: List[List[tuple[int, int]]] = [[] for _ in servers]
            for run_id, seed in chunk_run_specs:
                target = (run_id - 1) % len(servers)
                run_specs_per_server[target].append((run_id, seed))

            with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                futures = []
                for server, run_specs in zip(servers, run_specs_per_server):
                    if not run_specs:
                        continue
                    gpu_log_paths_by_run_id = None
                    if factor_mode:
                        gpu_log_paths_by_run_id = {}
                        for run_id, _seed in run_specs:
                            meta = schedule_map.get(run_id)
                            if not meta:
                                continue
                            combo_dir = meta["combo_dir"]
                            combo_energy_dir = combo_dir / "energy"
                            combo_gpu_logs_dir = combo_energy_dir / "gpu_logs"
                            combo_energy_dir.mkdir(parents=True, exist_ok=True)
                            combo_gpu_logs_dir.mkdir(parents=True, exist_ok=True)
                            gpu_log_paths_by_run_id[run_id] = combo_gpu_logs_dir / f"gpu_log_run_{run_id}.csv"
                    futures.append(
                        executor.submit(
                            worker_run_tests,
                            server,
                            run_specs,
                            args.python_exe,
                            test_script,
                            forwarded_args,
                            cpu_energy_script,
                            gpu_energy_script,
                            args.gpu_sample_interval,
                            gpu_logs_dir,
                                gpu_log_paths_by_run_id,
                                schedule_map,
                        )
                    )

                for future in as_completed(futures):
                    results.extend(future.result())

            if not args.keep_servers:
                for server in reversed(started_servers):
                    print(f"[servers] stopping server {server.slot_id}", flush=True)
                    stop_server(server.process)
                    if wait_for_tcp_close(server.host, server.rpc_port, args.server_start_timeout):
                        print(
                            f"[servers] server {server.slot_id} rpc port {server.rpc_port} closed",
                            flush=True,
                        )
                    else:
                        print(
                            f"[servers] warning: server {server.slot_id} rpc port {server.rpc_port} did not close in time",
                            file=sys.stderr,
                            flush=True,
                        )
                started_servers = []

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except RuntimeError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_servers:
            for server in reversed(started_servers):
                print(f"[servers] stopping server {server.slot_id}", flush=True)
                stop_server(server.process)
                wait_for_tcp_close(server.host, server.rpc_port, args.server_start_timeout)

    results.sort(key=lambda r: r.run_id)

    # Annotate results with schedule metadata if available.
    if schedule_map:
        for r in results:
            meta = schedule_map.get(r.run_id)
            if not meta:
                continue
            r.block_id = meta["block_id"]
            r.combo_id = meta["combo_id"]
            r.combo_factors = meta["combo_values"]

    def write_reports(target_results: List[TestResult], target_output_dir: Path, *, factor_view: bool = False) -> None:
        target_energy_dir = target_output_dir / "energy"
        target_gpu_logs_dir = target_energy_dir / "gpu_logs"
        target_output_dir.mkdir(parents=True, exist_ok=True)
        target_energy_dir.mkdir(parents=True, exist_ok=True)
        target_gpu_logs_dir.mkdir(parents=True, exist_ok=True)
        save_aggregate_files(target_results, target_output_dir)
        cpu_energy_table_path = save_cpu_energy_table_markdown(target_results, target_energy_dir)
        gpu_energy_table_path = save_gpu_energy_table_markdown(target_results, target_energy_dir)
        try:
            if factor_view:
                create_factor_plots(target_results, target_output_dir, target_energy_dir)
            else:
                create_plots(target_results, target_output_dir, target_energy_dir)
        except RuntimeError as exc:
            print(f"[plots] warning: {exc}", file=sys.stderr, flush=True)
        print_summary(target_results, target_output_dir, cpu_energy_table_path, gpu_energy_table_path)

    if factor_mode:
        # Emit a root-level aggregate report so the batch output shows
        # the full cross combination view in the same format as normal mode.
        write_reports(results, output_dir, factor_view=True)

        grouped_results: Dict[int, List[TestResult]] = {}
        for result in results:
            if result.combo_id is None:
                continue
            grouped_results.setdefault(result.combo_id, []).append(result)

        for combo_id, combo_results in grouped_results.items():
            combo_meta = combo_specs[combo_id - 1]
            combo_output_dir = combo_meta["dir"]
            combo_results_sorted = sorted(combo_results, key=lambda item: (item.block_id or 0, item.run_id))
            combo_results_local = [
                replace(result, run_id=index + 1)
                for index, result in enumerate(combo_results_sorted)
            ]
            write_reports(combo_results_local, combo_output_dir)

        summary_path = output_dir / "factor_summary.json"
        factor_summary = [
            {
                "combo_id": combo_id,
                "output_dir": str(combo_meta["dir"]),
                "factors": combo_meta["combo_values"],
            }
            for combo_id, combo_meta in sorted(((spec["combo_id"], spec) for spec in combo_specs), key=lambda item: item[0])
        ]
        summary_path.write_text(json.dumps(factor_summary, indent=2) + "\n", encoding="utf-8")
        print(f"[schedule] factor summary: {summary_path}", flush=True)
    else:
        write_reports(results, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
