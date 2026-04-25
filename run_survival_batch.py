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
import os
import random
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from batch_execution import worker_run_tests, worker_run_warmups
from batch_parsing_models import ServerSlot, TestResult
from batch_plotting import create_plots
from batch_reporting import (
    print_summary,
    save_aggregate_files,
    save_cpu_energy_table_markdown,
    save_gpu_energy_table_markdown,
)

CARLA_COMMAND = "-quality-level=Epic -nosound"


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
        CARLA_COMMAND,
        *carla_extra_args,
    ]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple survival_test.py instances by starting one/several local CARLA server(s) and aggregating results"
    )
    # Batch configuration parameters
    parser.add_argument("--runs", type=int, default=30, help="Number of survival test runs")
    parser.add_argument("--servers", type=int, default=1, help="Number of CARLA servers to launch")
    # Server connection and startup parameters
    parser.add_argument("--host", default="127.0.0.1", help="Host used by survival_test.py and startup checks")
    parser.add_argument("--rpc-base-port", type=int, default=2000, help="Base CARLA RPC port for server 1")
    parser.add_argument("--rpc-port-step", type=int, default=100, help="RPC port step between servers (default: 100)")
    parser.add_argument("--tm-base-port", type=int, default=8000, help="Base Traffic Manager port for server 1")
    parser.add_argument("--tm-port-step", type=int, default=100, help="TM port step between servers (default: 100)")
    # Startup timing parameters and CARLA launch configuration
    parser.add_argument("--server-startup-stagger", type=float, default=2.0, help="Seconds delay between server launches")
    parser.add_argument("--server-start-timeout", type=float, default=120.0, help="Seconds to wait for each server RPC port (default: 120)")
    parser.add_argument("--server-world-ready-timeout", type=float, default=120.0, help="Seconds to wait for each server to answer client.get_world()")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=6,
        help="Fallback warm-up runs per server when phase-specific options are not set (excluded from results)",
    )
    parser.add_argument(
        "--initial-warmup-runs",
        type=int,
        default=6,
        help="Warm-up runs per server before the first measured cycle (excluded from results)",
    )
    parser.add_argument(
        "--restart-warmup-runs",
        type=int,
        default=1,
        help="Warm-up runs per server after each server restart cycle (excluded from results)",
    )
    parser.add_argument("--server-restart-every-runs", type=int, default=0, help="Restart all CARLA servers every N runs (0 disables restarts)")
    parser.add_argument("--carla-script", default="~/Carla/CarlaUE4.sh", help="Path to CarlaUE4.sh used to launch servers")
    parser.add_argument("--carla-extra-args", default="", help="Extra args appended to CarlaUE4.sh command")
    parser.add_argument("--keep-servers", action="store_true", help="Do not stop CARLA server processes on exit")
    # Test script and forwarding parameters
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used to run survival_test.py")
    parser.add_argument("--test-script", default="survival_test.py", help="Path to survival test script")
    parser.add_argument("--cpu-energy-script", default="measure_cpu_energy.py", help="Path to the CPU energy wrapper script")
    parser.add_argument("--gpu-energy-script", default="measure_gpu_energy.py", help="Path to the GPU energy wrapper script")
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0, help="GPU power sampling interval in seconds")
    parser.add_argument("--base-seed", type=int, default=1000, help="Base seed; run i uses base_seed + i")
    parser.add_argument("--output-dir", default="out", help="Output folder for logs and plots (default: out)")
    parser.add_argument("--test-args", nargs=argparse.REMAINDER, help="Arguments forwarded to survival_test.py after --test-args")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.runs <= 0:
        print("--runs must be > 0", file=sys.stderr)
        return 2
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

    # Forward any extra args after --test-args to the test script
    forwarded_args = list(args.test_args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

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

    run_specs_all: List[tuple[int, int]] = []
    for i in range(args.runs):
        run_id = i + 1
        seed = args.base_seed + i if args.base_seed is not None else random.randint(1, 1_000_000)
        run_specs_all.append((run_id, seed))

    initial_warmup_runs = args.initial_warmup_runs if args.initial_warmup_runs is not None else args.warmup_runs
    restart_warmup_runs = args.restart_warmup_runs if args.restart_warmup_runs is not None else args.warmup_runs

    chunk_size = restart_every if restart_every > 0 else args.runs
    
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
                        warmup_seed_base = (args.base_seed + args.runs) if args.base_seed is not None else None
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

    save_aggregate_files(results, output_dir)
    cpu_energy_table_path = save_cpu_energy_table_markdown(results, energy_dir)
    gpu_energy_table_path = save_gpu_energy_table_markdown(results, energy_dir)
    try:
        create_plots(results, output_dir, energy_dir)
    except RuntimeError as exc:
        print(f"[plots] warning: {exc}", file=sys.stderr, flush=True)
    print_summary(results, output_dir, cpu_energy_table_path, gpu_energy_table_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
