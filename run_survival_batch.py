#!/usr/bin/env python3

"""Run multiple CARLA survival tests across several local CARLA servers.

This script launches several local CARLA servers (CarlaUE4.sh) on different
ports, dispatches survival_test.py runs in parallel, and aggregates results
with logs and plots for:
- collisions per run (+ average)
- lane invasions per run (+ average)
- distance breaches per run (+ average)
"""

import argparse
from collections import Counter
import json
import random
import re
import socket
import subprocess
import sys
import time
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RESULT_JSON_PREFIX = "RESULT_JSON:"
LANE_MARK_REASON_RE = re.compile(r"lane invasion detected lane_mark=\(([^)]*)\)")


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_result_json_payload(stdout_text: str) -> Optional[dict]:
    for line in reversed(stdout_text.splitlines()):
        if not line.startswith(RESULT_JSON_PREFIX):
            continue
        json_text = line[len(RESULT_JSON_PREFIX):].strip()
        if not json_text:
            return None
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        return payload
    return None


def _extract_lane_mark_counts_from_reasons(reasons: List[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()

    for reason in reasons:
        match = LANE_MARK_REASON_RE.search(reason)
        if not match:
            continue
        lane_mark = match.group(1).strip() or "unknown"
        counts[lane_mark] += 1
    return dict(counts)


def _parse_test_output(stdout_text: str) -> tuple[str, Optional[int], Optional[int], Optional[int], Optional[float], Optional[float], Dict[str, int], bool]:
    payload = _parse_result_json_payload(stdout_text)
    if isinstance(payload, dict):
        status = str(payload.get("status", "UNKNOWN"))
        collisions = _coerce_optional_int(payload.get("collisions"))
        lane_invasions = _coerce_optional_int(payload.get("lane_invasions"))
        distance_breaches = _coerce_optional_int(payload.get("distance_breaches"))
        min_observed_front_rear_distance = _coerce_optional_float(payload.get("min_observed_front_rear_distance"))
        min_observed_side_distance = _coerce_optional_float(payload.get("min_observed_side_distance"))

        payload_reasons = payload.get("reasons")
        reasons = [str(reason) for reason in payload_reasons] if isinstance(payload_reasons, list) else []
        lane_mark_counts = _extract_lane_mark_counts_from_reasons(reasons)
        return status, collisions, lane_invasions, distance_breaches, min_observed_front_rear_distance, min_observed_side_distance, lane_mark_counts, True

    return "UNKNOWN", None, None, None, None, None, {}, False


@dataclass
class TestResult:
    run_id: int
    seed: int
    command: List[str]
    return_code: int
    elapsed_seconds: float
    status: str
    collisions: Optional[int]
    lane_invasions: Optional[int]
    distance_breaches: Optional[int]
    min_observed_front_rear_distance: Optional[float]
    min_observed_side_distance: Optional[float]
    lane_mark_counts: Dict[str, int]
    parse_ok: bool
    stdout_text: str
    stderr_text: str


@dataclass
class ServerSlot:
    slot_id: int
    host: str
    rpc_port: int
    tm_port: int
    process: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None


def run_single_test(
    run_id: int,
    seed: int,
    python_exe: str,
    test_script: Path,
    forwarded_args: List[str],
    host: str,
    rpc_port: int,
    tm_port: int,
) -> TestResult:
    command = [
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
    status, collisions, lane_invasions, distance_breaches, min_observed_front_rear_distance, min_observed_side_distance, lane_mark_counts, parse_ok = _parse_test_output(stdout_text)

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
        min_observed_front_rear_distance=min_observed_front_rear_distance,
        min_observed_side_distance=min_observed_side_distance,
        lane_mark_counts=lane_mark_counts,
        parse_ok=parse_ok,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


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

def _plot_metric_with_average(
    plt,
    run_ids: List[int],
    values_optional: List[Optional[int]],
    label: str,
    plot_path: Path,
    *,
    average_values_optional: Optional[List[Optional[int]]] = None,
    average_label_prefix: str = "Average",
    line_label: Optional[str] = None,
    threshold: Optional[Tuple[float, str, str]] = None,
    highlight_points: Optional[Tuple[List[int], List[float], str, str]] = None,
) -> None:
    values = [v if v is not None else math.nan for v in values_optional]
    average_source = average_values_optional if average_values_optional is not None else values_optional
    average = mean_or_nan(average_source)

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


def create_plots(results: List[TestResult], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting"
        ) from exc

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
        "Minimum Observed Front/Rear Distance",
        output_dir / "min_front_rear_distance_plot.png",
    )
    _plot_metric_with_average(
        plt,
        run_ids,
        [r.min_observed_side_distance for r in results],
        "Minimum Observed Side Distance",
        output_dir / "min_side_distance_plot.png",
    )
    _plot_lane_marking_totals(plt, results, output_dir)
    _plot_lane_marking_by_run(plt, results, run_ids, output_dir)


def print_summary(results: List[TestResult], output_dir: Path) -> None:
    collisions_avg = mean_or_nan([r.collisions for r in results])
    lane_avg = mean_or_nan([r.lane_invasions for r in results])
    distance_avg = mean_or_nan([r.distance_breaches for r in results])

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    print("\n=== Batch Summary ===")
    print(f"runs: {len(results)}")
    print(f"PASS: {pass_count} | FAIL: {fail_count}")
    print(f"average collisions: {collisions_avg:.2f}" if math.isfinite(collisions_avg) else "average collisions: n/a")
    print(f"average lane invasions: {lane_avg:.2f}" if math.isfinite(lane_avg) else "average lane invasions: n/a")
    print(f"average distance breaches: {distance_avg:.2f}" if math.isfinite(distance_avg) else "average distance breaches: n/a")
    print(f"results folder: {output_dir}")


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
        "-quality-level=Low",
        "-RenderOffScreen",
        "-nosound",
        *carla_extra_args,
    ]

    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
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
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def worker_run_tests(
    slot: ServerSlot,
    run_specs: List[tuple[int, int]],
    python_exe: str,
    test_script: Path,
    forwarded_args: List[str],
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
        )
        local_results.append(result)
        print(
            f"[batch] run {result.run_id} on server {slot.slot_id} complete: "
            f"status={result.status} collisions={result.collisions} "
            f"lane={result.lane_invasions} distance={result.distance_breaches}",
            flush=True,
        )
    return local_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple survival_test.py instances by starting one/several local CARLA server(s) and aggregating results"
    )
    # Batch configuration parameters
    parser.add_argument("--runs", type=int, default=12, help="Number of survival test runs")
    parser.add_argument("--servers", type=int, default=2, help="Number of CARLA servers to launch")
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
    parser.add_argument("--carla-script", default="~/Carla/CarlaUE4.sh", help="Path to CarlaUE4.sh used to launch servers")
    parser.add_argument("--carla-extra-args", default="", help="Extra args appended to CarlaUE4.sh command")
    parser.add_argument("--keep-servers", action="store_true", help="Do not stop CARLA server processes on exit")
    # Test script and forwarding parameters
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used to run survival_test.py")
    parser.add_argument("--test-script", default="survival_test.py", help="Path to survival test script")
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

    this_file_dir = Path(__file__).resolve().parent
    test_script = Path(args.test_script)
    if not test_script.is_absolute():
        test_script = (this_file_dir / test_script).resolve()

    if not test_script.exists():
        print(f"test script not found: {test_script}", file=sys.stderr)
        return 2

    carla_script = Path(args.carla_script).expanduser().resolve()
    if not carla_script.exists():
        print(f"CARLA script not found: {carla_script}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()

    # Server logs are written during startup, so ensure folder exists first.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Forward any extra args after --test-args to the test script
    forwarded_args = list(args.test_args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    carla_extra_args = args.carla_extra_args.split() if args.carla_extra_args else []

    servers: List[ServerSlot] = []
    results: List[TestResult] = []
    started_servers: List[ServerSlot] = []
    
    try:
        # Start CARLA servers and verify world readiness
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

        # Distribute runs across available servers
        run_specs_per_server: List[List[tuple[int, int]]] = [[] for _ in servers]
        for i in range(args.runs):
            run_id = i + 1
            seed = args.base_seed + i if args.base_seed is not None else random.randint(1, 1_000_000)
            target = i % len(servers)
            run_specs_per_server[target].append((run_id, seed))

        # Run tests in parallel on available servers
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
                    )
                )

            for future in as_completed(futures):
                results.extend(future.result())
                
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

    results.sort(key=lambda r: r.run_id)

    save_aggregate_files(results, output_dir)
    create_plots(results, output_dir)
    print_summary(results, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
