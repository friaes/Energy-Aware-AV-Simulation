#!/usr/bin/env python3

"""Parsing helpers and data models"""

from collections import Counter
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


RESULT_JSON_PREFIX = "RESULT_JSON:"
CPU_ENERGY_JSON_PREFIX = "CPU_ENERGY_JSON:"
GPU_ENERGY_JSON_PREFIX = "GPU_ENERGY_JSON:"
LANE_MARK_REASON_RE = re.compile(r"lane invasion detected lane_mark=\(([^)]*)\)")


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
    cpu_energy_j: Optional[float]
    cpu_energy_uj: Optional[float]
    cpu_energy_before_uj: Optional[float]
    cpu_energy_after_uj: Optional[float]
    cpu_energy_error: Optional[str]
    cpu_energy_parse_ok: bool
    gpu_energy_j: Optional[float]
    gpu_average_power_w: Optional[float]
    gpu_sample_interval_seconds: Optional[float]
    gpu_sample_count: Optional[int]
    gpu_energy_error: Optional[str]
    gpu_energy_parse_ok: bool
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


def _parse_prefixed_json_payload(stdout_text: str, prefix: str) -> Optional[dict]:
    for line in reversed(stdout_text.splitlines()):
        if not line.startswith(prefix):
            continue
        json_text = line[len(prefix):].strip()
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


def parse_test_output(stdout_text: str) -> tuple[str, Optional[int], Optional[int], Optional[int], Optional[float], Dict[str, int], bool]:
    payload = _parse_prefixed_json_payload(stdout_text, RESULT_JSON_PREFIX)
    if isinstance(payload, dict):
        status = str(payload.get("status", "UNKNOWN"))
        collisions = _coerce_optional_int(payload.get("collisions"))
        lane_invasions = _coerce_optional_int(payload.get("lane_invasions"))
        distance_breaches = _coerce_optional_int(payload.get("distance_breaches"))
        min_observed_front_rear_distance = _coerce_optional_float(payload.get("min_observed_front_rear_distance"))

        payload_reasons = payload.get("reasons")
        reasons = [str(reason) for reason in payload_reasons] if isinstance(payload_reasons, list) else []
        lane_mark_counts = _extract_lane_mark_counts_from_reasons(reasons)
        return status, collisions, lane_invasions, distance_breaches, min_observed_front_rear_distance, lane_mark_counts, True

    return "UNKNOWN", None, None, None, None, {}, False


def parse_cpu_energy_output(stdout_text: str) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str], bool]:
    payload = _parse_prefixed_json_payload(stdout_text, CPU_ENERGY_JSON_PREFIX)
    if isinstance(payload, dict):
        energy_j = _coerce_optional_float(payload.get("energy_j"))
        energy_uj = _coerce_optional_float(payload.get("energy_uj"))
        energy_before_uj = _coerce_optional_float(payload.get("energy_before_uj"))
        energy_after_uj = _coerce_optional_float(payload.get("energy_after_uj"))
        error = payload.get("error")
        error_text = str(error) if error not in (None, "") else None
        return energy_j, energy_uj, energy_before_uj, energy_after_uj, error_text, True

    return None, None, None, None, None, False


def parse_gpu_energy_output(stdout_text: str) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[str], bool]:
    payload = _parse_prefixed_json_payload(stdout_text, GPU_ENERGY_JSON_PREFIX)
    if isinstance(payload, dict):
        energy_j = _coerce_optional_float(payload.get("energy_j"))
        average_power_w = _coerce_optional_float(payload.get("average_power_w"))
        sample_interval_seconds = _coerce_optional_float(payload.get("sample_interval_seconds"))
        sample_count = _coerce_optional_int(payload.get("sample_count"))
        error = payload.get("error")
        error_text = str(error) if error not in (None, "") else None
        return energy_j, average_power_w, sample_interval_seconds, sample_count, error_text, True

    return None, None, None, None, None, False


