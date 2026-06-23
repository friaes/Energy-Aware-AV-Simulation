# Energy-Aware Environment Configuration in Simulation-Based Testing of Autonomous Vehicles

A research toolkit for studying how **simulator configuration parameters** affect both the **safety verdict** (oracle pass/fail) and the **energy consumption** of CARLA-based autonomous-vehicle endurance tests.

The central research question is:

> Can we reduce the energy cost of simulation-based AV testing — by changing parameters such as physics step size, NPC traffic density, or LiDAR resolution — **without changing the test outcome** the oracle produces?

To answer this, the toolkit runs a fixed-duration "survival" test of an autopilot-driven ego vehicle, evaluates it against a safety oracle, measures CPU and GPU energy for each run, and aggregates everything across a randomized, blocked factorial experiment.

---

## Table of contents

- [Concept overview](#concept-overview)
- [The survival test and oracle](#the-survival-test-and-oracle)
- [Energy measurement](#energy-measurement)
- [Experimental design: randomized blocks](#experimental-design-randomized-blocks)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running a full factorial sweep](#running-a-full-factorial-sweep)
- [Command-line reference](#command-line-reference)
- [Output artifacts](#output-artifacts)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)

---

## Concept overview

The pipeline is built from three nested layers. From the inside out:

1. **A single survival test** (`survival_test.py`) drives one ego vehicle on autopilot through CARLA traffic for a fixed duration and evaluates it against a safety oracle, emitting a structured pass/fail verdict.
2. **Energy wrappers** (`measure_cpu_energy.py`, `measure_gpu_energy.py`) each wrap the layer beneath them, measuring CPU package energy (Intel RAPL) and GPU energy (NVIDIA power sampling) consumed during the run.
3. **A batch orchestrator** (`run_survival_batch.py`) launches one or more CARLA servers, schedules many runs across configurations in a randomized blocked design, collects the verdicts and energy figures, and produces aggregate reports and plots.

A single measured run is therefore a nested command:

```
measure_cpu_energy.py  ──  measure_gpu_energy.py  ──  survival_test.py
   (RAPL counters)            (nvidia-smi sampling)      (CARLA + oracle)
```

Each layer runs the next as a subprocess and prints a machine-readable JSON line that the orchestrator parses.

---

## The survival test and oracle

`survival_test.py` spawns a Tesla ego vehicle plus a configurable number of NPC vehicles, places all of them under CARLA Traffic Manager autopilot, and runs the world for a fixed duration. Throughout the run, a `SurvivalOracle` continuously monitors four safety-relevant signals:

- **Collisions** — detected via CARLA's collision sensor.
- **Lane invasions** — detected via the lane-invasion sensor; the type of lane marking crossed (e.g. broken vs solid) is recorded.
- **Minimum front clearance** — estimated from the ego LiDAR by selecting forward points within the ego's lane corridor and taking the closest return. A breach is recorded when clearance drops below `--min-front-distance`.
- **Distance traveled** — accumulated from the ego's per-tick displacement, optionally checked against a `--min-distance-traveled` floor at the end of the run.

The run is reported as **PASS** if none of the oracle's failure conditions trip, and **FAIL** otherwise. The result is printed both human-readably and as a single structured line prefixed with `RESULT_JSON:` that downstream tooling parses.

> **Note on the oracle definition.** The thresholds (`--min-front-distance`, `--min-distance-traveled`, etc.) define the *oracle envelope*. They are calibrated against a chosen baseline configuration; runs whose behaviour falls far outside that envelope should be interpreted with care, since the oracle was not calibrated for them.

---

## Energy measurement

Two independent wrappers measure energy for whatever command they are given:

**`measure_cpu_energy.py`** reads Intel RAPL energy counters from
`/sys/class/powercap/intel-rapl:0/energy_uj` immediately before and after the wrapped command, handling counter wrap-around, and reports the delta in joules. Output line prefix: `CPU_ENERGY_JSON:`.

**`measure_gpu_energy.py`** polls `nvidia-smi --query-gpu=power.draw` at a fixed interval while the wrapped command runs, then numerically integrates power over time to estimate total GPU energy in joules. It also writes a per-run CSV of raw power samples. Output line prefix: `GPU_ENERGY_JSON:`.

Because each wrapper simply runs an arbitrary command and measures around it, they compose cleanly: the batch runner nests CPU → GPU → TEST so that one invocation produces a verdict plus both energy figures.

---

## Experimental design: randomized blocks

When run with `--factor` arguments, the orchestrator builds a **full factorial** of all parameter combinations and executes them in a **randomized complete block design**:

- Each *block* contains exactly one run of every parameter combination.
- The order of combinations within each block is independently shuffled (seeded for reproducibility).
- The number of blocks (`--blocks`) equals the number of repetitions per combination.

This design controls for time-correlated confounds — most importantly **GPU thermal throttling**, where sustained load gradually changes clock speeds and therefore both timing and energy. By interleaving conditions rather than running all repetitions of one condition back-to-back, thermal drift is spread evenly across all conditions instead of biasing any single one.

The exact execution order is written to `schedule.csv` for full reproducibility. Optional warm-up runs (`--initial-warmup-runs`) further stabilize measurement conditions; warm-up runs are excluded from all reports.

---

## Repository structure

| File | Role |
|---|---|
| `survival_test.py` | **Core test.** Single CARLA endurance run + `SurvivalOracle`. Spawns ego + NPCs, monitors collisions/lane/clearance/distance. |
| `measure_cpu_energy.py` | CPU energy wrapper using Intel RAPL counters. |
| `measure_gpu_energy.py` | GPU energy wrapper using `nvidia-smi` power sampling + integration. |
| `run_survival_batch.py` | **Main orchestrator** Launches CARLA server(s), builds the blocked-factorial schedule, dispatches runs, aggregates results. |
| `batch_execution.py` | Builds and runs the nested CPU→GPU→TEST command per run; worker functions for measured and warm-up runs. |
| `batch_parsing_models.py` | `TestResult` / `ServerSlot` dataclasses and the parsers for the three `*_JSON:` output lines. |
| `batch_reporting.py` | Writes `results.json`, `combined.log`, CPU/GPU energy Markdown tables, and prints the batch summary. |
| `batch_plotting.py` | Generates per-run and per-combination plots (collisions, lane invasions, clearance, distance, CPU/GPU energy) via matplotlib. |

**Module dependency flow:**

```
run_survival_batch.py
├── batch_execution.py        → run_single_test (nests the 3 scripts)
│   └── batch_parsing_models  → TestResult, parse_*_output
├── batch_parsing_models.py   → dataclasses + JSON parsers
├── batch_plotting.py         → create_plots / create_factor_plots
└── batch_reporting.py        → save_aggregate_files / tables / summary
```

---

## Requirements

- **CARLA simulator** (a build providing `CarlaUE4.sh` and a matching `carla` Python API). The test was developed against CARLA 0.9.16.
- **Python 3.10+** supported python version for Carla.
- **`carla`** Python package matching your CARLA server version (required by `survival_test.py`).
- **`matplotlib`** (required only for plot generation in `batch_plotting.py`).
- **Linux** for energy measurement:
  - **Intel RAPL** exposed at `/sys/class/powercap/intel-rapl:0/energy_uj` for CPU energy. Reading these counters may require appropriate permissions.
  - **NVIDIA GPU + `nvidia-smi`** on `PATH` for GPU energy.

Energy measurement degrades gracefully: if RAPL or `nvidia-smi` is unavailable, the corresponding energy field is reported as `null` with an error message, and the rest of the pipeline still runs.

---

## Installation

```bash
# 1. Clone / unzip this repository
cd Energy-Aware-AV-Simulation-main

# 2. Install plotting dependency
pip install matplotlib
```

Make sure `CarlaUE4.sh` is reachable (default expected path is `~/Carla/CarlaUE4.sh`; override with `--carla-script`).

---

## Quick start

Run a single survival test directly against an already-running CARLA server, with no energy wrapping or batching:

```bash
# Terminal 1: start CARLA
~/Carla/CarlaUE4.sh -quality-level=Epic -nosound

# Terminal 2: run one 60-second test with 10 NPCs
python3 survival_test.py --town Town10HD --npc-count 10 --duration 60
```

The script prints a `=== Survival Test Result ===` block.

---

## Running a full factorial sweep

The orchestrator can launch the CARLA server(s) itself. The example below reproduces the kind of 3 × 3 × 2 sweep used in the study — NPC count × physics step × LiDAR points-per-second — with 10 blocks (i.e. 10 repetitions per combination):

```bash
python3 run_survival_batch.py \
  --carla-script ~/Carla/CarlaUE4.sh \
  --servers 1 \
  --blocks 10 \
  --factor npc-count=10,30,50 \
  --factor fixed-delta-seconds=0.05,0.10,0.02 \
  --factor lidar-points-per-second=10000,100000 \
  --initial-warmup-runs 3 \
  --output-dir out_sweep \
  --test-args --duration 60
```

What happens:

- A full factorial of 3 × 3 × 2 = **18 combinations** is built.
- With `--blocks 10`, the schedule contains **180 measured runs**, organised as 10 randomized blocks of 18.
- Each factor name maps to the corresponding `survival_test.py` argument (e.g. `fixed-delta-seconds` → `--fixed-delta-seconds`) and is forwarded automatically.
- Anything after `--test-args` is forwarded verbatim to every test run.
- Results are aggregated both **per combination** (one sub-folder each) and **across all combinations** at the output root.

---

## Command-line reference

### `run_survival_batch.py` (orchestrator)

| Argument | Default | Description |
|---|---|---|
| `--runs` | `5` | Number of runs in **non-factor** mode (ignored when `--factor` is used). |
| `--servers` | `1` | Number of CARLA servers to launch in parallel. |
| `--factor name=v1,v2,...` | — | Declare an experiment factor; repeatable. Triggers factorial mode. |
| `--blocks` | `1` | Number of randomized complete blocks (= repetitions per combination). |
| `--base-seed` | `500` | Base RNG seed; run *i* uses `base_seed + i`. |
| `--warmup-runs` | `3` | Fallback warm-up runs per server (excluded from results). |
| `--carla-script` | `~/Carla/CarlaUE4.sh` | Path to `CarlaUE4.sh`. |
| `--carla-extra-args` | `""` | Extra args appended to the CARLA launch command. |
| `--gpu-sample-interval` | `1.0` | GPU power sampling interval (s). |
| `--test-script` / `--cpu-energy-script` / `--gpu-energy-script` | the three `.py` files | Override script paths. |
| `--output-dir` | `out` | Output folder for logs, tables, and plots. |
| `--test-args ...` | — | Everything after is forwarded to `survival_test.py`. |

### `survival_test.py` (single run)

| Argument | Default | Description |
|---|---|---|
| `--host` / `--port` / `--tm-port` | `127.0.0.1` / `2000` / `8000` | CARLA connection. |
| `--town` | current map | Map to load (e.g. `Town10HD`). |
| `--duration` | `60.0` | Test window in seconds. |
| `--npc-count` | `10` | Number of NPC vehicles. |
| `--ego-filter` | `vehicle.tesla.*` | Blueprint filter for the ego. |
| `--sync` / `--no-sync` | sync on | Synchronous vs asynchronous world. |
| `--fixed-delta-seconds` | `0.05` | Physics step size (synchronous mode). |
| `--weather` | none | Weather preset (e.g. `ClearNoon`, `HardRainSunset`). |
| `--min-front-distance` | `3.0` | Minimum allowed forward clearance (m). |
| `--min-distance-traveled` | `150.0` | Minimum required distance traveled (m); 0 disables. |
| `--lidar-points-per-second` | `10000` | Ego LiDAR resolution. |
| `--lidar-tick` / `--camera-tick` / `--camera-resolution` | — | Optional sensor settings. |
| `--seed` | none | Random seed for reproducibility. |
| `--output-dir` | `out` | Per-run artifact directory. |

### Energy wrappers

```bash
# CPU only
python measure_cpu_energy.py -- <command> [args...]

# GPU only (1s sampling, log to gpu.csv)
python measure_gpu_energy.py --interval 1.0 --log-csv gpu.csv -- <command> [args...]
```

---

## Output artifacts

A factorial run produces, at the output root and within each `combo_*` sub-folder:

| Artifact | Description |
|---|---|
| `schedule.csv` | The exact run order: `run_id`, `block_id`, `combo_id`, `order_in_block`, factor levels. |
| `results.json` | All `TestResult` records (verdicts, metrics, energy, parse status). |
| `combined.log` | Human-readable per-run dump including raw stdout/stderr. |
| `factor_summary.json` | Map of each combo id to its factor values and output folder. |
| `energy/cpu_energy_table.md` | Per-run CPU energy table (ΔE, time, average power). |
| `energy/gpu_energy_table.md` | Per-run GPU energy table (ΔE, average power, samples). |
| `energy/gpu_logs/gpu_log_run_*.csv` | Raw per-run GPU power samples. |
| `collisions_plot.png`, `lane_invasions_plot.png`, `distance_breaches_plot.png`, `distance_traveled_plot.png` | Per-run / per-combination safety-metric plots. |
| `energy/cpu_energy_plot.png`, `energy/gpu_energy_plot.png` | Energy plots (with median and confidence intervals). |
| `carla_server_*.log` | CARLA server stdout/stderr for debugging startup issues. |

The terminal also prints a `=== Batch Summary ===` with pass/fail counts and averaged metrics.

---

## Reproducibility

- Every run's seed is derived deterministically from `--base-seed`.
- Block shuffling is driven by `--schedule-seed` (defaults to the base seed), so the same flags reproduce the same execution order.
- `schedule.csv` records the realized order; `results.json` records every input and output.
- CARLA Traffic Manager is seeded per run so NPC behaviour is repeatable.

---

## Troubleshooting

**`CARLA server did not expose RPC port ... within Ns`** — increase `--server-start-timeout`, confirm `--carla-script` points to a valid `CarlaUE4.sh`, and check `carla_server_*.log` in the output folder.

**CPU energy is `null`** — Intel RAPL is unavailable or unreadable. Confirm `/sys/class/powercap/intel-rapl:0/energy_uj` exists and is readable (RAPL is Intel-only and may need elevated permissions).

**GPU energy is `null`** — `nvidia-smi` is missing or the GPU does not report `power.draw`. Confirm `nvidia-smi --query-gpu=power.draw --format=csv` works.

**`matplotlib is required for plotting`** — `pip install matplotlib`. Data artifacts (`results.json`, tables) are still written without it.

**Failed to spawn ego vehicle** — the chosen map may be crowded; increase `--spawn-attempts` or reduce `--npc-count`.

**Unstable physics at small steps** — very small `--fixed-delta-seconds` values can make the simulation unstable and contaminate oracle results. Values at or above `0.02` are recommended.