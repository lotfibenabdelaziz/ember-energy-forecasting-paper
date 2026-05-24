"""
pipeline.py — Full Pipeline Orchestrator
Ember Energy Forecasting | IEEE Paper

Chains:
    Step 1: 01_eda.py           → ./outputs/eda/
    Step 2: 02_preprocessing.py → ./outputs/preprocessing/
    Step 3: 03_modeling.py      → ./outputs/modeling/
    Step 4: 04_forecasting.py   → ./outputs/forecasting/

Usage:
    python pipeline.py --csv data/ember_yearly.csv
    python pipeline.py --csv data/ember_yearly.csv --train_until 2019 --forecast_until 2025
    python pipeline.py --csv data/ember_yearly.csv --steps eda preprocessing
"""

import argparse
import logging
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ (no-op if file absent)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Step registry ─────────────────────────────────────────────────────────────
# Each step's args lambda now receives the full parsed namespace so it can
# forward --train_until and --forecast_until where needed.

_OUT = os.getenv("OUTPUTS_DIR", "outputs")

STEPS = {
    "eda": {
        "script": "./src/01_eda.py",
        "args": lambda a: [
            "--csv",
            a.csv,
            "--output_dir",
            f"{_OUT}/eda",
        ],
    },
    "preprocessing": {
        "script": "./src/02_preprocessing.py",
        "args": lambda a: [
            "--input_dir",
            f"{_OUT}/eda",
            "--output_dir",
            f"{_OUT}/preprocessing",
            "--train_until",
            str(a.train_until),
        ],
    },
    "modeling": {
        "script": "./src/03_modeling.py",
        "args": lambda a: [
            "--input_dir",
            f"{_OUT}/preprocessing",
            "--output_dir",
            f"{_OUT}/modeling",
        ],
    },
    "forecasting": {
        "script": "./src/04_forecasting.py",
        "args": lambda a: [
            "--pre_dir",
            f"{_OUT}/preprocessing",
            "--model_dir",
            f"{_OUT}/modeling",
            "--output_dir",
            f"{_OUT}/forecasting",
            "--forecast_until",
            str(a.forecast_until),
        ],
    },
    "deeplearning": {
        "script": "./src/05_deeplearning.py",
        "args": lambda a: [
            "--pre_dir",
            f"{_OUT}/preprocessing",
            "--output_dir",
            f"{_OUT}/deeplearning",
        ],
    },
}

STEP_ORDER = ["eda", "preprocessing", "modeling", "forecasting", "deeplearning"]


def run_step(name: str, script: str, extra_args: list) -> None:
    cmd = [sys.executable, script] + extra_args
    log.info("╔══ START: %s", name.upper())
    log.info("   cmd: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error("╚══ FAILED: %s (exit=%d) in %.1fs", name, result.returncode, elapsed)
        sys.exit(result.returncode)
    log.info("╚══ DONE: %s in %.1fs", name, elapsed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember full pipeline")
    p.add_argument(
        "--csv",
        default=os.getenv("CSV_PATH"),
        help="Path to Ember yearly CSV  [env: CSV_PATH]",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_ORDER,
        default=STEP_ORDER,
        help="Steps to run (default: all four)",
    )
    p.add_argument(
        "--train_until",
        type=int,
        default=int(os.getenv("TRAIN_UNTIL", 2019)),
        help="Last year of training window  [env: TRAIN_UNTIL, default: 2019]",
    )
    p.add_argument(
        "--forecast_until",
        type=int,
        default=int(os.getenv("FORECAST_UNTIL", 2025)),
        help="Last year of forecast horizon  [env: FORECAST_UNTIL, default: 2025]",
    )
    args = p.parse_args()
    if not args.csv:
        p.error("--csv is required (or set CSV_PATH in .env)")
    return args


def main() -> None:
    args = parse_args()
    log.info("=" * 60)
    log.info("  EMBER ENERGY PIPELINE")
    log.info("  CSV            : %s", args.csv)
    log.info("  Steps          : %s", args.steps)
    log.info("  train_until    : %d", args.train_until)
    log.info("  forecast_until : %d", args.forecast_until)

    log.info("=" * 60)

    t_total = time.time()
    for name in STEP_ORDER:
        if name not in args.steps:
            log.info("── Skipping: %s", name)
            continue
        step = STEPS[name]
        run_step(name, step["script"], step["args"](args))

    elapsed = time.time() - t_total
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE in %.1fs", elapsed)
    log.info("  ./outputs/eda/            ← EDA figures + ember_filtered.csv")
    log.info("  ./outputs/preprocessing/  ← ember_model_ready.csv + feature_meta.json")
    log.info("  ./outputs/modeling/       ← test_benchmarking.csv + best_models.csv")
    log.info("  ./outputs/forecasting/    ← demand_forecast_2025_2030.csv + CI")
    log.info("  ./outputs/deeplearning/   ← dl_benchmarking.csv + dl_forecast_2025_2030.csv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
