"""
pipeline.py — Full Pipeline Orchestrator  (with smart caching)
Ember Energy Forecasting | IEEE Paper

Chains:
    Step 1: 01_eda.py           → ./outputs/eda/
    Step 2: 02_preprocessing.py → ./outputs/preprocessing/
    Step 3: 03_modeling.py      → ./outputs/modeling/
    Step 4: 04_forecasting.py   → ./outputs/forecasting/
    Step 5: 05_deeplearning.py  → ./outputs/deeplearning/

Usage:
    python pipeline.py --csv data/ember_yearly.csv
    python pipeline.py --csv data/ember_yearly.csv --steps eda preprocessing
    python pipeline.py --csv data/ember_yearly.csv --force          # skip cache
    python pipeline.py --invalidate eda                             # clear one step
    python pipeline.py --cache-status                               # show cache state
"""

import argparse
import logging
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

from pipeline_cache import StepCache

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_OUT = os.getenv("OUTPUTS_DIR", "outputs")

# ── Step registry ─────────────────────────────────────────────────────────────
# Each step declares:
#   script  — the .py file to run
#   args    — lambda(namespace) → CLI args list
#   deps    — input files/dirs that affect the cache hash
#   params  — keys from params.yaml that affect the cache hash
#   outputs — files that must exist for the cache to be considered valid

STEPS = {
    "eda": {
        "script": "src/01_eda.py",
        "args": lambda a: [
            "--csv",
            a.csv,
            "--output_dir",
            f"{_OUT}/eda",
        ],
        "deps": lambda a: [a.csv, "src/01_eda.py"],
        "params": ["data", "countries"],
        "outputs": [f"{_OUT}/eda/ember_filtered.csv", f"{_OUT}/eda/table01_eda_statistics.csv"],
    },
    "preprocessing": {
        "script": "src/02_preprocessing.py",
        "args": lambda a: [
            "--input_dir",
            f"{_OUT}/eda",
            "--output_dir",
            f"{_OUT}/preprocessing",
            "--train_until",
            str(a.train_until),
        ],
        "deps": lambda a: [f"{_OUT}/eda/ember_filtered.csv", "src/02_preprocessing.py"],
        "params": ["splits", "features"],
        "outputs": [
            f"{_OUT}/preprocessing/ember_model_ready.csv",
            f"{_OUT}/preprocessing/feature_meta.json",
        ],
    },
    "modeling": {
        "script": "src/03_modeling.py",
        "args": lambda a: [
            "--input_dir",
            f"{_OUT}/preprocessing",
            "--output_dir",
            f"{_OUT}/modeling",
        ],
        "deps": lambda a: [
            f"{_OUT}/preprocessing/ember_model_ready.csv",
            f"{_OUT}/preprocessing/feature_meta.json",
            "src/03_modeling.py",
        ],
        "params": ["splits", "modeling"],
        "outputs": [f"{_OUT}/modeling/best_models.csv", f"{_OUT}/modeling/best_hp.json"],
    },
    "forecasting": {
        "script": "src/04_forecasting.py",
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
        "deps": lambda a: [
            f"{_OUT}/modeling/best_models.csv",
            f"{_OUT}/modeling/best_hp.json",
            "src/04_forecasting.py",
        ],
        "params": ["splits"],
        "outputs": [
            f"{_OUT}/forecasting/demand_forecast_2025_2030.csv",
            f"{_OUT}/forecasting/demand_growth_summary.csv",
        ],
    },
    "deeplearning": {
        "script": "src/05_deeplearning.py",
        "args": lambda a: [
            "--pre_dir",
            f"{_OUT}/preprocessing",
            "--output_dir",
            f"{_OUT}/deeplearning",
        ],
        "deps": lambda a: [
            f"{_OUT}/preprocessing/ember_model_ready.csv",
            f"{_OUT}/preprocessing/feature_meta.json",
            "src/05_deeplearning.py",
        ],
        "params": ["splits", "deeplearning"],
        "outputs": [
            f"{_OUT}/deeplearning/dl_forecast_2025_2030.csv",
            f"{_OUT}/deeplearning/dl_best_models.csv",
        ],
    },
}

STEP_ORDER = ["eda", "preprocessing", "modeling", "forecasting", "deeplearning"]


# ── Runner ────────────────────────────────────────────────────────────────────


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


# ── Arg parser ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember full pipeline")
    p.add_argument(
        "--csv", default=os.getenv("CSV_PATH"), help="Path to Ember yearly CSV  [env: CSV_PATH]"
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_ORDER,
        default=STEP_ORDER,
        help="Steps to run (default: all)",
    )
    p.add_argument(
        "--train_until",
        type=int,
        default=int(os.getenv("TRAIN_UNTIL", 2016)),
        help="Last year of training window",
    )
    p.add_argument(
        "--forecast_until",
        type=int,
        default=int(os.getenv("FORECAST_UNTIL", 2030)),
        help="Last year of forecast horizon",
    )
    # ── Cache controls ────────────────────────────────────────────────────────
    p.add_argument(
        "--force", action="store_true", help="Ignore cache and re-run all selected steps"
    )
    p.add_argument(
        "--invalidate",
        metavar="STEP",
        nargs="?",
        const="ALL",
        help="Invalidate cache for STEP (or ALL) then exit",
    )
    p.add_argument("--cache-status", action="store_true", help="Print cache status and exit")

    args = p.parse_args()

    # Allow --invalidate / --cache-status without --csv
    if not args.invalidate and not args.cache_status and not args.csv:
        p.error("--csv is required (or set CSV_PATH in .env)")
    return args


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cache = StepCache()

    # ── Cache utility commands ────────────────────────────────────────────────
    if args.cache_status:
        status = cache.status()
        log.info("═" * 50)
        log.info("  PIPELINE CACHE STATUS")
        log.info("═" * 50)
        if not status:
            log.info("  (empty — no steps cached yet)")
        for step in STEP_ORDER:
            h = status.get(step, "—")
            log.info("  %-16s %s", step, h)
        log.info("═" * 50)
        return

    if args.invalidate:
        if args.invalidate == "ALL":
            cache.invalidate()
        else:
            if args.invalidate not in STEPS:
                log.error("Unknown step: %s", args.invalidate)
                sys.exit(1)
            cache.invalidate(args.invalidate)
        return

    # ── Normal pipeline run ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  EMBER ENERGY PIPELINE")
    log.info("  CSV            : %s", args.csv)
    log.info("  Steps          : %s", args.steps)
    log.info("  train_until    : %d", args.train_until)
    log.info("  forecast_until : %d", args.forecast_until)
    log.info("  Cache          : %s", "DISABLED (--force)" if args.force else "ENABLED")
    log.info("=" * 60)

    t_total = time.time()
    skipped = []
    ran = []

    for name in STEP_ORDER:
        if name not in args.steps:
            log.info("── Skipping: %s", name)
            continue

        step = STEPS[name]
        script = step["script"]
        deps = step["deps"](args)
        params = step["params"]
        outputs = step["outputs"]

        # ── Cache check ───────────────────────────────────────────────────────
        if not args.force and cache.is_cached(name, script, deps, params, outputs):
            log.info("── CACHED:   %s  (inputs unchanged — skipping)", name)
            skipped.append(name)
            continue

        # ── Run ───────────────────────────────────────────────────────────────
        run_step(name, script, step["args"](args))
        cache.mark_done(name, script, deps, params)
        ran.append(name)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE in %.1fs", elapsed)
    if ran:
        log.info("  Ran     : %s", ", ".join(ran))
    if skipped:
        log.info("  Cached  : %s", ", ".join(skipped))
    log.info("  outputs/eda/            ← EDA figures + ember_filtered.csv")
    log.info("  outputs/preprocessing/  ← ember_model_ready.csv + feature_meta.json")
    log.info("  outputs/modeling/       ← test_benchmarking.csv + best_models.csv")
    log.info("  outputs/forecasting/    ← demand_forecast_2025_2030.csv + CI")
    log.info("  outputs/deeplearning/   ← dl_benchmarking.csv + dl_forecast.csv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
