"""
pipeline.py — Full Pipeline Orchestrator
=========================================
Ember Energy Forecasting | IEEE Paper

Chains:
    Step 1: 01_eda.py           → outputs/eda/
    Step 2: 02_preprocessing.py → outputs/preprocessing/
    Step 3: 03_modeling.py      → outputs/modeling/
    Step 4: 04_forecasting.py   → outputs/forecasting/
    Step 5: 05_deeplearning.py  → outputs/deeplearning/

Usage:
    python pipeline.py --csv data/ember_yearly.csv
    python pipeline.py --csv data/ember_yearly.csv --force
    python pipeline.py --invalidate eda
    python pipeline.py --cache-status
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

from pipeline_cache import StepCache, StepDefinition

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_OUT = os.getenv("OUTPUTS_DIR", "outputs")


# ── Step result dataclass ─────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""

    name: str
    status: str  # "ran" | "cached" | "skipped" | "failed"
    elapsed: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return self.status in {"ran", "cached", "skipped"}


# ── Pipeline config dataclass ─────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """All runtime configuration for a pipeline run."""

    csv: str
    steps: list[str]
    train_until: int
    forecast_until: int
    force: bool = False


# ── Step registry ─────────────────────────────────────────────────────────────


def build_step_definitions(cfg: PipelineConfig) -> list[StepDefinition]:
    """
    Build StepDefinition objects from the pipeline config.
    Each step declares its script, inputs, params, and expected outputs.
    """
    return [
        StepDefinition(
            name="eda",
            script="src/step01_eda.py",
            deps=[cfg.csv, "src/step01_eda.py"],
            params=["data", "countries"],
            outputs=[
                f"{_OUT}/eda/ember_filtered.csv",
                f"{_OUT}/eda/table01_eda_statistics.csv",
            ],
        ),
        StepDefinition(
            name="preprocessing",
            script="src/step02_preprocessing.py",
            deps=[
                f"{_OUT}/eda/ember_filtered.csv",
                "src/step02_preprocessing.py",
            ],
            params=["splits", "features"],
            outputs=[
                f"{_OUT}/preprocessing/ember_model_ready.csv",
                f"{_OUT}/preprocessing/feature_meta.json",
            ],
        ),
        StepDefinition(
            name="modeling",
            script="src/step03_modeling.py",
            deps=[
                f"{_OUT}/preprocessing/ember_model_ready.csv",
                f"{_OUT}/preprocessing/feature_meta.json",
                "src/03_modeling.py",
            ],
            params=["splits", "modeling"],
            outputs=[
                f"{_OUT}/modeling/best_models.csv",
                f"{_OUT}/modeling/best_hp.json",
            ],
        ),
        StepDefinition(
            name="forecasting",
            script="src/step04_forecasting.py",
            deps=[
                f"{_OUT}/modeling/best_models.csv",
                f"{_OUT}/modeling/best_hp.json",
                "src/step04_forecasting.py",
            ],
            params=["splits"],
            outputs=[
                f"{_OUT}/forecasting/demand_forecast_2025_2030.csv",
                f"{_OUT}/forecasting/demand_growth_summary.csv",
            ],
        ),
        StepDefinition(
            name="deeplearning",
            script="src/step05_deeplearning.py",
            deps=[
                f"{_OUT}/preprocessing/ember_model_ready.csv",
                f"{_OUT}/preprocessing/feature_meta.json",
                "src/step05_deeplearning.py",
            ],
            params=["splits", "deeplearning"],
            outputs=[
                f"{_OUT}/deeplearning/dl_forecast_2025_2030.csv",
                f"{_OUT}/deeplearning/dl_best_models.csv",
            ],
        ),
    ]


STEP_ORDER = ["eda", "preprocessing", "modeling", "forecasting", "deeplearning"]

# ── CLI args for each step ────────────────────────────────────────────────────


def build_step_args(name: str, cfg: PipelineConfig) -> list[str]:
    """Build the CLI argument list for a given step."""
    args_map: dict[str, list[str]] = {
        "eda": [
            "--csv",
            cfg.csv,
            "--output_dir",
            f"{_OUT}/eda",
        ],
        "preprocessing": [
            "--input_dir",
            f"{_OUT}/eda",
            "--output_dir",
            f"{_OUT}/preprocessing",
            "--train_until",
            str(cfg.train_until),
        ],
        "modeling": [
            "--input_dir",
            f"{_OUT}/preprocessing",
            "--output_dir",
            f"{_OUT}/modeling",
        ],
        "forecasting": [
            "--pre_dir",
            f"{_OUT}/preprocessing",
            "--model_dir",
            f"{_OUT}/modeling",
            "--output_dir",
            f"{_OUT}/forecasting",
            "--forecast_until",
            str(cfg.forecast_until),
        ],
        "deeplearning": [
            "--pre_dir",
            f"{_OUT}/preprocessing",
            "--output_dir",
            f"{_OUT}/deeplearning",
        ],
    }
    return args_map[name]


# ── Runner ────────────────────────────────────────────────────────────────────


def run_step(name: str, script: str, extra_args: list[str]) -> StepResult:
    """Execute a single pipeline step as a subprocess."""
    cmd = [sys.executable, script, *extra_args]
    log.info("╔══ START: %s", name.upper())
    log.info("   cmd: %s", " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error("╚══ FAILED: %s (exit=%d) in %.1fs", name, result.returncode, elapsed)
        return StepResult(
            name=name, status="failed", elapsed=elapsed, error=f"exit code {result.returncode}"
        )

    log.info("╚══ DONE: %s in %.1fs", name, elapsed)
    return StepResult(name=name, status="ran", elapsed=elapsed)


# ── Arg parser ────────────────────────────────────────────────────────────────


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
        help="Steps to run (default: all)",
    )
    p.add_argument(
        "--train_until",
        type=int,
        default=int(os.getenv("TRAIN_UNTIL", "2016")),
        help="Last year of training window",
    )
    p.add_argument(
        "--forecast_until",
        type=int,
        default=int(os.getenv("FORECAST_UNTIL", "2030")),
        help="Last year of forecast horizon",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-run all selected steps",
    )
    p.add_argument(
        "--invalidate",
        metavar="STEP",
        nargs="?",
        const="ALL",
        help="Invalidate cache for STEP (or ALL) then exit",
    )
    p.add_argument(
        "--cache-status",
        action="store_true",
        help="Print cache status and exit",
    )

    args = p.parse_args()

    # Allow utility commands without --csv
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
            log.info("  %-16s %s", step, status.get(step, "—"))
        log.info("═" * 50)
        return

    if args.invalidate:
        target = None if args.invalidate == "ALL" else args.invalidate
        if target and target not in STEP_ORDER:
            log.error("Unknown step: %s. Valid: %s", target, STEP_ORDER)
            sys.exit(1)
        cache.invalidate(target)
        return

    # ── Build config and step definitions ────────────────────────────────────
    cfg = PipelineConfig(
        csv=args.csv,
        steps=args.steps,
        train_until=args.train_until,
        forecast_until=args.forecast_until,
        force=args.force,
    )
    step_defs = {s.name: s for s in build_step_definitions(cfg)}

    # ── Run pipeline ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  EMBER ENERGY PIPELINE")
    log.info("  CSV            : %s", cfg.csv)
    log.info("  Steps          : %s", cfg.steps)
    log.info("  train_until    : %d", cfg.train_until)
    log.info("  forecast_until : %d", cfg.forecast_until)
    log.info("  Cache          : %s", "DISABLED (--force)" if cfg.force else "ENABLED")
    log.info("=" * 60)

    t_total = time.time()
    results: list[StepResult] = []

    for name in STEP_ORDER:
        if name not in cfg.steps:
            log.info("── Skipping: %s", name)
            results.append(StepResult(name=name, status="skipped"))
            continue

        step_def = step_defs[name]

        # Cache check
        if not cfg.force and cache.is_cached(step_def):
            log.info("── CACHED:   %s  (inputs unchanged — skipping)", name)
            results.append(StepResult(name=name, status="cached"))
            continue

        # Run
        result = run_step(name, step_def.script, build_step_args(name, cfg))
        results.append(result)

        if not result.success:
            log.error("Pipeline aborted at step: %s", name)
            sys.exit(1)

        cache.mark_done(step_def)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    ran = [r.name for r in results if r.status == "ran"]
    cached = [r.name for r in results if r.status == "cached"]
    skipped = [r.name for r in results if r.status == "skipped"]

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE in %.1fs", elapsed)
    if ran:
        log.info("  Ran     : %s", ", ".join(ran))
    if cached:
        log.info("  Cached  : %s", ", ".join(cached))
    if skipped:
        log.info("  Skipped : %s", ", ".join(skipped))
    log.info("  outputs/eda/            ← EDA + ember_filtered.csv")
    log.info("  outputs/preprocessing/  ← ember_model_ready.csv")
    log.info("  outputs/modeling/       ← best_models.csv")
    log.info("  outputs/forecasting/    ← demand_forecast_2025_2030.csv")
    log.info("  outputs/deeplearning/   ← dl_forecast_2025_2030.csv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
