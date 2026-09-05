"""
tests/test_pipeline_entrypoints.py — Smoke tests for the CLI entry points
============================================================================
Ember Energy | IEEE Paper

Prior to this file, step03_modeling.py, step04_forecasting.py, and
step05_deeplearning.py showed 0% coverage in coverage.xml — only the
library functions they import (src/modeling/*, src/forecasting/*) were
exercised. That means CI was never actually running the scripts that
produce the pipeline's reported numbers.

These tests run each entry point end-to-end via subprocess (same pattern
as TestPipelineCLI in test_pipeline.py) against the existing synthetic
`model_ready_df` / `feature_meta` fixtures from conftest.py, and assert the
documented output files actually get created. They're intentionally
smoke-level (does it run, does it produce the right files) rather than
numerically exhaustive — numerical correctness of each model is already
covered by test_modeling.py / test_forecasting.py / test_deeplearning.py.

step03/04 require statsmodels + xgboost; step05 requires torch — all three
are skipped automatically (not failed) if the dependency is missing, same
convention as test_deeplearning.py's existing torch skip.

TIMEOUTS: calibrated against a real `make run` on the full dataset, which
logged eda+preprocessing+modeling+forecasting = 275.2s combined (deep
learning alone took 753.8s separately, timed via `make run`'s per-stage
log). step03's synthetic-fixture timeout (600s) and step04's (300s) both
carry roughly 2x headroom over that real-data baseline for slower CI
machines — the original 180s was too tight and caused real timeouts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_STATSMODELS = _has_module("statsmodels")
HAS_XGBOOST = _has_module("xgboost")
HAS_TORCH = _has_module("torch")


@pytest.fixture()
def pre_dir(model_ready_df, feature_meta, tmp_path):
    """A tmp dir containing both ember_model_ready.csv and feature_meta.json —
    the two files every downstream step reads from --input_dir/--pre_dir."""
    _, _, meta_dir = feature_meta
    csv_path = os.path.join(meta_dir, "ember_model_ready.csv")
    model_ready_df.to_csv(csv_path, index=False)
    return meta_dir


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestStep03ModelingEntrypoint:
    def test_runs_and_produces_expected_outputs(self, pre_dir, tmp_path):
        out_dir = str(tmp_path / "modeling_out")
        r = subprocess.run(
            [
                sys.executable, "src/step03_modeling.py",
                "--input_dir", pre_dir,
                "--output_dir", out_dir,
            ],
            capture_output=True, text=True, timeout=600,
        )
        assert r.returncode == 0, r.stderr[-4000:]
        for fname in ["test_benchmarking.csv", "best_models.csv", "best_hp.json"]:
            assert os.path.exists(os.path.join(out_dir, fname)), f"missing {fname}"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestStep04ForecastingEntrypoint:
    def test_runs_and_produces_expected_outputs(self, pre_dir, tmp_path):
        model_dir = str(tmp_path / "modeling_out")
        r_model = subprocess.run(
            [sys.executable, "src/step03_modeling.py", "--input_dir", pre_dir, "--output_dir", model_dir],
            capture_output=True, text=True, timeout=600,
        )
        assert r_model.returncode == 0, r_model.stderr[-4000:]

        out_dir = str(tmp_path / "forecasting_out")
        r = subprocess.run(
            [
                sys.executable, "src/step04_forecasting.py",
                "--pre_dir", pre_dir,
                "--model_dir", model_dir,
                "--output_dir", out_dir,
                "--forecast_until", "2030",
            ],
            capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0, r.stderr[-4000:]
        assert os.path.exists(os.path.join(out_dir, "demand_forecast_2025_2030.csv"))


@pytest.mark.dl
@pytest.mark.slow
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestStep05DeepLearningEntrypoint:
    def test_quick_mode_runs_and_produces_expected_outputs(self, pre_dir, tmp_path):
        out_dir = str(tmp_path / "dl_out")
        r = subprocess.run(
            [
                sys.executable, "src/step05_deeplearning.py",
                "--pre_dir", pre_dir,
                "--output_dir", out_dir,
                "--quick",
                "--no-mlflow",
            ],
            capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0, r.stderr[-4000:]
        assert os.path.exists(os.path.join(out_dir, "dl_forecast_2025_2030.csv"))


@pytest.mark.unit
@pytest.mark.skipif(not (HAS_STATSMODELS and HAS_XGBOOST), reason="statsmodels/xgboost not installed")
def test_step03_04_expose_a_working_help_flag():
    """statsmodels is imported at module level by src/modeling/forecasters.py
    (ARIMA/Holt), so step03/04 require it even just to parse --help — this
    is NOT a dependency-light check, despite how it might look; it's gated
    the same as the full-run tests above."""
    for script in ["src/step03_modeling.py", "src/step04_forecasting.py"]:
        r = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, f"{script} --help failed:\n{r.stderr[-2000:]}"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_step05_expose_a_working_help_flag():
    r = subprocess.run(
        [sys.executable, "src/step05_deeplearning.py", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"step05 --help failed:\n{r.stderr[-2000:]}"
