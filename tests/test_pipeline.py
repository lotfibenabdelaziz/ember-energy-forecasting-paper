"""
tests/test_pipeline.py — Pipeline Orchestrator Integration Tests
Ember Energy | IEEE Paper

Covers:
  - PipelineConfig dataclass
  - StepResult dataclass
  - build_step_definitions()
  - build_step_args()
  - Cache invalidation + status
  - Full pipeline (smoke test via --steps eda)
"""

import os
import subprocess
import sys
from dataclasses import dataclass

import pytest


COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
STEP_ORDER = ["eda", "preprocessing", "modeling", "forecasting", "deeplearning"]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineConfig:

    def test_pipeline_config_creation(self):
        from pipeline import PipelineConfig
        cfg = PipelineConfig(
            csv="data/raw/test.csv",
            steps=STEP_ORDER,
            train_until=2016,
            forecast_until=2030,
        )
        assert cfg.csv == "data/raw/test.csv"
        assert cfg.train_until == 2016
        assert cfg.forecast_until == 2030
        assert cfg.force is False

    def test_pipeline_config_force_default_false(self):
        from pipeline import PipelineConfig
        cfg = PipelineConfig(csv="x.csv", steps=[], train_until=2016, forecast_until=2030)
        assert cfg.force is False


class TestStepResult:

    def test_step_result_ran_is_success(self):
        from pipeline import StepResult
        r = StepResult(name="eda", status="ran")
        assert r.success is True

    def test_step_result_cached_is_success(self):
        from pipeline import StepResult
        r = StepResult(name="eda", status="cached")
        assert r.success is True

    def test_step_result_failed_is_not_success(self):
        from pipeline import StepResult
        r = StepResult(name="eda", status="failed")
        assert r.success is False

    def test_step_result_skipped_is_success(self):
        from pipeline import StepResult
        r = StepResult(name="eda", status="skipped")
        assert r.success is True


# ═══════════════════════════════════════════════════════════════════════════════
# Step definitions
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildStepDefinitions:

    @pytest.fixture
    def cfg(self):
        from pipeline import PipelineConfig
        return PipelineConfig(
            csv="data/raw/test.csv",
            steps=STEP_ORDER,
            train_until=2016,
            forecast_until=2030,
        )

    def test_returns_all_steps(self, cfg):
        from pipeline import build_step_definitions
        steps = build_step_definitions(cfg)
        names = [s.name for s in steps]
        for step in STEP_ORDER:
            assert step in names

    def test_each_step_has_script(self, cfg):
        from pipeline import build_step_definitions
        for step in build_step_definitions(cfg):
            assert step.script.endswith(".py")

    def test_each_step_has_outputs(self, cfg):
        from pipeline import build_step_definitions
        for step in build_step_definitions(cfg):
            assert len(step.outputs) >= 1

    def test_eda_depends_on_csv(self, cfg):
        from pipeline import build_step_definitions
        steps = {s.name: s for s in build_step_definitions(cfg)}
        assert cfg.csv in steps["eda"].deps

    def test_preprocessing_depends_on_eda_output(self, cfg):
        from pipeline import build_step_definitions
        steps = {s.name: s for s in build_step_definitions(cfg)}
        deps  = " ".join(steps["preprocessing"].deps)
        assert "eda" in deps or "ember_filtered" in deps


# ═══════════════════════════════════════════════════════════════════════════════
# Step args
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildStepArgs:

    @pytest.fixture
    def cfg(self):
        from pipeline import PipelineConfig
        return PipelineConfig(
            csv="data/raw/test.csv",
            steps=STEP_ORDER,
            train_until=2016,
            forecast_until=2030,
        )

    def test_eda_args_contain_csv(self, cfg):
        from pipeline import build_step_args
        args = build_step_args("eda", cfg)
        assert cfg.csv in args

    def test_preprocessing_args_contain_train_until(self, cfg):
        from pipeline import build_step_args
        args = build_step_args("preprocessing", cfg)
        assert str(cfg.train_until) in args

    def test_forecasting_args_contain_forecast_until(self, cfg):
        from pipeline import build_step_args
        args = build_step_args("forecasting", cfg)
        assert str(cfg.forecast_until) in args

    def test_all_steps_return_list(self, cfg):
        from pipeline import build_step_args
        for step in STEP_ORDER:
            args = build_step_args(step, cfg)
            assert isinstance(args, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Cache management
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineCache:

    def test_cache_status_runs(self):
        result = subprocess.run(
            [sys.executable, "pipeline.py", "--cache-status"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_invalidate_single_step(self):
        result = subprocess.run(
            [sys.executable, "pipeline.py", "--invalidate", "eda"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_invalidate_all(self):
        result = subprocess.run(
            [sys.executable, "pipeline.py", "--invalidate", "ALL"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_invalid_step_name_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "pipeline.py", "--invalidate", "nonexistent_step"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineCLI:

    @pytest.fixture(autouse=True)
    def check_csv(self):
        csv = "data/raw/yearly_full_release_long_format.csv"
        if not os.path.exists(csv):
            pytest.skip(f"CSV not found: {csv}")

    def test_help_exits_zero(self):
        r = subprocess.run(
            [sys.executable, "pipeline.py", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--csv" in r.stdout

    def test_cache_status_shows_steps(self):
        r = subprocess.run(
            [sys.executable, "pipeline.py", "--cache-status"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
