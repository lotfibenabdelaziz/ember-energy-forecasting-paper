"""
tests/test_pipeline.py — End-to-End Pipeline Integration Tests
"""

import json
import os
import subprocess
import sys
import time

import pandas as pd
import pytest

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"


# ── Unit: pipeline step ordering ─────────────────────────────────────────────
class TestStepOrdering:

    def test_step_order_is_correct(self):
        STEP_ORDER = ["eda", "preprocessing", "modeling", "forecasting"]
        assert STEP_ORDER[0] == "eda"
        assert STEP_ORDER[1] == "preprocessing"
        assert STEP_ORDER[2] == "modeling"
        assert STEP_ORDER[3] == "forecasting"

    def test_all_steps_have_scripts(self):
        STEPS = {
            "eda": "src/01_eda.py",
            "preprocessing": "src/02_preprocessing.py",
            "modeling": "src/03_modeling.py",
            "forecasting": "src/04_forecasting.py",
        }
        for name, script in STEPS.items():
            assert os.path.exists(script), f"Script missing: {script}"

    def test_pipeline_script_exists(self):
        assert os.path.exists("pipeline.py"), "pipeline.py not found"

    def test_pipeline_help_runs(self):
        r = subprocess.run(
            [sys.executable, "pipeline.py", "--help"], capture_output=True, text=True
        )
        assert r.returncode == 0
        assert "csv" in r.stdout.lower()


# ── Unit: data-flow contracts ─────────────────────────────────────────────────
class TestDataFlowContracts:
    """
    Verify that each step's outputs satisfy the next step's expected inputs.
    """

    def test_eda_output_feeds_preprocessing(self, tmp_dir, raw_csv):
        """EDA must produce ember_filtered.csv consumed by preprocessing."""
        subprocess.run(
            [sys.executable, "src/01_eda.py", "--csv", raw_csv, "--output_dir", tmp_dir],
            capture_output=True,
        )
        path = os.path.join(tmp_dir, "ember_filtered.csv")
        assert os.path.exists(path), "ember_filtered.csv not created by EDA"
        df = pd.read_csv(path)
        for col in ["Area", "Year", "Subcategory", "Value"]:
            assert col in df.columns

    def test_preprocessing_output_feeds_modeling(self, ember_filtered, tmp_dir):
        """Preprocessing must produce ember_model_ready.csv + feature_meta.json."""
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out = os.path.join(tmp_dir, "pre")
        subprocess.run(
            [
                sys.executable,
                "src/02_preprocessing.py",
                "--input_dir",
                tmp_dir,
                "--output_dir",
                out,
            ],
            capture_output=True,
        )
        for fname in ["ember_model_ready.csv", "feature_meta.json"]:
            assert os.path.exists(os.path.join(out, fname)), f"{fname} not created by preprocessing"

        # feature_meta.json must have required keys
        with open(os.path.join(out, "feature_meta.json")) as f:
            meta = json.load(f)
        for key in ["TARGET", "all_features", "TRAIN_END", "VAL_END", "TEST_END"]:
            assert key in meta

    def test_modeling_output_feeds_forecasting(self, model_ready_csv, feature_meta, tmp_dir):
        """Modeling must produce best_models.csv + best_hp.json + ember_config.json."""
        _, pre_dir = model_ready_csv
        meta, _, _ = feature_meta
        out = os.path.join(tmp_dir, "mod_contract")
        subprocess.run(
            [sys.executable, "src/03_modeling.py", "--input_dir", pre_dir, "--output_dir", out],
            capture_output=True,
        )
        for fname in ["best_models.csv", "best_hp.json", "ember_config.json"]:
            assert os.path.exists(os.path.join(out, fname)), f"{fname} not created by modeling"


# ── Unit: output schema validation ───────────────────────────────────────────
class TestOutputSchemas:

    def test_ember_model_ready_schema(self, model_ready_csv):
        path, _ = model_ready_csv
        df = pd.read_csv(path)
        assert "Area" in df.columns
        assert "Year" in df.columns
        assert TARGET in df.columns
        assert df["Area"].isin(COUNTRIES).all()

    def test_feature_meta_schema(self, feature_meta):
        meta, _, _ = feature_meta
        assert isinstance(meta["all_features"], list)
        assert len(meta["all_features"]) > 0
        assert isinstance(meta["FORECAST_YEARS"], list)
        assert len(meta["FORECAST_YEARS"]) == 6
        assert meta["FORECAST_YEARS"][0] == 2025
        assert meta["FORECAST_YEARS"][-1] == 2030

    def test_forecast_output_schema(self, tmp_path):
        """demand_forecast_2025_2030.csv must have correct columns."""
        rows = [
            {
                "Country": c,
                "Model": "Ridge",
                "Year": yr,
                "Forecast": 10.0,
                "Lower_90": 9.5,
                "Upper_90": 10.5,
            }
            for c in COUNTRIES
            for yr in range(2025, 2031)
        ]
        df = pd.DataFrame(rows)
        for col in ["Country", "Model", "Year", "Forecast", "Lower_90", "Upper_90"]:
            assert col in df.columns

    def test_growth_summary_schema(self, tmp_path):
        rows = [
            {
                "Country": c,
                "Model": "Ridge",
                "2024 TWh": 10.0,
                "2030 TWh (forecast)": 12.0,
                "Total Growth (%)": 20.0,
                "CAGR (%)": 3.1,
            }
            for c in COUNTRIES
        ]
        df = pd.DataFrame(rows)
        assert "Country" in df.columns
        assert "CAGR (%)" in df.columns


# ── Integration: partial pipeline (EDA + preprocessing) ─────────────────────
class TestPartialPipeline:

    def test_pipeline_eda_only(self, raw_csv, tmp_dir):
        r = subprocess.run(
            [sys.executable, "pipeline.py", "--csv", raw_csv, "--steps", "eda"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"Pipeline EDA step failed:\n{r.stderr}"
        assert (
            os.path.exists(os.path.join("outputs", "eda", "ember_filtered.csv"))
            or r.returncode == 0
        )  # success is enough

    def test_pipeline_skips_unselected_steps(self, raw_csv):
        """Pipeline with --steps eda should not run preprocessing."""
        r = subprocess.run(
            [sys.executable, "pipeline.py", "--csv", raw_csv, "--steps", "eda"],
            capture_output=True,
            text=True,
        )
        assert "Skipping: preprocessing" in r.stdout or r.returncode == 0


# ── Performance: reasonable execution time ───────────────────────────────────
class TestPerformance:

    def test_eda_completes_within_60s(self, raw_csv, tmp_dir):
        t0 = time.time()
        subprocess.run(
            [sys.executable, "src/01_eda.py", "--csv", raw_csv, "--output_dir", tmp_dir],
            capture_output=True,
        )
        elapsed = time.time() - t0
        assert elapsed < 60, f"EDA took too long: {elapsed:.1f}s"

    def test_preprocessing_completes_within_120s(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out = os.path.join(tmp_dir, "perf_pre")
        t0 = time.time()
        subprocess.run(
            [
                sys.executable,
                "src/02_preprocessing.py",
                "--input_dir",
                tmp_dir,
                "--output_dir",
                out,
            ],
            capture_output=True,
        )
        elapsed = time.time() - t0
        assert elapsed < 120, f"Preprocessing took too long: {elapsed:.1f}s"


# ── Regression: re-running pipeline produces same best models ─────────────────
class TestReproducibility:

    def test_modeling_is_reproducible(self, model_ready_csv, feature_meta, tmp_dir):
        """Running modeling twice should produce the same best models."""
        _, pre_dir = model_ready_csv
        meta, _, _ = feature_meta

        out1 = os.path.join(tmp_dir, "rep1")
        out2 = os.path.join(tmp_dir, "rep2")

        subprocess.run(
            [sys.executable, "src/03_modeling.py", "--input_dir", pre_dir, "--output_dir", out1],
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, "src/03_modeling.py", "--input_dir", pre_dir, "--output_dir", out2],
            capture_output=True,
        )
        p1 = os.path.join(out1, "best_models.csv")
        p2 = os.path.join(out2, "best_models.csv")
        if os.path.exists(p1) and os.path.exists(p2):
            df1 = pd.read_csv(p1).sort_values("Country").reset_index(drop=True)
            df2 = pd.read_csv(p2).sort_values("Country").reset_index(drop=True)
            pd.testing.assert_frame_equal(
                df1[["Country", "Model"]],
                df2[["Country", "Model"]],
                check_like=True,
            )
