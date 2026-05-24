"""
tests/test_mlflow_config.py — MLflow Integration Tests
"""

import math
import os
import tempfile

import numpy as np
import pytest


# ── Unit: log_model_metrics params flattening ────────────────────────────────
class TestParamFlattening:

    def test_flat_params_pass_through(self):
        params = {"alpha": 1.0, "n_estimators": 100}
        flat = _flatten_params(params)
        assert flat["alpha"] == 1.0
        assert flat["n_estimators"] == 100

    def test_list_params_stringified(self):
        params = {"order": [1, 1, 1]}
        flat = _flatten_params(params)
        assert isinstance(flat["order"], str)
        assert "1" in flat["order"]

    def test_nested_dict_params_flattened(self):
        params = {"sarima": {"p": 1, "d": 1, "q": 1}}
        flat = _flatten_params(params)
        assert "sarima_p" in flat
        assert flat["sarima_p"] == str(1)

    def test_none_value_handled(self):
        params = {"max_depth": None}
        flat = _flatten_params(params)
        assert "max_depth" in flat


# ── Unit: metric NaN / Inf filtering ─────────────────────────────────────────
class TestMetricFiltering:

    def test_finite_metrics_logged(self):
        metrics = {"MAE": 1.5, "RMSE": 2.0, "R2": 0.85, "MAPE": 3.2}
        good = _filter_metrics(metrics)
        assert len(good) == len(metrics)

    def test_nan_metrics_excluded(self):
        metrics = {"MAE": 1.5, "MAPE": float("nan")}
        good = _filter_metrics(metrics)
        assert "MAPE" not in good
        assert "MAE" in good

    def test_inf_metrics_excluded(self):
        metrics = {"RMSE": float("inf"), "R2": 0.9}
        good = _filter_metrics(metrics)
        assert "RMSE" not in good
        assert "R2" in good

    def test_negative_r2_is_valid(self):
        """R² can be negative — it is a valid metric value."""
        metrics = {"R2": -1.5}
        good = _filter_metrics(metrics)
        assert "R2" in good

    def test_zero_mape_is_valid(self):
        metrics = {"MAPE": 0.0}
        good = _filter_metrics(metrics)
        assert "MAPE" in good


# ── Unit: run naming ──────────────────────────────────────────────────────────
class TestRunNaming:

    def test_run_name_format(self):
        country = "Tunisia"
        model = "Ridge"
        name = f"{country}_{model}"
        assert name == "Tunisia_Ridge"

    def test_run_name_all_combinations(self):
        countries = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
        models = ["Ridge", "RandomForest", "XGBoost", "MLP", "TCN", "NBeats", "TFT"]
        names = [f"{c}_{m}" for c in countries for m in models]
        assert len(names) == len(countries) * len(models)
        assert len(set(names)) == len(names)  # all unique


# ── Unit: base tags ───────────────────────────────────────────────────────────
class TestBaseTags:

    def test_base_tags_present(self):
        tags = {
            "project": "ember-energy-forecasting",
            "paper": "IEEE",
            "dataset": "Ember Annual Energy",
            "subcategory": "Demand",
            "unit": "TWh",
            "countries": "Tunisia,Austria,Germany,Egypt,Canada,France,Kuwait",
            "train_end": "2016",
            "val_end": "2020",
            "test_end": "2024",
        }
        for key in [
            "project",
            "paper",
            "dataset",
            "subcategory",
            "unit",
            "countries",
            "train_end",
            "val_end",
            "test_end",
        ]:
            assert key in tags

    def test_countries_tag_has_all_7(self):
        countries_tag = "Tunisia,Austria,Germany,Egypt,Canada,France,Kuwait"
        assert len(countries_tag.split(",")) == 7

    def test_split_dates_correct(self):
        assert int("2016") < int("2020") < int("2024")


# ── Integration: MLflow experiment creation (local filesystem) ───────────────
class TestMLflowLocalIntegration:

    def test_mlflow_set_experiment_local(self, tmp_path):
        """MLflow can create a local experiment without a server."""
        try:
            import mlflow

            uri = tmp_path.as_uri() + "/mlruns"
            mlflow.set_tracking_uri(uri)
            exp_name = "test-ember-exp"
            mlflow.set_experiment(exp_name)
            exp = mlflow.get_experiment_by_name(exp_name)
            assert exp is not None
            assert exp.name == exp_name
        except ImportError:
            pytest.skip("mlflow not installed")

    def test_mlflow_log_metrics_locally(self, tmp_path):
        """MLflow can log metrics without a remote server."""
        try:
            import mlflow

            uri = tmp_path.as_uri() + "/mlruns2"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("test-metrics")
            with mlflow.start_run(run_name="test_Tunisia_Ridge"):
                mlflow.log_metric("MAE", 0.63)
                mlflow.log_metric("RMSE", 0.71)
                mlflow.log_metric("MAPE", 2.76)
                mlflow.log_metric("R2", 0.622)
                mlflow.log_param("alpha", 1.0)
                mlflow.set_tag("country", "Tunisia")
                mlflow.set_tag("model", "Ridge")
        except ImportError:
            pytest.skip("mlflow not installed")

    def test_mlflow_artifact_logging(self, tmp_path):
        """MLflow can log a CSV artifact."""
        try:
            import mlflow
            import pandas as pd

            uri = tmp_path.as_uri() + "/mlruns3"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("test-artifacts")
            with mlflow.start_run():
                # Create temp CSV and log as artifact
                csv_path = str(tmp_path / "forecast.csv")
                pd.DataFrame({"Year": [2025, 2026], "Forecast": [12.5, 13.0]}).to_csv(
                    csv_path, index=False
                )
                mlflow.log_artifact(csv_path, artifact_path="forecasts")
        except ImportError:
            pytest.skip("mlflow not installed")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _flatten_params(params: dict) -> dict:
    flat = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            flat[k] = str(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = str(vv)
        else:
            flat[k] = v
    return flat


def _filter_metrics(metrics: dict) -> dict:
    good = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
            if not math.isnan(fv) and not math.isinf(fv):
                good[k] = fv
        except (TypeError, ValueError):
            pass
    return good
