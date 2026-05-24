"""
mlflow_config.py — MLflow Experiment Tracking
Ember Energy Forecasting | IEEE Paper

Experiment : ember-demand-forecasting
Run naming : {Country}_{Model}
Splits     : Train 2000-2016 | Val 2017-2020 | Test 2021-2024 | Forecast 2025-2030

Usage:
    from mlflow_config import init_mlflow, log_model_metrics
    from mlflow_config import log_dl_model_metrics, log_forecast_artifact, log_figure
"""

import math
import os
import tempfile
from typing import Any

import mlflow
import mlflow.sklearn

# ── Tracking config ───────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "ember-demand-forecasting"
ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns/artifacts")

# ── Data splits (canonical — must match conftest.py + params.yaml) ────────────
TRAIN_END = 2016  # last training year  (2000–2016, 17 years)
VAL_END = 2020  # last validation year(2017–2020,  4 years)
TEST_END = 2024  # last test year      (2021–2024,  4 years)
FORECAST_START = 2025  # first forecast year
FORECAST_END = 2030  # last forecast year  (2025–2030,  6 years)

# ── Countries ─────────────────────────────────────────────────────────────────
COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]

# ── Project tags — applied to every run ───────────────────────────────────────
BASE_TAGS: dict[str, str] = {
    "project": "ember-energy-forecasting",
    "paper": "IEEE",
    "dataset": "Ember Annual Energy",
    "subcategory": "Demand",
    "unit": "TWh",
    "countries": ",".join(COUNTRIES),
    "train_years": f"2000-{TRAIN_END}",
    "val_years": f"{TRAIN_END + 1}-{VAL_END}",
    "test_years": f"{VAL_END + 1}-{TEST_END}",
    "forecast_horizon": f"{FORECAST_START}-{FORECAST_END}",
    "train_end": str(TRAIN_END),
    "val_end": str(VAL_END),
    "test_end": str(TEST_END),
}


# ── Init ──────────────────────────────────────────────────────────────────────


def init_mlflow() -> mlflow.MlflowClient:
    """
    Set tracking URI and create experiment if it doesn't exist yet.
    Returns an MlflowClient for programmatic access.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_LOCATION,
            tags={
                "description": (
                    "Per-country electricity demand forecasting "
                    "using the Ember annual energy dataset."
                ),
                "splits": (
                    f"Train 2000-{TRAIN_END} | "
                    f"Val {TRAIN_END+1}-{VAL_END} | "
                    f"Test {VAL_END+1}-{TEST_END} | "
                    f"Forecast {FORECAST_START}-{FORECAST_END}"
                ),
            },
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# ── Param flattening ──────────────────────────────────────────────────────────


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested dicts / lists so MLflow can log them.
    MLflow only accepts scalar param values.
    """
    flat: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            flat[k] = str(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = str(vv)
        else:
            flat[k] = v
    return flat


# ── Metric filtering ──────────────────────────────────────────────────────────


def _filter_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Drop NaN / Inf values — MLflow rejects them."""
    good: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
            if not math.isnan(fv) and not math.isinf(fv):
                good[k] = fv
        except (TypeError, ValueError):
            pass
    return good


# ── Core logging function ─────────────────────────────────────────────────────


def log_model_metrics(
    country: str,
    model_name: str,
    phase: str,  # "benchmark" | "forecast" | "deeplearning"
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Log one walk-forward result as a named MLflow run.

    Parameters
    ----------
    country    : e.g. "Tunisia"
    model_name : e.g. "Ridge", "RandomForest", "XGBoost", "MLP", "TCN"
    phase      : "benchmark" | "forecast" | "deeplearning"
    metrics    : {metric_name: float} — NaN/Inf values are silently dropped
    params     : best hyperparameter dict — nested dicts/lists are flattened
    tags       : additional run tags

    Returns
    -------
    run_id : str
    """
    run_name = f"{country}_{model_name}"

    with mlflow.start_run(run_name=run_name) as run:
        # ── Tags ──────────────────────────────────────────────────────────────
        all_tags: dict[str, str] = {
            **BASE_TAGS,
            "country": country,
            "model": model_name,
            "phase": phase,
        }
        if tags:
            all_tags.update(tags)
        mlflow.set_tags(all_tags)

        # ── Params ────────────────────────────────────────────────────────────
        if params:
            mlflow.log_params(_flatten_params(params))

        # ── Metrics ───────────────────────────────────────────────────────────
        clean = _filter_metrics(metrics)
        for k, v in clean.items():
            mlflow.log_metric(k, v)

        return run.info.run_id


# ── Deep learning variant ─────────────────────────────────────────────────────


def log_dl_model_metrics(
    country: str,
    model_name: str,  # "MLP" | "TCN" | "N-BEATS" | "TFT"
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    seq_len: int = 5,
    epochs: int | None = None,
) -> str:
    """
    Log deep learning walk-forward metrics as a named MLflow run.
    Adds PyTorch-specific tags (framework, seq_len, epochs).

    Returns
    -------
    run_id : str
    """
    dl_tags: dict[str, str] = {
        "framework": "pytorch",
        "seq_len": str(seq_len),
    }
    if epochs is not None:
        dl_tags["epochs"] = str(epochs)

    return log_model_metrics(
        country=country,
        model_name=model_name,
        phase="deeplearning",
        metrics=metrics,
        params=params,
        tags=dl_tags,
    )


# ── Artifact helpers ──────────────────────────────────────────────────────────


def log_forecast_artifact(
    country: str,
    fc_df: Any,  # pd.DataFrame
    model_name: str,
) -> None:
    """
    Attach a forecast DataFrame as a CSV artifact in the active MLflow run.
    Must be called inside an active run context.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = f"forecast_{country.lower()}_{model_name.lower()}.csv"
        path = os.path.join(tmp, fname)
        fc_df.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path="forecasts")


def log_metrics_artifact(
    label: str,
    df: Any,  # pd.DataFrame
) -> None:
    """
    Attach a metrics DataFrame as a CSV artifact in the active MLflow run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{label}.csv")
        df.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path="metrics")


def log_figure(
    fig_path: str,
    artifact_subdir: str = "figures",
) -> None:
    """
    Log a saved figure (PDF / PNG / SVG) as an MLflow artifact.
    Silently skips if the file doesn't exist.
    """
    if os.path.exists(fig_path):
        mlflow.log_artifact(fig_path, artifact_path=artifact_subdir)
    else:
        import logging

        logging.getLogger(__name__).warning("Figure not found, skipping MLflow log: %s", fig_path)


def log_params_from_yaml(param_keys: list[str]) -> None:
    """
    Log specific top-level keys from params.yaml into the active MLflow run.
    Useful for reproducibility tracking.
    """
    import yaml

    params_path = "dvc_params.yaml"
    if not os.path.exists(params_path):
        return
    with open(params_path) as f:
        all_params = yaml.safe_load(f) or {}
    selected = {k: all_params.get(k) for k in param_keys if k in all_params}
    if selected:
        mlflow.log_params(_flatten_params(selected))
