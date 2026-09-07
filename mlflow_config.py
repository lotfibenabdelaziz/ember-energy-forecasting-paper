"""
mlflow_config.py — MLflow Experiment Tracking + Model Registry
================================================================
Ember Energy Forecasting | IEEE Paper

Responsibilities:
  - setup_experiment()          : create/get MLflow experiment
  - log_classical_run()         : log one classical model run
  - log_dl_model_metrics()      : log all DL benchmarking results
  - log_params_from_yaml()      : log params.yaml to active run
  - register_best_model()       : register best model per country to Registry
  - promote_model()             : Staging → Production transition
  - get_production_model()      : load Production model for inference
  - list_registered_models()    : show registry status
  - compare_and_promote()       : auto-promote if new model beats Production MAPE

Cross-platform fix:
  - Windows file:// URI uses Path.as_uri() not f"file://{path}"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)

# ── MLflow import (graceful degradation) ─────────────────────────────────────
try:
    import mlflow
    from mlflow.exceptions import MlflowException
    import mlflow.pytorch
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning("MLflow not installed — tracking disabled")

EXPERIMENT_NAME = "ember-demand-forecasting"
REGISTRY_NAME = "ember-demand-model"  # base name; suffixed with country

# ── Timeout — fail fast when server unreachable ───────────────────────────────
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")


# ═══════════════════════════════════════════════════════════════════════════════
# Tracking URI
# ═══════════════════════════════════════════════════════════════════════════════


def _local_uri(path: Path | None = None) -> str:
    """
    Cross-platform local file URI.
    Windows requires file:///C:/... (triple slash).
    Path.as_uri() handles this correctly on all platforms.
    """
    p = path or Path("./mlruns").resolve()
    return p.as_uri()


def get_tracking_uri() -> str:
    """Return MLFLOW_TRACKING_URI from env or fall back to local ./mlruns."""
    return os.getenv("MLFLOW_TRACKING_URI", _local_uri())


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment setup
# ═══════════════════════════════════════════════════════════════════════════════


def setup_experiment(name: str = EXPERIMENT_NAME) -> str | None:
    """
    Create or get an MLflow experiment.

    Returns experiment_id or None if MLflow unavailable/server unreachable.
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment(name)
        exp = mlflow.get_experiment_by_name(name)
        exp_id = exp.experiment_id if exp else None
        log.info("MLflow experiment: '%s' (id=%s)", name, exp_id)
        return exp_id
    except Exception as e:
        log.warning("MLflow setup failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Classical model logging
# ═══════════════════════════════════════════════════════════════════════════════


def log_classical_run(
    country: str,
    model_name: str,
    params: dict,
    metrics: dict,
    experiment_name: str = EXPERIMENT_NAME,
    tags: dict | None = None,
) -> str | None:
    """
    Log one classical model walk-forward result to MLflow.

    Returns run_id or None on failure.
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment(experiment_name)
        run_name = f"{country}_{model_name}_classical"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("country", country)
            mlflow.set_tag("model_type", "classical")
            mlflow.set_tag("model_name", model_name)
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(
                {k: round(float(v), 4) for k, v in metrics.items() if v is not None and v == v}
            )  # skip NaN
            return str(run.info.run_id)
    except Exception as e:
        log.warning("MLflow classical logging failed (%s %s): %s", country, model_name, e)
        return None


def log_all_classical_to_mlflow(
    test_metrics: pd.DataFrame,
    best_hp: dict,
    experiment_name: str = EXPERIMENT_NAME,
) -> None:
    """Log all classical benchmarking results — one run per (country, model)."""
    if not MLFLOW_AVAILABLE:
        return
    for _, row in test_metrics.iterrows():
        model_name = row["Model"]
        params = best_hp.get(model_name, {})
        metrics = {k: row[k] for k in ["MAE", "RMSE", "MAPE"] if k in row and row[k] == row[k]}
        log_classical_run(
            country=row["Country"],
            model_name=model_name,
            params=params,
            metrics=metrics,
            experiment_name=experiment_name,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Deep learning metrics logging
# ═══════════════════════════════════════════════════════════════════════════════


def log_dl_model_metrics(
    metrics_df: pd.DataFrame,
    params: dict,
    experiment_name: str = EXPERIMENT_NAME,
) -> None:
    """Log all DL benchmarking results — one run per (country, model)."""
    if not MLFLOW_AVAILABLE or metrics_df.empty:
        return
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment(experiment_name)
        for _, row in metrics_df.iterrows():
            run_name = f"{row['Country']}_{row['Model']}_DL"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("country", row["Country"])
                mlflow.set_tag("model_type", "deep_learning")
                mlflow.set_tag("model_name", row["Model"])
                mlflow.log_params({k: str(v) for k, v in params.items()})
                for metric in ["MAE", "RMSE", "MAPE", "SMAPE"]:
                    if metric in row and row[metric] == row[metric]:
                        mlflow.log_metric(metric, round(float(row[metric]), 4))
    except Exception as e:
        log.warning("MLflow DL logging failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# params.yaml logging
# ═══════════════════════════════════════════════════════════════════════════════


def log_params_from_yaml(
    yaml_path: str = "params.yaml",
    prefix: str = "",
) -> None:
    """Log all params from params.yaml to the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    if not Path(yaml_path).exists():
        log.warning("params.yaml not found: %s", yaml_path)
        return
    try:
        with open(yaml_path) as f:
            params = yaml.safe_load(f) or {}

        flat: dict[str, str] = {}

        def _flatten(d: dict, parent: str = "") -> None:
            for k, v in d.items():
                key = f"{parent}.{k}" if parent else k
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    flat[key] = str(v)

        _flatten(params)
        if prefix:
            flat = {f"{prefix}.{k}": v for k, v in flat.items()}

        mlflow.log_params(flat)
        log.info("Logged %d params from %s", len(flat), yaml_path)
    except Exception as e:
        log.warning("Failed to log params from YAML: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


def _registry_name(country: str, model_type: str = "classical") -> str:
    """
    Naming convention for registry:
        ember-demand-model-Tunisia-classical
        ember-demand-model-Tunisia-dl
    """
    return f"{REGISTRY_NAME}-{country}-{model_type}"


def register_best_model(
    run_id: str,
    country: str,
    model_name: str,
    model_type: str = "classical",
    artifact_path: str = "model",
) -> str | None:
    """
    Register a trained model in the MLflow Model Registry.

    Stages: None → Staging → Production

    Parameters
    ----------
    run_id        : MLflow run ID containing the logged model artifact
    country       : e.g. "Tunisia"
    model_name    : e.g. "Ridge" or "TCN"
    model_type    : "classical" or "dl"
    artifact_path : artifact subdirectory name in the run

    Returns
    -------
    model_version string or None on failure
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        client = MlflowClient()
        reg_name = _registry_name(country, model_type)

        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(
                name=reg_name,
                description=(
                    f"Electricity demand forecast model for {country}. "
                    f"Type: {model_type}, Algorithm: {model_name}."
                ),
                tags={
                    "country": country,
                    "model_name": model_name,
                    "model_type": model_type,
                },
            )
            log.info("Created registered model: %s", reg_name)
        except MlflowException:
            log.info("Registered model already exists: %s", reg_name)

        # Register new version
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = client.create_model_version(
            name=reg_name,
            source=model_uri,
            run_id=run_id,
            description=f"{model_name} trained for {country} — {model_type}",
            tags={"algorithm": model_name, "country": country},
        )
        log.info("Registered: %s v%s (run_id=%s)", reg_name, mv.version, run_id)
        return mv.version

    except Exception as e:
        log.warning("Model registration failed (%s %s): %s", country, model_name, e)
        return None


def promote_model(
    country: str,
    version: str,
    stage: str = "Staging",
    model_type: str = "classical",
    archive_existing: bool = True,
) -> bool:
    """
    Promote a model version to a new stage.

    stage options: "Staging" | "Production" | "Archived"

    If archive_existing=True, all existing versions in the target stage
    are archived before promotion.
    """
    if not MLFLOW_AVAILABLE:
        return False
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        client = MlflowClient()
        reg_name = _registry_name(country, model_type)

        if archive_existing:
            existing = client.search_model_versions(
                f"name='{reg_name}' and current_stage='{stage}'"
            )
            for mv in existing:
                client.transition_model_version_stage(
                    name=reg_name,
                    version=mv.version,
                    stage="Archived",
                )
                log.info("Archived: %s v%s", reg_name, mv.version)

        client.transition_model_version_stage(
            name=reg_name,
            version=version,
            stage=stage,
        )
        log.info("Promoted: %s v%s → %s", reg_name, version, stage)
        return True

    except Exception as e:
        log.warning("Promotion failed (%s v%s → %s): %s", country, version, stage, e)
        return False


def compare_and_promote(
    country: str,
    new_run_id: str,
    new_mape: float,
    new_version: str,
    model_type: str = "classical",
    threshold: float = 0.0,
) -> bool:
    """
    Auto-promote to Production if new model beats current Production MAPE.

    threshold: minimum improvement required (default 0 = any improvement)

    Flow:
      1. Get current Production model MAPE from registry tags
      2. If new MAPE < current MAPE - threshold → promote to Production
      3. Otherwise → keep in Staging

    Returns True if promoted to Production.
    """
    if not MLFLOW_AVAILABLE:
        return False
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        client = MlflowClient()
        reg_name = _registry_name(country, model_type)

        prod_versions = client.search_model_versions(
            f"name='{reg_name}' and current_stage='Production'"
        )
        if not prod_versions:
            # No production model yet — promote directly
            log.info("No Production model found for %s — promoting directly.", country)
            promote_model(country, new_version, "Production", model_type)
            return True

        prod_mv = prod_versions[0]
        assert prod_mv.run_id is not None, "registered model version has no run_id"
        prod_run = client.get_run(prod_mv.run_id)
        prod_mape = prod_run.data.metrics.get("MAPE", float("inf"))

        log.info(
            "%s: new MAPE=%.2f%% vs Production MAPE=%.2f%%",
            country,
            new_mape,
            prod_mape,
        )

        if new_mape < prod_mape - threshold:
            log.info("New model wins — promoting to Production.")
            promote_model(country, new_version, "Production", model_type)
            return True
        else:
            log.info("Production model retained (no improvement).")
            promote_model(country, new_version, "Staging", model_type)
            return False

    except Exception as e:
        log.warning("compare_and_promote failed (%s): %s", country, e)
        return False


def get_production_model(
    country: str,
    model_type: str = "classical",
) -> Any:
    """
    Load the Production model from the Registry for inference.

    Returns fitted sklearn/pytorch model or None if not found.
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        reg_name = _registry_name(country, model_type)
        model_uri = f"models:/{reg_name}@production"

        if model_type == "dl":
            model = mlflow.pytorch.load_model(model_uri)
        else:
            model = mlflow.sklearn.load_model(model_uri)

        log.info("Loaded Production model: %s", reg_name)
        return model

    except Exception as e:
        log.warning("Failed to load Production model (%s %s): %s", country, model_type, e)
        return None


def list_registered_models(model_type: str | None = None) -> pd.DataFrame:
    """
    List all registered models and their latest versions per stage.

    Returns DataFrame with columns:
        [name, country, model_type, version, stage, mape, run_id]
    """
    if not MLFLOW_AVAILABLE:
        return pd.DataFrame()
    try:
        mlflow.set_tracking_uri(get_tracking_uri())
        client = MlflowClient()
        rows = []

        for rm in client.search_registered_models():
            if model_type and model_type not in rm.name:
                continue
            for mv in client.search_model_versions(f"name='{rm.name}'"):
                try:
                    assert mv.run_id is not None, "registered model version has no run_id"
                    run = client.get_run(mv.run_id)
                    mape = run.data.metrics.get("MAPE", None)
                    ctry = run.data.tags.get("country", "unknown")
                    mtype = run.data.tags.get("model_type", "unknown")
                    mname = run.data.tags.get("model_name", "unknown")
                except Exception:
                    mape = None
                    ctry = rm.tags.get("country", "unknown")
                    mtype = rm.tags.get("model_type", "unknown")
                    mname = mv.tags.get("algorithm", "unknown")

                rows.append(
                    {
                        "registered_name": rm.name,
                        "country": ctry,
                        "model_type": mtype,
                        "algorithm": mname,
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "MAPE": round(mape, 3) if mape else None,
                        "run_id": mv.run_id[:8] if mv.run_id else "unknown",
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["country", "stage"])
        return df

    except Exception as e:
        log.warning("list_registered_models failed: %s", e)
        return pd.DataFrame()


def registry_summary() -> None:
    """Print a human-readable registry status table to the log."""
    df = list_registered_models()
    if df.empty:
        log.info("No registered models found.")
        return
    log.info("═" * 70)
    log.info("  MODEL REGISTRY STATUS")
    log.info("═" * 70)
    log.info(df.to_string(index=False))
    log.info("═" * 70)
