"""
src/register_models.py — Register Best Models to MLflow Registry
=================================================================
Ember Energy | IEEE Paper

Registers ALL best models per country — both statistical and ML.
Statistical models (Naive, Holt, ARIMA) are registered with metadata only
(no artifact — they have no learnable weights to persist).
ML models (Ridge, RF, XGBoost) are registered with full sklearn artifact.

Stages: None → Staging → Production → Archived
"""

from __future__ import annotations

import os as _os
import sys as _sys
from typing import Any

from sklearn.base import BaseEstimator

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"
VAL_END = 2020

# Models that have no sklearn artifact to persist
STAT_MODELS = {"Naive", "Naïve", "LinearTrend", "Holt", "ARIMA_1_1_1", "ARIMA(1,1,1)"}
ML_MODELS = {"Ridge", "RandomForest", "XGBoost"}


def _build_sklearn_model(model_name: str, best_hp: dict) -> BaseEstimator:
    """Instantiate and return a fitted-ready sklearn model."""
    if model_name == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**best_hp.get("Ridge", {"alpha": 1.0}))
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            **best_hp.get("RandomForest", {"n_estimators": 100}),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        import xgboost as xgb

        return xgb.XGBRegressor(
            **best_hp.get("XGBoost", {}),
            verbosity=0,
            tree_method="hist",
        )
    raise ValueError(f"Unknown ML model: {model_name}")


def register_classical(
    model_dir: str,
    pre_dir: str,
) -> None:
    """
    Register ALL best classical models — statistical and ML alike.

    Statistical models  → logged with params + MAPE, no artifact
    ML models (sklearn) → logged with params + MAPE + model artifact
    """
    from mlflow_config import (
        compare_and_promote,
        register_best_model,
        setup_experiment,
    )
    from src.modeling.features import prepare_xy

    # ── Load artifacts ────────────────────────────────────────────────────────
    best_models_path = os.path.join(model_dir, "best_models.csv")
    best_hp_path = os.path.join(model_dir, "best_hp.json")
    meta_path = os.path.join(pre_dir, "feature_meta.json")
    data_path = os.path.join(pre_dir, "ember_model_ready.csv")

    for path in [best_models_path, best_hp_path, meta_path, data_path]:
        if not os.path.exists(path):
            log.error("Missing: %s — run make run first", path)
            return

    best_df = pd.read_csv(best_models_path)
    df = pd.read_csv(data_path)
    with open(best_hp_path) as f:
        best_hp = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    good_features = meta["all_features"]
    setup_experiment()

    log.info("── Registering classical models (%d countries)…", len(COUNTRIES))

    for _, row in best_df.iterrows():
        country = row["Country"]
        model_name = row["Model"]
        mape = float(row["MAPE"])

        log.info("  %-10s | %-16s | MAPE=%.2f%%", country, model_name, mape)

        try:
            with mlflow.start_run(run_name=f"{country}_{model_name}_registry") as run:
                # Tags
                mlflow.set_tag("country", country)
                mlflow.set_tag("model_type", "classical")
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("is_statistical", str(model_name in STAT_MODELS))

                # Metrics
                mlflow.log_metric("MAPE", mape)

                # Params
                if model_name in ML_MODELS:
                    mlflow.log_params(best_hp.get(model_name, {}))
                else:
                    # Statistical model params
                    stat_params: dict[str, dict[str, Any]] = {
                        "Holt": {"trend": "add", "damped_trend": True},
                        "ARIMA_1_1_1": {"p": 1, "d": 1, "q": 1},
                        "ARIMA(1,1,1)": {"p": 1, "d": 1, "q": 1},
                        "Naive": {"type": "persistence"},
                        "Naïve": {"type": "persistence"},
                        "LinearTrend": {"type": "OLS_on_time_index"},
                    }
                    mlflow.log_params(stat_params.get(model_name, {}))

                # Artifact — only for sklearn models
                artifact_path = "none"
                if model_name in ML_MODELS:
                    sub = df[df["Area"] == country]
                    train_df = sub[sub["Year"] <= VAL_END].reset_index(drop=True)
                    X_tr, y_tr = prepare_xy(train_df, good_features, TARGET)

                    if model_name == "Ridge":
                        from sklearn.preprocessing import StandardScaler

                        X_tr = StandardScaler().fit_transform(X_tr)

                    model = _build_sklearn_model(model_name, best_hp)
                    model.fit(X_tr, y_tr)
                    mlflow.sklearn.log_model(model, "model")
                    artifact_path = "model"

                # Register to Model Registry
                version = register_best_model(
                    run_id=run.info.run_id,
                    country=country,
                    model_name=model_name,
                    model_type="classical",
                    artifact_path=artifact_path,
                )

                if version:
                    promoted = compare_and_promote(
                        country=country,
                        new_run_id=run.info.run_id,
                        new_mape=mape,
                        new_version=version,
                        model_type="classical",
                    )
                    stage = "Production" if promoted else "Staging"
                    log.info("  ✓ %-10s %-16s → %s v%s", country, model_name, stage, version)
                else:
                    log.warning("  ✗ %-10s %-16s registration failed", country, model_name)

        except Exception as e:
            log.error("  ✗ %-10s %-16s error: %s", country, model_name, e)


def register_dl(dl_dir: str) -> None:
    """Register best DL models — metadata + MAPE (no PyTorch artifact yet)."""
    from mlflow_config import (
        compare_and_promote,
        register_best_model,
        setup_experiment,
    )

    best_dl_path = os.path.join(dl_dir, "dl_best_models.csv")
    if not os.path.exists(best_dl_path):
        log.warning("dl_best_models.csv not found — skipping DL registration")
        return

    best_dl = pd.read_csv(best_dl_path)
    setup_experiment()
    log.info("── Registering DL models (%d entries)…", len(best_dl))

    for _, row in best_dl.iterrows():
        country = row["Country"]
        model_name = row["Model"]
        mape = float(row["MAPE"]) if "MAPE" in row else float("nan")

        try:
            with mlflow.start_run(run_name=f"{country}_{model_name}_DL_registry") as run:
                mlflow.set_tag("country", country)
                mlflow.set_tag("model_type", "dl")
                mlflow.set_tag("model_name", model_name)
                if mape == mape:  # not NaN
                    mlflow.log_metric("MAPE", mape)

                version = register_best_model(
                    run_id=run.info.run_id,
                    country=country,
                    model_name=model_name,
                    model_type="dl",
                    artifact_path="none",
                )

                if version and mape == mape:
                    promoted = compare_and_promote(
                        country=country,
                        new_run_id=run.info.run_id,
                        new_mape=mape,
                        new_version=version,
                        model_type="dl",
                    )
                    stage = "Production" if promoted else "Staging"
                    log.info(
                        "  ✓ %-10s %-14s MAPE=%.2f%% → %s v%s",
                        country,
                        model_name,
                        mape,
                        stage,
                        version,
                    )

        except Exception as e:
            log.error("  ✗ %-10s %-14s error: %s", country, model_name, e)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Register best models to MLflow Registry")
    p.add_argument("--model_dir", default="outputs/modeling", help="Classical model dir")
    p.add_argument("--pre_dir", default="outputs/preprocessing", help="Preprocessing dir")
    p.add_argument("--dl_dir", default="outputs/deeplearning", help="DL output dir")
    p.add_argument("--dl-only", action="store_true", help="Register DL only")
    p.add_argument("--classic-only", action="store_true", help="Register classical only")
    args = p.parse_args()

    log.info("=" * 60)
    log.info("  MLflow Model Registry — Registration")
    log.info("  Tracking URI: %s", os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    log.info("=" * 60)

    if not args.dl_only:
        register_classical(args.model_dir, args.pre_dir)

    if not args.classic_only:
        register_dl(args.dl_dir)

    # Print summary
    try:
        from mlflow_config import registry_summary

        registry_summary()
    except Exception:
        pass

    log.info("=" * 60)
    log.info("  Registration complete.")
    log.info("  View: http://localhost:5000/#/models")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
