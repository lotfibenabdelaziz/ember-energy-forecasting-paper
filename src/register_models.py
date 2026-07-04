"""
src/register_models.py — Register Best Models to MLflow Registry
=================================================================
Ember Energy | IEEE Paper

Run after make run to register and promote best models:
    python src/register_models.py

What it does:
  1. Loads best_models.csv + best_hp.json from outputs/modeling/
  2. Loads dl_best_models.csv from outputs/deeplearning/
  3. Refits each best classical model on full Train+Val data
  4. Logs model artifact + MAPE to MLflow
  5. Registers to MLflow Model Registry
  6. Auto-promotes to Production if beats existing Production MAPE
  7. Prints registry summary

Stages:
    None → Staging → Production → Archived
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

COUNTRIES  = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET     = "Demand"
TRAIN_END  = 2016
VAL_END    = 2020


def register_classical(
    model_dir: str,
    pre_dir:   str,
    outputs_dir: str,
) -> None:
    """Register best classical models from outputs/modeling/."""
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor

        from mlflow_config import (
            setup_experiment, compare_and_promote,
            register_best_model, registry_summary,
        )
        from src.modeling.features import prepare_xy
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        return

    # Load artifacts
    best_models_path = os.path.join(model_dir, "best_models.csv")
    best_hp_path     = os.path.join(model_dir, "best_hp.json")
    meta_path        = os.path.join(pre_dir, "feature_meta.json")
    data_path        = os.path.join(pre_dir, "ember_model_ready.csv")

    for path in [best_models_path, best_hp_path, meta_path, data_path]:
        if not os.path.exists(path):
            log.error("Missing: %s — run make run first", path)
            return

    best_df  = pd.read_csv(best_models_path)
    df       = pd.read_csv(data_path)
    with open(best_hp_path) as f:
        best_hp = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    good_features = meta["all_features"]
    setup_experiment()

    log.info("── Registering classical models (%d countries)…", len(COUNTRIES))

    for _, row in best_df.iterrows():
        country    = row["Country"]
        model_name = row["Model"]
        mape       = float(row["MAPE"])

        # Only register sklearn models (not statistical)
        if model_name not in ("Ridge", "RandomForest", "XGBoost"):
            log.info("  %-10s %-14s — skipped (statistical model)", country, model_name)
            continue

        sub      = df[df["Area"] == country]
        train_df = sub[sub["Year"] <= VAL_END].reset_index(drop=True)
        X_tr, y_tr = prepare_xy(train_df, good_features, TARGET)

        # Build model
        if model_name == "Ridge":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            model  = Ridge(**best_hp.get("Ridge", {"alpha": 1.0}))
        elif model_name == "RandomForest":
            model  = RandomForestRegressor(
                **best_hp.get("RandomForest", {"n_estimators": 100}),
                random_state=42, n_jobs=-1,
            )
        elif model_name == "XGBoost":
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    **best_hp.get("XGBoost", {}),
                    verbosity=0, tree_method="hist",
                )
            except ImportError:
                log.warning("XGBoost not installed — skipping %s", country)
                continue
        else:
            continue

        model.fit(X_tr, y_tr)

        # Log + register
        try:
            with mlflow.start_run(
                run_name=f"{country}_{model_name}_registry"
            ) as run:
                mlflow.set_tag("country",    country)
                mlflow.set_tag("model_type", "classical")
                mlflow.set_tag("model_name", model_name)
                mlflow.log_metric("MAPE", mape)
                mlflow.log_params(best_hp.get(model_name, {}))
                mlflow.sklearn.log_model(model, "model")

                version = register_best_model(
                    run_id      = run.info.run_id,
                    country     = country,
                    model_name  = model_name,
                    model_type  = "classical",
                )

                if version:
                    promoted = compare_and_promote(
                        country     = country,
                        new_run_id  = run.info.run_id,
                        new_mape    = mape,
                        new_version = version,
                        model_type  = "classical",
                    )
                    stage = "Production" if promoted else "Staging"
                    log.info("  ✓ %-10s %-14s MAPE=%.2f%% → %s v%s",
                             country, model_name, mape, stage, version)
                else:
                    log.warning("  ✗ %-10s %-14s registration failed", country, model_name)

        except Exception as e:
            log.error("  ✗ %-10s %-14s error: %s", country, model_name, e)


def register_dl(dl_dir: str) -> None:
    """Register best DL models from outputs/deeplearning/."""
    try:
        import mlflow
        import mlflow.pytorch
        from mlflow_config import (
            setup_experiment, compare_and_promote,
            register_best_model,
        )
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        return

    best_dl_path = os.path.join(dl_dir, "dl_best_models.csv")
    if not os.path.exists(best_dl_path):
        log.warning("dl_best_models.csv not found — skipping DL registration")
        return

    best_dl = pd.read_csv(best_dl_path)
    setup_experiment()

    log.info("── Registering DL models (%d countries)…", len(best_dl))

    for _, row in best_dl.iterrows():
        country    = row["Country"]
        model_name = row["Model"]
        mape       = float(row["MAPE"]) if "MAPE" in row else float("nan")

        try:
            with mlflow.start_run(
                run_name=f"{country}_{model_name}_DL_registry"
            ) as run:
                mlflow.set_tag("country",    country)
                mlflow.set_tag("model_type", "dl")
                mlflow.set_tag("model_name", model_name)
                if mape == mape:  # not NaN
                    mlflow.log_metric("MAPE", mape)

                # Note: actual PyTorch model logging requires the fitted model object
                # which is not persisted after 05_deeplearning.py runs.
                # To enable: modify 05_deeplearning.py to save models with torch.save()
                # and load them here before mlflow.pytorch.log_model()

                version = register_best_model(
                    run_id     = run.info.run_id,
                    country    = country,
                    model_name = model_name,
                    model_type = "dl",
                )

                if version and mape == mape:
                    compare_and_promote(
                        country     = country,
                        new_run_id  = run.info.run_id,
                        new_mape    = mape,
                        new_version = version,
                        model_type  = "dl",
                    )
                    log.info("  ✓ %-10s %-14s MAPE=%.2f%% v%s", country, model_name, mape, version)

        except Exception as e:
            log.error("  ✗ %-10s %-14s error: %s", country, model_name, e)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Register best models to MLflow Registry")
    p.add_argument("--model_dir",   default="outputs/modeling",     help="Classical model dir")
    p.add_argument("--pre_dir",     default="outputs/preprocessing", help="Preprocessing dir")
    p.add_argument("--dl_dir",      default="outputs/deeplearning",  help="DL output dir")
    p.add_argument("--output_dir",  default="outputs",               help="Root outputs dir")
    p.add_argument("--dl-only",     action="store_true",             help="Register DL only")
    p.add_argument("--classic-only",action="store_true",             help="Register classical only")
    args = p.parse_args()

    log.info("=" * 60)
    log.info("  MLflow Model Registry — Registration")
    log.info("  Tracking URI: %s", os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    log.info("=" * 60)

    if not args.dl_only:
        register_classical(args.model_dir, args.pre_dir, args.output_dir)

    if not args.classic_only:
        register_dl(args.dl_dir)

    # Summary
    try:
        from mlflow_config import registry_summary
        registry_summary()
    except Exception:
        pass

    log.info("=" * 60)
    log.info("  Registration complete.")
    log.info("  View registry: http://localhost:5000/#/models")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
