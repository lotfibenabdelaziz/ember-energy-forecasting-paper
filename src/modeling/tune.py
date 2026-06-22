"""
src/modeling/tune.py — Hyperparameter Tuning (Ridge/RF/XGBoost)
==================================================================
Ember Energy | IEEE Paper — Classical modeling module

tune_hyperparameters() — GridSearchCV with TimeSeriesSplit on Train->Val.
Used by 03_modeling.py.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.modeling.features import prepare_xy

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

log = logging.getLogger(__name__)


def tune_hyperparameters(
    df: pd.DataFrame,
    all_features: list[str],
    target: str,
    val_end: int,
    model_dir: str,
) -> tuple[dict, list[str]]:
    """
    Tune Ridge, RF, XGBoost on Train->Val via TimeSeriesSplit.

    Also drops features with >80% NaN/Inf across the tuning window
    (GOOD_FEATURES filtering).

    Returns
    -------
    best_hp        : dict {model_name: {param: value}}
    good_features  : list[str] — cleaned feature list
    """
    df_tune    = df[df["Year"] <= val_end].copy()
    feat_check = df_tune[all_features].replace([np.inf, -np.inf], np.nan)
    null_frac  = feat_check.isnull().mean()
    good_features = [c for c in all_features if null_frac[c] < 0.80]

    dropped = set(all_features) - set(good_features)
    if dropped:
        log.warning("Dropped %d high-null features: %s", len(dropped), dropped)
    log.info("Features used for tuning: %d", len(good_features))

    X_tune, y_tune = prepare_xy(df_tune, good_features, target)
    log.info("Tune set shape: X=%s, y=%s", X_tune.shape, y_tune.shape)

    tscv = TimeSeriesSplit(n_splits=3)

    # Ridge
    ridge_grid = GridSearchCV(
        Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]},
        cv=tscv, scoring="neg_mean_absolute_error", refit=True,
    )
    ridge_grid.fit(X_tune, y_tune)
    best_ridge_alpha = ridge_grid.best_params_["alpha"]
    log.info("Best Ridge alpha: %s", best_ridge_alpha)

    # Random Forest
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        cv=tscv, scoring="neg_mean_absolute_error", refit=True,
    )
    rf_grid.fit(X_tune, y_tune)
    best_rf_params = rf_grid.best_params_
    log.info("Best RF params: %s", best_rf_params)

    # XGBoost
    best_xgb_params: dict = {}
    if HAS_XGB:
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(verbosity=0, random_state=42, tree_method="hist"),
            {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            cv=tscv, scoring="neg_mean_absolute_error", refit=True,
        )
        xgb_grid.fit(X_tune, y_tune)
        best_xgb_params = xgb_grid.best_params_
        log.info("Best XGB params: %s", best_xgb_params)
    else:
        log.warning("XGBoost not installed — skipping tuning")

    best_hp = {
        "Ridge":        {"alpha": best_ridge_alpha},
        "RandomForest": best_rf_params,
        "XGBoost":      best_xgb_params,
    }

    with open(os.path.join(model_dir, "best_hp.json"), "w") as f:
        json.dump(best_hp, f, indent=2)
    log.info("Saved best_hp.json")

    return best_hp, good_features
