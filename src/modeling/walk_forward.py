"""
src/modeling/walk_forward.py — Walk-Forward Evaluation Loop
================================================================
Ember Energy | IEEE Paper — Classical modeling module

walk_forward_evaluate() — expanding-window evaluation over a set of years.
Used by 03_modeling.py for both Test (2021-2024) and Val (2017-2020) evaluation.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.modeling.forecasters import (
    arima_1step, holt_1step, linear_trend_1step, ml_1step, naive_1step,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

log = logging.getLogger(__name__)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]


def walk_forward_evaluate(
    df: pd.DataFrame,
    all_features: list[str],
    target: str,
    years: list[int],
    best_hp: dict,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward over `years`.

    For each (country, year):
      1. Train on all data strictly before `year`
      2. Predict `year` with 7 models: Naive, LinearTrend, Holt,
         ARIMA(1,1,1), Ridge, RandomForest, XGBoost
      3. Record actual vs predicted

    Returns long-format DataFrame:
        [Country, Year, Model, y_actual, y_pred, error, abs_pct_error]
    """
    best_ridge_alpha = best_hp["Ridge"]["alpha"]
    best_rf_params    = best_hp["RandomForest"]
    best_xgb_params   = best_hp["XGBoost"]

    all_results: list[dict] = []

    for country in COUNTRIES:
        sub = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)

        for t_year in years:
            train_df = sub[sub["Year"] < t_year]
            test_row = sub[sub["Year"] == t_year]

            if test_row.empty or len(train_df) < 5:
                continue

            y_actual     = test_row[target].values[0]
            train_series = train_df[target]

            preds: dict[str, float] = {}
            preds["Naive"]       = naive_1step(train_series)
            preds["LinearTrend"] = linear_trend_1step(train_series)
            preds["Holt"]        = holt_1step(train_series)
            preds["ARIMA_1_1_1"] = arima_1step(train_series)

            try:
                preds["Ridge"] = ml_1step(
                    train_df, test_row.iloc[0], Ridge,
                    all_features, target, scaler=StandardScaler(),
                    alpha=best_ridge_alpha,
                )
            except Exception:
                preds["Ridge"] = train_series.iloc[-1]

            try:
                preds["RandomForest"] = ml_1step(
                    train_df, test_row.iloc[0], RandomForestRegressor,
                    all_features, target, scaler=None,
                    random_state=42, n_jobs=-1, **best_rf_params,
                )
            except Exception:
                preds["RandomForest"] = train_series.iloc[-1]

            if HAS_XGB:
                try:
                    preds["XGBoost"] = ml_1step(
                        train_df, test_row.iloc[0], xgb.XGBRegressor,
                        all_features, target, scaler=None,
                        verbosity=0, random_state=42, tree_method="hist",
                        **best_xgb_params,
                    )
                except Exception:
                    preds["XGBoost"] = train_series.iloc[-1]

            for model_name, y_pred in preds.items():
                all_results.append({
                    "Country":  country,
                    "Year":     t_year,
                    "Model":    model_name,
                    "y_actual": y_actual,
                    "y_pred":   float(y_pred),
                    "error":    y_actual - float(y_pred),
                    "abs_pct_error": abs(y_actual - float(y_pred)) / (abs(y_actual) + 1e-9) * 100,
                })

    return pd.DataFrame(all_results)
