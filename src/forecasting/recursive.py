"""
src/forecasting/recursive.py — Recursive ML forecast + residual computation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.forecasting.forecasters import clean_x, extrapolate_exog, forecast_statistical

STAT_MODELS = {"Naive", "Naïve", "LinearTrend", "Holt", "ARIMA(1,1,1)", "ARIMA_1_1_1"}


def _build_future_row(
    history_df: pd.DataFrame,
    exog_future: dict,
    demand_preds: list[float],
    step: int,
    all_feature_cols: list[str],
    target: str,
) -> dict[str, float]:
    hist_demand = list(history_df[target].values.astype(float))
    full_demand = hist_demand + list(demand_preds)
    n = len(full_demand)
    row: dict[str, float] = {}

    for lag in [1, 2, 3]:
        k   = f"{target}_lag{lag}"
        idx = n - lag
        row[k] = full_demand[idx] if idx >= 0 else np.nan  # always set

    for w in [3, 5]:
        k      = f"{target}_ma{w}"
        window = full_demand[max(0, n - w) : n]
        row[k] = float(np.mean(window)) if window else np.nan  # always set

    k = f"{target}_yoy"
    if k in all_feature_cols:
        if n >= 2 and full_demand[n - 2] != 0:
            row[k] = (full_demand[n - 1] - full_demand[n - 2]) / abs(full_demand[n - 2]) * 100
        else:
            row[k] = 0.0

    for col in exog_future:
        hist_exog = list(history_df[col].values.astype(float))
        full_exog = hist_exog + list(exog_future[col][: step + 1])
        n_ex = len(full_exog)
        for lag in [1, 2, 3]:
            k = f"{col}_lag{lag}"
            if k in all_feature_cols:
                row[k] = full_exog[n_ex - lag] if n_ex - lag >= 0 else np.nan
        for w in [3, 5]:
            k = f"{col}_ma{w}"
            if k in all_feature_cols:
                window = full_exog[max(0, n_ex - w) : n_ex]
                row[k] = float(np.mean(window)) if window else np.nan
        k = f"{col}_yoy"
        if k in all_feature_cols:
            if n_ex >= 2 and full_exog[n_ex - 2] != 0:
                row[k] = (
                    (full_exog[n_ex - 1] - full_exog[n_ex - 2]) / abs(full_exog[n_ex - 2]) * 100
                )
            else:
                row[k] = 0.0

    year_min = int(history_df["Year"].min())
    future_year = int(history_df["Year"].max()) + step + 1
    row["trend"] = future_year - year_min
    row["trend_sq"] = row["trend"] ** 2
    if "country_code" in all_feature_cols and "country_code" in history_df.columns:
        row["country_code"] = int(history_df["country_code"].iloc[0])
    for col in all_feature_cols:
        if col not in row:
            row[col] = 0.0
    return row


def forecast_ml_recursive(
    history_df: pd.DataFrame,
    model_cls,
    all_feature_cols: list[str],
    exog_cols: list[str],
    horizon: int,
    target: str,
    scaler=None,
    **kwargs,
) -> np.ndarray:
    tr = history_df[all_feature_cols + [target]].copy()
    tr = tr[tr[target].notna()]
    X_tr = clean_x(tr[all_feature_cols].values)
    y_tr = tr[target].values.astype(float)
    if len(X_tr) < 3:
        raise ValueError("Not enough training rows")
    if scaler:
        X_tr = scaler.fit_transform(X_tr)
    model = model_cls(**kwargs)
    model.fit(X_tr, y_tr)
    exog_future = extrapolate_exog(history_df, exog_cols, horizon)
    demand_preds: list[float] = []
    for step in range(horizon):
        feat_row = _build_future_row(
            history_df, exog_future, demand_preds, step, all_feature_cols, target
        )
        x_pred = np.array([[feat_row[c] for c in all_feature_cols]], dtype=float)
        x_pred = clean_x(x_pred)
        if scaler:
            x_pred = scaler.transform(x_pred)
        demand_preds.append(float(model.predict(x_pred)[0]))
    return np.array(demand_preds)


def point_forecast(
    history_df,
    model_name,
    all_feature_cols,
    exog_cols,
    params,
    horizon,
    target,
    ml_cls_map,
) -> np.ndarray:
    ts = history_df[target].values.astype(float)
    if model_name in STAT_MODELS:
        return forecast_statistical(ts, model_name, horizon)
    cls, needs_scaler = ml_cls_map[model_name]
    if cls is None:
        return np.full(horizon, ts[-1])
    scaler = StandardScaler() if needs_scaler else None
    kw = dict(params)
    if model_name == "XGBoost":
        kw.setdefault("tree_method", "hist")
        kw.setdefault("verbosity", 0)
    return forecast_ml_recursive(
        history_df, cls, all_feature_cols, exog_cols, horizon, target, scaler, **kw
    )


def get_insample_residuals(
    history_df,
    model_name,
    all_feature_cols,
    exog_cols,
    params,
    target,
    ml_cls_map,
    n_eval=8,
) -> np.ndarray:
    residuals = []
    n = len(history_df)
    start = max(10, n - n_eval)
    for t in range(start, n):
        tr = history_df.iloc[:t].reset_index(drop=True)
        y_t = float(history_df.iloc[t][target])
        try:
            fc1 = point_forecast(
                tr, model_name, all_feature_cols, exog_cols, params, 1, target, ml_cls_map
            )
            residuals.append(y_t - float(fc1[0]))
        except Exception:
            pass
    return np.array(residuals)
