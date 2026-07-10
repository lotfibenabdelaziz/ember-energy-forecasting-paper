"""
src/forecasting/forecasters.py — Statistical + ML Multi-Step Forecasters
=========================================================================
Ember Energy | IEEE Paper

Shared between 04_forecasting.py and test_forecasting.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def clean_x(arr: np.ndarray) -> np.ndarray:
    """Replace Inf/-Inf with NaN, impute column medians."""
    arr = np.array(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    col_medians = np.nanmedian(arr, axis=0)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return arr


def forecast_statistical(series: np.ndarray, model_name: str, horizon: int) -> np.ndarray:
    """Multi-step statistical forecast. Mirrors 04_forecasting.py exactly."""
    series = np.array(series, dtype=float)
    if model_name in ("Naive", "Naïve", "LinearTrend"):
        X = np.arange(len(series)).reshape(-1, 1)
        m = LinearRegression().fit(X, series)
        return m.predict(np.arange(len(series), len(series) + horizon).reshape(-1, 1)).flatten()
    elif model_name == "Holt":
        m = ExponentialSmoothing(series, trend="add", damped_trend=True).fit(optimized=True)
        return np.array(m.forecast(horizon))
    elif model_name in ("ARIMA(1,1,1)", "ARIMA_1_1_1"):
        m = ARIMA(series, order=(1, 1, 1)).fit()
        pm = m.get_forecast(steps=horizon).predicted_mean
        # statsmodels >= 0.14 returns ndarray directly; older returns Series
        return np.array(pm)
    return np.full(horizon, series[-1])


def extrapolate_exog(
    history_df: pd.DataFrame, exog_cols: list[str], n_steps: int
) -> dict[str, np.ndarray]:
    """Linear trend extrapolation per exogenous feature."""
    preds: dict[str, np.ndarray] = {}
    n = len(history_df)
    for col in exog_cols:
        y = history_df[col].values.astype(float)
        y = np.where(np.isfinite(y), y, np.nanmedian(y))
        X = np.arange(n).reshape(-1, 1)
        try:
            m = LinearRegression().fit(X, y)
            preds[col] = m.predict(np.arange(n, n + n_steps).reshape(-1, 1))
        except Exception:
            preds[col] = np.full(n_steps, y[-1])
    return preds


def get_ml_cls_map() -> dict:
    """Return model class map for dispatch in bootstrap_forecast_ci."""
    ml_map = {
        "Ridge": (Ridge, True),
        "RandomForest": (None, False),
        "XGBoost": (None, False),
    }
    try:
        from sklearn.ensemble import RandomForestRegressor

        ml_map["RandomForest"] = (RandomForestRegressor, False)
    except ImportError:
        pass
    if HAS_XGB:
        ml_map["XGBoost"] = (xgb.XGBRegressor, False)
    return ml_map


def bootstrap_forecast_ci(
    history_df: pd.DataFrame,
    model_name: str,
    all_feature_cols: list[str],
    exog_cols: list[str],
    params: dict,
    horizon: int,
    target: str,
    ml_cls_map: dict,
    n_boot: int = 200,
    ci: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual bootstrap CI. Returns (point, lower, upper)."""
    from src.forecasting.recursive import get_insample_residuals, point_forecast

    base_fc = point_forecast(
        history_df, model_name, all_feature_cols, exog_cols, params, horizon, target, ml_cls_map
    )
    residuals = get_insample_residuals(
        history_df, model_name, all_feature_cols, exog_cols, params, target, ml_cls_map
    )
    ts = history_df[target].values.astype(float)
    alpha = (100 - ci) / 2

    if len(residuals) < 3:
        std = np.std(ts) * 0.10
        return base_fc, base_fc - 1.645 * std, base_fc + 1.645 * std

    boot_fcs = []
    for _ in range(n_boot):
        noise = np.random.choice(residuals, size=len(ts), replace=True)
        ts_pert = np.clip(ts + noise * 0.5, ts.min() * 0.3, None)
        h_pert = history_df.copy()
        h_pert[target] = ts_pert
        try:
            fc_b = point_forecast(
                h_pert,
                model_name,
                all_feature_cols,
                exog_cols,
                params,
                horizon,
                target,
                ml_cls_map,
            )
            boot_fcs.append(fc_b)
        except Exception:
            boot_fcs.append(base_fc)

    boot_arr = np.array(boot_fcs)
    raw_lo = np.percentile(boot_arr, alpha, axis=0)
    raw_hi = np.percentile(boot_arr, 100 - alpha, axis=0)

    # Guarantee: lower <= point forecast <= upper
    lo = np.minimum(raw_lo, base_fc)
    hi = np.maximum(raw_hi, base_fc)

    return base_fc, lo, hi
