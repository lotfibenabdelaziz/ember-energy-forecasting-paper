"""
src/modeling/forecasters.py — 1-Step Statistical & ML Forecasters
=====================================================================
Ember Energy | IEEE Paper — Classical modeling module

naive_1step(), linear_trend_1step(), holt_1step(), arima_1step(), ml_1step()
Used by 03_modeling.py for walk-forward evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.modeling.features import clean_features


def naive_1step(train_series: pd.Series) -> float:
    """Persistence baseline — predict last observed value."""
    return train_series.iloc[-1]


def linear_trend_1step(train_series: pd.Series) -> float:
    """Fit linear trend on full series, predict next point."""
    from sklearn.linear_model import LinearRegression

    X = np.arange(len(train_series)).reshape(-1, 1)
    m = LinearRegression().fit(X, train_series.values)
    return float(m.predict([[len(train_series)]])[0])


def holt_1step(train_series: pd.Series) -> float:
    """Holt's damped trend exponential smoothing, 1-step forecast."""
    try:
        m = ExponentialSmoothing(train_series.values, trend="add", damped_trend=True).fit(
            optimized=True
        )
        return float(m.forecast(1)[0])
    except Exception:
        return train_series.iloc[-1]


def arima_1step(train_series: pd.Series) -> float:
    """ARIMA(1,1,1), 1-step forecast."""
    try:
        m = ARIMA(np.asarray(train_series.values), order=(1, 1, 1)).fit()
        return float(m.forecast(1)[0])
    except Exception:
        return train_series.iloc[-1]


def ml_1step(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    model_cls,
    feature_cols: list[str],
    target: str,
    scaler: StandardScaler | None = None,
    **kwargs,
) -> float:
    """
    1-step ML prediction — robust to Inf/NaN via clean_features().

    Fits model_cls(**kwargs) on train_df, predicts test_row.
    Falls back to last training value if fewer than 3 clean rows.
    """
    tr = train_df[[*feature_cols, target]].copy()
    tr = tr[tr[target].notna()]
    X_tr = clean_features(tr[feature_cols].values)
    y_tr = tr[target].values

    if len(X_tr) < 3:
        return float(y_tr[-1]) if len(y_tr) else 0.0

    if scaler:
        X_tr = scaler.fit_transform(X_tr)

    model = model_cls(**kwargs)
    model.fit(X_tr, y_tr)

    x_pred = clean_features(np.array(test_row[feature_cols].values, dtype=float).reshape(1, -1))
    if scaler:
        x_pred = scaler.transform(x_pred)

    return float(model.predict(x_pred)[0])
