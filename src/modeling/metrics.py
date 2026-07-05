"""
src/modeling/metrics.py — Classical Model Metrics
===================================================
Ember Energy | IEEE Paper — Classical modeling module

mape(), rmse(), agg_metrics() — shared by 03_modeling.py and 04_forecasting.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (excludes zero actuals)."""
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100)


def rmse(yt, yp) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(yt, yp)))


def agg_metrics(df_r: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MAE/RMSE/MAPE per Country x Model."""
    rows = []
    for (country, model), g in df_r.groupby(["Country", "Model"]):
        rows.append(
            {
                "Country": country,
                "Model": model,
                "MAE": mean_absolute_error(g["y_actual"], g["y_pred"]),
                "RMSE": rmse(g["y_actual"], g["y_pred"]),
                "MAPE": mape(g["y_actual"], g["y_pred"]),
            }
        )
    return pd.DataFrame(rows)
