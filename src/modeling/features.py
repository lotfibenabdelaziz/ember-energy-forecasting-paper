"""
src/modeling/features.py — Feature Cleaning Utilities
========================================================
Ember Energy | IEEE Paper — Classical modeling module

clean_features(), prepare_xy() — shared by step03_modeling.py and step04_forecasting.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_features(X: np.ndarray) -> np.ndarray:
    """Replace Inf/-Inf with NaN, then impute column medians."""
    X = X.astype(float)
    X[~np.isfinite(X)] = np.nan
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


def prepare_xy(
    subset: pd.DataFrame, feature_cols: list[str], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return clean X, y arrays — drop rows where target is NaN."""
    sub = subset[[*feature_cols, target]].copy()
    sub = sub[sub[target].notna()]
    X = clean_features(sub[feature_cols].values)
    y = sub[target].values
    return X, y
