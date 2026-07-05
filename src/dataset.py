"""
src/dataset.py — Dataset Loading, Sliding-Window Dataset, Scalers
==================================================================
Ember Energy | IEEE Paper — Deep Learning module

Mirrors notebook 04_deeplearning_enhanced_patched.ipynb sections:
    - Load data + feature_meta.json
    - clean_arr() utility
    - DemandDataset (sliding window torch.Dataset)
    - make_loaders() — train/val DataLoader builder with scalers
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# ── Constants — must match notebook exactly ───────────────────────────────────
COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"

TRAIN_END = 2016
VAL_END = 2020
TEST_END = 2024
FORECAST_YEARS = list(range(2025, 2031))

SEQ_LEN = 5  # look-back window (years)
HORIZON = 1  # one-step-ahead
EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 16
LR = 1e-3
SEED = 42

PALETTE = {
    "Tunisia": "#e63946",
    "Austria": "#2196F3",
    "Germany": "#4CAF50",
    "Egypt": "#9C27B0",
    "Canada": "#00BCD4",
    "France": "#797148",
    "Kuwait": "#4C4879",
}


# ── Loading ───────────────────────────────────────────────────────────────────


def load_model_ready(pre_dir: str) -> tuple[pd.DataFrame, dict]:
    """
    Load ember_model_ready.csv + feature_meta.json from preprocessing output.

    Returns
    -------
    df   : DataFrame — model-ready data (all countries, lag-NaN rows dropped)
    meta : dict — feature_meta.json contents (all_subs, all_features, etc.)
    """
    df_path = os.path.join(pre_dir, "ember_model_ready.csv")
    meta_path = os.path.join(pre_dir, "feature_meta.json")

    df = pd.read_csv(df_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return df, meta


def get_all_features(meta: dict) -> list[str]:
    """Extract the full engineered feature list from feature_meta.json."""
    return meta["all_features"]


# ── Cleaning utility ──────────────────────────────────────────────────────────


def clean_arr(arr: np.ndarray) -> np.ndarray:
    """
    Replace Inf/NaN with column median.
    Mirrors notebook clean_arr() exactly.
    """
    arr = np.array(arr, dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    col_med = np.nanmedian(arr, axis=0)
    mask = np.isnan(arr)
    arr[mask] = np.take(col_med, np.where(mask)[1])
    return arr


# ── Sliding-window dataset ─────────────────────────────────────────────────────


class DemandDataset(Dataset):
    """
    Sliding-window dataset for a single country.

    X : [SEQ_LEN, n_features]  (StandardScaler-normalised)
    y : scalar                 (StandardScaler-normalised Demand)

    If scaler_X / scaler_y are not provided, new scalers are fit on this data
    (used for training sets). Pass training scalers when constructing
    validation/test datasets to avoid data leakage.
    """

    def __init__(
        self,
        df_country: pd.DataFrame,
        feature_cols: list[str],
        target: str,
        seq_len: int,
        scaler_X: StandardScaler | None = None,
        scaler_y: StandardScaler | None = None,
    ) -> None:
        sub = df_country.sort_values("Year").reset_index(drop=True)
        X_raw = clean_arr(sub[feature_cols].values)
        y_raw = sub[target].values.reshape(-1, 1).astype(np.float32)

        # Fit or apply scalers
        self.scaler_X = scaler_X or StandardScaler()
        self.scaler_y = scaler_y or StandardScaler()

        if scaler_X is None:
            X_sc = self.scaler_X.fit_transform(X_raw)
        else:
            X_sc = self.scaler_X.transform(X_raw)

        if scaler_y is None:
            y_sc = self.scaler_y.fit_transform(y_raw).flatten()
        else:
            y_sc = self.scaler_y.transform(y_raw).flatten()

        self.X = torch.tensor(X_sc, dtype=torch.float32)
        self.y = torch.tensor(y_sc, dtype=torch.float32)
        self.seq_len = seq_len
        self.n_samples = max(0, len(self.X) - self.seq_len)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i : i + self.seq_len], self.y[i + self.seq_len]


# ── DataLoader builder ─────────────────────────────────────────────────────────


def make_loaders(
    df_country: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    seq_len: int,
    train_end: int,
    val_end: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Split a single country's data by year and return
    (train_loader, val_loader, scaler_X, scaler_y).

    Mirrors notebook make_loaders() exactly:
      - Train dataset fits its own scalers
      - Val dataset reuses train scalers (no leakage)
      - Val loader only includes windows AFTER the train cutoff
      - Falls back to train loader if no val windows exist
    """
    tr = df_country[df_country["Year"] <= train_end]
    val = df_country[df_country["Year"] <= val_end]

    if len(tr) < seq_len + 2:
        raise ValueError(f"Insufficient training data: {len(tr)} rows, need >= {seq_len + 2}")

    ds_tr = DemandDataset(tr, feature_cols, target, seq_len)
    ds_val = DemandDataset(
        val,
        feature_cols,
        target,
        seq_len,
        scaler_X=ds_tr.scaler_X,
        scaler_y=ds_tr.scaler_y,
    )

    n_tr = len(ds_tr)
    idx_val = list(range(n_tr, len(ds_val)))

    # drop_last=True avoids a final batch of size 1, which crashes BatchNorm.
    # If the dataset is too small for even one full batch, fall back to
    # drop_last=False and let safe_loader() cap the batch size instead.
    use_drop_last = n_tr > batch_size

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=use_drop_last)

    # Guard: if drop_last produced an EMPTY loader (n_tr <= batch_size),
    # rebuild with a safe (capped) batch size instead.
    if len(loader_tr) == 0:
        safe_bs = max(2, min(batch_size, n_tr))
        loader_tr = DataLoader(ds_tr, batch_size=safe_bs, shuffle=True, drop_last=False)

    val_subset = Subset(ds_val, idx_val) if idx_val else ds_tr
    val_bs = max(2, min(batch_size, len(val_subset)))
    loader_val = DataLoader(val_subset, batch_size=val_bs, shuffle=False, drop_last=False)

    return loader_tr, loader_val, ds_tr.scaler_X, ds_tr.scaler_y


def safe_loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """
    Cap batch_size to dataset length to prevent batch_size=1 crashes
    (LayerNorm/BatchNorm require >1 sample). Mirrors notebook _safe_loader().
    """
    n = len(dataset)
    bs = max(2, min(batch_size, n))
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=False)
