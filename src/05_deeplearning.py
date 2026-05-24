"""
src/05_deeplearning.py — Deep Learning Demand Forecasting
==========================================================
Ember Energy | IEEE Paper

Models: MLP, TCN, N-BEATS, TFT (per-country, walk-forward)

Reads:   outputs/preprocessing/ember_model_ready.csv
         outputs/preprocessing/feature_meta.json
Writes:  outputs/deeplearning/data/dl_benchmarking.csv
         outputs/deeplearning/data/dl_best_models.csv
         outputs/deeplearning/data/dl_wf_predictions.csv
         outputs/deeplearning/data/dl_forecast_2025_2030.csv
         outputs/deeplearning/data/dl_summary.json
         outputs/deeplearning/figures/*.png

Usage:
    python src/05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning
    python src/05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning --quick
    python src/05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning --models MLP TCN
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, Subset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.error("PyTorch not installed. Run: pip install torch")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ── Constants ─────────────────────────────────────────────────────────────────
COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"

PALETTE = {
    "Tunisia": "#e63946",
    "Austria": "#2196F3",
    "Germany": "#4CAF50",
    "Egypt": "#9C27B0",
    "Canada": "#00BCD4",
    "France": "#797148",
    "Kuwait": "#4C4879",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════


def mape(yt: np.ndarray, yp: np.ndarray) -> float:
    yt, yp = np.array(yt, float), np.array(yp, float)
    m = yt != 0
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100)


def rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(yt, yp)))


def smape(yt: np.ndarray, yp: np.ndarray) -> float:
    yt, yp = np.array(yt, float), np.array(yp, float)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100)


def safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    ss_tot = np.sum((a - a.mean()) ** 2)
    if ss_tot < 1e-10:
        return np.nan
    return float(1.0 - np.sum((a - b) ** 2) / ss_tot)


def compute_metrics(country: str, model: str, yt: np.ndarray, yp: np.ndarray) -> dict:
    a, b = np.array(yt, float), np.array(yp, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {
            "Country": country,
            "Model": model,
            "MAPE": np.nan,
            "SMAPE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }
    r2 = safe_r2(a, b)
    return {
        "Country": country,
        "Model": model,
        "MAPE": round(mape(a, b), 2),
        "SMAPE": round(smape(a, b), 2),
        "RMSE": round(rmse(a, b), 3),
        "R2": round(r2, 4) if np.isfinite(r2) else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def clean_arr(arr: np.ndarray) -> np.ndarray:
    """Replace Inf/NaN with column median."""
    arr = np.array(arr, dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    col_med = np.nanmedian(arr, axis=0)
    mask = np.isnan(arr)
    arr[mask] = np.take(col_med, np.where(mask)[1])
    return arr


if HAS_TORCH:

    class DemandDataset(Dataset):
        """
        Sliding-window dataset for a single country.
        X : [SEQ_LEN, n_features]  (normalised)
        y : scalar  (normalised Demand)
        """

        def __init__(
            self,
            df_country: pd.DataFrame,
            feature_cols: list,
            target: str,
            seq_len: int,
            scaler_X=None,
            scaler_y=None,
        ):
            sub = df_country.sort_values("Year").reset_index(drop=True)
            X_raw = clean_arr(sub[feature_cols].values)
            y_raw = sub[target].values.reshape(-1, 1).astype(np.float32)

            self.scaler_X = scaler_X or StandardScaler()
            self.scaler_y = scaler_y or StandardScaler()
            X_sc = (
                self.scaler_X.fit_transform(X_raw)
                if scaler_X is None
                else self.scaler_X.transform(X_raw)
            )
            y_sc = (
                self.scaler_y.fit_transform(y_raw).flatten()
                if scaler_y is None
                else self.scaler_y.transform(y_raw).flatten()
            )

            self.X = torch.tensor(X_sc, dtype=torch.float32)
            self.y = torch.tensor(y_sc, dtype=torch.float32)
            self.seq_len = seq_len
            self.n_samples = max(0, len(self.X) - self.seq_len)

        def __len__(self):
            return self.n_samples

        def __getitem__(self, i):
            return self.X[i : i + self.seq_len], self.y[i + self.seq_len]

    def make_loaders(df_country, feature_cols, target, seq_len, train_end, val_end, batch_size):
        tr = df_country[df_country["Year"] <= train_end]
        val = df_country[df_country["Year"] <= val_end]
        if len(tr) < seq_len + 2:
            raise ValueError(f"Insufficient training data: {len(tr)} rows")

        ds_tr = DemandDataset(tr, feature_cols, target, seq_len)
        ds_val = DemandDataset(
            val, feature_cols, target, seq_len, scaler_X=ds_tr.scaler_X, scaler_y=ds_tr.scaler_y
        )

        n_tr = len(ds_tr)
        idx_val = list(range(n_tr, len(ds_val)))
        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
        loader_val = DataLoader(
            Subset(ds_val, idx_val) if idx_val else ds_tr,
            batch_size=batch_size,
            shuffle=False,
        )
        return loader_tr, loader_val, ds_tr.scaler_X, ds_tr.scaler_y

    def _safe_loader(dataset, batch_size, shuffle=False):
        bs = max(2, min(batch_size, len(dataset)))
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def train_model(model, loader_tr, loader_val, epochs, patience, lr=1e-3, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=15, factor=0.5, verbose=False
        )
        criterion = nn.MSELoss()
        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
        tr_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            tr_loss = 0.0
            for X_b, y_b in loader_tr:
                X_b, y_b = X_b.to(device), y_b.to(device)
                opt.zero_grad()
                loss = criterion(model(X_b).squeeze(-1), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item() * len(X_b)
            tr_loss /= max(len(loader_tr.dataset), 1)

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in loader_val:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    v_loss += criterion(model(X_b).squeeze(-1), y_b).item() * len(X_b)
            v_loss = (
                v_loss / max(len(loader_val.dataset), 1) if len(loader_val.dataset) > 0 else tr_loss
            )

            tr_losses.append(tr_loss)
            val_losses.append(v_loss)
            scheduler.step(v_loss)

            if v_loss < best_val - 1e-6:
                best_val = v_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        return tr_losses, val_losses, epoch + 1


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    # ── MLP ───────────────────────────────────────────────────────────────────
    class MLPForecaster(nn.Module):
        def __init__(self, seq_len, n_features, hidden=(256, 128, 64), dropout=(0.3, 0.2)):
            super().__init__()
            in_dim = seq_len * n_features
            layers, prev = [], in_dim
            drops = list(dropout) + [0.0] * len(hidden)
            for i, h in enumerate(hidden):
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
                if drops[i] > 0:
                    layers.append(nn.Dropout(drops[i]))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x.flatten(1))

    # ── TCN ───────────────────────────────────────────────────────────────────
    class CausalConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.utils.weight_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
            )

        def forward(self, x):
            x = torch.nn.functional.pad(x, (self.padding, 0))
            return self.conv(x)

    class TCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
            super().__init__()
            self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
            self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.bn2 = nn.BatchNorm1d(out_ch)

        def forward(self, x):
            res = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.drop(out)
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.drop(out)
            return self.relu(out + res)

    class TCNForecaster(nn.Module):
        def __init__(
            self, n_features, n_channels=64, kernel_size=3, dilations=(1, 2, 4), dropout=0.2
        ):
            super().__init__()
            layers, in_ch = [], n_features
            for d in dilations:
                layers.append(TCNBlock(in_ch, n_channels, kernel_size, d, dropout))
                in_ch = n_channels
            self.tcn = nn.Sequential(*layers)
            self.linear = nn.Linear(n_channels, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.tcn(x)
            return self.linear(x[:, :, -1])

    # ── N-BEATS ───────────────────────────────────────────────────────────────
    class NBeatsBlock(nn.Module):
        def __init__(
            self, input_size, theta_size, horizon, n_layers=4, hidden=128, basis_type="generic"
        ):
            super().__init__()
            self.horizon = horizon
            self.basis_type = basis_type
            self.theta_size = theta_size
            fc_layers, prev = [], input_size
            for _ in range(n_layers):
                fc_layers += [nn.Linear(prev, hidden), nn.ReLU()]
                prev = hidden
            self.fc = nn.Sequential(*fc_layers)
            self.theta_b = nn.Linear(hidden, theta_size, bias=False)
            self.theta_f = nn.Linear(hidden, theta_size, bias=False)

        def basis_expansion(self, theta, t):
            if self.basis_type == "trend":
                p = torch.arange(self.theta_size, dtype=torch.float32, device=theta.device)
                T = t.unsqueeze(-1) ** p.unsqueeze(0)
                return torch.einsum("bt,Tt->bT", theta, T) if theta.dim() == 2 else theta @ T.t()
            else:
                L = len(t)
                if self.theta_size == L:
                    return theta
                return torch.nn.functional.interpolate(
                    theta.unsqueeze(1), size=L, mode="linear", align_corners=False
                ).squeeze(1)

        def forward(self, x, backcast_t, forecast_t):
            h = self.fc(x)
            backcast = self.basis_expansion(self.theta_b(h), backcast_t)
            forecast = self.basis_expansion(self.theta_f(h), forecast_t)
            return backcast, forecast

    class NBeatsForecaster(nn.Module):
        def __init__(self, seq_len, horizon=1, n_features=1, trend_degree=3, hidden=128):
            super().__init__()
            self.seq_len = seq_len
            self.horizon = horizon
            in_size = seq_len * n_features
            self.trend_block = NBeatsBlock(
                in_size, trend_degree + 1, horizon, n_layers=4, hidden=hidden, basis_type="trend"
            )
            self.generic_block = NBeatsBlock(
                in_size, max(horizon, 4), horizon, n_layers=4, hidden=hidden, basis_type="generic"
            )
            self.register_buffer("backcast_t", torch.linspace(-1, 1, seq_len))
            self.register_buffer("forecast_t", torch.linspace(0, 1, horizon))

        def forward(self, x):
            x_flat = x.flatten(1)
            backcast1, forecast1 = self.trend_block(x_flat, self.backcast_t, self.forecast_t)
            residual = x[:, :, 0] - backcast1
            x_res = x_flat.clone()
            x_res[:, : self.seq_len] = residual
            _, forecast2 = self.generic_block(x_res, self.backcast_t, self.forecast_t)
            return (forecast1 + forecast2).unsqueeze(-1)

    # ── TFT ───────────────────────────────────────────────────────────────────
    class GatedResidualNetwork(nn.Module):
        def __init__(self, d_in, d_hidden, d_out, dropout=0.1, context_dim=None):
            super().__init__()
            self.fc1 = nn.Linear(d_in + (context_dim or 0), d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_out * 2)
            self.gate = nn.GLU(dim=-1)
            self.norm = nn.LayerNorm(d_out)
            self.drop = nn.Dropout(dropout)
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        def forward(self, x, context=None):
            h = x if context is None else torch.cat([x, context], dim=-1)
            h = torch.nn.functional.elu(self.fc1(h))
            h = self.drop(self.gate(self.fc2(h)))
            return self.norm(h + self.skip(x))

    class VariableSelectionNetwork(nn.Module):
        def __init__(self, n_features, d_model, dropout=0.1):
            super().__init__()
            self.grns = nn.ModuleList(
                [GatedResidualNetwork(1, d_model, d_model, dropout) for _ in range(n_features)]
            )
            self.weight = GatedResidualNetwork(n_features, d_model, n_features, dropout)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            B, T, F = x.shape
            feat_emb = torch.stack([self.grns[i](x[..., i : i + 1]) for i in range(F)], dim=-2)
            weights = self.softmax(self.weight(x))
            out = (feat_emb * weights.unsqueeze(-1)).sum(dim=-2)
            return out, weights

    class TFTForecaster(nn.Module):
        def __init__(self, n_features, d_model=32, n_heads=2, seq_len=5, dropout=0.1):
            super().__init__()
            self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
            self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, dropout=0.0)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)
            self._weights = None

        def forward(self, x):
            x_sel, _ = self.vsn(x)
            enc, _ = self.lstm(x_sel)
            attn_out, weights = self.attn(enc, enc, enc, need_weights=True)
            self._weights = weights.detach().cpu()
            out = self.grn(self.norm(attn_out + enc))
            return self.head(out[:, -1, :])

        def get_attention_weights(self):
            return self._weights


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════


def predict_one_step(model, X_window, scaler_y, device) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0).to(device)
        y_sc = model(x).squeeze().cpu().item()
    return float(scaler_y.inverse_transform([[y_sc]])[0][0])


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def walk_forward_dl(
    country: str,
    df: pd.DataFrame,
    model_cls,
    model_kwargs: dict,
    feature_cols: list,
    target: str,
    seq_len: int,
    train_end: int,
    val_end: int,
    test_end: int,
    epochs: int,
    patience: int,
    lr: float = 1e-3,
    batch_size: int = 16,
    device=None,
) -> list[dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
    test_years = [y for y in sub["Year"].values if val_end < y <= test_end]
    results = []

    for t_year in test_years:
        df_tr = sub[sub["Year"] < t_year].reset_index(drop=True)
        if len(df_tr) < seq_len + 2:
            continue

        loader_tr, loader_val, scaler_X, scaler_y = make_loaders(
            df_tr,
            feature_cols,
            target,
            seq_len,
            train_end=min(train_end, t_year - 1),
            val_end=t_year - 1,
            batch_size=batch_size,
        )
        model = model_cls(**model_kwargs).to(device)
        train_model(
            model, loader_tr, loader_val, epochs=epochs, patience=patience, lr=lr, device=device
        )

        window_raw = clean_arr(df_tr.tail(seq_len)[feature_cols].values)
        window_sc = scaler_X.transform(window_raw)
        y_pred = predict_one_step(model, window_sc, scaler_y, device)
        y_true = float(sub[sub["Year"] == t_year][target].values[0])
        results.append({"Year": t_year, "y_true": y_true, "y_pred": y_pred})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# RECURSIVE FORECAST
# ══════════════════════════════════════════════════════════════════════════════


def forecast_dl_recursive(
    country: str,
    df: pd.DataFrame,
    model_cls,
    model_kwargs: dict,
    feature_cols: list,
    target: str,
    seq_len: int,
    forecast_years: list,
    epochs: int,
    patience: int,
    batch_size: int = 16,
    device=None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
    ds_full = DemandDataset(sub, feature_cols, target, seq_len)
    ds_last = DemandDataset(
        sub.tail(seq_len + 1),
        feature_cols,
        target,
        seq_len,
        scaler_X=ds_full.scaler_X,
        scaler_y=ds_full.scaler_y,
    )
    loader_tr = _safe_loader(ds_full, batch_size, shuffle=True)
    loader_val = _safe_loader(ds_last, batch_size, shuffle=False)

    model = model_cls(**model_kwargs).to(device)
    train_model(model, loader_tr, loader_val, epochs=epochs, patience=patience, device=device)

    scaler_X = ds_full.scaler_X
    scaler_y = ds_full.scaler_y
    running_raw = clean_arr(sub.tail(seq_len)[feature_cols].values.copy())
    running_sc = scaler_X.transform(running_raw)

    preds = []
    for step, _ in enumerate(forecast_years):
        y_pred = predict_one_step(model, running_sc, scaler_y, device)
        preds.append(y_pred)

        new_row = running_raw[-1].copy()
        for fi, fname in enumerate(feature_cols):
            if fname == f"{target}_lag1":
                new_row[fi] = y_pred
            elif fname == f"{target}_lag2" and step > 0:
                new_row[fi] = preds[-2]
            elif fname == f"{target}_lag3" and step > 1:
                new_row[fi] = preds[-3]

        new_row_sc = scaler_X.transform(new_row.reshape(1, -1))[0]
        running_raw = np.vstack([running_raw[1:], new_row])
        running_sc = np.vstack([running_sc[1:], new_row_sc])

    return np.array(preds)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════


def plot_loss_curves(loss_curves: dict, countries: list, model_names: list, fig_dir: str) -> None:
    fig, axes = plt.subplots(len(model_names), len(countries), figsize=(22, 3 * len(model_names)))
    if len(model_names) == 1:
        axes = axes[np.newaxis, :]

    for mi, m in enumerate(model_names):
        for ci, c in enumerate(countries):
            ax = axes[mi, ci]
            key = (c, m)
            if key in loss_curves:
                ax.plot(loss_curves[key]["train"], label="Train", lw=1.2)
                ax.plot(loss_curves[key]["val"], label="Val", lw=1.2, ls="--")
                ax.set_yscale("log")
            ax.set_title(f"{m} — {c}", fontsize=8)
            if ci == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Training Loss Curves (last test fold)", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_loss_curves.png"))


def plot_mape_heatmap(dl_metrics: pd.DataFrame, fig_dir: str) -> None:
    pivot = dl_metrics.pivot_table(index="Country", columns="Model", values="MAPE", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4, ax=ax, annot_kws={"size": 9}
    )
    ax.set_title("Test MAPE (%) per Model × Country", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_mape_heatmap.png"))


def plot_scatter(dl_res: pd.DataFrame, fig_dir: str) -> None:
    models = dl_res["Model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        sub = dl_res[dl_res["Model"] == m]
        ax.scatter(
            sub["y_actual"],
            sub["y_pred"],
            c=[PALETTE.get(c, "#999") for c in sub["Country"]],
            alpha=0.75,
            edgecolors="white",
            s=60,
        )
        lo = min(sub["y_actual"].min(), sub["y_pred"].min()) * 0.95
        hi = max(sub["y_actual"].max(), sub["y_pred"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="Perfect")
        ax.set_title(m, fontweight="bold")
        ax.set_xlabel("Actual (TWh)")
        ax.set_ylabel("Predicted (TWh)")
    fig.suptitle("Actual vs Predicted — Test Years", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_scatter_actual_vs_pred.png"))


def plot_forecast(
    dl_fc_df: pd.DataFrame,
    df: pd.DataFrame,
    train_end: int,
    val_end: int,
    test_end: int,
    fig_dir: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        col = PALETTE[country]
        hist = df[df["Area"] == country].sort_values("Year")
        frow = dl_fc_df[dl_fc_df["Country"] == country]

        tr = hist[hist["Year"] <= train_end]
        va = hist[(hist["Year"] > train_end) & (hist["Year"] <= val_end)]
        te = hist[hist["Year"] > val_end]

        ax.plot(tr["Year"], tr[TARGET], color="#3a86ff", lw=2, label="Train")
        ax.plot(va["Year"], va[TARGET], color="#ff9f1c", lw=2, label="Val")
        ax.plot(te["Year"], te[TARGET], color="#e63946", lw=2, label="Test")

        if not frow.empty:
            ax.plot(
                [hist["Year"].values[-1], frow["Year"].values[0]],
                [hist[TARGET].values[-1], frow["Forecast"].values[0]],
                color=col,
                lw=1,
                ls="--",
                alpha=0.4,
            )
            ax.plot(
                frow["Year"],
                frow["Forecast"],
                color=col,
                lw=2.5,
                ls="--",
                marker="D",
                ms=5,
                label=f"DL ({frow['Model'].values[0]})",
            )
            fc_arr = frow["Forecast"].values
            ax.fill_between(
                frow["Year"], fc_arr * 0.90, fc_arr * 1.10, color=col, alpha=0.12, label="±10% band"
            )

        ax.axvline(test_end, color="grey", ls=":", lw=0.8)
        ax.set_title(country, fontweight="bold", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    axes[-1].set_visible(False)
    fig.suptitle(
        "Deep Learning Demand Forecast\n(Best DL model per country)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_forecast_2025_2030.png"))


# ══════════════════════════════════════════════════════════════════════════════
# HOLT FALLBACK
# ══════════════════════════════════════════════════════════════════════════════


def holt_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    if not HAS_STATSMODELS:
        return np.full(horizon, series[-1])
    try:
        m = ExponentialSmoothing(series, trend="add", damped_trend=True).fit(optimized=True)
        return m.forecast(horizon)
    except Exception:
        return np.full(horizon, series[-1])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep Learning demand forecasting")
    p.add_argument("--pre_dir", default="outputs/preprocessing", help="Preprocessing output dir")
    p.add_argument("--output_dir", default="outputs/deeplearning", help="Output dir")
    p.add_argument("--quick", action="store_true", help="Quick run: 1 country, 100 epochs")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["MLP", "TCN", "N-BEATS", "TFT"],
        default=["MLP", "TCN", "N-BEATS", "TFT"],
        help="Models to run",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_until", type=int, default=None)
    p.add_argument("--forecast_until", type=int, default=None)
    return p.parse_args()


def main() -> None:
    if not HAS_TORCH:
        log.error("PyTorch is required. Install with: pip install torch")
        return

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig_dir = os.path.join(args.output_dir, "figures")
    data_dir = args.output_dir
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(args.pre_dir, "ember_model_ready.csv"))
    with open(os.path.join(args.pre_dir, "feature_meta.json")) as f:
        meta = json.load(f)

    ALL_FEATURES = meta["all_features"]
    TRAIN_END = args.train_until or meta["TRAIN_END"]
    VAL_END = TRAIN_END + 4
    TEST_END = TRAIN_END + 8
    fc_end = args.forecast_until or TEST_END + 6
    FC_YEARS = list(range(TEST_END + 1, fc_end + 1))
    N_FEAT = len(ALL_FEATURES)

    log.info("Device      : %s", device)
    log.info("Features    : %d", N_FEAT)
    log.info("Train/Val/Test: %d/%d/%d", TRAIN_END, VAL_END, TEST_END)
    log.info("Forecast    : %d–%d", FC_YEARS[0], FC_YEARS[-1])
    log.info("Models      : %s", args.models)
    log.info("Quick mode  : %s", args.quick)

    EPOCHS = 100 if args.quick else args.epochs
    COUNTRIES_ = COUNTRIES[:1] if args.quick else COUNTRIES
    SEQ_LEN = args.seq_len
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience

    # ── Model registry ────────────────────────────────────────────────────────
    MODEL_REGISTRY = {
        "MLP": {
            "cls": MLPForecaster,
            "kwargs": {
                "seq_len": SEQ_LEN,
                "n_features": N_FEAT,
                "hidden": (256, 128, 64),
                "dropout": (0.3, 0.2),
            },
        },
        "TCN": {
            "cls": TCNForecaster,
            "kwargs": {
                "n_features": N_FEAT,
                "n_channels": 64,
                "kernel_size": 3,
                "dilations": (1, 2, 4),
                "dropout": 0.2,
            },
        },
        "N-BEATS": {
            "cls": NBeatsForecaster,
            "kwargs": {
                "seq_len": SEQ_LEN,
                "horizon": 1,
                "n_features": N_FEAT,
                "trend_degree": 3,
                "hidden": 128,
            },
        },
        "TFT": {
            "cls": TFTForecaster,
            "kwargs": {
                "n_features": N_FEAT,
                "d_model": 32,
                "n_heads": 2,
                "seq_len": SEQ_LEN,
                "dropout": 0.1,
            },
        },
    }

    # ── Walk-forward evaluation ────────────────────────────────────────────────
    all_results: list[dict] = []
    loss_curves: dict = {}

    for model_name in args.models:
        reg = MODEL_REGISTRY[model_name]
        log.info("══ %s ══", model_name)
        for country in COUNTRIES_:
            log.info("  %s ...", country)
            try:
                results = walk_forward_dl(
                    country,
                    df,
                    model_cls=reg["cls"],
                    model_kwargs=reg["kwargs"],
                    feature_cols=ALL_FEATURES,
                    target=TARGET,
                    seq_len=SEQ_LEN,
                    train_end=TRAIN_END,
                    val_end=VAL_END,
                    test_end=TEST_END,
                    epochs=EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    device=device,
                )
                for r in results:
                    all_results.append(
                        {
                            "Country": country,
                            "Model": model_name,
                            "Year": r["Year"],
                            "y_actual": r["y_true"],
                            "y_pred": r["y_pred"],
                            "abs_pct_error": abs(r["y_true"] - r["y_pred"])
                            / (abs(r["y_true"]) + 1e-9)
                            * 100,
                        }
                    )
                avg_err = (
                    np.mean(
                        [
                            abs(r["y_true"] - r["y_pred"]) / (abs(r["y_true"]) + 1e-9) * 100
                            for r in results
                        ]
                    )
                    if results
                    else np.nan
                )
                log.info("  %s done — avg err %.1f%%", country, avg_err)

            except Exception as e:
                log.error("  %s FAILED: %s", country, e)

    dl_res = pd.DataFrame(all_results)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_rows = []
    for model_name in args.models:
        for country in COUNTRIES_:
            sub = dl_res[(dl_res["Country"] == country) & (dl_res["Model"] == model_name)]
            if sub.empty:
                continue
            metrics_rows.append(
                compute_metrics(country, model_name, sub["y_actual"].values, sub["y_pred"].values)
            )
    dl_metrics = pd.DataFrame(metrics_rows)

    nan_countries = dl_metrics.groupby("Country")["MAPE"].apply(lambda x: x.isna().all())
    if nan_countries.any():
        bad = nan_countries[nan_countries].index.tolist()
        print(f"⚠️  All models produced NaN MAPE for: {bad} — using Holt fallback")
        dl_metrics = dl_metrics.dropna(subset=["MAPE"])
    best_dl = dl_metrics.loc[dl_metrics.groupby("Country")["MAPE"].idxmin().dropna()]

    log.info("=== Best DL model per country ===")
    log.info("\n%s", best_dl[["Country", "Model", "MAPE", "RMSE", "R2"]].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not dl_res.empty:
        plot_mape_heatmap(dl_metrics, fig_dir)
        plot_scatter(dl_res, fig_dir)
        if loss_curves:
            plot_loss_curves(loss_curves, COUNTRIES_, args.models, fig_dir)

    # ── Recursive forecast ────────────────────────────────────────────────────
    fc_records: list[dict] = []
    log.info("── Forecasting %d–%d ──", FC_YEARS[0], FC_YEARS[-1])

    for country in COUNTRIES_:
        row = best_dl[best_dl["Country"] == country]
        if row.empty or row["Model"].values[0] not in MODEL_REGISTRY:
            log.warning("%s: no best model — Holt fallback", country)
            hist = df[df["Area"] == country].sort_values("Year")[TARGET].values.astype(float)
            fc_arr = holt_forecast(hist, len(FC_YEARS))
            for i, yr in enumerate(FC_YEARS):
                fc_records.append(
                    {
                        "Country": country,
                        "Year": yr,
                        "Forecast": round(float(fc_arr[i]), 2),
                        "Model": "Holt(fallback)",
                    }
                )
            continue

        best_m = row["Model"].values[0]
        reg = MODEL_REGISTRY[best_m]
        log.info("  %s → %s", country, best_m)
        try:
            fc = forecast_dl_recursive(
                country,
                df,
                model_cls=reg["cls"],
                model_kwargs=reg["kwargs"],
                feature_cols=ALL_FEATURES,
                target=TARGET,
                seq_len=SEQ_LEN,
                forecast_years=FC_YEARS,
                epochs=EPOCHS,
                patience=PATIENCE,
                batch_size=BATCH_SIZE,
                device=device,
            )
            for i, yr in enumerate(FC_YEARS):
                fc_records.append(
                    {
                        "Country": country,
                        "Year": yr,
                        "Forecast": round(float(fc[i]), 2),
                        "Model": best_m,
                    }
                )
        except Exception as e:
            log.error("  %s forecast failed: %s — Holt fallback", country, e)
            hist = df[df["Area"] == country].sort_values("Year")[TARGET].values.astype(float)
            fc_arr = holt_forecast(hist, len(FC_YEARS))
            for i, yr in enumerate(FC_YEARS):
                fc_records.append(
                    {
                        "Country": country,
                        "Year": yr,
                        "Forecast": round(float(fc_arr[i]), 2),
                        "Model": f"{best_m}→Holt",
                    }
                )

    dl_fc_df = pd.DataFrame(fc_records)

    if not dl_fc_df.empty:
        plot_forecast(dl_fc_df, df, TRAIN_END, VAL_END, TEST_END, fig_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    dl_metrics.to_csv(os.path.join(data_dir, "dl_benchmarking.csv"), index=False)
    best_dl.to_csv(os.path.join(data_dir, "dl_best_models.csv"), index=False)
    dl_res.to_csv(os.path.join(data_dir, "dl_wf_predictions.csv"), index=False)
    dl_fc_df.to_csv(os.path.join(data_dir, "dl_forecast_2025_2030.csv"), index=False)

    summary = {
        "best_models": best_dl[["Country", "Model", "MAPE"]].to_dict("records"),
        "mean_test_mape": float(best_dl["MAPE"].mean()) if not best_dl.empty else None,
        "forecast_years": FC_YEARS,
    }
    with open(os.path.join(data_dir, "dl_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=== Deep Learning complete ===")
    log.info("  dl_benchmarking.csv      : %s", dl_metrics.shape)
    log.info("  dl_best_models.csv       : %s", best_dl.shape)
    log.info("  dl_wf_predictions.csv    : %s", dl_res.shape)
    log.info("  dl_forecast_2025_2030.csv: %s", dl_fc_df.shape)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
