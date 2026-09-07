"""
src/train.py — Training Loop with Early Stopping + MLflow Logging
==================================================================
Ember Energy | IEEE Paper — Deep Learning module

Mirrors notebook 04_deeplearning_enhanced_patched.ipynb Section 2:
    - Adam optimiser + MSELoss
    - ReduceLROnPlateau scheduler
    - Gradient clipping (norm=1.0)
    - Early stopping (restore best weights)
    - Optional MLflow run logging
"""

from __future__ import annotations

import copy
import logging
import os

# mlflow imported lazily inside logging functions only
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Fail fast instead of retrying for ~90s when MLflow server is unreachable
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

log = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    loader_tr: DataLoader,
    loader_val: DataLoader,
    epochs: int = 300,
    patience: int = 40,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = False,
    desc: str = "",
) -> tuple[list[float], list[float], int]:
    """
    Train with Adam + MSE loss.

    • Early stopping on val loss (restore best weights)
    • ReduceLROnPlateau scheduler
    • Gradient clipping (norm=1.0)

    Mirrors notebook train_model() exactly.

    Returns
    -------
    tr_losses  : list of per-epoch training MSE loss
    val_losses : list of per-epoch validation MSE loss
    epochs_run : number of epochs actually run (early stopping)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    tr_losses: list[float] = []
    val_losses: list[float] = []
    epoch = 0

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for X_b, y_b in loader_tr:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            pred = model(X_b).squeeze(-1)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(X_b)
        # DataLoader.dataset is typed as Dataset, whose stubs don't declare
        # __len__ as required — every dataset actually used here does.
        tr_loss /= max(len(loader_tr.dataset), 1)  # type: ignore[arg-type]

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in loader_val:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b).squeeze(-1)
                v_loss += criterion(pred, y_b).item() * len(X_b)

        n_val = (
            len(loader_val.dataset)  # type: ignore[arg-type]
            if len(loader_val.dataset) > 0  # type: ignore[arg-type]
            else 1
        )
        v_loss = v_loss / n_val if n_val > 0 else tr_loss

        tr_losses.append(tr_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if verbose and epoch % 50 == 0:
            log.debug("%s  epoch %4d  tr=%.4f  val=%.4f", desc, epoch, tr_loss, v_loss)

        # ── Early stopping ─────────────────────────────────────────────
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


def log_dl_run_to_mlflow(
    country: str,
    model_name: str,
    metrics: dict,
    params: dict,
    run_name: str | None = None,
    experiment: str = "ember-demand-forecasting",
) -> None:
    """
    Log one DL walk-forward result to MLflow.

    metrics: {"MAE": ..., "RMSE": ..., "MAPE": ..., "SMAPE": ...}
    params:  {"seq_len": ..., "epochs": ..., "patience": ..., ...}
    """
    try:
        import mlflow  # lazy import — optional dependency

        mlflow.set_experiment(experiment)
        run_name = run_name or f"{country}_{model_name}_DL"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("country", country)
            mlflow.set_tag("model_type", "deep_learning")
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})
    except ImportError:
        log.debug("mlflow not installed — skipping DL run logging")
    except Exception as e:
        log.warning("MLflow logging failed for %s %s: %s", country, model_name, e)


def log_all_dl_to_mlflow(
    dl_metrics: pd.DataFrame,
    params: dict,
    experiment: str = "ember-demand-forecasting",
) -> None:
    """
    Log all DL benchmarking results to MLflow — one run per (country, model).
    Mirrors mlflow_config.log_dl_model_metrics() logic.
    """

    for _, row in dl_metrics.iterrows():
        metrics = {
            "MAE": row.get("MAE", float("nan")),
            "RMSE": row.get("RMSE", float("nan")),
            "MAPE": row.get("MAPE", float("nan")),
            "SMAPE": row.get("SMAPE", float("nan")),
        }
        log_dl_run_to_mlflow(
            country=row["Country"],
            model_name=row["Model"],
            metrics=metrics,
            params=params,
            experiment=experiment,
        )
