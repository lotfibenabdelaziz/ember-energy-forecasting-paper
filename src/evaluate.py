"""
src/evaluate.py — Metrics & Walk-Forward Evaluation
====================================================
Ember Energy | IEEE Paper — Deep Learning module

Mirrors notebook 04_deeplearning_enhanced_patched.ipynb sections:
    - mape(), rmse(), smape()
    - predict_one_step()
    - walk_forward_dl()
    - benchmarking aggregation (best model per country)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch

from src.dataset import clean_arr, make_loaders
from src.train import train_model

log = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (excludes zero actuals)."""
    yt, yp = np.array(y_true), np.array(y_pred)
    mask = yt != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE — bounded, robust to near-zero values."""
    yt, yp = np.array(y_true), np.array(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100)


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


# ── Single-step prediction ────────────────────────────────────────────────────


def predict_one_step(model, X_window: np.ndarray, scaler_y, device: torch.device) -> float:
    """
    Run one forward pass; inverse-transform the output.
    Mirrors notebook predict_one_step() exactly.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0).to(device)
        y_sc = model(x).squeeze().cpu().item()
    return float(scaler_y.inverse_transform([[y_sc]])[0][0])


# ── Walk-forward evaluation ────────────────────────────────────────────────────


def walk_forward_dl(
    country: str,
    df: pd.DataFrame,
    model_cls,
    model_kwargs: dict,
    feature_cols: list[str],
    target: str,
    seq_len: int,
    train_end: int,
    val_end: int,
    test_end: int,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    verbose: bool = False,
) -> list[dict]:
    """
    Expanding walk-forward over test years.

    For each test year t:
      1. Train on 2000..(t-1), validate on (train_end+1)..(t-1)
      2. Build input window from last seq_len steps
      3. Predict year t

    Mirrors notebook walk_forward_dl() exactly.
    Returns list of dicts: {Year, y_actual, y_pred, tr_loss, val_loss}
    """
    sub = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
    years = sub["Year"].values
    results: list[dict] = []

    test_years = [y for y in years if val_end < y <= test_end]

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

        tr_losses, val_losses, _ep = train_model(
            model,
            loader_tr,
            loader_val,
            epochs=epochs,
            patience=patience,
            lr=lr,
            device=device,
        )

        window_raw = clean_arr(df_tr.tail(seq_len)[feature_cols].values)
        window_sc = scaler_X.transform(window_raw)

        y_pred = predict_one_step(model, window_sc, scaler_y, device)
        y_true = float(sub[sub["Year"] == t_year][target].values[0])

        results.append(
            {
                "Year": t_year,
                "y_actual": y_true,
                "y_pred": y_pred,
                "tr_loss": tr_losses,
                "val_loss": val_losses,
            }
        )

        if verbose:
            err = abs(y_true - y_pred) / y_true * 100 if y_true != 0 else float("nan")
            log.info("    %d: actual=%.2f  pred=%.2f  err=%.1f%%", t_year, y_true, y_pred, err)

    return results


def run_all_walk_forward(
    df: pd.DataFrame,
    countries: list[str],
    model_registry: dict,
    feature_cols: list[str],
    target: str,
    seq_len: int,
    train_end: int,
    val_end: int,
    test_end: int,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Run walk-forward for all models x all countries.
    Mirrors notebook Section 7 driver loop exactly.

    Returns
    -------
    dl_res       : DataFrame — long format {Country, Model, Year, y_actual, y_pred, abs_pct_error}
    loss_curves  : dict {(country, model_name): {"train": [...], "val": [...]}}
    model_timing : dict {model_name: seconds}
    """
    all_dl_results: list[dict] = []
    loss_curves: dict = {}
    model_timing: dict = {}

    import time

    t0 = time.time()

    for model_name, reg in model_registry.items():
        t_model = time.time()
        log.info("══ %s ══", model_name)

        for country in countries:
            try:
                results = walk_forward_dl(
                    country,
                    df,
                    reg["cls"],
                    reg["kwargs"],
                    feature_cols,
                    target,
                    seq_len,
                    train_end,
                    val_end,
                    test_end,
                    epochs,
                    patience,
                    lr,
                    batch_size,
                    device,
                )

                for r in results:
                    all_dl_results.append(
                        {
                            "Country": country,
                            "Model": model_name,
                            "Year": r["Year"],
                            "y_actual": r["y_actual"],
                            "y_pred": r["y_pred"],
                            "abs_pct_error": (
                                abs(r["y_actual"] - r["y_pred"]) / (abs(r["y_actual"]) + 1e-9) * 100
                            ),
                        }
                    )

                if results:
                    loss_curves[(country, model_name)] = {
                        "train": results[-1]["tr_loss"],
                        "val": results[-1]["val_loss"],
                    }

                avg_err = (
                    np.mean(
                        [
                            abs(r["y_actual"] - r["y_pred"]) / (abs(r["y_actual"]) + 1e-9) * 100
                            for r in results
                        ]
                    )
                    if results
                    else float("nan")
                )
                log.info("  %-10s done (%d steps, avg err %.1f%%)", country, len(results), avg_err)

            except Exception as e:
                log.error("  %-10s ERROR: %s", country, e)

        model_timing[model_name] = time.time() - t_model

    elapsed = time.time() - t0
    log.info("Total walk-forward elapsed: %.1f min", elapsed / 60)

    dl_res = pd.DataFrame(all_dl_results)
    return dl_res, loss_curves, model_timing


# ── Benchmarking aggregation ──────────────────────────────────────────────────


def compute_benchmarking(dl_res: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (Country, Model) metrics from walk-forward results.
    Mirrors notebook Section 9 exactly.
    """
    rows = []
    for (country, model), g in dl_res.groupby(["Country", "Model"]):
        g = g[["y_actual", "y_pred"]].dropna()
        if len(g) == 0:
            continue
        rows.append(
            {
                "Country": country,
                "Model": model,
                "MAE": mae(g["y_actual"], g["y_pred"]),
                "RMSE": rmse(g["y_actual"], g["y_pred"]),
                "MAPE": mape(g["y_actual"], g["y_pred"]),
                "SMAPE": smape(g["y_actual"], g["y_pred"]),
                "N": len(g),
            }
        )
    return pd.DataFrame(rows)


def compute_best_models(dl_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Best model per country by lowest Test MAPE.
    Mirrors notebook best_dl computation exactly.
    """
    if dl_metrics.empty:
        return dl_metrics
    return dl_metrics.loc[dl_metrics.groupby("Country")["MAPE"].idxmin()].reset_index(drop=True)


def compare_with_classical(
    best_dl: pd.DataFrame, classical_benchmark: pd.DataFrame
) -> pd.DataFrame:
    """
    DL vs Classical head-to-head — mirrors notebook Section 9 comparison.

    classical_benchmark: DataFrame with columns [Country, Model, MAPE]
    (typically from outputs/modeling/test_benchmarking.csv)
    """
    classic_best = classical_benchmark.loc[
        classical_benchmark.groupby("Country")["MAPE"].idxmin(), ["Country", "Model", "MAPE"]
    ].rename(columns={"Model": "Best_Classic", "MAPE": "Classic_MAPE"})

    dl_best_merge = best_dl[["Country", "Model", "MAPE"]].rename(
        columns={"Model": "Best_DL", "MAPE": "DL_MAPE"}
    )

    compare = classic_best.merge(dl_best_merge, on="Country")
    compare["Winner"] = compare.apply(
        lambda r: r["Best_DL"] if r["DL_MAPE"] < r["Classic_MAPE"] else r["Best_Classic"],
        axis=1,
    )
    compare["DL_gain_%"] = compare["Classic_MAPE"] - compare["DL_MAPE"]
    return compare
