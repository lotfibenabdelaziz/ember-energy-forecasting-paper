"""
src/04_forecasting.py — Demand Forecasting 2025–2030
Ember Energy | IEEE Paper

Models  : Best ML model per country (from 03_modeling)
          + statistical fallbacks (Naive, LinearTrend, Holt, ARIMA)
Horizon : 2025–2030 (6 steps)
CI      : Residual-bootstrap (N_BOOT=300, CI=90%)

Reads:  outputs/preprocessing/ember_model_ready.csv
        outputs/modeling/best_models.csv
        outputs/modeling/best_hp.json
        outputs/modeling/ember_config.json
Writes: outputs/forecasting/demand_forecast_2025_2030.csv
        outputs/forecasting/demand_growth_summary.csv
        outputs/forecasting/figures/*.pdf
        outputs/forecasting/figures/forecast_per_country.png
        outputs/forecasting/figures/forecast_overlay.png
        outputs/forecasting/figures/growth_uncertainty.png

Conversion notes (notebook → script)
--------------------------------------
* clean_X()                  — ported verbatim from notebook cell 1
* build_future_feature_row() — ported verbatim; correctly recomputes lag-1/2/3,
                               MA-3/5, YoY for demand AND every exog col each step
* forecast_ml_recursive()    — fits model ONCE on full history, then walks forward
                               using build_future_feature_row (matches notebook)
* Holt model                 — ExponentialSmoothing(trend='add', damped_trend=True)
                               (matches notebook; NOT SimpleExpSmoothing)
* Naive / LinearTrend        — both handled by LinearRegression (matches notebook)
* Bootstrap                  — residual-bootstrap via get_insample_residuals()
                               (matches notebook; NOT case-resampling)
* PALETTE                    — hex values copied from notebook
* Per-country subplot grid   — reproduced as forecast_per_country.png
* _safe_float_array()        — kept as extra dtype guard (not in notebook but
                               harmless; wraps clean_X for DataFrame inputs)
* _append_row()              — kept; used only in the dtype-safe history append
"""

import argparse
import json
import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # ← matches notebook

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]

# Palette copied verbatim from notebook
PALETTE = {
    "Tunisia": "#e63946",
    "Austria": "#2196F3",
    "Germany": "#4CAF50",
    "Egypt": "#9C27B0",
    "Canada": "#00BCD4",
    "France": "#797148",
    "Kuwait": "#4C4879",
}

TARGET = "Demand"
FORECAST_YEARS = list(range(2025, 2031))
N_BOOT = 300
CI = 90

# Model dispatch table — mirrors notebook cell "Bootstrap CI"
STAT_MODELS = {"Naive", "Naïve", "LinearTrend", "Holt", "ARIMA(1,1,1)", "ARIMA_1_1_1"}
ML_CLS_MAP: dict = {
    "Ridge": (Ridge, True),  # (class, needs_scaler)
    "RandomForest": (RandomForestRegressor, False),
    "XGBoost": (None, False),  # filled after HAS_XGB check
}
if HAS_XGB:
    ML_CLS_MAP["XGBoost"] = (xgb.XGBRegressor, False)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", palette="tab10")
pd.set_option("display.float_format", "{:.2f}".format)


# =============================================================================
# Array / DataFrame helpers
# =============================================================================


def clean_X(arr: np.ndarray) -> np.ndarray:
    """
    Replace Inf/-Inf with NaN, impute column medians. Safe for model input.
    Ported verbatim from notebook cell 1.
    """
    arr = np.array(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    col_medians = np.nanmedian(arr, axis=0)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return arr


def savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# =============================================================================
# Exogenous feature extrapolation
# =============================================================================


def extrapolate_exog(history_df: pd.DataFrame, exog_cols: list, n_steps: int) -> dict:
    """
    For each exogenous column fit a linear trend on history and project
    n_steps forward. Returns dict: {col: ndarray[n_steps]}.
    Ported verbatim from notebook.
    """
    preds: dict = {}
    n = len(history_df)
    for col in exog_cols:
        if col not in history_df.columns:
            continue
        y = history_df[col].values.astype(float)
        y = np.where(np.isfinite(y), y, np.nanmedian(y))  # sanitize
        X = np.arange(n).reshape(-1, 1)
        try:
            m = LinearRegression().fit(X, y)
            preds[col] = m.predict(np.arange(n, n + n_steps).reshape(-1, 1))
        except Exception:
            preds[col] = np.full(n_steps, y[-1])
    return preds


# =============================================================================
# Future feature-row builder  ← KEY function ported from notebook
# =============================================================================


def build_future_feature_row(
    history_df: pd.DataFrame,
    exog_future: dict,
    demand_preds_so_far: list,
    step: int,
    all_feature_cols: list,
    target_col: str,
) -> dict:
    """
    Build a single feature row for *step* (0-indexed) into the future.

    Uses actual history + demand_preds_so_far (already predicted steps).
    Recomputes lag-1/2/3, MA-3/5, YoY for BOTH the demand target and every
    exogenous column, then fills trend/trend_sq/country_code.

    Ported verbatim from notebook — this is the function that was missing
    from the previous .py conversion, causing stale lag values.
    """
    # Full demand series: history + predictions so far
    hist_demand = list(history_df[target_col].values.astype(float))
    full_demand = hist_demand + list(demand_preds_so_far)  # length = n_hist + step
    n_full = len(full_demand)

    row: dict = {}

    # ── Lag features for target ───────────────────────────────────────────────
    for lag in [1, 2, 3]:
        k = f"{target_col}_lag{lag}"
        if k in all_feature_cols:
            idx = n_full - lag
            row[k] = full_demand[idx] if idx >= 0 else np.nan

    # ── Rolling MAs for target (shift=1 already baked in: use full_demand) ───
    for w in [3, 5]:
        k = f"{target_col}_ma{w}"
        if k in all_feature_cols:
            window = full_demand[max(0, n_full - w):n_full]
            row[k] = float(np.mean(window)) if window else np.nan

    # ── YoY for target ────────────────────────────────────────────────────────
    k = f"{target_col}_yoy"
    if k in all_feature_cols:
        if n_full >= 2 and full_demand[n_full - 2] != 0:
            row[k] = (
                (full_demand[n_full - 1] - full_demand[n_full - 2])
                / abs(full_demand[n_full - 2])
                * 100
            )
        else:
            row[k] = 0.0

    # ── Exog features (each with their projected future values) ──────────────
    for col, future_vals in exog_future.items():
        hist_exog = list(history_df[col].values.astype(float))
        # full exog series up to current step (inclusive)
        full_exog = hist_exog + list(future_vals[: step + 1])
        n_ex = len(full_exog)

        for lag in [1, 2, 3]:
            k = f"{col}_lag{lag}"
            if k in all_feature_cols:
                idx = n_ex - lag
                row[k] = full_exog[idx] if idx >= 0 else np.nan

        for w in [3, 5]:
            k = f"{col}_ma{w}"
            if k in all_feature_cols:
                window = full_exog[max(0, n_ex - w):n_ex]
                row[k] = float(np.mean(window)) if window else np.nan

        k = f"{col}_yoy"
        if k in all_feature_cols:
            if n_ex >= 2 and full_exog[n_ex - 2] != 0:
                row[k] = (
                    (full_exog[n_ex - 1] - full_exog[n_ex - 2]) / abs(full_exog[n_ex - 2]) * 100
                )
            else:
                row[k] = 0.0

    # ── Trend features ────────────────────────────────────────────────────────
    year_min = int(history_df["Year"].min())
    future_year = int(history_df["Year"].max()) + step + 1
    row["trend"] = future_year - year_min
    row["trend_sq"] = row["trend"] ** 2

    if "country_code" in all_feature_cols and "country_code" in history_df.columns:
        row["country_code"] = int(history_df["country_code"].iloc[0])

    # ── Fill any remaining feature columns with 0 ─────────────────────────────
    for col in all_feature_cols:
        if col not in row:
            row[col] = 0.0

    return row


# =============================================================================
# Statistical forecast
# =============================================================================


def forecast_statistical(series: np.ndarray, model_name: str, horizon: int) -> np.ndarray:
    """
    Statistical multi-step forecast (Holt, ARIMA, LinearTrend, Naive).

    NOTE — matches notebook: both 'Naive' and 'LinearTrend' use LinearRegression
    (notebook treats Naive as a linear-trend model, not last-value repeat).
    Holt uses ExponentialSmoothing with trend='add', damped_trend=True.
    """
    series = np.array(series, dtype=float)

    if model_name in {"Naive", "Naïve", "LinearTrend"}:
        # Notebook behaviour: both use LinearRegression
        X = np.arange(len(series)).reshape(-1, 1)
        m = LinearRegression().fit(X, series)
        return m.predict(np.arange(len(series), len(series) + horizon).reshape(-1, 1)).flatten()

    elif model_name == "Holt":
        try:
            # ExponentialSmoothing with damped linear trend — matches notebook
            m = ExponentialSmoothing(series, trend="add", damped_trend=True).fit(optimized=True)
            return np.array(m.forecast(horizon))
        except Exception:
            return np.full(horizon, series[-1])

    elif model_name in {"ARIMA(1,1,1)", "ARIMA_1_1_1"}:
        try:
            m = ARIMA(series, order=(1, 1, 1)).fit()
            return m.get_forecast(steps=horizon).predicted_mean.values
        except Exception:
            return np.full(horizon, series[-1])

    return np.full(horizon, series[-1])


# =============================================================================
# ML recursive forecast  ← matches notebook's forecast_ml_recursive()
# =============================================================================


def forecast_ml_recursive(
    history_df: pd.DataFrame,
    model_cls,
    all_feature_cols: list,
    exog_cols: list,
    horizon: int,
    scaler=None,
    **kwargs,
) -> np.ndarray:
    """
    Recursive ML multi-step forecast.

    1. Fit model ONCE on full clean history.
    2. Extrapolate exog features forward (linear trend per feature).
    3. At each step: build feature row via build_future_feature_row()
       (recomputes all lags/MAs from history + predictions so far),
       clean Inf/NaN with clean_X(), predict, append to running list.

    Matches notebook's forecast_ml_recursive() exactly.
    """
    # ── Fit ───────────────────────────────────────────────────────────────────
    tr = history_df[all_feature_cols + [TARGET]].copy()
    tr = tr[tr[TARGET].notna()]
    X_tr = clean_X(tr[all_feature_cols].values)
    y_tr = tr[TARGET].values.astype(float)

    if len(X_tr) < 3:
        raise ValueError("Not enough training rows after cleaning")

    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)

    model = model_cls(**kwargs)
    model.fit(X_tr, y_tr)

    # ── Extrapolate exog ──────────────────────────────────────────────────────
    valid_exog = [c for c in exog_cols if c in history_df.columns]
    exog_future = extrapolate_exog(history_df, valid_exog, horizon)

    # ── Recursive prediction ──────────────────────────────────────────────────
    demand_preds: list[float] = []

    for step in range(horizon):
        feat_row = build_future_feature_row(
            history_df, exog_future, demand_preds, step, all_feature_cols, TARGET
        )
        x_pred = np.array([[feat_row[c] for c in all_feature_cols]], dtype=float)
        x_pred = clean_X(x_pred)  # final Inf/NaN guard

        if scaler is not None:
            x_pred = scaler.transform(x_pred)

        p = float(model.predict(x_pred)[0])
        demand_preds.append(p)

    return np.array(demand_preds, dtype=np.float64)


# =============================================================================
# Point-forecast dispatcher
# =============================================================================


def _point_forecast(
    history_df: pd.DataFrame,
    model_name: str,
    all_feature_cols: list,
    exog_cols: list,
    params: dict,
    horizon: int,
) -> np.ndarray:
    """Single point forecast dispatcher — stat or ML. Mirrors notebook."""
    ts = history_df[TARGET].values.astype(float)

    if model_name in STAT_MODELS:
        return forecast_statistical(ts, model_name, horizon)

    cls, needs_scaler = ML_CLS_MAP.get(model_name, (None, False))
    if cls is None:
        log.warning("Model class unavailable for '%s' — falling back to LinearTrend", model_name)
        return forecast_statistical(ts, "LinearTrend", horizon)

    scaler = StandardScaler() if needs_scaler else None
    kw = dict(params)
    if model_name == "XGBoost":
        kw.setdefault("tree_method", "hist")
        kw.setdefault("verbosity", 0)

    return forecast_ml_recursive(
        history_df,
        cls,
        all_feature_cols,
        exog_cols,
        horizon,
        scaler=scaler,
        **kw,
    )


# =============================================================================
# In-sample residuals  ← ported from notebook
# =============================================================================


def get_insample_residuals(
    history_df: pd.DataFrame,
    model_name: str,
    all_feature_cols: list,
    exog_cols: list,
    params: dict,
    n_eval: int = 8,
) -> np.ndarray:
    """
    Walk-forward 1-step residuals on the last n_eval years of history.
    Ported verbatim from notebook.
    """
    residuals: list[float] = []
    n = len(history_df)
    start = max(10, n - n_eval)

    for t in range(start, n):
        tr = history_df.iloc[:t].reset_index(drop=True)
        y_t = float(history_df.iloc[t][TARGET])
        try:
            fc1 = _point_forecast(tr, model_name, all_feature_cols, exog_cols, params, 1)
            residuals.append(y_t - float(fc1[0]))
        except Exception:
            pass

    return np.array(residuals)


# =============================================================================
# Residual-bootstrap CI  ← matches notebook's bootstrap_forecast_ci()
# =============================================================================


def bootstrap_forecast_ci(
    history_df: pd.DataFrame,
    model_name: str,
    all_feature_cols: list,
    exog_cols: list,
    params: dict,
    horizon: int,
    n_boot: int = N_BOOT,
    ci: int = CI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Residual-bootstrap confidence intervals for multi-step forecast.
    Returns (point_forecast, lower, upper) each of length horizon.

    Matches notebook: perturbs the target series with resampled in-sample
    residuals, then re-forecasts N_BOOT times.
    """
    # Point forecast
    base_fc = _point_forecast(history_df, model_name, all_feature_cols, exog_cols, params, horizon)

    # In-sample residuals
    residuals = get_insample_residuals(history_df, model_name, all_feature_cols, exog_cols, params)

    ts = history_df[TARGET].values.astype(float)
    alpha = (100 - ci) / 2

    if len(residuals) < 3:
        std = np.std(ts) * 0.10
        return base_fc, base_fc - 1.645 * std, base_fc + 1.645 * std

    boot_fcs: list[np.ndarray] = []

    for _ in range(n_boot):
        noise = np.random.choice(residuals, size=len(ts), replace=True)
        ts_pert = np.clip(ts + noise * 0.5, ts.min() * 0.3, None)
        h_pert = history_df.copy()
        h_pert[TARGET] = ts_pert
        try:
            fc_b = _point_forecast(h_pert, model_name, all_feature_cols, exog_cols, params, horizon)
            boot_fcs.append(fc_b)
        except Exception:
            boot_fcs.append(base_fc)

    boot_arr = np.array(boot_fcs, dtype=np.float64)
    return (
        base_fc,
        np.percentile(boot_arr, alpha, axis=0),
        np.percentile(boot_arr, 100 - alpha, axis=0),
    )


# =============================================================================
# Plots
# =============================================================================


def plot_per_country(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    model_used: dict,
    fig_dir: str,
    train_end: int = 2016,
    val_end: int = 2020,
) -> None:
    """
    2×4 subplot grid — one panel per country, train/val/test colour-coded.
    Matches notebook section 5.
    """
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        col = clrs[i % len(clrs)]
        hist = df[df["Area"] == country].sort_values("Year")
        frow = fc_df[fc_df["Country"] == country]
        model_label = model_used.get(country, "?")

        tr = hist[hist["Year"] <= train_end]
        va = hist[(hist["Year"] > train_end) & (hist["Year"] <= val_end)]
        te = hist[hist["Year"] > val_end]

        ax.plot(tr["Year"], tr[TARGET], color="#3a86ff", lw=2, label="Train")
        ax.plot(va["Year"], va[TARGET], color="#ff9f1c", lw=2, label="Val")
        ax.plot(te["Year"], te[TARGET], color="#e63946", lw=2, label="Test")

        # Bridge last actual → first forecast
        ax.plot(
            [hist["Year"].values[-1], frow["Year"].values[0]],
            [hist[TARGET].values[-1], frow["Forecast"].values[0]],
            color=col,
            lw=1.5,
            ls="--",
            alpha=0.5,
        )

        ax.plot(
            frow["Year"],
            frow["Forecast"],
            color=col,
            lw=2.5,
            ls="--",
            marker="D",
            ms=5,
            label=f"Forecast ({model_label})",
        )
        ax.fill_between(
            frow["Year"],
            frow["Lower_90"],
            frow["Upper_90"],
            alpha=0.18,
            color=col,
            label="90% CI",
        )

        ax.axvspan(train_end + 0.5, val_end + 0.5, alpha=0.05, color="orange")
        ax.axvspan(val_end + 0.5, val_end + 4.5, alpha=0.05, color="red")
        ax.axvline(val_end + 4, color="grey", ls=":", lw=1)

        ax.set_title(country, fontweight="bold", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    axes[-1].set_visible(False)
    fig.suptitle(
        "Electricity Demand Forecast 2025–2030\n" "(Train/Val/Test + Recursive Forecast, 90% CI)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "forecast_per_country.png"))


def plot_overlay(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    model_used: dict,
    test_end: int,
    fig_dir: str,
) -> None:
    """All-country overlay. Matches notebook section 5."""
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, country in enumerate(COUNTRIES):
        col = clrs[i % len(clrs)]
        hist = df[df["Area"] == country].sort_values("Year")
        frow = fc_df[fc_df["Country"] == country]
        ax.plot(hist["Year"], hist[TARGET], color=col, lw=1.5, alpha=0.6)
        ax.plot(
            frow["Year"],
            frow["Forecast"],
            color=col,
            lw=2.5,
            ls="--",
            label=country,
        )
        ax.fill_between(
            frow["Year"],
            frow["Lower_90"],
            frow["Upper_90"],
            alpha=0.09,
            color=col,
        )

    ax.axvline(test_end, color="black", ls=":", lw=1.2)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("TWh", fontsize=12)
    ax.set_title(
        "All Countries — Demand History & Forecast 2025–2030",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(ncol=2, fontsize=9, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "forecast_overlay.png"))


def plot_growth_uncertainty(
    growth_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """CAGR bar + CI-width heatmap. Matches notebook section 6."""
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CAGR bar
    ax = axes[0]
    cagr_col = "CAGR 24-30 %"
    bars = ax.bar(
        growth_df["Country"],
        growth_df[cagr_col],
        color=[clrs[i] for i in range(len(growth_df))],
        edgecolor="white",
    )
    for bar, val in zip(bars, growth_df[cagr_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Electricity Demand CAGR 2024–2030", fontsize=12, fontweight="bold")
    ax.set_ylabel("CAGR (%)")
    ax.tick_params(axis="x", rotation=30)

    # CI-width heatmap
    fc_df = fc_df.copy()
    fc_df["CI_Width"] = fc_df["Upper_90"] - fc_df["Lower_90"]
    ci_piv = fc_df.pivot(index="Country", columns="Year", values="CI_Width")
    sns.heatmap(
        ci_piv,
        cmap="OrRd",
        annot=True,
        fmt=".1f",
        ax=axes[1],
        linewidths=0.3,
        cbar_kws={"label": "CI Width (TWh)"},
    )
    axes[1].set_title(
        "Forecast Uncertainty — 90% CI Width per Year",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "growth_uncertainty.png"))


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember demand forecasting 2025–2030")
    p.add_argument("--pre_dir", default="outputs/preprocessing", help="Preprocessing dir")
    p.add_argument("--model_dir", default="outputs/modeling", help="Modeling dir")
    p.add_argument("--output_dir", default="outputs/forecasting", help="Output dir")
    p.add_argument(
        "--forecast_until",
        type=int,
        default=None,
        help="Last forecast year (default: taken from ember_config.json)",
    )
    p.add_argument("--train_end", type=int, default=2016, help="Last training year")
    p.add_argument("--val_end", type=int, default=2020, help="Last validation year")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load artifacts ────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(args.pre_dir, "ember_model_ready.csv"))

    with open(os.path.join(args.model_dir, "ember_config.json")) as f:
        meta = json.load(f)

    best_df = pd.read_csv(os.path.join(args.model_dir, "best_models.csv"))

    with open(os.path.join(args.model_dir, "best_hp.json")) as f:
        best_hp = json.load(f)

    ALL_FEATURES: list = meta["all_features"]
    FEATURES: list = meta.get("raw_features", meta.get("FEATURES", []))
    TEST_END: int = meta["TEST_END"]

    _fc_end = (
        args.forecast_until if args.forecast_until else max(meta.get("FORECAST_YEARS", [2030]))
    )
    FORECAST_YEARS_RUN = list(range(TEST_END + 1, _fc_end + 1))
    horizon = len(FORECAST_YEARS_RUN)

    # Build PARAMS_MAP — defaults first, then overlay tuned HPs from best_hp.json
    PARAMS_MAP: dict = {
        "Ridge": {"alpha": 1.0},
        "RandomForest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "XGBoost": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "tree_method": "hist",
            "verbosity": 0,
            "random_state": 42,
        },
        "Holt": {},
        "ARIMA(1,1,1)": {},
        "ARIMA_1_1_1": {},
        "LinearTrend": {},
        "Naive": {},
        "Naïve": {},
    }
    for k, v in best_hp.items():
        PARAMS_MAP[k] = v
    log.info("Hyperparameters loaded from best_hp.json ✓")

    fc_records: list = []
    model_used: dict = {}

    log.info(
        "=== Forecasting %d–%d (N_BOOT=%d, CI=%d%%) ===",
        TEST_END + 1,
        _fc_end,
        N_BOOT,
        CI,
    )

    for country in COUNTRIES:
        hist_df = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
        if hist_df.empty:
            log.warning("No data for %s — skipping.", country)
            continue

        row = best_df[best_df["Country"] == country]
        best_model = row["Model"].values[0] if not row.empty else "Holt"
        params = PARAMS_MAP.get(best_model, {})

        log.info("  %-12s | %-16s ...", country, best_model)

        fc, lo, hi = None, None, None
        used_model = best_model

        try:
            fc, lo, hi = bootstrap_forecast_ci(
                hist_df,
                best_model,
                ALL_FEATURES,
                FEATURES,
                params,
                horizon,
                n_boot=N_BOOT,
                ci=CI,
            )
            log.info("    ✓ done")

        except Exception as e1:
            log.warning("    Best model failed (%s: %s) — trying Holt …", type(e1).__name__, e1)
            try:
                fc, lo, hi = bootstrap_forecast_ci(
                    hist_df,
                    "Holt",
                    ALL_FEATURES,
                    FEATURES,
                    {},
                    horizon,
                    n_boot=N_BOOT,
                    ci=CI,
                )
                used_model = "Holt (fallback)"
                log.info("    ✓ Holt fallback succeeded")
            except Exception as e2:
                log.warning("    Holt failed too (%s) — using LinearTrend last resort", e2)
                ts = hist_df[TARGET].values.astype(float)
                fc = forecast_statistical(ts, "LinearTrend", horizon)
                std = np.std(ts) * 0.10
                lo = fc - 1.645 * std
                hi = fc + 1.645 * std
                used_model = "LinearTrend (last resort)"

        model_used[country] = used_model

        for i, yr in enumerate(FORECAST_YEARS_RUN):
            fc_records.append(
                {
                    "Country": country,
                    "Year": yr,
                    "Forecast": round(float(fc[i]), 2),
                    "Lower_90": round(float(lo[i]), 2),
                    "Upper_90": round(float(hi[i]), 2),
                    "Model": used_model,
                }
            )

    fc_df = pd.DataFrame(fc_records)

    log.info("\n--- Model used per country ---")
    for c, m in model_used.items():
        flag = "✅" if "fallback" not in m and "last resort" not in m else "⚠️"
        log.info("  %s  %-12s: %s", flag, c, m)

    # ── Growth summary ────────────────────────────────────────────────────────
    growth_rows: list = []
    for country in COUNTRIES:
        if country not in model_used:
            continue
        hist_last = df[df["Area"] == country].sort_values("Year")[TARGET].values[-1]
        frows = fc_df[fc_df["Country"] == country]
        fc_start = frows[frows["Year"] == FORECAST_YEARS_RUN[0]]["Forecast"].values[0]
        fc_end_v = frows[frows["Year"] == FORECAST_YEARS_RUN[-1]]["Forecast"].values[0]
        lo_end = frows[frows["Year"] == FORECAST_YEARS_RUN[-1]]["Lower_90"].values[0]
        hi_end = frows[frows["Year"] == FORECAST_YEARS_RUN[-1]]["Upper_90"].values[0]
        cagr = (
            ((fc_end_v / hist_last) ** (1 / horizon) - 1) * 100 if hist_last != 0 else float("nan")
        )
        total_g = (fc_end_v - hist_last) / hist_last * 100 if hist_last != 0 else float("nan")

        # test_mape = (
        #     float(row["MAPE"].values[0])
        #     if "MAPE" in best_df.columns and not best_df[best_df["Country"] == country].empty
        #     else float("nan")
        # )

        growth_rows.append(
            {
                "Country": country,
                f"{TEST_END} (TWh)": round(hist_last, 1),
                f"{FORECAST_YEARS_RUN[0]} Forecast": round(fc_start, 1),
                f"{FORECAST_YEARS_RUN[-1]} Forecast": round(fc_end_v, 1),
                f"{FORECAST_YEARS_RUN[-1]} 90% Lo": round(lo_end, 1),
                f"{FORECAST_YEARS_RUN[-1]} 90% Hi": round(hi_end, 1),
                "Total Growth %": round(total_g, 1),
                "CAGR 24-30 %": round(cagr, 2),
                "Model": model_used[country],
            }
        )

    growth_df = pd.DataFrame(growth_rows).sort_values("CAGR 24-30 %", ascending=False)
    log.info("=== Demand Growth Summary ===\n%s", growth_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_per_country(df, fc_df, model_used, fig_dir, train_end=args.train_end, val_end=args.val_end)
    plot_overlay(df, fc_df, model_used, TEST_END, fig_dir)
    plot_growth_uncertainty(growth_df, fc_df, fig_dir)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    fc_df.to_csv(os.path.join(args.output_dir, "demand_forecast_2025_2030.csv"), index=False)
    growth_df.to_csv(os.path.join(args.output_dir, "demand_growth_summary.csv"), index=False)

    log.info("=== Forecasting Complete ===")
    log.info("  demand_forecast_2025_2030.csv : %s", fc_df.shape)
    log.info("  demand_growth_summary.csv     : %s", growth_df.shape)
    log.info("  Bootstrap CI : N_BOOT=%d, CI=%d%%", N_BOOT, CI)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
