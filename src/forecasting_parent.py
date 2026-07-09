"""
src/step04_forecasting.py — Future Demand Forecasting (2025-2030)
================================================================
Ember Energy | IEEE Paper

Mirrors exactly: notebooks/033_forecasting_patched.ipynb

Reads:   outputs/preprocessing/ember_model_ready.csv
         outputs/preprocessing/feature_meta.json
         outputs/modeling/best_models.csv
         outputs/modeling/best_hp.json
Writes:  outputs/forecasting/demand_forecast_2025_2030.csv
         outputs/forecasting/demand_growth_summary.csv
         outputs/forecasting/figures/*.png

Strategy:
    - Retrain best model per country on ALL history (2000-2024)
    - Statistical models (Holt, ARIMA, LinearTrend, Naive): native multi-step
    - ML models (Ridge, RF, XGBoost): RECURSIVE multi-step
        Step 1: predict 2025 -> append to history
        Step 2: predict 2026 using 2025 prediction as lag-1
        ... exogenous (other subcategory) features extrapolated via linear trend
    - Bootstrap residuals -> 90% Confidence Intervals
    - Fallback chain: best_model -> Holt -> LinearTrend (last resort)
"""

from __future__ import annotations

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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"
N_BOOT = 300

PALETTE = {
    "Tunisia": "#e63946",
    "Austria": "#2196F3",
    "Germany": "#4CAF50",
    "Egypt": "#9C27B0",
    "Canada": "#00BCD4",
    "France": "#797148",
    "Kuwait": "#4C4879",
}

STAT_MODELS = {"Naive", "Naïve", "LinearTrend", "Holt", "ARIMA(1,1,1)", "ARIMA_1_1_1"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    }
)
sns.set_theme(style="whitegrid", palette="tab10")


def savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ── Feature cleaning ───────────────────────────────────────────────────────────


def clean_x(arr: np.ndarray) -> np.ndarray:
    """Replace Inf/-Inf with NaN, impute column medians. Mirrors notebook clean_x()."""
    arr = np.array(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    col_medians = np.nanmedian(arr, axis=0)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return arr


# ── Exogenous feature extrapolation ───────────────────────────────────────────


def extrapolate_exog(
    history_df: pd.DataFrame, exog_cols: list[str], n_steps: int
) -> dict[str, np.ndarray]:
    """
    Linear trend extrapolation per exogenous feature.
    Mirrors notebook extrapolate_exog() exactly.
    """
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


def build_future_feature_row(
    history_df: pd.DataFrame,
    exog_future: dict[str, np.ndarray],
    demand_preds_so_far: list[float],
    step: int,
    all_feature_cols: list[str],
    target_col: str,
) -> dict[str, float]:
    """
    Build a single feature row for `step` (0-indexed) into the future.
    Mirrors notebook build_future_feature_row() exactly.
    """
    hist_demand = list(history_df[target_col].values.astype(float))
    full_demand = hist_demand + list(demand_preds_so_far)
    n_full = len(full_demand)

    row: dict[str, float] = {}

    # Lag features for target
    for lag in [1, 2, 3]:
        k = f"{target_col}_lag{lag}"
        if k in all_feature_cols:
            idx = n_full - lag
            row[k] = full_demand[idx] if idx >= 0 else np.nan

    # Rolling MAs for target (shift=1 baked in: use full_demand[:-1] window logic)
    for w in [3, 5]:
        k = f"{target_col}_ma{w}"
        if k in all_feature_cols:
            window = full_demand[max(0, n_full - w) : n_full]
            row[k] = float(np.mean(window)) if window else np.nan

    # YoY for target
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

    # Exogenous features (each with their projected future values)
    for col in exog_future:
        hist_exog = list(history_df[col].values.astype(float))
        full_exog = hist_exog + list(exog_future[col][: step + 1])
        n_ex = len(full_exog)

        for lag in [1, 2, 3]:
            k = f"{col}_lag{lag}"
            if k in all_feature_cols:
                idx = n_ex - lag
                row[k] = full_exog[idx] if idx >= 0 else np.nan

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

    # Trend + country code
    year_min = int(history_df["Year"].min())
    future_year = int(history_df["Year"].max()) + step + 1
    row["trend"] = future_year - year_min
    row["trend_sq"] = row["trend"] ** 2
    row["country_code"] = int(history_df["country_code"].iloc[0])

    # Fill any remaining feature columns with 0 (cross-feature interactions etc.)
    for col in all_feature_cols:
        if col not in row:
            row[col] = 0.0

    return row


# ── Forecast functions ────────────────────────────────────────────────────────


def forecast_statistical(series: np.ndarray, model_name: str, horizon: int) -> np.ndarray:
    """Statistical multi-step forecast. Mirrors notebook forecast_statistical()."""
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
        return m.get_forecast(steps=horizon).predicted_mean.values
    return np.full(horizon, series[-1])


def forecast_ml_recursive(
    history_df: pd.DataFrame,
    model_cls,
    all_feature_cols: list[str],
    exog_cols: list[str],
    horizon: int,
    target: str,
    scaler: StandardScaler | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Recursive ML multi-step forecast. Mirrors notebook forecast_ml_recursive() exactly.
    """
    tr = history_df[[*all_feature_cols, target]].copy()
    tr = tr[tr[target].notna()]
    X_tr = clean_x(tr[all_feature_cols].values)
    y_tr = tr[target].values.astype(float)

    if len(X_tr) < 3:
        raise ValueError("Not enough training rows after cleaning")

    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)

    model = model_cls(**kwargs)
    model.fit(X_tr, y_tr)

    exog_future = extrapolate_exog(history_df, exog_cols, horizon)

    demand_preds: list[float] = []
    for step in range(horizon):
        feat_row = build_future_feature_row(
            history_df, exog_future, demand_preds, step, all_feature_cols, target
        )
        x_pred = np.array([[feat_row[c] for c in all_feature_cols]], dtype=float)
        x_pred = clean_x(x_pred)

        if scaler is not None:
            x_pred = scaler.transform(x_pred)

        p = float(model.predict(x_pred)[0])
        demand_preds.append(p)

    return np.array(demand_preds)


# ── Bootstrap CI ────────────────────────────────────────────────────────────────


def get_ml_cls_map() -> dict[str, tuple]:
    ml_map: dict[str, tuple] = {
        "Ridge": (Ridge, True),
        "RandomForest": (RandomForestRegressor, False),
        "XGBoost": (None, False),
    }
    if HAS_XGB:
        ml_map["XGBoost"] = (xgb.XGBRegressor, False)
    return ml_map


def point_forecast(
    history_df: pd.DataFrame,
    model_name: str,
    all_feature_cols: list[str],
    exog_cols: list[str],
    params: dict,
    horizon: int,
    target: str,
    ml_cls_map: dict,
) -> np.ndarray:
    """Single point forecast dispatcher — stat or ML. Mirrors notebook _point_forecast()."""
    ts = history_df[target].values.astype(float)
    if model_name in STAT_MODELS:
        return forecast_statistical(ts, model_name, horizon)

    cls, needs_scaler = ml_cls_map[model_name]
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
        target,
        scaler=scaler,
        **kw,
    )


def get_insample_residuals(
    history_df: pd.DataFrame,
    model_name: str,
    all_feature_cols: list[str],
    exog_cols: list[str],
    params: dict,
    target: str,
    ml_cls_map: dict,
    n_eval: int = 8,
) -> np.ndarray:
    """Walk-forward 1-step residuals on last n_eval years. Mirrors notebook."""
    residuals: list[float] = []
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


def bootstrap_forecast_ci(
    history_df: pd.DataFrame,
    model_name: str,
    all_feature_cols: list[str],
    exog_cols: list[str],
    params: dict,
    horizon: int,
    target: str,
    ml_cls_map: dict,
    n_boot: int = N_BOOT,
    ci: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Residual-bootstrap CI for multi-step forecast.
    Mirrors notebook bootstrap_forecast_ci() exactly.
    Returns (point_forecast, lower, upper).
    """
    base_fc = point_forecast(
        history_df, model_name, all_feature_cols, exog_cols, params, horizon, target, ml_cls_map
    )

    residuals = get_insample_residuals(
        history_df, model_name, all_feature_cols, exog_cols, params, target, ml_cls_map
    )

    ts = history_df[target].values.astype(float)

    if len(residuals) < 3:
        std = np.std(ts) * 0.10
        return base_fc, base_fc - 1.645 * std, base_fc + 1.645 * std

    boot_fcs = []
    alpha = (100 - ci) / 2

    for _ in range(n_boot):
        noise = np.random.choice(residuals, size=len(ts), replace=True)
        ts_pert = np.clip(ts + noise * 0.5, ts.min() * 0.3, None)
        h_pert = history_df.copy()
        h_pert[target] = ts_pert
        try:
            fc_b = point_forecast(
                h_pert, model_name, all_feature_cols, exog_cols, params, horizon, target, ml_cls_map
            )
            boot_fcs.append(fc_b)
        except Exception:
            boot_fcs.append(base_fc)

    boot_arr = np.array(boot_fcs)
    return (
        base_fc,
        np.percentile(boot_arr, alpha, axis=0),
        np.percentile(boot_arr, 100 - alpha, axis=0),
    )


# ── Generate forecasts for all countries (with fallback chain) ───────────────


def generate_all_forecasts(
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    all_features: list[str],
    raw_features: list[str],
    forecast_years: list[int],
    target: str,
    params_map: dict,
    ml_cls_map: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate forecasts for all countries with fallback chain:
    best_model -> Holt -> LinearTrend (last resort).
    Mirrors notebook Section 4 exactly.
    """
    horizon = len(forecast_years)
    fc_records: list[dict] = []
    model_used: dict[str, str] = {}

    for country in COUNTRIES:
        hist_df = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
        best_model = best_df[best_df["Country"] == country]["Model"].values[0]
        params = params_map.get(best_model, {})

        log.info("  %-10s | %-16s ...", country, best_model)

        fc, lo, hi = None, None, None
        used_model = best_model

        try:
            fc, lo, hi = bootstrap_forecast_ci(
                hist_df,
                best_model,
                all_features,
                raw_features,
                params,
                horizon,
                target,
                ml_cls_map,
            )
        except Exception as e1:
            log.warning("    Best model failed: %s: %s", type(e1).__name__, e1)
            log.info("    Trying Holt fallback...")
            try:
                fc, lo, hi = bootstrap_forecast_ci(
                    hist_df,
                    "Holt",
                    all_features,
                    raw_features,
                    {},
                    horizon,
                    target,
                    ml_cls_map,
                )
                used_model = "Holt (fallback)"
            except Exception as e2:
                log.warning("    Holt failed too: %s", e2)
                ts = hist_df[target].values.astype(float)
                fc = forecast_statistical(ts, "LinearTrend", horizon)
                std = np.std(ts) * 0.10
                lo = fc - 1.645 * std
                hi = fc + 1.645 * std
                used_model = "LinearTrend (last resort)"

        model_used[country] = used_model

        for i, yr in enumerate(forecast_years):
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

    for c, m in model_used.items():
        flag = "OK" if "fallback" not in m and "last resort" not in m else "WARN"
        log.info("  [%s] %-10s: %s", flag, c, m)

    return fc_df, model_used


# ── Growth summary ─────────────────────────────────────────────────────────────


def compute_growth_summary(
    df: pd.DataFrame, fc_df: pd.DataFrame, best_df: pd.DataFrame, target: str
) -> pd.DataFrame:
    """Mirrors notebook Section 6 growth summary exactly."""
    rows = []
    for country in COUNTRIES:
        hist_last = df[df["Area"] == country].sort_values("Year")[target].values[-1]
        fc_2025 = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == 2025)]["Forecast"].values[
            0
        ]
        fc_2030 = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == 2030)]["Forecast"].values[
            0
        ]
        lo_2030 = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == 2030)]["Lower_90"].values[
            0
        ]
        hi_2030 = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == 2030)]["Upper_90"].values[
            0
        ]

        cagr = ((fc_2030 / hist_last) ** (1 / 6) - 1) * 100
        total_g = (fc_2030 - hist_last) / hist_last * 100

        test_mape = best_df[best_df["Country"] == country]["MAPE"].values[0]
        quality = (
            "Excellent"
            if test_mape < 3
            else "Good"
            if test_mape < 7
            else "Moderate"
            if test_mape < 15
            else "Poor"
        )
        model = fc_df[fc_df["Country"] == country]["Model"].values[0]

        rows.append(
            {
                "Country": country,
                "2024 (TWh)": round(hist_last, 1),
                "2025 Forecast": round(fc_2025, 1),
                "2030 Forecast": round(fc_2030, 1),
                "2030 90% Lo": round(lo_2030, 1),
                "2030 90% Hi": round(hi_2030, 1),
                "Total Growth %": round(total_g, 1),
                "CAGR 24-30 %": round(cagr, 2),
                "Test MAPE %": round(test_mape, 2),
                "Forecast Quality": quality,
                "Model": model,
            }
        )

    return pd.DataFrame(rows).sort_values("CAGR 24-30 %", ascending=False)


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_forecast_per_country(
    df: pd.DataFrame, fc_df: pd.DataFrame, target: str, fig_dir: str
) -> None:
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        col = clrs[i % len(clrs)]
        hist = df[df["Area"] == country].sort_values("Year")
        frow = fc_df[fc_df["Country"] == country]
        model = frow["Model"].values[0]

        tr = hist[hist["Year"] <= 2024]
        va = hist[(hist["Year"] > 2016) & (hist["Year"] <= 2020)]
        te = hist[hist["Year"] > 2020]

        ax.plot(tr["Year"], tr[target], color="#3a86ff", lw=2, label="Train")
        ax.plot(va["Year"], va[target], color="#ff9f1c", lw=2, label="Val")
        ax.plot(te["Year"], te[target], color="#e63946", lw=2, label="Test")

        ax.plot(
            [hist["Year"].values[-1], frow["Year"].values[0]],
            [hist[target].values[-1], frow["Forecast"].values[0]],
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
            label=f"Forecast ({model})",
        )
        ax.fill_between(
            frow["Year"], frow["Lower_90"], frow["Upper_90"], alpha=0.18, color=col, label="90% CI"
        )

        ax.axvspan(2024.5, 2020.5, alpha=0.05, color="orange")
        ax.axvspan(2020.5, 2024.5, alpha=0.05, color="red")
        ax.axvline(2024, color="grey", ls=":", lw=1)

        ax.set_title(country, fontweight="bold", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    axes[-1].set_visible(False)
    fig.suptitle(
        "Electricity Demand Forecast 2025–2030\n(Train/Val/Test + Recursive Forecast, 90% CI)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "forecast_per_country.png"))


def plot_forecast_overlay(df: pd.DataFrame, fc_df: pd.DataFrame, target: str, fig_dir: str) -> None:
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, country in enumerate(COUNTRIES):
        col = clrs[i % len(clrs)]
        hist = df[df["Area"] == country].sort_values("Year")
        frow = fc_df[fc_df["Country"] == country]
        ax.plot(hist["Year"], hist[target], color=col, lw=1.5, alpha=0.6)
        ax.plot(frow["Year"], frow["Forecast"], color=col, lw=2.5, ls="--", label=country)
        ax.fill_between(frow["Year"], frow["Lower_90"], frow["Upper_90"], alpha=0.09, color=col)
    ax.axvline(2024, color="black", ls=":", lw=1.2)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("TWh", fontsize=12)
    ax.set_title(
        "All Countries — Demand History & Forecast 2025–2030", fontsize=13, fontweight="bold"
    )
    ax.legend(ncol=2, fontsize=9, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "forecast_overlay.png"))


def plot_growth_uncertainty(fc_df: pd.DataFrame, growth_df: pd.DataFrame, fig_dir: str) -> None:
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(
        growth_df["Country"],
        growth_df["CAGR 24-30 %"],
        color=[clrs[i % len(clrs)] for i in range(len(growth_df))],
        edgecolor="white",
    )
    for bar, val in zip(bars, growth_df["CAGR 24-30 %"], strict=False):
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
        "Forecast Uncertainty — 90% CI Width per Year", fontsize=11, fontweight="bold"
    )

    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "growth_uncertainty.png"))


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember forecasting step")
    p.add_argument("--pre_dir", default="outputs/preprocessing", help="Preprocessing dir")
    p.add_argument("--model_dir", default="outputs/modeling", help="Modeling dir")
    p.add_argument("--output_dir", default="outputs/forecasting", help="Output dir")
    p.add_argument("--forecast_until", type=int, default=2030, help="Last forecast year")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    forecast_years = list(range(2025, args.forecast_until + 1))

    # Load
    df = pd.read_csv(os.path.join(args.pre_dir, "ember_model_ready.csv"))
    best_df = pd.read_csv(os.path.join(args.model_dir, "best_models.csv"))
    with open(os.path.join(args.pre_dir, "feature_meta.json")) as f:
        meta = json.load(f)

    all_features = meta["all_features"]
    raw_features = meta["raw_features"]

    log.info("Data: %s", df.shape)
    log.info("Best models:\n%s", best_df[["Country", "Model", "MAPE"]].to_string())

    # Load tuned hyperparameters
    params_map = {
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
    best_hp_path = os.path.join(args.model_dir, "best_hp.json")
    if os.path.exists(best_hp_path):
        with open(best_hp_path) as f:
            best_hp = json.load(f)
        for k, v in best_hp.items():
            params_map[k] = v
        log.info("Loaded tuned hyperparameters from best_hp.json")
    else:
        log.warning("best_hp.json not found — using default hyperparameters")

    ml_cls_map = get_ml_cls_map()

    # Generate forecasts
    log.info(
        "── Generating forecasts for %d countries x %d years…", len(COUNTRIES), len(forecast_years)
    )
    fc_df, _model_used = generate_all_forecasts(
        df,
        best_df,
        all_features,
        raw_features,
        forecast_years,
        TARGET,
        params_map,
        ml_cls_map,
    )
    log.info("Forecast table: %s", fc_df.shape)

    # Plots
    plot_forecast_per_country(df, fc_df, TARGET, fig_dir)
    plot_forecast_overlay(df, fc_df, TARGET, fig_dir)

    # Growth summary
    growth_df = compute_growth_summary(df, fc_df, best_df, TARGET)
    log.info("Growth summary:\n%s", growth_df.to_string())

    plot_growth_uncertainty(fc_df, growth_df, fig_dir)

    # Export
    fc_df.to_csv(os.path.join(args.output_dir, "demand_forecast_2025_2030.csv"), index=False)
    growth_df.to_csv(os.path.join(args.output_dir, "demand_growth_summary.csv"), index=False)

    log.info("=== Forecasting Complete ===")
    log.info("  demand_forecast_2025_2030.csv : %s", fc_df.shape)
    log.info("  demand_growth_summary.csv     : %s", growth_df.shape)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
