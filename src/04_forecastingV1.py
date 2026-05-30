"""
src/04_forecasting.py — Demand Forecasting 2025–2030
Ember Energy | IEEE Paper

Models  : Best ML model per country (from 03_modeling)
          + statistical fallbacks (Naive, LinearTrend, Holt, ARIMA)
Horizon : 2025–2030 (6 steps)
CI      : Bootstrap (N_BOOT=300, CI=90%)

Reads:  outputs/preprocessing/ember_model_ready.csv
        outputs/modeling/best_models.csv
        outputs/modeling/best_hp.json
        outputs/modeling/ember_config.json
Writes: outputs/forecasting/demand_forecast_2025_2030.csv
        outputs/forecasting/demand_growth_summary.csv
        outputs/forecasting/figures/*.pdf
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

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
PALETTE = {
    "Tunisia": "#1f77b4",
    "Austria": "#d62728",
    "Germany": "#2ca02c",
    "Egypt": "#9467bd",
    "Canada": "#e6a817",
    "France": "#8c564b",
    "Kuwait": "#17becf",
}
TARGET = "Demand"
FORECAST_YEARS = list(range(2025, 2030))
N_BOOT = 300
CI = 90

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


def savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ── Metric helpers ─────────
def safe_mape(a, b):
    a, b = np.array(a, float), np.array(b, float)
    mask = np.abs(a) > np.abs(a).mean() * 0.01
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


# ── Statistical models ─────────────
def forecast_statistical(series: np.ndarray, model_name: str, horizon: int) -> np.ndarray:
    if model_name in {"Naive", "Naïve"}:
        return np.full(horizon, series[-1])

    elif model_name == "LinearTrend":
        t = np.arange(len(series)).reshape(-1, 1)
        lr = LinearRegression().fit(t, series)
        t_f = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        return lr.predict(t_f).flatten()

    elif model_name == "Holt":
        try:
            m = SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True)
            return m.forecast(horizon)
        except Exception:
            return np.full(horizon, series[-1])

    elif model_name in {"ARIMA(1,1,1)", "ARIMA_1_1_1"}:
        try:
            m = ARIMA(series, order=(1, 1, 1)).fit()
            return m.get_forecast(steps=horizon).predicted_mean.values
        except Exception:
            return np.full(horizon, series[-1])

    return np.full(horizon, series[-1])


# ── Exogenous feature extrapolation ──────────────────────────────────────────
def extrapolate_exog(history_df: pd.DataFrame, exog_cols: list, horizon: int) -> pd.DataFrame:
    rows = []
    for step in range(1, horizon + 1):
        row = {}
        for col in exog_cols:
            try:
                t = np.arange(len(history_df)).reshape(-1, 1)
                lr = LinearRegression().fit(t, history_df[col].fillna(0).values)
                row[col] = float(lr.predict([[len(history_df) + step - 1]])[0])
            except Exception:
                row[col] = float(history_df[col].iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows, columns=exog_cols)


# ── ML auto-regressive forecast ───────────────────────────────────────────────
def forecast_ml(
    history_df: pd.DataFrame,
    model_cls,
    all_feature_cols: list,
    exog_cols: list,
    horizon: int,
    scaler=None,
    **model_kwargs,
) -> np.ndarray:
    df_h = history_df.copy()
    preds = []
    exog_f = extrapolate_exog(df_h, exog_cols, horizon)

    for step in range(horizon):
        X_tr = df_h[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_tr = df_h[TARGET].values
        if scaler is not None:
            sc = scaler.__class__().fit(X_tr)
            X_tr = sc.transform(X_tr)

        m = model_cls(**model_kwargs).fit(X_tr, y_tr)

        # Build next feature row
        next_row = df_h[all_feature_cols].iloc[-1].copy()
        # Update exogenous
        for col in exog_cols:
            if col in next_row.index:
                next_row[col] = exog_f.iloc[step][col]

        X_te = next_row.values.reshape(1, -1)
        X_te = np.where(np.isfinite(X_te), X_te, 0)
        if scaler is not None:
            X_te = sc.transform(X_te)

        pred = float(m.predict(X_te)[0])
        preds.append(pred)

        # Append predicted row to history for next step
        new_row = df_h.iloc[-1].copy()
        new_row[TARGET] = pred
        new_row["Year"] += 1
        df_h = pd.concat([df_h, new_row.to_frame().T], ignore_index=True)

    return np.array(preds)


def forecast_country(
    model_name: str,
    history_df: pd.DataFrame,
    all_feature_cols: list,
    exog_cols: list,
    params: dict,
    horizon: int,
) -> np.ndarray:
    ts = history_df[TARGET].values

    STAT_MODELS = {"Naive", "Naïve", "LinearTrend", "Holt", "ARIMA(1,1,1)", "ARIMA_1_1_1"}
    if model_name in STAT_MODELS:
        return forecast_statistical(ts, model_name, horizon)

    ML_CLS_MAP = {
        "Ridge": (Ridge, True),
        "RandomForest": (RandomForestRegressor, False),
        "XGBoost": (xgb.XGBRegressor if HAS_XGB else None, False),
    }
    if model_name not in ML_CLS_MAP:
        log.warning("Unknown model %s — falling back to Holt", model_name)
        return forecast_statistical(ts, "Holt", horizon)

    cls, needs_scaler = ML_CLS_MAP[model_name]
    if cls is None:
        return forecast_statistical(ts, "LinearTrend", horizon)

    scaler = StandardScaler() if needs_scaler else None
    try:
        return forecast_ml(
            history_df, cls, all_feature_cols, exog_cols, horizon, scaler=scaler, **params
        )
    except Exception as e:
        log.warning(
            "ML forecast failed for %s/%s: %s — falling back to Holt",
            model_name,
            history_df["Area"].iloc[0],
            e,
        )
        return forecast_statistical(ts, "Holt", horizon)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_forecast(
    model_name: str,
    history_df: pd.DataFrame,
    all_feature_cols: list,
    exog_cols: list,
    params: dict,
    horizon: int,
    n_boot: int = N_BOOT,
    ci: int = CI,
):
    """Returns (point_forecast, lower, upper) each of length horizon."""
    boot_preds = []
    for _ in range(n_boot):
        # Resample training rows with replacement
        n = len(history_df)
        idx = np.random.choice(n, n, replace=True)
        idx = np.sort(idx)  # keep temporal order
        df_b = history_df.iloc[idx].reset_index(drop=True)
        try:
            fc = forecast_country(model_name, df_b, all_feature_cols, exog_cols, params, horizon)
            boot_preds.append(fc)
        except Exception:
            continue

    point = forecast_country(model_name, history_df, all_feature_cols, exog_cols, params, horizon)
    if not boot_preds:
        return point, point * 0.95, point * 1.05

    arr = np.array(boot_preds)
    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = np.percentile(arr, lo, axis=0)
    upper = np.percentile(arr, hi, axis=0)
    return point, lower, upper


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_country_forecast(
    country: str,
    hist: pd.DataFrame,
    fc_years: list,
    fc_point: np.ndarray,
    fc_lower: np.ndarray,
    fc_upper: np.ndarray,
    model_name: str,
    test_end: int,
    fig_dir: str,
) -> None:
    color = PALETTE[country]
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(hist["Year"], hist[TARGET], color="black", lw=1.8, label="Actual")
    ax.axvline(test_end + 0.5, color="grey", ls=":", lw=1.2, label="Forecast start")
    ax.axvspan(test_end + 0.5, max(fc_years) + 0.5, alpha=0.05, color=color)

    # Bridge
    last_yr = int(hist["Year"].iloc[-1])
    last_val = float(hist[TARGET].iloc[-1])
    ax.plot([last_yr, fc_years[0]], [last_val, fc_point[0]], color=color, lw=1.8, ls="--")

    ax.plot(
        fc_years,
        fc_point,
        color=color,
        lw=2.2,
        ls="-",
        marker="D",
        ms=6,
        label=f"{model_name} forecast",
    )
    ax.fill_between(fc_years, fc_lower, fc_upper, alpha=0.15, color=color, label=f"{CI}% CI")

    ax.set_xlim(hist["Year"].min() - 0.5, max(fc_years) + 0.5)
    ax.set_title(f"{country} — {TARGET} Forecast 2025–2030 ({model_name})", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("TWh")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"fc_fig_{country.lower()}.pdf"))


def plot_overlay(results: dict, df: pd.DataFrame, test_end: int, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for country in COUNTRIES:
        if country not in results:
            continue
        color = PALETTE[country]
        hist = df[df["Area"] == country].sort_values("Year")
        r = results[country]
        ax.plot(hist["Year"], hist[TARGET], color=color, lw=1.6, alpha=0.9)
        ax.plot(
            [hist["Year"].iloc[-1], r["fc_years"][0]],
            [hist[TARGET].iloc[-1], r["fc_point"][0]],
            color=color,
            lw=1.6,
            ls="--",
        )
        ax.plot(
            r["fc_years"],
            r["fc_point"],
            color=color,
            lw=2.0,
            ls="--",
            marker="D",
            ms=5,
            label=f"{country} ({r['model']})",
        )
        ax.fill_between(r["fc_years"], r["fc_lower"], r["fc_upper"], alpha=0.10, color=color)
    ax.axvline(test_end + 0.5, color="grey", ls=":", lw=1.2)
    ax.set_title(f"{TARGET} Forecast 2025–2030 — All Countries", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("TWh")
    ax.legend(ncol=4, frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fc_fig_overlay.pdf"))


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember demand forecasting 2025–2030")
    p.add_argument("--pre_dir", default="outputs/preprocessing", help="Preprocessing dir")
    p.add_argument("--model_dir", default="outputs/modeling", help="Modeling dir")
    p.add_argument("--output_dir", default="outputs/forecasting", help="Output dir")
    p.add_argument(
        "--forecast_until",
        type=int,
        default=None,
        help="Last forecast year (default: uses meta FORECAST_YEARS)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load
    df = pd.read_csv(os.path.join(args.pre_dir, "ember_model_ready.csv"))
    with open(os.path.join(args.model_dir, "ember_config.json")) as f:
        meta = json.load(f)
    best_df = pd.read_csv(os.path.join(args.model_dir, "best_models.csv"))
    with open(os.path.join(args.model_dir, "best_hp.json")) as f:
        best_hp = json.load(f)

    ALL_FEATURES = meta["all_features"]
    FEATURES = meta.get("FEATURES", [])  # non-target subcategory cols
    TEST_END = meta["TEST_END"]  # 2024
    _fc_end = (
        args.forecast_until if args.forecast_until else max(meta.get("FORECAST_YEARS", [2030]))
    )
    FORECAST_YEARS_RUN = list(range(TEST_END + 1, _fc_end + 1))
    horizon = len(FORECAST_YEARS_RUN)

    PARAMS_MAP = {
        "Ridge": best_hp.get("Ridge", {}),
        "RandomForest": best_hp.get("RandomForest", {}),
        "XGBoost": best_hp.get("XGBoost", {}),
    }

    results = {}
    fc_rows = []

    log.info("=== Forecasting 2025–2030 (N_BOOT=%d, CI=%d%%) ===", N_BOOT, CI)

    for country in COUNTRIES:
        hist = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
        if hist.empty:
            continue

        row = best_df[best_df["Country"] == country]
        best_m = row["Model"].values[0] if not row.empty else "Holt"
        params = PARAMS_MAP.get(best_m, {})

        log.info("  %-12s | %-16s ...", country, best_m)

        fc_point, fc_lower, fc_upper = bootstrap_forecast(
            best_m, hist, ALL_FEATURES, FEATURES, params, horizon, n_boot=N_BOOT, ci=CI
        )

        results[country] = {
            "model": best_m,
            "fc_years": FORECAST_YEARS_RUN,
            "fc_point": fc_point,
            "fc_lower": fc_lower,
            "fc_upper": fc_upper,
        }

        for yr, pt, lo, hi in zip(FORECAST_YEARS_RUN, fc_point, fc_lower, fc_upper):
            fc_rows.append(
                {
                    "Country": country,
                    "Model": best_m,
                    "Year": yr,
                    "Forecast": round(float(pt), 2),
                    "Lower_90": round(float(lo), 2),
                    "Upper_90": round(float(hi), 2),
                }
            )

        plot_country_forecast(
            country,
            hist,
            FORECAST_YEARS_RUN,
            fc_point,
            fc_lower,
            fc_upper,
            best_m,
            TEST_END,
            fig_dir,
        )

    # Growth summary
    growth_rows = []
    for country, r in results.items():
        hist = df[df["Area"] == country].sort_values("Year")
        base = float(hist[TARGET].iloc[-1])
        end = float(r["fc_point"][-1])
        cagr = (end / base) ** (1 / horizon) - 1
        growth_rows.append(
            {
                "Country": country,
                "Model": r["model"],
                f"{TEST_END} TWh": round(base, 1),
                "2030 TWh (forecast)": round(end, 1),
                "Total Growth (%)": round((end / base - 1) * 100, 1),
                "CAGR (%)": round(cagr * 100, 2),
            }
        )
    df_growth = pd.DataFrame(growth_rows)
    log.info("=== Demand Growth Summary ===\n%s", df_growth.to_string(index=False))

    # Overlay plot
    plot_overlay(results, df, TEST_END, fig_dir)

    # Save
    df_fc = pd.DataFrame(fc_rows)
    df_fc.to_csv(os.path.join(args.output_dir, "demand_forecast_2025_2030.csv"), index=False)
    df_growth.to_csv(os.path.join(args.output_dir, "demand_growth_summary.csv"), index=False)

    log.info("=== Forecasting Complete ===")
    log.info("  demand_forecast_2025_2030.csv : %s", df_fc.shape)
    log.info("  demand_growth_summary.csv     : %s", df_growth.shape)
    log.info("  Bootstrap CI : N_BOOT=%d, CI=%d%%", N_BOOT, CI)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
