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

Fix log
-------
2024-xx-xx  forecast_ml: pd.concat(.to_frame().T) silently casts all columns
            to object dtype after the first auto-regressive step, causing
            np.isfinite to raise "ufunc not supported for input types".
            Fix: use _append_row() which rebuilds a typed DataFrame row via
            a dict + explicit astype(float64) on the feature block, and calls
            _safe_float_array() before every sklearn call.
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
FORECAST_YEARS = list(range(2025, 2031))
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


# ── Metric helpers ────────────────────────────────────────────────────────────
def safe_mape(a, b):
    a, b = np.array(a, float), np.array(b, float)
    mask = np.abs(a) > np.abs(a).mean() * 0.01
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


# ── dtype-safe float array ────────────────────────────────────────────────────
def _safe_float_array(df: pd.DataFrame, cols: list) -> np.ndarray:
    """
    Extract *cols* from *df* as a clean float64 ndarray.

    Handles every silent-cast trap that causes
    "ufunc 'isfinite' not supported for input types":
      1. object columns produced by pd.concat(.to_frame().T)
      2. categorical / boolean columns
      3. string-valued columns left in the feature set
      4. inf / NaN values
    """
    # Step 1 – select only requested columns (they may be a subset)
    sub = df[cols].copy()

    # Step 2 – drop any purely non-numeric columns (str, category …)
    #           that survived feature engineering
    non_numeric = [c for c in sub.columns if not pd.api.types.is_numeric_dtype(sub[c])]
    if non_numeric:
        log.debug("_safe_float_array: dropping non-numeric cols %s", non_numeric)
        sub = sub.drop(columns=non_numeric)

    # Step 3 – coerce everything to float64; unconvertible → NaN
    sub = sub.apply(pd.to_numeric, errors="coerce")

    # Step 4 – replace inf/-inf, then fill NaN with column median (or 0)
    sub = sub.replace([np.inf, -np.inf], np.nan)
    col_medians = sub.median()
    sub = sub.fillna(col_medians).fillna(0)

    # Step 5 – cast explicitly; should be a no-op after step 3 but guarantees
    X = sub.to_numpy(dtype=np.float64)

    # Step 6 – final sanity: replace any surviving non-finite with 0
    if not np.isfinite(X).all():
        X = np.where(np.isfinite(X), X, 0.0)

    return X


# ── Safe row append (preserves dtypes) ───────────────────────────────────────
def _append_row(df: pd.DataFrame, new_vals: dict) -> pd.DataFrame:
    """
    Append one row to *df* without losing dtype information.

    pd.concat([df, series.to_frame().T]) converts every column to object.
    This helper uses pd.concat with a properly-typed single-row DataFrame
    built from a dict, then re-casts numeric columns explicitly.
    """
    new_row_df = pd.DataFrame([new_vals], columns=df.columns)

    # Re-apply original dtypes where possible
    for col in df.columns:
        orig_dtype = df[col].dtype
        try:
            new_row_df[col] = new_row_df[col].astype(orig_dtype)
        except (ValueError, TypeError):
            pass  # leave as-is; _safe_float_array will handle it

    return pd.concat([df, new_row_df], ignore_index=True)


# ── Statistical models ────────────────────────────────────────────────────────
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
    """
    Linearly extrapolate each exogenous column for *horizon* future steps.
    Only operates on columns that actually exist in *history_df*.
    """
    valid_cols = [c for c in exog_cols if c in history_df.columns]
    rows = []
    for step in range(1, horizon + 1):
        row = {}
        for col in valid_cols:
            try:
                vals = pd.to_numeric(history_df[col], errors="coerce").fillna(0).values
                t = np.arange(len(history_df)).reshape(-1, 1)
                lr = LinearRegression().fit(t, vals)
                row[col] = float(lr.predict([[len(history_df) + step - 1]])[0])
            except Exception:
                row[col] = float(pd.to_numeric(history_df[col], errors="coerce").fillna(0).iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows, columns=valid_cols)


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
    """
    Walk-forward auto-regressive ML forecast.

    Key fix: every sklearn call goes through _safe_float_array() which
    guarantees a float64 array free of object-dtype columns, inf, and NaN —
    regardless of what pd.concat did to the dtypes internally.
    """
    df_h = history_df.copy()

    # Limit feature cols to those that actually exist in the DataFrame
    feat_cols = [c for c in all_feature_cols if c in df_h.columns]
    exog_valid = [c for c in exog_cols if c in df_h.columns]

    preds: list[float] = []
    exog_f = extrapolate_exog(df_h, exog_valid, horizon)

    for step in range(horizon):
        # ── Training matrix (dtype-safe) ──────────────────────────────────
        X_tr = _safe_float_array(df_h, feat_cols)
        y_tr = pd.to_numeric(df_h[TARGET], errors="coerce").fillna(0).to_numpy(dtype=np.float64)

        sc = None
        if scaler is not None:
            sc = scaler.__class__().fit(X_tr)
            X_tr = sc.transform(X_tr)

        model = model_cls(**model_kwargs).fit(X_tr, y_tr)

        # ── Test row (last known row + updated exogenous) ─────────────────
        # Build as a plain dict to avoid dtype erasure from Series.copy()
        last_row_dict = {col: df_h[col].iloc[-1] for col in feat_cols}
        for col in exog_valid:
            if col in last_row_dict:
                last_row_dict[col] = exog_f.iloc[step][col]

        X_te_df = pd.DataFrame([last_row_dict], columns=feat_cols)
        X_te = _safe_float_array(X_te_df, feat_cols)  # shape (1, n_features)

        if sc is not None:
            X_te = sc.transform(X_te)

        pred = float(model.predict(X_te)[0])
        preds.append(pred)

        # ── Append predicted row to history for next step ─────────────────
        # Use _append_row to preserve column dtypes (avoids object cast)
        last_full = {col: df_h[col].iloc[-1] for col in df_h.columns}
        last_full[TARGET] = pred
        last_full["Year"] = int(df_h["Year"].iloc[-1]) + 1
        # Update exogenous in the appended row as well
        for col in exog_valid:
            last_full[col] = exog_f.iloc[step][col]

        df_h = _append_row(df_h, last_full)

    return np.array(preds, dtype=np.float64)


def forecast_country(
    model_name: str,
    history_df: pd.DataFrame,
    all_feature_cols: list,
    exog_cols: list,
    params: dict,
    horizon: int,
) -> np.ndarray:
    ts = pd.to_numeric(history_df[TARGET], errors="coerce").fillna(0).to_numpy(dtype=np.float64)

    STAT_MODELS = {"Naive", "Naïve", "LinearTrend", "Holt", "ARIMA(1,1,1)", "ARIMA_1_1_1"}
    if model_name in STAT_MODELS:
        return forecast_statistical(ts, model_name, horizon)

    ML_CLS_MAP = {
        "Ridge": (Ridge, True),
        "RandomForest": (RandomForestRegressor, False),
        "XGBoost": (xgb.XGBRegressor if HAS_XGB else None, False),
    }
    if model_name not in ML_CLS_MAP:
        log.warning("Unknown model '%s' — falling back to Holt", model_name)
        return forecast_statistical(ts, "Holt", horizon)

    cls, needs_scaler = ML_CLS_MAP[model_name]
    if cls is None:
        log.warning("XGBoost not installed — falling back to LinearTrend")
        return forecast_statistical(ts, "LinearTrend", horizon)

    scaler = StandardScaler() if needs_scaler else None
    try:
        return forecast_ml(
            history_df, cls, all_feature_cols, exog_cols, horizon, scaler=scaler, **params
        )
    except Exception as e:
        country_label = history_df["Area"].iloc[0] if "Area" in history_df.columns else "?"
        log.warning(
            "ML forecast failed for %s/%s: %s — falling back to Holt",
            model_name,
            country_label,
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
    """Returns (point_forecast, lower, upper) each of length *horizon*."""
    boot_preds: list[np.ndarray] = []

    for _ in range(n_boot):
        n = len(history_df)
        idx = np.sort(np.random.choice(n, n, replace=True))
        df_b = history_df.iloc[idx].reset_index(drop=True)
        try:
            fc = forecast_country(model_name, df_b, all_feature_cols, exog_cols, params, horizon)
            boot_preds.append(fc)
        except Exception:
            continue

    point = forecast_country(model_name, history_df, all_feature_cols, exog_cols, params, horizon)

    if not boot_preds:
        return point, point * 0.95, point * 1.05

    arr = np.array(boot_preds, dtype=np.float64)
    lo_pct = (100 - ci) / 2
    hi_pct = 100 - lo_pct
    lower = np.percentile(arr, lo_pct, axis=0)
    upper = np.percentile(arr, hi_pct, axis=0)
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
    color = PALETTE.get(country, "#333333")
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(hist["Year"], hist[TARGET], color="black", lw=1.8, label="Actual")
    ax.axvline(test_end + 0.5, color="grey", ls=":", lw=1.2, label="Forecast start")
    ax.axvspan(test_end + 0.5, max(fc_years) + 0.5, alpha=0.05, color=color)

    # Bridge last actual → first forecast point
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
    ax.fill_between(
        fc_years,
        fc_lower,
        fc_upper,
        alpha=0.15,
        color=color,
        label=f"{CI}% CI",
    )

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
        color = PALETTE.get(country, "#333333")
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
        ax.fill_between(
            r["fc_years"],
            r["fc_lower"],
            r["fc_upper"],
            alpha=0.10,
            color=color,
        )
    ax.axvline(test_end + 0.5, color="grey", ls=":", lw=1.2)
    ax.set_title(f"{TARGET} Forecast 2025–2030 — All Countries", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("TWh")
    ax.legend(ncol=4, frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "fc_fig_overlay.pdf"))


# ── CLI ───────────────────────────────────────────────────────────────────────
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
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
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
    FEATURES: list = meta.get("FEATURES", [])  # exogenous / subcategory cols
    TEST_END: int = meta["TEST_END"]  # e.g. 2024

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

    results: dict = {}
    fc_rows: list = []

    log.info("=== Forecasting %d–%d (N_BOOT=%d, CI=%d%%) ===", TEST_END + 1, _fc_end, N_BOOT, CI)

    for country in COUNTRIES:
        hist = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
        if hist.empty:
            log.warning("No data for %s — skipping.", country)
            continue

        row = best_df[best_df["Country"] == country]
        best_m = row["Model"].values[0] if not row.empty else "Holt"
        params = PARAMS_MAP.get(best_m, {})

        log.info("  %-12s | %-16s | horizon=%d", country, best_m, horizon)

        fc_point, fc_lower, fc_upper = bootstrap_forecast(
            best_m,
            hist,
            ALL_FEATURES,
            FEATURES,
            params,
            horizon,
            n_boot=N_BOOT,
            ci=CI,
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

    # ── Growth summary ────────────────────────────────────────────────────────
    growth_rows = []
    for country, r in results.items():
        hist = df[df["Area"] == country].sort_values("Year")
        base = float(hist[TARGET].iloc[-1])
        end = float(r["fc_point"][-1])
        cagr = (end / base) ** (1 / horizon) - 1 if base != 0 else float("nan")
        growth_rows.append(
            {
                "Country": country,
                "Model": r["model"],
                f"{TEST_END} TWh": round(base, 1),
                "2030 TWh (forecast)": round(end, 1),
                "Total Growth (%)": round((end / base - 1) * 100, 1) if base != 0 else float("nan"),
                "CAGR (%)": round(cagr * 100, 2),
            }
        )
    df_growth = pd.DataFrame(growth_rows)
    log.info("=== Demand Growth Summary ===\n%s", df_growth.to_string(index=False))

    plot_overlay(results, df, TEST_END, fig_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
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
