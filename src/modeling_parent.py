"""
src/03_modeling.py — Train/Val/Test Split + Walk-Forward Benchmarking
=========================================================================
Ember Energy | IEEE Paper

Mirrors exactly: notebooks/02_benchmarking_patched.ipynb

Reads:   outputs/preprocessing/ember_model_ready.csv
         outputs/preprocessing/feature_meta.json
Writes:  outputs/modeling/test_benchmarking.csv
         outputs/modeling/val_benchmarking.csv
         outputs/modeling/best_models.csv
         outputs/modeling/wf_test_predictions.csv
         outputs/modeling/best_hp.json
         outputs/modeling/figures/*.pdf

Models (7):
    Naive | LinearTrend | Holt | ARIMA(1,1,1) | Ridge | RandomForest | XGBoost

Protocol:
    1. Hyperparameter tuning (Ridge/RF/XGB) on Train→Val via TimeSeriesSplit
    2. Walk-forward evaluation on Test set (2021-2024), expanding window
    3. Walk-forward evaluation on Val set (2017-2020), for overfit check
    4. Best model per country selected by lowest Test MAPE
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
TARGET    = "Demand"

PALETTE = {
    "Tunisia": "#e63946", "Austria": "#2196F3", "Germany": "#4CAF50",
    "Egypt":   "#9C27B0", "Canada":  "#00BCD4", "France":  "#797148",
    "Kuwait":  "#4C4879",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
})
sns.set_theme(style="whitegrid", palette="tab10")


def savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true, y_pred) -> float:
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100)


def rmse(yt, yp) -> float:
    return float(np.sqrt(mean_squared_error(yt, yp)))


# ── Feature cleaning ───────────────────────────────────────────────────────────

def clean_features(X: np.ndarray) -> np.ndarray:
    """Replace Inf/-Inf with NaN, then impute column medians. Mirrors notebook."""
    X = X.astype(float)
    X[~np.isfinite(X)] = np.nan
    col_medians = np.nanmedian(X, axis=0)
    nan_mask    = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


def prepare_xy(
    subset: pd.DataFrame, feature_cols: list[str], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return clean X, y arrays — drop rows where target is NaN."""
    sub = subset[feature_cols + [target]].copy()
    sub = sub[sub[target].notna()]
    X = clean_features(sub[feature_cols].values)
    y = sub[target].values
    return X, y


# ── Statistical 1-step forecasters ────────────────────────────────────────────

def naive_1step(train_series: pd.Series) -> float:
    return train_series.iloc[-1]


def linear_trend_1step(train_series: pd.Series) -> float:
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(train_series)).reshape(-1, 1)
    m = LinearRegression().fit(X, train_series.values)
    return float(m.predict([[len(train_series)]])[0])


def holt_1step(train_series: pd.Series) -> float:
    try:
        m = ExponentialSmoothing(
            train_series.values, trend="add", damped_trend=True
        ).fit(optimized=True)
        return float(m.forecast(1)[0])
    except Exception:
        return train_series.iloc[-1]


def arima_1step(train_series: pd.Series) -> float:
    try:
        m = ARIMA(np.asarray(train_series.values), order=(1, 1, 1)).fit()
        return float(m.forecast(1)[0])
    except Exception:
        return train_series.iloc[-1]


def ml_1step(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    model_cls,
    feature_cols: list[str],
    target: str,
    scaler: StandardScaler | None = None,
    **kwargs,
) -> float:
    """1-step ML prediction — robust to Inf/NaN via clean_features()."""
    tr   = train_df[feature_cols + [target]].copy()
    tr   = tr[tr[target].notna()]
    X_tr = clean_features(tr[feature_cols].values)
    y_tr = tr[target].values

    if len(X_tr) < 3:
        return float(y_tr[-1]) if len(y_tr) else 0.0

    if scaler:
        X_tr = scaler.fit_transform(X_tr)

    model = model_cls(**kwargs)
    model.fit(X_tr, y_tr)

    x_pred = clean_features(
        np.array(test_row[feature_cols].values, dtype=float).reshape(1, -1)
    )
    if scaler:
        x_pred = scaler.transform(x_pred)

    return float(model.predict(x_pred)[0])


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_hyperparameters(
    df: pd.DataFrame, all_features: list[str], target: str, val_end: int, model_dir: str
) -> tuple[dict, list[str]]:
    """
    Tune Ridge, RF, XGBoost on Train→Val via TimeSeriesSplit.
    Mirrors notebook Section 2 exactly.

    Returns (best_hp dict, GOOD_FEATURES list)
    """
    df_tune    = df[df["Year"] <= val_end].copy()
    feat_check = df_tune[all_features].replace([np.inf, -np.inf], np.nan)
    null_frac  = feat_check.isnull().mean()
    good_features = [c for c in all_features if null_frac[c] < 0.80]
    dropped = set(all_features) - set(good_features)
    if dropped:
        log.warning("Dropped %d high-null features: %s", len(dropped), dropped)
    log.info("Features used for tuning: %d", len(good_features))

    X_tune, y_tune = prepare_xy(df_tune, good_features, target)
    log.info("Tune set shape: X=%s, y=%s", X_tune.shape, y_tune.shape)

    tscv = TimeSeriesSplit(n_splits=3)

    # Ridge
    ridge_grid = GridSearchCV(
        Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]},
        cv=tscv, scoring="neg_mean_absolute_error", refit=True,
    )
    ridge_grid.fit(X_tune, y_tune)
    best_ridge_alpha = ridge_grid.best_params_["alpha"]
    log.info("Best Ridge alpha: %s", best_ridge_alpha)

    # Random Forest
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        cv=tscv, scoring="neg_mean_absolute_error", refit=True,
    )
    rf_grid.fit(X_tune, y_tune)
    best_rf_params = rf_grid.best_params_
    log.info("Best RF params: %s", best_rf_params)

    # XGBoost
    best_xgb_params: dict = {}
    if HAS_XGB:
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(verbosity=0, random_state=42, tree_method="hist"),
            {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            cv=tscv, scoring="neg_mean_absolute_error", refit=True,
        )
        xgb_grid.fit(X_tune, y_tune)
        best_xgb_params = xgb_grid.best_params_
        log.info("Best XGB params: %s", best_xgb_params)
    else:
        log.warning("XGBoost not installed — skipping tuning")

    best_hp = {
        "Ridge":        {"alpha": best_ridge_alpha},
        "RandomForest": best_rf_params,
        "XGBoost":      best_xgb_params,
    }

    with open(os.path.join(model_dir, "best_hp.json"), "w") as f:
        json.dump(best_hp, f, indent=2)
    log.info("Saved best_hp.json")

    return best_hp, good_features


# ── Walk-forward evaluation ────────────────────────────────────────────────────

def walk_forward_evaluate(
    df: pd.DataFrame,
    all_features: list[str],
    target: str,
    years: list[int],
    best_hp: dict,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward over `years`.
    Mirrors notebook Sections 3 & 4 (shared logic for test + val).
    """
    best_ridge_alpha = best_hp["Ridge"]["alpha"]
    best_rf_params    = best_hp["RandomForest"]
    best_xgb_params   = best_hp["XGBoost"]

    all_results: list[dict] = []

    for country in COUNTRIES:
        sub = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)

        for t_year in years:
            train_df = sub[sub["Year"] < t_year]
            test_row = sub[sub["Year"] == t_year]

            if test_row.empty or len(train_df) < 5:
                continue

            y_actual     = test_row[target].values[0]
            train_series = train_df[target]

            preds: dict[str, float] = {}
            preds["Naive"]       = naive_1step(train_series)
            preds["LinearTrend"] = linear_trend_1step(train_series)
            preds["Holt"]        = holt_1step(train_series)
            preds["ARIMA_1_1_1"] = arima_1step(train_series)

            try:
                preds["Ridge"] = ml_1step(
                    train_df, test_row.iloc[0], Ridge,
                    all_features, target, scaler=StandardScaler(),
                    alpha=best_ridge_alpha,
                )
            except Exception:
                preds["Ridge"] = train_series.iloc[-1]

            try:
                preds["RandomForest"] = ml_1step(
                    train_df, test_row.iloc[0], RandomForestRegressor,
                    all_features, target, scaler=None,
                    random_state=42, n_jobs=-1, **best_rf_params,
                )
            except Exception:
                preds["RandomForest"] = train_series.iloc[-1]

            if HAS_XGB:
                try:
                    preds["XGBoost"] = ml_1step(
                        train_df, test_row.iloc[0], xgb.XGBRegressor,
                        all_features, target, scaler=None,
                        verbosity=0, random_state=42, tree_method="hist",
                        **best_xgb_params,
                    )
                except Exception:
                    preds["XGBoost"] = train_series.iloc[-1]

            for model_name, y_pred in preds.items():
                all_results.append({
                    "Country":  country,
                    "Year":     t_year,
                    "Model":    model_name,
                    "y_actual": y_actual,
                    "y_pred":   float(y_pred),
                    "error":    y_actual - float(y_pred),
                    "abs_pct_error": abs(y_actual - float(y_pred)) / (abs(y_actual) + 1e-9) * 100,
                })

    return pd.DataFrame(all_results)


def agg_metrics(df_r: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MAE/RMSE/MAPE per Country x Model. Mirrors notebook agg_metrics()."""
    rows = []
    for (country, model), g in df_r.groupby(["Country", "Model"]):
        rows.append({
            "Country": country,
            "Model":   model,
            "MAE":     mean_absolute_error(g["y_actual"], g["y_pred"]),
            "RMSE":    rmse(g["y_actual"], g["y_pred"]),
            "MAPE":    mape(g["y_actual"], g["y_pred"]),
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_split_viz(df: pd.DataFrame, target: str, train_end: int, val_end: int, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
    axes = axes.flatten()
    zone_c = {"Train": "#3a86ff", "Val": "#ff9f1c", "Test": "#e63946"}
    for i, country in enumerate(COUNTRIES):
        ax  = axes[i]
        sub = df[df["Area"] == country].sort_values("Year")
        tr  = sub[sub["Year"] <= train_end]
        va  = sub[(sub["Year"] > train_end) & (sub["Year"] <= val_end)]
        te  = sub[sub["Year"] > val_end]
        ax.plot(tr["Year"], tr[target], color=zone_c["Train"], lw=2, label="Train")
        ax.plot(va["Year"], va[target], color=zone_c["Val"],   lw=2, label="Val")
        ax.plot(te["Year"], te[target], color=zone_c["Test"],  lw=2, label="Test")
        ax.axvline(train_end + 0.5, color="grey", ls="--", lw=1)
        ax.axvline(val_end   + 0.5, color="grey", ls=":",  lw=1)
        ax.set_title(country, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle("Train / Validation / Test Split by Country", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_split.pdf"))


def plot_mape_heatmaps(test_metrics: pd.DataFrame, val_metrics: pd.DataFrame, fig_dir: str) -> None:
    mape_piv     = test_metrics.pivot(index="Country", columns="Model", values="MAPE")
    val_mape_piv = val_metrics.pivot(index="Country",  columns="Model", values="MAPE")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.heatmap(mape_piv, cmap="RdYlGn_r", annot=True, fmt=".1f", ax=axes[0],
                linewidths=0.4, cbar_kws={"label": "MAPE %"})
    axes[0].set_title("TEST — MAPE (%) per Country × Model", fontweight="bold")
    sns.heatmap(val_mape_piv, cmap="RdYlGn_r", annot=True, fmt=".1f", ax=axes[1],
                linewidths=0.4, cbar_kws={"label": "MAPE %"})
    axes[1].set_title("VALIDATION — MAPE (%) per Country × Model", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_mape_heatmap.pdf"))


def plot_val_vs_test(val_metrics: pd.DataFrame, test_metrics: pd.DataFrame, fig_dir: str) -> None:
    merged = (
        val_metrics[["Country", "Model", "MAPE"]].rename(columns={"MAPE": "Val_MAPE"})
        .merge(
            test_metrics[["Country", "Model", "MAPE"]].rename(columns={"MAPE": "Test_MAPE"}),
            on=["Country", "Model"],
        )
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in merged["Model"].unique():
        sub = merged[merged["Model"] == m]
        ax.scatter(sub["Val_MAPE"], sub["Test_MAPE"], label=m, s=60)
    max_v = merged[["Val_MAPE", "Test_MAPE"]].max().max()
    ax.plot([0, max_v], [0, max_v], "k--", lw=0.8, label="Val=Test line")
    ax.set_xlabel("Val MAPE (%)")
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Val vs Test MAPE — Overfitting Check", fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_val_vs_test.pdf"))


def plot_walk_forward_test(
    df: pd.DataFrame, res: pd.DataFrame, best_test: pd.DataFrame,
    target: str, train_end: int, val_end: int, test_end: int, fig_dir: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.flatten()
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        full   = df[df["Area"] == country].sort_values("Year")
        best_m = best_test[best_test["Country"] == country]["Model"].values[0]
        ax.plot(full["Year"], full[target], "k-", lw=2, label="Actual", zorder=5)
        c_res = res[res["Country"] == country]
        for j, model in enumerate(c_res["Model"].unique()):
            m_res = c_res[c_res["Model"] == model]
            lw = 2.5 if model == best_m else 1
            ls = "-"  if model == best_m else "--"
            ax.plot(m_res["Year"], m_res["y_pred"], ls=ls, lw=lw,
                    color=clrs[j % len(clrs)], label=model, alpha=0.85)
        ax.axvspan(val_end + 0.5, test_end + 0.5, alpha=0.06, color="red", label="Test zone")
        ax.axvspan(train_end + 0.5, val_end + 0.5, alpha=0.06, color="orange", label="Val zone")
        ax.set_title(f"{country}\n[Best: {best_m}]", fontweight="bold", fontsize=9)
        ax.set_xlabel("Year"); ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        if i == 0:
            ax.legend(fontsize=6, ncol=2)
    axes[-1].set_visible(False)
    fig.suptitle("Walk-Forward Test Predictions vs Actual (all models)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_walk_forward.pdf"))


def plot_residuals(
    df: pd.DataFrame, res: pd.DataFrame, best_test: pd.DataFrame, fig_dir: str
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    axes = axes.flatten()
    for i, country in enumerate(COUNTRIES):
        best_m = best_test[best_test["Country"] == country]["Model"].values[0]
        r = res[(res["Country"] == country) & (res["Model"] == best_m)]
        clrs_bar = ["tomato" if e < 0 else "steelblue" for e in r["error"]]
        axes[i].bar(r["Year"], r["error"], color=clrs_bar)
        axes[i].axhline(0, color="black", lw=0.8)
        axes[i].set_title(f"{country} — {best_m}", fontweight="bold", fontsize=9)
        axes[i].set_xlabel("Year"); axes[i].set_ylabel("Residual (TWh)")
    axes[-1].set_visible(False)
    fig.suptitle("Test Residuals — Best Model per Country", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_residuals.pdf"))


def plot_skill_score(test_metrics: pd.DataFrame, fig_dir: str) -> None:
    naive_mape = (
        test_metrics[test_metrics["Model"] == "Naive"][["Country", "MAPE"]]
        .rename(columns={"MAPE": "naive_mape"})
    )
    skill_df = test_metrics.merge(naive_mape, on="Country", how="inner")
    skill_df["skill"] = (1 - skill_df["MAPE"] / skill_df["naive_mape"]) * 100
    skill_df = skill_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["skill"])
    skill_piv = skill_df.pivot(index="Country", columns="Model", values="skill")

    if skill_piv.empty:
        log.warning("Skill score pivot empty — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(skill_piv, cmap="RdYlGn", annot=True, fmt=".1f", ax=ax,
                center=0, linewidths=0.4, cbar_kws={"label": "Skill vs Naïve (%)"})
    ax.set_title("Skill Score vs Naïve Baseline (Test Set) — Higher is Better",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_skill_score.pdf"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember modeling/benchmarking step")
    p.add_argument("--input_dir",  default="outputs/preprocessing", help="Preprocessing output dir")
    p.add_argument("--output_dir", default="outputs/modeling",      help="Output dir")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir,         exist_ok=True)

    # Load
    df = pd.read_csv(os.path.join(args.input_dir, "ember_model_ready.csv"))
    with open(os.path.join(args.input_dir, "feature_meta.json")) as f:
        meta = json.load(f)

    all_features = meta["all_features"]
    train_end = meta.get("TRAIN_END", 2016)
    val_end   = meta.get("VAL_END", 2020)
    test_end  = meta.get("TEST_END", 2024)

    log.info("Dataset shape: %s | Features: %d", df.shape, len(all_features))
    log.info("Split: Train≤%d | Val %d-%d | Test %d-%d",
             train_end, train_end + 1, val_end, val_end + 1, test_end)

    # 1. Split visualization
    plot_split_viz(df, TARGET, train_end, val_end, fig_dir)

    # 2. Hyperparameter tuning
    best_hp, good_features = tune_hyperparameters(
        df, all_features, TARGET, val_end, args.output_dir
    )

    # 3. Walk-forward on TEST
    test_years = list(range(val_end + 1, test_end + 1))
    log.info("── Walk-forward TEST evaluation: years %s", test_years)
    res = walk_forward_evaluate(df, good_features, TARGET, test_years, best_hp)
    log.info("Test walk-forward results: %s", res.shape)

    # 4. Walk-forward on VAL
    val_years = list(range(train_end + 1, val_end + 1))
    log.info("── Walk-forward VAL evaluation: years %s", val_years)
    val_res = walk_forward_evaluate(df, good_features, TARGET, val_years, best_hp)
    log.info("Val walk-forward results: %s", val_res.shape)

    # 5. Aggregate metrics
    test_metrics = agg_metrics(res)
    val_metrics  = agg_metrics(val_res)
    best_test = test_metrics.loc[
        test_metrics.groupby("Country")["MAPE"].idxmin()
    ].reset_index(drop=True)
    log.info("Best model per country (Test MAPE):\n%s", best_test.to_string())

    # Plots
    plot_mape_heatmaps(test_metrics, val_metrics, fig_dir)
    plot_val_vs_test(val_metrics, test_metrics, fig_dir)
    plot_walk_forward_test(df, res, best_test, TARGET, train_end, val_end, test_end, fig_dir)
    plot_residuals(df, res, best_test, fig_dir)
    plot_skill_score(test_metrics, fig_dir)

    # 6. Save
    test_metrics.to_csv(os.path.join(args.output_dir, "test_benchmarking.csv"), index=False)
    val_metrics.to_csv(os.path.join(args.output_dir, "val_benchmarking.csv"),   index=False)
    best_test.to_csv(os.path.join(args.output_dir, "best_models.csv"),          index=False)
    res.to_csv(os.path.join(args.output_dir, "wf_test_predictions.csv"),        index=False)

    log.info("=== Modeling Complete ===")
    log.info("  test_benchmarking.csv   : %s", test_metrics.shape)
    log.info("  val_benchmarking.csv    : %s", val_metrics.shape)
    log.info("  best_models.csv         : %s", best_test.shape)
    log.info("  wf_test_predictions.csv : %s", res.shape)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
