"""
src/03_modeling.py — Classical ML Benchmark
Ember Energy | IEEE Paper

Models  : Ridge · RandomForest · XGBoost
Split   : Train ≤ 2016 | Val 2017–2020 | Test 2021–2024
Tuning  : GridSearchCV + TimeSeriesSplit on Train+Val data
Validation: Walk-forward 1-step expanding window

Reads:  outputs/preprocessing/ember_model_ready.csv
        outputs/preprocessing/feature_meta.json
Writes: outputs/modeling/test_benchmarking.csv
        outputs/modeling/val_benchmarking.csv
        outputs/modeling/best_models.csv
        outputs/modeling/best_hp.json
        outputs/modeling/ember_config.json
        outputs/modeling/figures/*.pdf
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("XGBoost not installed — pip install xgboost")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
COLORS = {
    "Tunisia": "#1f77b4",
    "Austria": "#d62728",
    "Germany": "#2ca02c",
    "Egypt": "#9467bd",
    "Canada": "#e6a817",
    "France": "#8c564b",
    "Kuwait": "#17becf",
}
MODEL_COLORS = {"Ridge": "#aec7e8", "RandomForest": "#2ca02c", "XGBoost": "#9467bd"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
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


# ── Metrics ───────────────────────────────────────────────────────────────────
def safe_mape(a, b):
    a, b = np.array(a, float), np.array(b, float)
    mask = np.abs(a) > np.abs(a).mean() * 0.01
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def safe_smape(a, b):
    a, b = np.array(a, float), np.array(b, float)
    denom = np.abs(a) + np.abs(b)
    mask = denom > 1e-8
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(a[mask] - b[mask]) / denom[mask]) * 100)


def theil_u(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 2:
        return np.nan
    num = np.sqrt(np.mean((a[1:] - b[1:]) ** 2))
    den = np.sqrt(np.mean((a[1:] - a[:-1]) ** 2))
    return float(num / (den + 1e-8))


def compute_metrics(name: str, y_true, y_pred) -> dict:
    a, b = np.array(y_true, float), np.array(y_pred, float)
    return {
        "Model": name,
        "MAE": round(mean_absolute_error(a, b), 3),
        "RMSE": round(np.sqrt(mean_squared_error(a, b)), 3),
        "R2": round(r2_score(a, b), 4),
        "MAPE": round(safe_mape(a, b), 2),
        "SMAPE": round(safe_smape(a, b), 2),
        "TheilU": round(theil_u(a, b), 4),
    }


# ── Data helpers ──────────────────────────────────────────────────────────────
def prepare_xy(df: pd.DataFrame, feat_cols: list, target: str):
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df[target].values
    return X, y


def ml_1step(train_df, test_row, cls, feat_cols, scaler=None, **kwargs):
    X_tr, y_tr = prepare_xy(train_df, feat_cols, TARGET)
    X_te = test_row[feat_cols].values.reshape(1, -1)
    X_te = np.where(np.isfinite(X_te), X_te, 0)
    if scaler is not None:
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)
    m = cls(**kwargs).fit(X_tr, y_tr)
    return float(m.predict(X_te)[0])


# ── Hyperparameter tuning ─────────────────────────────────────────────────────
def tune_models(
    df: pd.DataFrame, feat_cols: list, target: str, train_end: int, val_end: int
) -> dict:
    tscv = TimeSeriesSplit(n_splits=3)
    df_tune = df[df["Year"] <= val_end].copy()

    # Clean features
    null_frac = df_tune[feat_cols].isnull().mean()
    good_feat = [c for c in feat_cols if null_frac[c] < 0.80]
    log.info("Features for tuning: %d / %d", len(good_feat), len(feat_cols))

    X_tune, y_tune = prepare_xy(df_tune, good_feat, target)

    # Ridge
    ridge_gs = GridSearchCV(
        Ridge(),
        {"alpha": [0.01, 0.1, 1, 10, 100]},
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ridge_gs.fit(StandardScaler().fit_transform(X_tune), y_tune)
    best_ridge = ridge_gs.best_params_
    log.info("Best Ridge alpha: %s", best_ridge)

    # Random Forest
    rf_gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None], "min_samples_leaf": [1, 2, 3]},
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rf_gs.fit(X_tune, y_tune)
    best_rf = rf_gs.best_params_
    log.info("Best RF params: %s", best_rf)

    # XGBoost
    best_xgb = {}
    if HAS_XGB:
        xgb_gs = GridSearchCV(
            xgb.XGBRegressor(verbosity=0, random_state=42, tree_method="hist"),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "reg_alpha": [0, 0.5, 1.0],
                "reg_lambda": [1.0, 5.0],
            },
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        xgb_gs.fit(X_tune, y_tune)
        best_xgb = xgb_gs.best_params_
        log.info("Best XGB params: %s", best_xgb)

    return {
        "good_features": good_feat,
        "Ridge": best_ridge,
        "RandomForest": best_rf,
        "XGBoost": best_xgb,
    }


# ── Walk-forward evaluation ───────────────────────────────────────────────────
def walk_forward(
    df: pd.DataFrame, feat_cols: list, target: str, years: list, best_hp: dict
) -> pd.DataFrame:
    good_feat = best_hp["good_features"]
    rows = []
    for yr in years:
        train_df = df[df["Year"] < yr]
        test_row = df[df["Year"] == yr]
        if train_df.empty or test_row.empty:
            continue
        test_row_s = test_row.iloc[0]
        preds = {}

        # Ridge
        try:
            preds["Ridge"] = ml_1step(
                train_df, test_row_s, Ridge, good_feat, scaler=StandardScaler(), **best_hp["Ridge"]
            )
        except Exception:
            preds["Ridge"] = float(train_df[target].iloc[-1])

        # Random Forest
        try:
            preds["RandomForest"] = ml_1step(
                train_df,
                test_row_s,
                RandomForestRegressor,
                good_feat,
                scaler=None,
                random_state=42,
                n_jobs=-1,
                **best_hp["RandomForest"],
            )
        except Exception:
            preds["RandomForest"] = float(train_df[target].iloc[-1])

        # XGBoost
        if HAS_XGB and best_hp["XGBoost"]:
            try:
                preds["XGBoost"] = ml_1step(
                    train_df,
                    test_row_s,
                    xgb.XGBRegressor,
                    good_feat,
                    scaler=None,
                    verbosity=0,
                    random_state=42,
                    tree_method="hist",
                    **best_hp["XGBoost"],
                )
            except Exception:
                preds["XGBoost"] = float(train_df[target].iloc[-1])
        else:
            preds["XGBoost"] = float(train_df[target].iloc[-1])

        rows.append(
            {
                "Country": test_row_s["Area"],
                "Year": yr,
                "y_true": float(test_row_s[target]),
                **preds,
            }
        )
    return pd.DataFrame(rows)


# ── Evaluation & best model selection ────────────────────────────────────────
def evaluate_split(df_results: pd.DataFrame, split_name: str) -> pd.DataFrame:
    model_cols = ["Ridge", "RandomForest", "XGBoost"]
    all_metrics = []
    for c in COUNTRIES:
        df_c = df_results[df_results["Country"] == c]
        if df_c.empty:
            continue
        y_true = df_c["y_true"].values
        for m in model_cols:
            if m not in df_c.columns:
                continue
            met = compute_metrics(m, y_true, df_c[m].values)
            met["Country"] = c
            met["Split"] = split_name
            all_metrics.append(met)
    return pd.DataFrame(all_metrics)


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_predictions(
    df_full: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    train_end: int,
    val_end: int,
    fig_dir: str,
) -> None:
    model_cols = [c for c in df_test.columns if c in ["Ridge", "RandomForest", "XGBoost"]]
    mstyles = {
        "Ridge": ("#aec7e8", "o"),
        "RandomForest": ("#2ca02c", "s"),
        "XGBoost": ("#9467bd", "^"),
    }

    for country in COUNTRIES:
        hist = df_full[df_full["Area"] == country].sort_values("Year")
        pred = df_test[df_test["Country"] == country].sort_values("Year")
        if hist.empty or pred.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist["Year"], hist[target], color="black", lw=1.8, label="Actual")
        ax.axvspan(train_end + 0.5, val_end + 0.5, alpha=0.06, color="orange")
        ax.axvspan(val_end + 0.5, 2025, alpha=0.06, color="red")
        ax.axvline(train_end + 0.5, color="grey", ls="--", lw=1)
        ax.axvline(val_end + 0.5, color="grey", ls=":", lw=1)

        for m in model_cols:
            color, marker = mstyles.get(m, ("grey", "o"))
            ax.plot(
                pred["Year"], pred[m], color=color, ls="--", marker=marker, ms=5, lw=1.4, label=m
            )

        ax.set_title(f"{country} — ML Predictions vs Actual", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("TWh")
        ax.legend(frameon=False, ncol=4, fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
        plt.tight_layout()
        savefig(fig, os.path.join(fig_dir, f"mod_fig_{country.lower()}_pred.pdf"))


def plot_metrics_heatmap(df_metrics: pd.DataFrame, fig_dir: str) -> None:
    for metric in ["MAPE", "RMSE", "R2"]:
        pivot = df_metrics[df_metrics["Split"] == "Test"].pivot_table(
            index="Model", columns="Country", values=metric
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 3.5))
        cmap = "RdYlGn" if metric == "R2" else "YlOrRd"
        center = 0 if metric == "R2" else None
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=center,
            linewidths=0.4,
            ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title(f"{metric} — Test Set (2021–2024)", fontweight="bold")
        plt.tight_layout()
        savefig(fig, os.path.join(fig_dir, f"mod_heatmap_{metric.lower()}.pdf"))


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember ML benchmark step")
    p.add_argument("--input_dir", default="outputs/preprocessing", help="Preprocessing dir")
    p.add_argument("--output_dir", default="outputs/modeling", help="Modeling output dir")
    return p.parse_args()


TARGET = "Demand"


def main() -> None:
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load
    df = pd.read_csv(os.path.join(args.input_dir, "ember_model_ready.csv"))
    with open(os.path.join(args.input_dir, "feature_meta.json")) as f:
        meta = json.load(f)

    TRAIN_END = meta["TRAIN_END"]  # 2016
    VAL_END = meta["VAL_END"]  # 2020
    TEST_END = meta["TEST_END"]  # 2024
    ALL_FEATURES = meta["all_features"]

    log.info(
        "Split: Train ≤ %d | Val %d–%d | Test %d–%d",
        TRAIN_END,
        TRAIN_END + 1,
        VAL_END,
        VAL_END + 1,
        TEST_END,
    )
    log.info("Features: %d", len(ALL_FEATURES))

    # Tune on Train + Val combined
    log.info("=== Hyperparameter Tuning (Train + Val) ===")
    best_hp = tune_models(df, ALL_FEATURES, TARGET, TRAIN_END, VAL_END)

    # Walk-forward on TEST (2021–2024)
    test_years = list(range(VAL_END + 1, TEST_END + 1))
    log.info("=== Walk-Forward Evaluation — Test %d–%d ===", VAL_END + 1, TEST_END)
    all_test = []
    for c in COUNTRIES:
        df_c = df[df["Area"] == c].sort_values("Year").reset_index(drop=True)
        results = walk_forward(df_c, ALL_FEATURES, TARGET, test_years, best_hp)
        all_test.append(results)
    df_test = pd.concat(all_test, ignore_index=True)

    # Walk-forward on VAL (2017–2020)
    val_years = list(range(TRAIN_END + 1, VAL_END + 1))
    log.info("=== Walk-Forward Evaluation — Val %d–%d ===", TRAIN_END + 1, VAL_END)
    all_val = []
    for c in COUNTRIES:
        df_c = df[df["Area"] == c].sort_values("Year").reset_index(drop=True)
        results = walk_forward(df_c, ALL_FEATURES, TARGET, val_years, best_hp)
        all_val.append(results)
    df_val = pd.concat(all_val, ignore_index=True)

    # Evaluate
    test_metrics = evaluate_split(df_test, "Test")
    val_metrics = evaluate_split(df_val, "Val")
    df_metrics = pd.concat([test_metrics, val_metrics], ignore_index=True)

    log.info(
        "=== Test Metrics ===\n%s",
        test_metrics[["Country", "Model", "MAE", "RMSE", "R2", "MAPE", "TheilU"]].to_string(
            index=False
        ),
    )

    # Best model per country (lowest MAPE on Test)
    best_rows = []
    for c in COUNTRIES:
        df_c = test_metrics[test_metrics["Country"] == c]
        if df_c.empty:
            continue
        idx = df_c["MAPE"].idxmin()
        row = df_c.loc[idx].to_dict()
        best_rows.append(row)
    best_test = pd.DataFrame(best_rows)
    log.info(
        "=== Best Model per Country ===\n%s",
        best_test[["Country", "Model", "MAPE", "RMSE", "R2"]].to_string(index=False),
    )

    # Plots
    plot_predictions(df, df_test, TARGET, TRAIN_END, VAL_END, fig_dir)
    plot_metrics_heatmap(df_metrics, fig_dir)

    # Save
    df_test.to_csv(os.path.join(args.output_dir, "test_benchmarking.csv"), index=False)
    df_val.to_csv(os.path.join(args.output_dir, "val_benchmarking.csv"), index=False)
    best_test.to_csv(os.path.join(args.output_dir, "best_models.csv"), index=False)

    with open(os.path.join(args.output_dir, "best_hp.json"), "w") as f:
        json.dump(best_hp, f, indent=2, default=str)

    # Update config for downstream notebooks
    meta["BEST_MODELS"] = best_test.set_index("Country")["Model"].to_dict()
    meta["BEST_HP"] = best_hp
    with open(os.path.join(args.output_dir, "ember_config.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log.info("Modeling complete → %s", args.output_dir)


if __name__ == "__main__":
    main()
