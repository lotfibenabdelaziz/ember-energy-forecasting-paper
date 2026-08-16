"""
compute_dl_robustness.py — Extend RRS scoring to deep learning forecasts
============================================================================
robustness.py's compute_robustness_score() was only ever run on
all_models_forecast.csv (classical models). This runs the exact same
function on dl_forecast_2025_2030.csv, reusing all thresholds/logic
unchanged, to produce a genuinely comparable robustness classification
for the DL forecasts shown in Fig. dl_forecast_per_country.

Usage: python compute_dl_robustness.py
Reads:  outputs/deeplearning/dl_forecast_2025_2030.csv
        outputs/deeplearning/dl_best_models.csv
        outputs/preprocessing/ember_model_ready.csv  (for last_known anchor)
Writes: outputs/deeplearning/dl_robustness_summary.csv
"""

import pandas as pd
from src.modeling.robustness import compute_robustness_score

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"


def main():
    dl_fc = pd.read_csv("outputs/deeplearning/dl_forecast_2025_2030.csv")
    dl_best = pd.read_csv("outputs/deeplearning/dl_best_models.csv")
    hist = pd.read_csv("outputs/preprocessing/ember_model_ready.csv")

    # Same anchor logic as compute_robustness_score()'s classical usage:
    # last observed real value per country (2024)
    last_known = {}
    for c in COUNTRIES:
        sub = hist[hist["Area"] == c].sort_values("Year")
        if not sub.empty:
            last_known[c] = float(sub[TARGET].values[-1])

    # Reuse the exact same function, unmodified, on DL data
    dl_robustness = compute_robustness_score(dl_fc, last_known)

    # Merge with accuracy (MAPE) — same structure as robustness_vs_accuracy.csv
    merged = dl_best.merge(
        dl_robustness[["Country", "Model", "max_deviation_pct", "max_yoy_change_pct",
                        "direction_flips", "robustness_flag"]],
        on=["Country", "Model"], how="left",
    )

    print(merged[["Country", "Model", "MAPE", "max_deviation_pct",
                   "max_yoy_change_pct", "direction_flips", "robustness_flag"]]
          .sort_values("max_deviation_pct", ascending=False).to_string(index=False))

    merged.to_csv("outputs/deeplearning/dl_robustness_summary.csv", index=False)
    print("\nSaved -> outputs/deeplearning/dl_robustness_summary.csv")


if __name__ == "__main__":
    main()
