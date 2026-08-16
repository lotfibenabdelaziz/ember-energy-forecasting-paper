"""
plot_direction_flip.py — Classical vs. DL Forecast Direction-Flip Figure
============================================================================
Ember Energy | IEEE Paper — standalone figure generation, not part of the
main pipeline (step04/step05 run in sequence, so DL forecasts don't exist
yet at the point step04's plots.py functions run — this script runs AFTER
both steps have completed, reading their real CSV outputs directly).

Produces the 4-panel figure for the 4 countries where classical and DL
models disagree on forecast DIRECTION (not just magnitude): Egypt,
Tunisia, France, Canada — see Table \\ref{tab:direction_flip} in the paper.

Usage:
    python plot_direction_flip.py
    (run from project root, after both `make run` and the DL step have
    completed — reads outputs/forecasting/ and outputs/deeplearning/ directly)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})

# The 4 countries where classical and DL CAGR have opposite signs —
# confirmed directly from demand_growth_summary.csv / dl_growth_summary.csv:
#   Egypt:   classical +2.49%  vs  DL -2.56%
#   Tunisia: classical +1.94%  vs  DL -1.10%
#   France:  classical +0.60%  vs  DL -0.06%
#   Canada:  classical -0.03%  vs  DL +0.18%
FLIP_COUNTRIES = ["Egypt", "Tunisia", "France", "Canada"]

CLASSICAL_COLOR = "#3a86ff"
DL_COLOR        = "#00d4a0"
HIST_COLOR      = "#2b2b2b"


def load_data(
    hist_path: str = "outputs/preprocessing/ember_model_ready.csv",
    classical_fc_path: str = "outputs/forecasting/demand_forecast_2025_2030.csv",
    dl_fc_path: str = "outputs/deeplearning/dl_forecast_2025_2030.csv",
    target: str = "Demand",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load historical, classical forecast, and DL forecast data."""
    hist = pd.read_csv(hist_path)
    classical_fc = pd.read_csv(classical_fc_path)
    dl_fc = pd.read_csv(dl_fc_path)
    return hist, classical_fc, dl_fc, target


def plot_direction_flip_comparison(
    hist: pd.DataFrame,
    classical_fc: pd.DataFrame,
    dl_fc: pd.DataFrame,
    target: str,
    fig_dir: str = "outputs/forecasting/figures",
    countries: list[str] = FLIP_COUNTRIES,
    hist_years_shown: int = 10,
) -> None:
    """
    2x2 grid, one panel per direction-flip country, overlaying the
    classical forecast (solid blue) and DL forecast (dashed green) on the
    same axes against recent historical context — makes the direction
    disagreement visually immediate rather than only readable from a
    CAGR table.
    """
    n_cols = 2
    n_rows = (len(countries) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.5 * n_rows))
    axes = axes.flatten()

    for i, country in enumerate(countries):
        ax = axes[i]

        h = hist[hist["Area"] == country].sort_values("Year")
        h_recent = h[h["Year"] >= h["Year"].max() - hist_years_shown]

        c_fc = classical_fc[classical_fc["Country"] == country].sort_values("Year")
        d_fc = dl_fc[dl_fc["Country"] == country].sort_values("Year")

        if h_recent.empty or c_fc.empty or d_fc.empty:
            ax.set_visible(False)
            continue

        # Historical context (solid black)
        ax.plot(h_recent["Year"], h_recent[target], color=HIST_COLOR,
                lw=2, marker="o", ms=4, label="Historical", zorder=5)

        # Bridge line from last historical point to each forecast's first year
        last_year, last_val = h_recent["Year"].values[-1], h_recent[target].values[-1]

        # Classical forecast (solid blue)
        c_model = c_fc["Model"].values[0] if "Model" in c_fc.columns else "Classical"
        ax.plot([last_year, c_fc["Year"].values[0]], [last_val, c_fc["Forecast"].values[0]],
                color=CLASSICAL_COLOR, lw=1.2, ls="--", alpha=0.5)
        ax.plot(c_fc["Year"], c_fc["Forecast"], color=CLASSICAL_COLOR,
                lw=2.5, marker="D", ms=5, label=f"Classical ({c_model})")
        if "Lower_90" in c_fc.columns and "Upper_90" in c_fc.columns:
            ax.fill_between(c_fc["Year"], c_fc["Lower_90"], c_fc["Upper_90"],
                             alpha=0.12, color=CLASSICAL_COLOR)

        # DL forecast (dashed green)
        d_model = d_fc["Model"].values[0] if "Model" in d_fc.columns else "DL"
        ax.plot([last_year, d_fc["Year"].values[0]], [last_val, d_fc["Forecast"].values[0]],
                color=DL_COLOR, lw=1.2, ls="--", alpha=0.5)
        ax.plot(d_fc["Year"], d_fc["Forecast"], color=DL_COLOR,
                lw=2.5, ls="--", marker="^", ms=6, label=f"DL ({d_model})")
        if "Lower_90" in d_fc.columns and "Upper_90" in d_fc.columns:
            ax.fill_between(d_fc["Year"], d_fc["Lower_90"], d_fc["Upper_90"],
                             alpha=0.12, color=DL_COLOR)

        ax.axvline(last_year + 0.5, color="grey", ls=":", lw=1)
        ax.set_title(country, fontweight="bold", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Demand (TWh)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
        ax.legend(fontsize=7.5, frameon=False, loc="best")

    for j in range(len(countries), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Classical vs. Deep Learning Forecast Direction Divergence (2025\u20132030)\n"
        "Countries where the two model families disagree on growth vs. decline",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "direction_flip_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    hist, classical_fc, dl_fc, target = load_data()
    plot_direction_flip_comparison(hist, classical_fc, dl_fc, target)
