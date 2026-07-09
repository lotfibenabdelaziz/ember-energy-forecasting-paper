"""
src/config.py — Central Configuration Reader
==============================================
Ember Energy | IEEE Paper

Single source of truth for all runtime configuration.
Reads from environment variables (loaded from .env by pipeline.py).

Usage in any module:
    from src.config import cfg

    print(cfg.train_end)      # 2024
    print(cfg.countries)      # ["Tunisia", "Austria", ...]
    print(cfg.forecast_years) # [2025, 2026, ..., 2030]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _list(key: str, default: list[str]) -> list[str]:
    val = os.getenv(key, "")
    return [v.strip() for v in val.split(",")] if val else default


@dataclass(frozen=True)
class Config:
    """
    All project configuration — frozen dataclass (immutable after init).
    Populated once from environment variables at import time.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    csv_path: str = field(
        default_factory=lambda: _str(
            "CSV_PATH",
            "data/raw/yearly_full_release_long_format.csv"
        )
    )
    target: str = field(default_factory=lambda: _str("TARGET", "Demand"))
    countries: list[str] = field(
        default_factory=lambda: _list(
            "COUNTRIES",
            ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
        )
    )
    drop_cols: list[str] = field(
        default_factory=lambda: _list("DROP_COLS", ["Total", "Aggregate_fuel"])
    )

    # ── Splits ────────────────────────────────────────────────────────────────
    train_end:      int = field(default_factory=lambda: _int("TRAIN_END",  2024))
    val_end:        int = field(default_factory=lambda: _int("VAL_END",    2020))
    test_end:       int = field(default_factory=lambda: _int("TEST_END",   2024))
    forecast_start: int = field(default_factory=lambda: _int("FORECAST_START", 2025))
    forecast_end:   int = field(default_factory=lambda: _int("FORECAST_END",   2030))

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_uri:        str = field(default_factory=lambda: _str("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow_experiment: str = field(default_factory=lambda: _str("MLFLOW_EXPERIMENT", "ember-demand-forecasting"))
    mlflow_artifact:   str = field(default_factory=lambda: _str("MLFLOW_ARTIFACT_ROOT", "./mlruns/artifacts"))

    # ── Outputs ───────────────────────────────────────────────────────────────
    output_root: str = field(default_factory=lambda: _str("OUTPUT_ROOT", "outputs"))

    # ── Deep Learning ─────────────────────────────────────────────────────────
    seq_len:    int   = field(default_factory=lambda: _int("SEQ_LEN",    5))
    epochs:     int   = field(default_factory=lambda: _int("EPOCHS",     300))
    patience:   int   = field(default_factory=lambda: _int("PATIENCE",   40))
    batch_size: int   = field(default_factory=lambda: _int("BATCH_SIZE", 16))
    lr:         float = field(default_factory=lambda: _float("LR",       1e-3))
    seed:       int   = field(default_factory=lambda: _int("SEED",       42))

    # ── API ───────────────────────────────────────────────────────────────────
    port:           int = field(default_factory=lambda: _int("PORT", 8000))
    env:            str = field(default_factory=lambda: _str("ENV", "development"))
    admin_username: str = field(default_factory=lambda: _str("ADMIN_USERNAME", "admin"))

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def forecast_years(self) -> list[int]:
        return list(range(self.forecast_start, self.forecast_end + 1))

    @property
    def n_forecast(self) -> int:
        return len(self.forecast_years)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def eda_dir(self) -> str:
        return os.path.join(self.output_root, "eda")

    @property
    def pre_dir(self) -> str:
        return os.path.join(self.output_root, "preprocessing")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.output_root, "modeling")

    @property
    def forecast_dir(self) -> str:
        return os.path.join(self.output_root, "forecasting")

    @property
    def dl_dir(self) -> str:
        return os.path.join(self.output_root, "deeplearning")

    def summary(self) -> str:
        return (
            f"Config("
            f"train={self.train_end}, val={self.val_end}, "
            f"test={self.test_end}, forecast={self.forecast_years}, "
            f"countries={len(self.countries)}, "
            f"seq_len={self.seq_len}, epochs={self.epochs}"
            f")"
        )


# ── Singleton — import this everywhere ────────────────────────────────────────
cfg = Config()
