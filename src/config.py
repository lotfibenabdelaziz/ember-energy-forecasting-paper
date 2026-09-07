"""
src/config.py — Central Configuration Reader
==============================================
Ember Energy | IEEE Paper

SINGLE SOURCE OF TRUTH: params.yaml (DVC-tracked, repo root).
Every pipeline parameter — countries, splits, feature engineering, model
hyperparameters — is defined there ONCE. This module reads params.yaml
and exposes it as a typed, importable object.

Environment variables (.env) are for things that legitimately vary per
deployment and should NOT live in params.yaml: paths, secrets, ports,
MLflow URI, dev/prod switches. An env var, if set, overrides the
matching params.yaml value — this lets CI/Docker/k8s override a path or
a port without editing params.yaml, while params.yaml stays the one
place pipeline *behavior* (splits, countries, hyperparameters) is
defined and reviewed.

Import anywhere:
    from src.config import cfg
    cfg.train_end       # 2016 (from params.yaml, unless TRAIN_END env overrides)
    cfg.forecast_years  # [2025, ..., 2030]
"""

from __future__ import annotations

import os
import tempfile

# ── Windows home/cache-dir fix ────────────────────────────────────────────────
# pathlib.Path.home() on Windows does NOT check HOME — it checks USERPROFILE,
# then HOMEDRIVE+HOMEPATH. parso (pulled in by seaborn -> ipywidgets -> IPython
# -> jedi -> parso) additionally checks LOCALAPPDATA before ever calling
# Path.home(). In this environment one or more of these vars isn't reaching
# the Python subprocess, so we backstop all of them with a guaranteed-valid
# temp directory before any heavy import happens anywhere downstream.
if os.name == "nt":
    _fallback_dir = tempfile.gettempdir()
    os.environ.setdefault("USERPROFILE", _fallback_dir)
    os.environ.setdefault("LOCALAPPDATA", _fallback_dir)
    os.environ.setdefault("HOMEDRIVE", os.path.splitdrive(_fallback_dir)[0] or "C:")
    os.environ.setdefault("HOMEPATH", os.path.splitdrive(_fallback_dir)[1] or "\\Temp")
    os.environ.setdefault("HOME", _fallback_dir)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_fallback_dir, "mplcache"))

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

yaml: ModuleType | None
try:
    import yaml
except ImportError:  # pragma: no cover — PyYAML is a core dependency; this
    yaml = None  # only guards against a broken/partial install.


# ── Load params.yaml once, at import time ────────────────────────────────────
def _find_params_yaml() -> Path | None:
    """Look for params.yaml starting at CWD and walking up — works whether
    the process is launched from repo root, src/, or a Docker WORKDIR."""
    here = Path(__file__).resolve().parent
    for candidate_dir in [Path.cwd(), here, here.parent]:
        p = candidate_dir / "params.yaml"
        if p.exists():
            return p
    return None


def _load_params() -> dict:
    path = _find_params_yaml()
    if path is None or yaml is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_PARAMS = _load_params()


def _yaml(path: str, default: Any) -> Any:
    """Dotted-path lookup into the loaded params.yaml, e.g. 'splits.train_end'."""
    node = _PARAMS
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


# ── env-var helpers (override params.yaml when explicitly set) ───────────────


def _int(key: str, yaml_path: str, default: int) -> int:
    return int(os.getenv(key, str(_yaml(yaml_path, default))))


def _float(key: str, yaml_path: str, default: float) -> float:
    return float(os.getenv(key, str(_yaml(yaml_path, default))))


def _str(key: str, yaml_path: str, default: str) -> str:
    val = os.getenv(key)
    if val is not None:
        return val.strip()
    return str(_yaml(yaml_path, default)).strip()


def _list(key: str, yaml_path: str, default: list[str]) -> list[str]:
    val = os.getenv(key, "")
    if val:
        return [v.strip() for v in val.split(",")]
    return list(_yaml(yaml_path, default))


@dataclass(frozen=True)
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    csv_path: str = field(
        default_factory=lambda: _str(
            "CSV_PATH", "data.raw_csv", "data/raw/yearly_full_release_long_format.csv"
        )
    )
    target: str = field(default_factory=lambda: _str("TARGET", "countries.target", "Demand"))
    countries: list[str] = field(
        default_factory=lambda: _list(
            "COUNTRIES",
            "countries.list",
            ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"],
        )
    )
    drop_cols: list[str] = field(
        default_factory=lambda: _list(
            "DROP_COLS", "countries.drop_cols", ["Total", "Aggregate_fuel"]
        )
    )

    # ── Splits (source of truth: params.yaml `splits:`) ─────────────────────────
    train_end: int = field(default_factory=lambda: _int("TRAIN_END", "splits.train_end", 2016))
    val_end: int = field(default_factory=lambda: _int("VAL_END", "splits.val_end", 2020))
    test_end: int = field(default_factory=lambda: _int("TEST_END", "splits.test_end", 2024))
    forecast_start: int = field(
        default_factory=lambda: _int("FORECAST_START", "splits.forecast_start", 2025)
    )
    forecast_end: int = field(
        default_factory=lambda: _int("FORECAST_END", "splits.forecast_end", 2030)
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_uri: str = field(
        default_factory=lambda: _str(
            "MLFLOW_TRACKING_URI", "mlflow.tracking_uri", "http://localhost:5000"
        )
    )
    mlflow_experiment: str = field(
        default_factory=lambda: _str(
            "MLFLOW_EXPERIMENT", "mlflow.experiment", "ember-demand-forecasting"
        )
    )

    # ── Outputs (source of truth: params.yaml `paths:`, flattened to one root) ──
    output_root: str = field(default_factory=lambda: _str("OUTPUT_ROOT", "paths.root", "outputs"))

    # ── Deep learning (source of truth: params.yaml `deeplearning:`) ────────────
    seq_len: int = field(default_factory=lambda: _int("SEQ_LEN", "deeplearning.seq_len", 5))
    epochs: int = field(default_factory=lambda: _int("EPOCHS", "deeplearning.epochs", 300))
    patience: int = field(default_factory=lambda: _int("PATIENCE", "deeplearning.patience", 40))
    batch_size: int = field(
        default_factory=lambda: _int("BATCH_SIZE", "deeplearning.batch_size", 16)
    )
    lr: float = field(default_factory=lambda: _float("LR", "deeplearning.lr", 1e-3))
    seed: int = field(default_factory=lambda: _int("SEED", "deeplearning.seed", 42))

    # ── API / deployment (env-only — these never belong in params.yaml) ─────────
    port: int = field(default_factory=lambda: _int("PORT", "__no_yaml__", 8000))
    env: str = field(default_factory=lambda: _str("ENV", "__no_yaml__", "development"))

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

    @property
    def params_yaml_loaded(self) -> bool:
        """True if params.yaml was actually found and parsed. False means every
        value above is falling back to its hardcoded default — worth knowing
        when debugging a config drift."""
        return bool(_PARAMS)

    def summary(self) -> str:
        return (
            f"Config(train={self.train_end}, val={self.val_end}, "
            f"test={self.test_end}, forecast={self.forecast_years}, "
            f"countries={len(self.countries)}, "
            f"params_yaml_loaded={self.params_yaml_loaded})"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
cfg = Config()
