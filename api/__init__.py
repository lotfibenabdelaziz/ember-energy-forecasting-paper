"""
api/__init__.py — Ember Energy Forecasting API package
=======================================================
Exposes the FastAPI app and key constants at package level so that:

    from api import app                   # use in uvicorn / tests
    from api import COUNTRIES             # use in other modules
    from api.main import app              # also works (direct import)

The heavy loading (CSV reads) lives in api/main.py and happens
once at import time — this file just re-exports the public surface.
"""

from api.main import (
    BEST_MODELS_CSV,
    COUNTRIES,
    DL_BEST_CSV,
    DL_FORECAST_CSV,
    DL_METRICS_CSV,
    FORECAST_CSV,
    GROWTH_CSV,
    METRICS_CSV,
    app,
)

__all__ = [
    "app",
    "COUNTRIES",
    "FORECAST_CSV",
    "METRICS_CSV",
    "GROWTH_CSV",
    "BEST_MODELS_CSV",
    "DL_FORECAST_CSV",
    "DL_METRICS_CSV",
    "DL_BEST_CSV",
]
