"""
api/main.py — Ember Energy Forecasting REST API
================================================
Serves all pipeline outputs as JSON endpoints.

Splits  : Train 2000-2016 | Val 2017-2020 | Test 2021-2024 | Forecast 2025-2030
Countries: Tunisia · Austria · Germany · Egypt · Canada · France · Kuwait

Public endpoints  (no auth required):
    GET  /health                      — liveness check
    GET  /countries                   — list of 7 countries
    POST /auth/login                  — obtain JWT token
    GET  /docs                        — Swagger UI
    GET  /redoc                       — ReDoc

Protected endpoints (Bearer JWT required):
    GET  /forecast/{country}          — classical forecast 2025-2030 + CI
    GET  /forecast/dl/{country}       — deep learning forecast 2025-2030
    GET  /metrics/{country}           — classical model test metrics
    GET  /metrics/dl/{country}        — deep learning test metrics
    GET  /models/{country}            — best model (classical + DL)
    GET  /growth/{country}            — CAGR + growth summary
    GET  /compare/{country}           — classical vs DL side by side
    GET  /figures                     — list all pipeline figures
    GET  /figures/{filename}          — serve a figure file
    POST /auth/logout                 — revoke current token
    GET  /auth/me                     — current user info
    POST /ask                         — LangChain Q&A (rate limited)

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from datetime import timedelta
import logging
import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from api.auth import (
    _admin_password,
    _admin_username,
    _token_expire_minutes,
    bearer_scheme,
    create_access_token,
    decode_token,
    get_current_user,
    rate_limit,
    revoke_token,
)

load_dotenv()

log = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
_ROOT = Path(os.getenv("OUTPUT_ROOT", os.getenv("OUTPUTS_DIR", "outputs")))

FORECAST_CSV = _ROOT / "forecasting" / "demand_forecast_2025_2030.csv"
ALL_MODELS_FORECAST_CSV = _ROOT / "forecasting" / "all_models_forecast.csv"
METRICS_CSV = _ROOT / "modeling" / "test_benchmarking.csv"
GROWTH_CSV = _ROOT / "forecasting" / "demand_growth_summary.csv"
BEST_MODELS_CSV = _ROOT / "modeling" / "best_models.csv"
BENCHMARKING_CSV = _ROOT / "modeling" / "test_benchmarking.csv"
SIGNIFICANCE_TESTS_CSV = _ROOT / "modeling" / "significance_tests.csv"
SIGNIFICANCE_SUMMARY_CSV = _ROOT / "modeling" / "significance_summary.csv"
ROBUSTNESS_SUMMARY_CSV = _ROOT / "forecasting" / "robustness_summary.csv"
ROBUSTNESS_VS_ACCURACY_CSV = _ROOT / "forecasting" / "robustness_vs_accuracy.csv"
BEST_MODEL_CONFLICTS_CSV = _ROOT / "forecasting" / "best_model_conflicts.csv"
DL_FORECAST_CSV = _ROOT / "deeplearning" / "dl_forecast_2025_2030.csv"
DL_METRICS_CSV = _ROOT / "deeplearning" / "dl_benchmarking.csv"
DL_BEST_CSV = _ROOT / "deeplearning" / "dl_best_models.csv"
HISTORY_CSV = _ROOT / "preprocessing" / "ember_model_ready.csv"
SUBCATEGORY_CSV = _ROOT / "eda" / "ember_filtered.csv"
SUMMARY_STATS_CSV = _ROOT / "eda" / "table01_eda_statistics.csv"
MULTIFEATURE_CSV = _ROOT / "preprocessing" / "ember_multifeature.csv"

FIGURES_DIRS = [
    _ROOT / "eda" / "figures",
    _ROOT / "preprocessing" / "figures",
    _ROOT / "modeling" / "figures",
    _ROOT / "forecasting" / "figures",
    _ROOT / "deeplearning" / "figures",
]

COUNTRIES = [
    "Tunisia",
    "Austria",
    "Germany",
    "Egypt",
    "Canada",
    "France",
    "Kuwait",
]

# ── Data loader ───────────────────────────────────────────────────────────────


def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.warning("%s not found at %s", label, path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_all() -> dict[str, pd.DataFrame]:
    return {
        "forecast": _load(FORECAST_CSV, "forecast"),
        "all_models_forecast": _load(ALL_MODELS_FORECAST_CSV, "all_models_forecast"),
        "metrics": _load(METRICS_CSV, "metrics"),
        "growth": _load(GROWTH_CSV, "growth"),
        "best_models": _load(BEST_MODELS_CSV, "best_models"),
        "benchmark": _load(BENCHMARKING_CSV, "benchmark"),
        "dl_forecast": _load(DL_FORECAST_CSV, "dl_forecast"),
        "dl_metrics": _load(DL_METRICS_CSV, "dl_metrics"),
        "dl_best": _load(DL_BEST_CSV, "dl_best"),
        "significance_tests": _load(SIGNIFICANCE_TESTS_CSV, "significance_tests"),
        "significance_summary": _load(SIGNIFICANCE_SUMMARY_CSV, "significance_summary"),
        "robustness_summary": _load(ROBUSTNESS_SUMMARY_CSV, "robustness_summary"),
        "robustness_vs_accuracy": _load(ROBUSTNESS_VS_ACCURACY_CSV, "robustness_vs_accuracy"),
        "best_model_conflicts": _load(BEST_MODEL_CONFLICTS_CSV, "best_model_conflicts"),
        "history": _load(HISTORY_CSV, "history"),
        "subcategories": _load(SUBCATEGORY_CSV, "subcategories"),
        "summary_stats": _load(SUMMARY_STATS_CSV, "summary_stats"),
        "multifeature": _load(MULTIFEATURE_CSV, "multifeature"),
    }


_DATA: dict[str, pd.DataFrame] = _load_all()

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Ember Energy Forecasting API",
    description=(
        "REST API for per-country electricity demand forecasts (2025-2030) "
        "based on the Ember annual energy dataset. IEEE Paper.\n\n"
        "**Authentication:** POST `/auth/login` → copy `access_token` → "
        "click 🔒 Authorize → paste `Bearer <token>`"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*" if os.getenv("ENV") == "development" else "http://localhost:8000,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type", "*"],
    allow_credentials=True,
    max_age=600,
)


# ── Security headers ──────────────────────────────────────────────────────────
# NOTE: /docs, /redoc, /openapi.json are FastAPI's own auto-generated debug
# pages — their Swagger UI assets load from cdn.jsdelivr.net (a different
# domain than the dashboard's own cdnjs.cloudflare.com/Google Fonts CSP
# below), so they're deliberately excluded from this strict policy rather
# than loosening the dashboard's real CSP to accommodate a debug-only page.
_DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.path not in _DOCS_PATHS:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com "
                "https://fonts.googleapis.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com "
                "https://fonts.gstatic.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data:; "
                "connect-src 'self';"
            )
        if os.getenv("ENV") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ── Static files ──────────────────────────────────────────────────────────────
_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")

# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600  # seconds


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_country(country: str) -> str:
    match = next((c for c in COUNTRIES if c.lower() == country.lower()), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Country '{country}' not found. Available: {COUNTRIES}",
        )
    return match


def _require(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"{label} not available — run the pipeline first.",
        )
    return df


def _is_finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _metrics_dict(row: pd.Series) -> dict[str, Any]:
    return {
        str(k): round(float(v), 4)
        for k, v in row.items()
        if k not in ("Country", "Model") and _is_finite(v)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Global exception handler — never leak stack traces
# ═══════════════════════════════════════════════════════════════════════════════


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception: %s %s — %s", request.method, request.url, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC endpoints — no auth required
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/", include_in_schema=False)
def root() -> Response:
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/static/index.html")


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness check — returns API status and which output files are loaded."""
    return {
        "status": "ok",
        "outputs": {k: not v.empty for k, v in _DATA.items()},
    }


@app.get("/countries", tags=["meta"])
def get_countries() -> dict:
    """Return the list of 7 countries covered by the forecasts."""
    return {"countries": COUNTRIES}


# ── Auth ──────────────────────────────────────────────────────────────────────


@app.post("/auth/login", tags=["auth"], response_model=TokenResponse)
def login(body: LoginRequest, request: Request) -> TokenResponse:
    """
    Exchange username + password for a JWT Bearer token.
    Set ADMIN_USERNAME and ADMIN_PASSWORD in .env before use.
    """
    # Rate-limit login attempts
    rate_limit(request)

    pwd = _admin_password()
    if not pwd:
        raise HTTPException(
            status_code=503,
            detail="Auth not configured — set ADMIN_PASSWORD in .env",
        )
    if body.username != _admin_username() or body.password != pwd:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    expire_mins = _token_expire_minutes()
    token = create_access_token(
        data={"sub": body.username, "role": "admin"},
        expires_delta=timedelta(minutes=expire_mins),
    )
    return TokenResponse(
        access_token=token,
        expires_in=expire_mins * 60,
    )


@app.post("/auth/logout", tags=["auth"], dependencies=[Depends(get_current_user)])
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Revoke the current JWT token."""
    if credentials:
        revoke_token(credentials.credentials)
    return {"message": "Logged out successfully"}


@app.get("/auth/me", tags=["auth"], dependencies=[Depends(get_current_user)])
def me(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Return current authenticated user info."""
    user = decode_token(credentials.credentials)
    return {"username": user.get("sub"), "role": user.get("role")}


# ═══════════════════════════════════════════════════════════════════════════════
# PROTECTED endpoints — Bearer JWT required
# ═══════════════════════════════════════════════════════════════════════════════

# ── Classical forecast ────────────────────────────────────────────────────────


@app.get("/forecast/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)])
def get_forecast(country: str) -> dict:
    """Classical ML demand forecast (2025-2030) + 90% Bootstrap CI."""
    country = _validate_country(country)
    df = _require(_DATA["forecast"], "Forecast")

    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No forecast data for {country}")

    sub = sub.sort_values("Year")

    # Use actual model from forecast CSV (may differ from best_models.csv
    # if fallback was triggered e.g. "LinearTrend (fallback from Ridge)")
    result: dict[str, Any] = {
        "country": country,
        "model": sub["Model"].iloc[0] if "Model" in sub.columns and not sub.empty else "unknown",
        "forecast_years": sub["Year"].tolist(),
        "forecast_twh": [round(v, 3) for v in sub["Forecast"].tolist()],
    }
    if "Lower_90" in sub.columns:
        result["lower_90"] = [round(v, 3) for v in sub["Lower_90"].tolist()]
        result["upper_90"] = [round(v, 3) for v in sub["Upper_90"].tolist()]
    return result


# ── Deep learning forecast ────────────────────────────────────────────────────


@app.get("/forecast/dl/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)])
def get_dl_forecast(country: str) -> dict:
    """Deep learning demand forecast (2025-2030)."""
    country = _validate_country(country)
    df = _require(_DATA["dl_forecast"], "DL Forecast")

    sub = df[df["Country"] == country].sort_values("Year")
    if sub.empty:
        raise HTTPException(404, f"No DL forecast data for {country}")

    return {
        "country": country,
        "model": sub["Model"].iloc[0] if "Model" in sub.columns else "dl",
        "forecast_years": sub["Year"].tolist(),
        "forecast_twh": [round(v, 3) for v in sub["Forecast"].tolist()],
    }


# ── Classical metrics ─────────────────────────────────────────────────────────


@app.get("/history/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)])
def get_history(country: str) -> dict:
    """Historical electricity demand 2000-2024 with event annotations."""
    country = _validate_country(country)
    df = _require(_DATA["history"], "History")
    sub = df[df["Area"] == country].sort_values("Year")
    if sub.empty:
        raise HTTPException(404, f"No history data for {country}")
    events = [
        {"year": 2009, "label": "2008 Financial crisis", "color": "#6b7280"},
        {"year": 2020, "label": "COVID-19 pandemic", "color": "#e63946"},
        {"year": 2022, "label": "Ukraine-Russia war", "color": "#ff9f1c"},
    ]
    return {
        "country": country,
        "years": sub["Year"].tolist(),
        "demand_twh": [round(float(v), 3) for v in sub["Demand"].tolist()],
        "events": events,
    }


# ── Data Overview (presentation) ──────────────────────────────────────────────

# Subcategories with a single clean value per country-year (safe to chart
# directly). "Aggregate fuel" is excluded — it mixes THREE different units
# (GW, %, TWh) under the same Subcategory label, so summing/averaging it
# would be meaningless. "Fuel" IS summed below (see FUEL_SUBCAT) since its
# rows are all in a consistent unit (GW) — summing gives total generation
# capacity across fuel-type components.
CLEAN_SUBCATS = ["Demand", "CO2 intensity", "Demand per capita", "Electricity imports"]
FUEL_SUBCAT = "Fuel"  # aggregated via sum(), not mean() — see get_subcategories_all()


@app.get("/data/demand-history", tags=["data"], dependencies=[Depends(get_current_user)])
def get_demand_history_all() -> dict:
    """Electricity demand 2000-2024 for ALL 7 countries — one chart, all lines."""
    df = _require(_DATA["history"], "History")
    series = {}
    for country in COUNTRIES:
        sub = df[df["Area"] == country].sort_values("Year")
        if sub.empty:
            continue
        series[country] = {
            "years": sub["Year"].tolist(),
            "demand_twh": [round(float(v), 3) for v in sub["Demand"].tolist()],
        }
    return {"countries": series}


@app.get("/data/subcategories", tags=["data"], dependencies=[Depends(get_current_user)])
def get_subcategories_all() -> dict:
    """
    Each clean subcategory (CO2 intensity, Demand per capita,
    Electricity imports, Demand) as its own time series per country —
    powers small-multiple charts comparing all 7 countries per metric.
    """
    df = _require(_DATA["subcategories"], "Subcategories")
    result: dict[str, Any] = {}
    for subcat in CLEAN_SUBCATS:
        sub_df = df[df["Subcategory"] == subcat]
        if sub_df.empty:
            continue
        unit = sub_df["Unit"].iloc[0] if "Unit" in sub_df.columns else ""
        countries_data = {}
        for country in COUNTRIES:
            c_sub = (
                sub_df[sub_df["Area"] == country]
                .groupby("Year", as_index=False)["Value"]
                .mean()
                .sort_values("Year")  # type: ignore[call-overload]
                # pandas-stubs limitation: as_index=False + single-column
                # selection returns a DataFrame at runtime (verified), but
                # the stubs infer Series here, whose sort_values() has no
                # `by` param — this is a false positive, not a real bug.
            )
            if c_sub.empty:
                continue
            countries_data[country] = {
                "years": c_sub["Year"].tolist(),
                "values": [
                    round(float(v), 3) if pd.notna(v) else None for v in c_sub["Value"].tolist()
                ],
            }
        result[subcat] = {"unit": unit, "countries": countries_data}

    # Fuel — summed (not averaged) per country-year, since its rows are all
    # in GW and represent generation-mix components without a type label.
    # Sum = total generation capacity across those unlabelled components.
    fuel_df = df[df["Subcategory"] == FUEL_SUBCAT]
    if not fuel_df.empty:
        unit = fuel_df["Unit"].iloc[0] if "Unit" in fuel_df.columns else "GW"
        countries_data = {}
        for country in COUNTRIES:
            c_sub = (
                fuel_df[fuel_df["Area"] == country]
                .groupby("Year", as_index=False)["Value"]
                .sum()
                .sort_values("Year")  # type: ignore[call-overload]
                # same pandas-stubs false positive as get_subcategories_all() above
            )
            if c_sub.empty:
                continue
            countries_data[country] = {
                "years": c_sub["Year"].tolist(),
                "values": [
                    round(float(v), 3) if pd.notna(v) else None for v in c_sub["Value"].tolist()
                ],
            }
        result["Total Generation Capacity"] = {"unit": unit, "countries": countries_data}
    return {"subcategories": result}


@app.get("/data/summary", tags=["data"], dependencies=[Depends(get_current_user)])
def get_summary_stats() -> dict:
    """Pre-computed 2000 vs 2024 summary statistics per country (EDA table01)."""
    df = _require(_DATA["summary_stats"], "Summary statistics")
    return {"rows": df.to_dict(orient="records")}


# Crisis-period windows — widened slightly from the single-year markers used
# elsewhere (2009/2020/2022) since a scatter needs multiple points per period
# to show a trend shift, not just one isolated year.
CRISIS_PERIODS = {
    "2008 Financial Crisis": (2008, 2009),
    "COVID-19 Pandemic": (2020, 2021),
    "Ukraine War / Energy Crisis": (2022, 2023),
}


def _tag_crisis_period(year: int) -> str:
    for label, (start, end) in CRISIS_PERIODS.items():
        if start <= year <= end:
            return label
    return "Normal"


@app.get("/data/fuel-vs-demand/{country}", tags=["data"], dependencies=[Depends(get_current_user)])
def get_fuel_vs_demand(country: str) -> dict:
    """
    Fuel vs Demand scatter data for one country, with each point tagged by
    crisis period. Demand (TWh) and Fuel (GW) are on genuinely different
    scales/units — the frontend plots them on independent X/Y axes rather
    than a shared scale, and labels each axis with its actual unit.
    """
    country = _validate_country(country)
    df = _require(_DATA["multifeature"], "Multifeature dataset")
    sub = df[df["Area"] == country].sort_values("Year")
    if sub.empty:
        raise HTTPException(404, f"No multifeature data for {country}")

    points = []
    for _, row in sub.iterrows():
        if pd.isna(row["Demand"]) or pd.isna(row["Fuel"]):
            continue
        year = int(row["Year"])
        points.append(
            {
                "year": year,
                "demand": round(float(row["Demand"]), 3),
                "fuel": round(float(row["Fuel"]), 3),
                "period": _tag_crisis_period(year),
            }
        )

    return {
        "country": country,
        "demand_unit": "TWh",
        "fuel_unit": "GW",
        "periods": list(CRISIS_PERIODS.keys()) + ["Normal"],
        "points": points,
    }


@app.get("/forecast/all/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)])
def get_all_models_forecast(country: str) -> dict:
    """
    Point forecast (2025-2030) for EVERY classical model — not just the
    winner. Lets the dashboard overlay all 11 models on one chart to
    visualise divergence between short-horizon (1-step) test accuracy
    and long-horizon (6-step recursive) forecast stability.
    """
    country = _validate_country(country)
    df = _require(_DATA["all_models_forecast"], "All-models forecast")
    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No all-models forecast data for {country}")

    series = {}
    for model_name, grp in sub.groupby("Model"):
        grp = grp.sort_values("Year")
        series[model_name] = {
            "years": grp["Year"].tolist(),
            "forecast_twh": [round(float(v), 3) for v in grp["Forecast"].tolist()],
        }

    return {"country": country, "models": series}


@app.get("/metrics/{country}", tags=["metrics"], dependencies=[Depends(get_current_user)])
def get_metrics(country: str) -> dict:
    """Classical model test-set metrics (MAE, RMSE, MAPE — see test_benchmarking.csv)."""
    country = _validate_country(country)
    df = _require(_DATA["metrics"], "Metrics")

    if "Country" not in df.columns:
        df = df.reset_index()
    if "Country" not in df.columns:
        raise HTTPException(503, "Metrics CSV has no Country column")

    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No metrics for {country}")

    return {
        "country": country,
        "model": sub["Model"].iloc[0] if "Model" in sub.columns else "unknown",
        "metrics": _metrics_dict(sub.iloc[0]),
    }


# ── Deep learning metrics ─────────────────────────────────────────────────────


@app.get("/metrics/dl/{country}", tags=["metrics"], dependencies=[Depends(get_current_user)])
def get_dl_metrics(country: str) -> dict:
    """Deep learning test-set metrics for the best model."""
    country = _validate_country(country)
    df = _require(_DATA["dl_metrics"], "DL Metrics")

    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No DL metrics for {country}")

    if "MAPE" in sub.columns:
        valid = sub.dropna(subset=["MAPE"])
        if not valid.empty:
            sub = valid.loc[[valid["MAPE"].idxmin()]]

    return {
        "country": country,
        "model": sub["Model"].iloc[0] if "Model" in sub.columns else "dl",
        "metrics": _metrics_dict(sub.iloc[0]),
    }


# ── Model diagnostics — significance & robustness ─────────────────────────────


@app.get("/significance/{country}", tags=["diagnostics"], dependencies=[Depends(get_current_user)])
def get_significance(country: str) -> dict:
    """
    Statistical significance (DM test + Wilcoxon) of MAPE differences
    between every model pair for this country — answers "is Model A
    really better than Model B, or is that just test-set noise?"
    """
    country = _validate_country(country)
    df = _require(_DATA["significance_tests"], "Significance tests")
    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No significance test data for {country}")
    return {"country": country, "pairs": sub.drop(columns=["Country"]).to_dict(orient="records")}


@app.get("/robustness/conflicts", tags=["diagnostics"], dependencies=[Depends(get_current_user)])
def get_robustness_conflicts() -> dict:
    """
    Across ALL countries: does the model selected as "best" purely on
    MAPE actually get flagged WATCH/UNSTABLE on the robustness axis?
    This is the direct dashboard answer to "accuracy alone would have
    hidden this" — surfaced globally, not per-country.

    NOTE: this route MUST be declared before /robustness/{country} —
    FastAPI matches routes in declaration order, so a parameterized
    route declared first will capture "conflicts" as a country name
    and 404 before this handler is ever reached.
    """
    df = _require(_DATA["best_model_conflicts"], "Best-model conflicts")
    n_conflicts = int(df["conflict"].sum()) if "conflict" in df.columns else 0
    return {
        "total_countries": len(df),
        "n_conflicts": n_conflicts,
        "rows": df.to_dict(orient="records"),
    }


@app.get("/robustness/{country}", tags=["diagnostics"], dependencies=[Depends(get_current_user)])
def get_robustness(country: str) -> dict:
    """
    Recursive extrapolation robustness score per model for this country —
    independent of MAPE. Flags STABLE / WATCH / UNSTABLE based on how far
    the 2025-2030 point forecast strays from the last known value.
    """
    country = _validate_country(country)
    df = _require(_DATA["robustness_summary"], "Robustness summary")
    sub = df[df["Country"] == country].sort_values("max_deviation_pct")
    if sub.empty:
        raise HTTPException(404, f"No robustness data for {country}")

    result = {"country": country, "models": sub.drop(columns=["Country"]).to_dict(orient="records")}

    # Merge in accuracy rank if available, so the dashboard can show both
    # axes side by side — the whole point of this metric.
    combo = _DATA["robustness_vs_accuracy"]
    if not combo.empty:
        combo_sub = combo[combo["Country"] == country]
        if not combo_sub.empty:
            result["accuracy_vs_robustness"] = combo_sub.drop(columns=["Country"]).to_dict(
                orient="records"
            )

    return result


# ── Best models ───────────────────────────────────────────────────────────────


@app.get("/models/{country}", tags=["meta"], dependencies=[Depends(get_current_user)])
def get_best_models(country: str) -> dict:
    """Return the best classical and DL model for a country."""
    country = _validate_country(country)
    result: dict[str, Any] = {"country": country}

    if not _DATA["best_models"].empty:
        row = _DATA["best_models"]
        row = row[row["Country"] == country]
        if not row.empty:
            result["classical_model"] = row.iloc[0]["Model"]
            if "MAPE" in row.columns:
                result["classical_mape"] = round(float(row.iloc[0]["MAPE"]), 4)

    if not _DATA["dl_best"].empty:
        row = _DATA["dl_best"]
        row = row[row["Country"] == country]
        if not row.empty:
            result["dl_model"] = row.iloc[0]["Model"]
            if "MAPE" in row.columns:
                result["dl_mape"] = round(float(row.iloc[0]["MAPE"]), 4)

    if len(result) == 1:
        raise HTTPException(404, f"No model data for {country}")
    return result


# ── Growth summary ────────────────────────────────────────────────────────────


@app.get("/growth/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)])
def get_growth(country: str) -> dict:
    """CAGR and total growth summary (2024 → 2030)."""
    country = _validate_country(country)
    df = _require(_DATA["growth"], "Growth summary")

    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No growth data for {country}")

    row = sub.iloc[0].to_dict()
    return {
        "country": country,
        "summary": {
            k: (round(float(v), 4) if _is_finite(v) else v)
            for k, v in row.items()
            if k != "Country"
        },
    }


# ── Compare classical vs DL ───────────────────────────────────────────────────


@app.get("/compare/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)])
def compare_forecasts(country: str) -> dict:
    """Side-by-side comparison of classical and DL forecasts + metrics."""
    country = _validate_country(country)
    result: dict[str, Any] = {"country": country}

    if not _DATA["forecast"].empty:
        sub = _DATA["forecast"][_DATA["forecast"]["Country"] == country]
        if not sub.empty:
            sub = sub.sort_values("Year")
            result["classical"] = {
                "model": sub["Model"].iloc[0] if "Model" in sub.columns else "unknown",
                "forecast_years": sub["Year"].tolist(),
                "forecast_twh": [round(v, 3) for v in sub["Forecast"].tolist()],
            }

    if not _DATA["dl_forecast"].empty:
        sub = _DATA["dl_forecast"][_DATA["dl_forecast"]["Country"] == country].sort_values("Year")
        if not sub.empty:
            result["deeplearning"] = {
                "model": sub["Model"].iloc[0] if "Model" in sub.columns else "dl",
                "forecast_years": sub["Year"].tolist(),
                "forecast_twh": [round(v, 3) for v in sub["Forecast"].tolist()],
            }

    metrics_cmp: dict[str, Any] = {}
    if not _DATA["metrics"].empty:
        df = _DATA["metrics"]
        if "Country" not in df.columns:
            df = df.reset_index()
        sub = df[df["Country"] == country]
        if not sub.empty:
            metrics_cmp["classical"] = _metrics_dict(sub.iloc[0])

    if not _DATA["dl_metrics"].empty:
        sub = _DATA["dl_metrics"][_DATA["dl_metrics"]["Country"] == country]
        if not sub.empty:
            if "MAPE" in sub.columns:
                valid = sub.dropna(subset=["MAPE"])
                if not valid.empty:
                    sub = valid.loc[[valid["MAPE"].idxmin()]]
            metrics_cmp["deeplearning"] = _metrics_dict(sub.iloc[0])

    if metrics_cmp:
        result["metrics"] = metrics_cmp

    if len(result) == 1:
        raise HTTPException(404, f"No comparison data for {country}")
    return result


# ── Figures ───────────────────────────────────────────────────────────────────


@app.get("/figures", tags=["figures"], dependencies=[Depends(get_current_user)])
def list_figures() -> dict:
    """List all available figure files across all pipeline steps."""
    figures = []
    for fig_dir in FIGURES_DIRS:
        if fig_dir.exists():
            for f in sorted(fig_dir.iterdir()):
                if f.suffix.lower() in {".png", ".pdf", ".svg", ".html"}:
                    figures.append(
                        {
                            "filename": f.name,
                            "step": fig_dir.parent.name,
                            "path": str(f.relative_to(_ROOT)),
                        }
                    )
    return {"figures": figures, "total": len(figures)}


@app.get("/figures/{filename}", tags=["figures"], dependencies=[Depends(get_current_user)])
def get_figure(filename: str) -> FileResponse:
    """Serve a figure file (PNG / PDF / SVG / HTML)."""
    for fig_dir in FIGURES_DIRS:
        candidate = fig_dir / filename
        if candidate.exists():
            media = {
                ".png": "image/png",
                ".pdf": "application/pdf",
                ".svg": "image/svg+xml",
                ".html": "text/html",
            }.get(candidate.suffix.lower(), "application/octet-stream")
            return FileResponse(str(candidate), media_type=media)
    raise HTTPException(404, f"Figure '{filename}' not found")


# ── LangChain Q&A ─────────────────────────────────────────────────────────────


@app.post(
    "/ask",
    tags=["ai"],
    response_model=AskResponse,
    dependencies=[Depends(get_current_user), Depends(rate_limit)],
)
def ask(body: AskRequest) -> AskResponse:
    """
    Natural language Q&A over forecast data.
    Uses LangChain + OpenAI when OPENAI_API_KEY is set,
    falls back to a rule-based engine otherwise.
    """
    question = body.question.strip()
    if not question:
        return AskResponse(question=question, answer="Please provide a question.")

    key = os.getenv("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-..."):
        try:
            return AskResponse(question=question, answer=_langchain_answer(question))
        except Exception as e:
            log.warning("LangChain failed: %s", e)

    return AskResponse(question=question, answer=_rule_based_answer(question))


def _build_context() -> str:
    lines = ["Ember Energy Forecasting — Summary\n"]
    if not _DATA["forecast"].empty:
        for country in COUNTRIES:
            sub = _DATA["forecast"][_DATA["forecast"]["Country"] == country].sort_values("Year")
            if not sub.empty:
                lines.append(
                    f"{country}: {sub['Year'].iloc[0]}={round(sub['Forecast'].iloc[0], 2)} TWh"
                    f" → {sub['Year'].iloc[-1]}={round(sub['Forecast'].iloc[-1], 2)} TWh"
                )
    if not _DATA["growth"].empty:
        for country in COUNTRIES:
            sub = _DATA["growth"][_DATA["growth"]["Country"] == country]
            if not sub.empty and "CAGR (%)" in sub.columns:
                lines.append(f"{country}: CAGR={round(float(sub.iloc[0]['CAGR (%)']), 2)}%")
    return "\n".join(lines)


def _langchain_answer(question: str) -> str:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    content = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an energy analyst. Answer using only this data:\n\n" + _build_context()
                )
            ),
            HumanMessage(content=question),
        ]
    ).content
    # .content is typed str | list[str | dict] for multimodal support, but
    # with plain-text-only messages (as used here) it's always a plain str.
    return content if isinstance(content, str) else str(content)


def _rule_based_answer(question: str) -> str:
    q = question.lower()

    # if any(w in q for w in ["highest", "largest", "most", "biggest"]):
    #     if not _DATA["forecast"].empty:
    #         avg = _DATA["forecast"].groupby("Country")["Forecast"].mean()
    #         top = avg.idxmax()
    #         return f"{top} has the highest average demand at {round(avg[top], 1)} TWh (2025-2030)."

    # if any(w in q for w in ["growth", "cagr", "growing", "fastest"]):
    #     if not _DATA["growth"].empty:
    #         df = _DATA["growth"].copy()
    #         if "CAGR (%)" in df.columns:
    #             df["CAGR (%)"] = pd.to_numeric(df["CAGR (%)"], errors="coerce")
    #             row = df.loc[df["CAGR (%)"].idxmax()]
    #             return (
    #                 f"{row['Country']} has the highest CAGR at "
    #                 f"{round(float(row['CAGR (%)']), 2)}% (2024-2030)."
    #             )
    if (
        any(w in q for w in ["highest", "largest", "most", "biggest"])
        and not _DATA["forecast"].empty
    ):
        avg = _DATA["forecast"].groupby("Country")["Forecast"].mean()
        top = avg.idxmax()
        return f"{top} has the highest average demand at {round(avg[top], 1)} TWh (2025-2030)."
    if any(w in q for w in ["growth", "cagr", "growing", "fastest"]) and not _DATA["growth"].empty:
        df = _DATA["growth"].copy()
        if "CAGR (%)" in df.columns:
            df["CAGR (%)"] = pd.to_numeric(df["CAGR (%)"], errors="coerce")
            row = df.loc[df["CAGR (%)"].idxmax()]
            return (
                f"{row['Country']} has the highest CAGR at "
                f"{round(float(row['CAGR (%)']), 2)}% (2024-2030)."
            )

    for country in COUNTRIES:
        if country.lower() in q and not _DATA["forecast"].empty:
            sub = _DATA["forecast"][_DATA["forecast"]["Country"] == country].sort_values("Year")
            if not sub.empty:
                return f"{country} demand forecast: " + ", ".join(
                    f"{y}: {round(v, 1)} TWh"
                    for y, v in zip(sub["Year"], sub["Forecast"], strict=False)
                )

    return (
        f"I can answer questions about electricity demand forecasts for "
        f"{', '.join(COUNTRIES)} from 2025 to 2030. "
        "Try asking about a specific country, growth rates, or comparisons."
    )


# ── Dev entry point ───────────────────────────────────────────────────────────


def start() -> None:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
    )


if __name__ == "__main__":
    start()
