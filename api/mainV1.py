"""
api/main.py — Ember Energy Forecasting REST API
================================================
Serves all pipeline outputs as JSON endpoints.

Endpoints:
    GET  /health                      — liveness check
    GET  /countries                   — list of 7 countries
    GET  /forecast/{country}          — classical forecast 2025-2030
    GET  /forecast/dl/{country}       — deep learning forecast 2025-2030
    GET  /metrics/{country}           — classical model metrics
    GET  /metrics/dl/{country}        — deep learning metrics
    GET  /models/{country}            — best model (classical + DL)
    GET  /growth/{country}            — CAGR + growth summary
    GET  /compare/{country}           — classical vs DL side by side
    GET  /figures                     — list all available figures
    GET  /figures/{filename}          — serve a figure file
    POST /ask                         — LangChain Q&A over forecasts

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv()  
from datetime import timedelta
 

from fastapi import Depends, Request, status
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# ── Security headers middleware ───────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
 
from api.auth import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    check_rate_limit,
    create_access_token,
    get_current_user,
    optional_auth,
    rate_limit,
    require_api_key,
    revoke_token,
)
log = logging.getLogger(__name__)

# ── Output paths ───────────────

_ROOT = Path(os.getenv("OUTPUT_ROOT", os.getenv("OUTPUTS_DIR", "outputs")))

FORECAST_CSV = _ROOT / "forecasting" / "demand_forecast_2025_2030.csv"
METRICS_CSV = _ROOT / "forecasting" / "forecast_metrics.csv"
GROWTH_CSV = _ROOT / "forecasting" / "demand_growth_summary.csv"
BEST_MODELS_CSV = _ROOT / "modeling" / "best_models.csv"
BENCHMARKING_CSV = _ROOT / "modeling" / "test_benchmarking.csv"

DL_FORECAST_CSV = _ROOT / "deeplearning" / "dl_forecast_2025_2030.csv"
DL_METRICS_CSV = _ROOT / "deeplearning" / "dl_benchmarking.csv"
DL_BEST_CSV = _ROOT / "deeplearning" / "dl_best_models.csv"

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


# ── Data loader helpers ───────────────────────────────────────────────────────


def _load(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV; return empty DataFrame with a warning if missing."""
    if not path.exists():
        log.warning("%s not found at %s", label, path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_all() -> dict[str, pd.DataFrame]:
    return {
        "forecast": _load(FORECAST_CSV, "forecast"),
        "metrics": _load(METRICS_CSV, "metrics"),
        "growth": _load(GROWTH_CSV, "growth"),
        "best_models": _load(BEST_MODELS_CSV, "best_models"),
        "benchmark": _load(BENCHMARKING_CSV, "benchmark"),
        "dl_forecast": _load(DL_FORECAST_CSV, "dl_forecast"),
        "dl_metrics": _load(DL_METRICS_CSV, "dl_metrics"),
        "dl_best": _load(DL_BEST_CSV, "dl_best"),
    }


# Load once at startup — reloaded on module reload (test monkeypatch friendly)
_DATA: dict[str, pd.DataFrame] = _load_all()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ember Energy Forecasting API",
    description=(
        "REST API for per-country electricity demand forecasts "
        "(2025-2030) based on the Ember dataset. IEEE Paper."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

#middkeware without JWT
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# ── CORS — restrict to your domains in production ─────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://localhost:3000,http://127.0.0.1:8000"
).split(",")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Authorization", "X-API-Key", "Content-Type"],
    allow_credentials = True,
    max_age           = 600,
)
 
# ── Trusted hosts — prevents Host header injection ────────────────
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS + ["*"])
 
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"]    = "nosniff"
        response.headers["X-Frame-Options"]           = "DENY"
        response.headers["X-XSS-Protection"]          = "1; mode=block"
        response.headers["Referrer-Policy"]            = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]         = "geolocation=(), microphone=()"
        response.headers["Content-Security-Policy"]    = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        if os.getenv("ENV") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


 
app.add_middleware(SecurityHeadersMiddleware)

#jwtException
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception: %s %s — %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},   # never expose stack trace
    )

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str

# ══════════════════════════════════════════════════════════════════
# JWT— Auth schemas 
# ══════════════════════════════════════════════════════════════════

 
class LoginRequest(BaseModel):
    username: str
    password: str
    
 
class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_country(country: str) -> str:
    """Normalise and validate country name; raise 404 if unknown."""
    match = next((c for c in COUNTRIES if c.lower() == country.lower()), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Country '{country}' not found. " f"Available: {COUNTRIES}",
        )
    return match


def _require(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Raise 503 if a required DataFrame is empty (file not yet generated)."""
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"{label} outputs not available yet. Run the pipeline first.",
        )
    return df


def _metrics_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a metrics row to a clean dict, dropping NaN values."""
    return {
        k: round(float(v), 4)
        for k, v in row.items()
        if k not in ("Country", "Model") and _is_finite(v)
    }


def _is_finite(v: Any) -> bool:
    try:
        import math

        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness check."""
    loaded = {k: not v.empty for k, v in _DATA.items()}
    return {
        "status": "ok",
        "outputs": loaded,
    }
# ══════════════════════════════════════════════════════════════════
# JWT — Auth endpoints 
# ══════════════════════════════════════════════════════════════════
 
@app.post("/auth/login", tags=["auth"], response_model=TokenResponse)
def login(body: LoginRequest, request: Request) -> TokenResponse:
    """
    Exchange username + password for a JWT Bearer token.
    Set ADMIN_USERNAME and ADMIN_PASSWORD in .env.
    """
    check_rate_limit(request)          # rate-limit login attempts
 
    from api.auth import _admin_username, _admin_password

    if not _admin_password():
        raise HTTPException(503, "Auth not configured — set ADMIN_PASSWORD in .env")
    if body.username != _admin_username() or body.password != _admin_password():
 
    token = create_access_token(
        data={"sub": body.username, "role": "admin"},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(
        access_token=token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
 
 
@app.post("/auth/logout", tags=["auth"])
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> dict:
    """Revoke the current JWT token (adds to blacklist)."""
    revoke_token(credentials.credentials)
    return {"message": "Logged out successfully"}
 
 
@app.get("/auth/me", tags=["auth"])
def me(user: dict = Depends(get_current_user)) -> dict:
    """Return current authenticated user info."""
    return {"username": user.get("sub"), "role": user.get("role")}

# ── Countries ─────────────────────────────────────────────────────────────────


@app.get("/countries", tags=["meta"])
def get_countries() -> dict:
    """Return the list of 7 countries covered by the forecasts."""
    return {"countries": COUNTRIES}


# ── Classical forecast ────────────────────────────────────────────────────────


@app.get("/forecast/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)])         
def get_forecast(country: str) -> dict:
    """
    Classical ML demand forecast (2025-2030) for a country.
    Returns point estimate + 90% confidence interval.
    """
    country = _validate_country(country)
    df = _require(_DATA["forecast"], "Forecast")

    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No forecast data for {country}")

    # Pick the best-model rows if multiple models exist
    if "Model" in sub.columns and not _DATA["best_models"].empty:
        bm = _DATA["best_models"]
        bm_row = bm[bm["Country"] == country]
        if not bm_row.empty:
            best = bm_row.iloc[0]["Model"]
            sub = sub[sub["Model"] == best]

    sub = sub.sort_values("Year")
    result: dict[str, Any] = {
        "country": country,
        "model": sub["Model"].iloc[0] if "Model" in sub.columns else "unknown",
        "forecast_years": sub["Year"].tolist(),
        "forecast_twh": [round(v, 3) for v in sub["Forecast"].tolist()],
    }
    if "Lower_90" in sub.columns:
        result["lower_90"] = [round(v, 3) for v in sub["Lower_90"].tolist()]
        result["upper_90"] = [round(v, 3) for v in sub["Upper_90"].tolist()]
    return result


# ── Deep learning forecast ────────────────────────────────────────────────────


@app.get("/forecast/dl/{country}", tags=["forecast"], dependencies=[Depends(get_current_user)] )      
def get_dl_forecast(country: str) -> dict:
    """
    Deep learning demand forecast (2025-2030) for a country.
    """
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


@app.get("/metrics/{country}", tags=["metrics"], dependencies=[Depends(get_current_user)])
def get_metrics(country: str) -> dict:
    """
    Classical model test-set metrics (MAE, RMSE, MAPE, SMAPE, R2, TheilU)
    for a country.
    """
    country = _validate_country(country)
    df = _require(_DATA["metrics"], "Metrics")

    # metrics CSV may be indexed by Country or have a Country column
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
    """
    Deep learning test-set metrics for a country.
    """
    country = _validate_country(country)
    df = _require(_DATA["dl_metrics"], "DL Metrics")

    sub = df[df["Country"] == country]
    if sub.empty:
        raise HTTPException(404, f"No DL metrics for {country}")

    # Return best model's metrics
    if "MAPE" in sub.columns:
        sub = sub.loc[[sub["MAPE"].idxmin()]]

    return {
        "country": country,
        "model": sub["Model"].iloc[0] if "Model" in sub.columns else "dl",
        "metrics": _metrics_dict(sub.iloc[0]),
    }


# ── Best models ───────────────────────────────────────────────────────────────


@app.get("/models/{country}", tags=["meta"], dependencies=[Depends(get_current_user)])
def get_best_models(country: str) -> dict:
    """
    Return the best classical and DL model for a country.
    """
    country = _validate_country(country)

    result: dict[str, Any] = {"country": country}

    # Classical
    if not _DATA["best_models"].empty:
        bm = _DATA["best_models"]
        row = bm[bm["Country"] == country]
        if not row.empty:
            result["classical_model"] = row.iloc[0]["Model"]
            if "MAPE" in row.columns:
                result["classical_mape"] = round(float(row.iloc[0]["MAPE"]), 4)

    # Deep learning
    if not _DATA["dl_best"].empty:
        dl = _DATA["dl_best"]
        row = dl[dl["Country"] == country]
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
    """
    CAGR and total growth summary for a country (2024 → 2030).
    """
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
    """
    Side-by-side comparison of classical and DL forecasts + metrics.
    """
    country = _validate_country(country)

    result: dict[str, Any] = {"country": country}

    # Classical forecast
    if not _DATA["forecast"].empty:
        sub = _DATA["forecast"]
        sub = sub[sub["Country"] == country]
        if not sub.empty:
            if "Model" in sub.columns and not _DATA["best_models"].empty:
                bm = _DATA["best_models"]
                bm_row = bm[bm["Country"] == country]
                if not bm_row.empty:
                    sub = sub[sub["Model"] == bm_row.iloc[0]["Model"]]
            sub = sub.sort_values("Year")
            result["classical"] = {
                "model": sub["Model"].iloc[0] if "Model" in sub.columns else "unknown",
                "forecast_years": sub["Year"].tolist(),
                "forecast_twh": [round(v, 3) for v in sub["Forecast"].tolist()],
            }

    # DL forecast
    if not _DATA["dl_forecast"].empty:
        sub = _DATA["dl_forecast"]
        sub = sub[sub["Country"] == country].sort_values("Year")
        if not sub.empty:
            result["deeplearning"] = {
                "model": sub["Model"].iloc[0] if "Model" in sub.columns else "dl",
                "forecast_years": sub["Year"].tolist(),
                "forecast_twh": [round(v, 3) for v in sub["Forecast"].tolist()],
            }

    # Metrics comparison
    metrics_comparison: dict[str, Any] = {}
    if not _DATA["metrics"].empty:
        df = _DATA["metrics"]
        if "Country" not in df.columns:
            df = df.reset_index()
        sub = df[df["Country"] == country]
        if not sub.empty:
            metrics_comparison["classical"] = _metrics_dict(sub.iloc[0])

    if not _DATA["dl_metrics"].empty:
        sub = _DATA["dl_metrics"][_DATA["dl_metrics"]["Country"] == country]
        if not sub.empty:
            if "MAPE" in sub.columns:
                sub = sub.loc[[sub["MAPE"].idxmin()]]
            metrics_comparison["deeplearning"] = _metrics_dict(sub.iloc[0])

    if metrics_comparison:
        result["metrics"] = metrics_comparison

    if len(result) == 1:
        raise HTTPException(404, f"No comparison data available for {country}")

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
    """Serve a figure file by name."""
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
    raise HTTPException(404, f"Figure '{filename}' not found.")


# ── LangChain Q&A ─────────────────────────────────────────────────────────────


@app.post(
    "/ask",
    tags=["ai"],
    response_model=AskResponse,
    dependencies=[Depends(rate_limit)],          # ← rate limit every /ask call
)
def ask(body: AskRequest, request: Request) -> AskResponse:
    """
    Natural language Q&A over the forecast data using LangChain + OpenAI.
    Falls back to a rule-based answer if no API key is configured.
    """
    question = body.question.strip()
    if not question:
        return AskResponse(question=question, answer="Please provide a question.")

    # Try LangChain if key available
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and not openai_key.startswith("sk-..."):
        try:
            answer = _langchain_answer(question)
            return AskResponse(question=question, answer=answer)
        except Exception as e:
            log.warning("LangChain failed: %s", e)

    # Fallback: rule-based answer from data
    answer = _rule_based_answer(question)
    return AskResponse(question=question, answer=answer)


def _build_context() -> str:
    """Build a text summary of current forecast data for the LLM."""
    lines = ["Ember Energy Forecasting — Summary\n"]

    if not _DATA["forecast"].empty:
        df = _DATA["forecast"]
        for country in COUNTRIES:
            sub = df[df["Country"] == country].sort_values("Year")
            if not sub.empty:
                first = round(sub["Forecast"].iloc[0], 2)
                last = round(sub["Forecast"].iloc[-1], 2)
                lines.append(
                    f"{country}: forecast {sub['Year'].iloc[0]}={first} TWh "
                    f"→ {sub['Year'].iloc[-1]}={last} TWh"
                )

    if not _DATA["growth"].empty:
        df = _DATA["growth"]
        for country in COUNTRIES:
            sub = df[df["Country"] == country]
            if not sub.empty and "CAGR (%)" in sub.columns:
                cagr = round(float(sub.iloc[0]["CAGR (%)"]), 2)
                lines.append(f"{country}: CAGR = {cagr}%")

    return "\n".join(lines)


def _langchain_answer(question: str) -> str:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    context = _build_context()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(
            content=(
                "You are an energy analyst assistant. "
                "Answer questions about electricity demand forecasts "
                "using only the data provided below.\n\n"
                f"{context}"
            )
        ),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content


def _rule_based_answer(question: str) -> str:
    """Simple keyword-based fallback when no LLM key is available."""
    q = question.lower()

    # Highest demand
    if any(w in q for w in ["highest", "largest", "most", "biggest"]):
        if not _DATA["forecast"].empty:
            df = _DATA["forecast"]
            avg = df.groupby("Country")["Forecast"].mean()
            top = avg.idxmax()
            return (
                f"Based on 2025-2030 forecasts, {top} has the highest "
                f"average demand at {round(avg[top], 1)} TWh."
            )

    # CAGR / growth
    if any(w in q for w in ["growth", "cagr", "growing", "fastest"]):
        if not _DATA["growth"].empty:
            df = _DATA["growth"]
            if "CAGR (%)" in df.columns:
                df["CAGR (%)"] = pd.to_numeric(df["CAGR (%)"], errors="coerce")
                top = df.loc[df["CAGR (%)"].idxmax()]
                return (
                    f"{top['Country']} has the highest projected CAGR "
                    f"at {round(float(top['CAGR (%)']), 2)}% (2024-2030)."
                )

    # Country-specific
    for country in COUNTRIES:
        if country.lower() in q:
            if not _DATA["forecast"].empty:
                sub = _DATA["forecast"][_DATA["forecast"]["Country"] == country].sort_values("Year")
                if not sub.empty:
                    vals = sub["Forecast"].tolist()
                    yrs = sub["Year"].tolist()
                    return f"{country} demand forecast: " + ", ".join(
                        f"{y}: {round(v,1)} TWh" for y, v in zip(yrs, vals)
                    )

    # Default
    return (
        "I can answer questions about electricity demand forecasts for "
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
