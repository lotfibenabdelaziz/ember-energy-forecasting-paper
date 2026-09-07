"""
api/auth.py — JWT Authentication & Security
============================================
Provides:
  - JWT token creation and validation
  - HTTP Bearer dependency for FastAPI routes
  - API key validation
  - Rate limiting per client IP

Usage in main.py:
    from api.auth import (
        get_current_user, require_api_key, rate_limit,
        create_access_token, revoke_token,
        _admin_username, _admin_password, _token_expire_minutes,
    )
"""

from datetime import datetime, timedelta, timezone
import os
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

# ── Optional JWT support ──────────────────────────────────────────────────────
try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# ── Algorithm (fixed constant — never changes) ────────────────────────────────
ALGORITHM = "HS256"

# ── All config read at call-time so monkeypatch works in tests ────────────────


def _secret_key() -> str:
    return os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-32chars")


def _admin_username() -> str:
    return os.getenv("ADMIN_USERNAME", "admin")


def _admin_password() -> str:
    return os.getenv("ADMIN_PASSWORD", "")


def _token_expire_minutes() -> int:
    return int(os.getenv("TOKEN_EXPIRE_MINUTES", 60))


def _api_key() -> str:
    return os.getenv("DASHBOARD_API_KEY", "")


def _rate_limit_requests() -> int:
    return int(os.getenv("RATE_LIMIT_REQUESTS", 60))


def _rate_limit_window() -> int:
    return int(os.getenv("RATE_LIMIT_WINDOW", 60))


# ── Schemes ───────────────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── In-memory stores (module-level — survive reloads within same process) ─────
_rate_store: dict[str, list[float]] = {}
_blacklist: set[str] = set()


# ═══════════════════════════════════════════════════════════════════════════════
# JWT helpers
# ═══════════════════════════════════════════════════════════════════════════════


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token."""
    if not JWT_AVAILABLE:
        raise RuntimeError("python-jose not installed — run: pip install python-jose[cryptography]")
    payload = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=_token_expire_minutes()))
    payload.update({"exp": expire, "iat": now})
    return str(jwt.encode(payload, _secret_key(), algorithm=ALGORITHM))


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises HTTPException 401 on any failure.
    """
    if not JWT_AVAILABLE:
        raise HTTPException(status_code=501, detail="JWT not available — install python-jose")
    if token in _blacklist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
        return dict(payload)
    except JWTError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err


def revoke_token(token: str) -> None:
    """Add token to the blacklist — used on logout."""
    _blacklist.add(token)


# ═══════════════════════════════════════════════════════════════════════════════
# Rate limiter
# ═══════════════════════════════════════════════════════════════════════════════


def check_rate_limit(request: Request) -> None:
    """
    Sliding-window rate limiter.
    Raises 429 if the client IP exceeds RATE_LIMIT_REQUESTS
    within RATE_LIMIT_WINDOW seconds.
    Uses X-Forwarded-For when behind a reverse proxy.
    """
    ip = request.headers.get("X-Forwarded-For", "") or (
        request.client.host if request.client else "unknown"
    )
    ip = ip.split(",")[0].strip()
    now = time.time()

    window = _rate_limit_window()
    limit = _rate_limit_requests()

    hits = [t for t in _rate_store.get(ip, []) if now - t < window]
    hits.append(now)
    _rate_store[ip] = hits

    if len(hits) > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window}s",
            headers={"Retry-After": str(window)},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI dependencies
# ═══════════════════════════════════════════════════════════════════════════════


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> dict:
    """
    FastAPI dependency — validates Bearer JWT token.

    Usage:
        @app.get("/endpoint", dependencies=[Depends(get_current_user)])
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required — please log in",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def require_api_key(
    key: Annotated[str | None, Security(api_key_header)],
) -> str:
    """
    FastAPI dependency — validates X-API-Key header.
    Simpler alternative to JWT for server-to-server calls.
    Disabled (passthrough) if DASHBOARD_API_KEY not set in .env.
    """
    stored = _api_key()
    if not stored:
        return "no-key-configured"
    if key != stored:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return key


def rate_limit(request: Request) -> None:
    """FastAPI dependency — apply sliding-window rate limiting."""
    check_rate_limit(request)


def optional_auth(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    key: Annotated[str | None, Security(api_key_header)],
) -> dict | None:
    """
    Optional auth dependency — accepts JWT Bearer OR API key.
    Returns user dict if authenticated, None if no credentials provided.
    Raises 401/403 if credentials are provided but invalid.
    """
    if credentials:
        return decode_token(credentials.credentials)
    stored = _api_key()
    if key and stored and key == stored:
        return {"sub": "api-key-user", "role": "reader"}
    if not credentials and not key:
        return None  # no credentials — public access
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid credentials")
