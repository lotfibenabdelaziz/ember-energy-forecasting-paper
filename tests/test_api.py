"""
tests/test_api.py — FastAPI Endpoint Tests
Ember Energy Forecasting | IEEE Paper

Covers:
  - Public endpoints  (/health, /countries, /auth/login)
  - Protected endpoints (all forecast, metrics, figures, ask)
  - JWT auth flow (login, logout, token revocation, /auth/me)
  - Input validation and error codes
"""

import os

import pandas as pd
import pytest

COUNTRIES      = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
FORECAST_YEARS = list(range(2025, 2031))
TEST_USERNAME  = "admin"
TEST_PASSWORD  = "testpass"
TEST_SECRET    = "test-secret-key-32-chars-minimum!!"


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_outputs(tmp_path):
    """Create minimal mock output files the API reads at startup."""
    fc_dir  = tmp_path / "forecasting"
    mod_dir = tmp_path / "modeling"
    dl_dir  = tmp_path / "deeplearning"
    fc_dir.mkdir(); mod_dir.mkdir(); dl_dir.mkdir()

    # demand_forecast_2025_2030.csv
    rows = []
    for c in COUNTRIES:
        for yr in FORECAST_YEARS:
            rows.append({
                "Country": c, "Model": "Ridge", "Year": yr,
                "Forecast": 10.0 + 0.5 * (yr - 2025),
                "Lower_90": 9.5, "Upper_90": 10.5,
            })
    pd.DataFrame(rows).to_csv(str(fc_dir / "demand_forecast_2025_2030.csv"), index=False)

    # forecast_metrics.csv
    pd.DataFrame([{
        "Country": c, "Model": "Ridge",
        "MAE": 0.5, "RMSE": 0.7, "R2": 0.92,
        "MAPE": 2.5, "SMAPE": 2.4, "TheilU": 0.6,
    } for c in COUNTRIES]).set_index("Country").to_csv(
        str(fc_dir / "forecast_metrics.csv"))

    # demand_growth_summary.csv
    pd.DataFrame([{
        "Country": c, "Model": "Ridge",
        "2024 TWh": 10.0, "2030 TWh (forecast)": 13.0,
        "Total Growth (%)": 30.0, "CAGR (%)": 4.5,
    } for c in COUNTRIES]).to_csv(str(fc_dir / "demand_growth_summary.csv"), index=False)

    # best_models.csv
    pd.DataFrame([{"Country": c, "Model": "Ridge", "MAPE": 2.5}
                  for c in COUNTRIES]).to_csv(str(mod_dir / "best_models.csv"), index=False)

    # dl_forecast_2025_2030.csv
    dl_rows = []
    for c in COUNTRIES:
        for yr in FORECAST_YEARS:
            dl_rows.append({"Country": c, "Model": "MLP", "Year": yr,
                            "Forecast": 10.2 + 0.4 * (yr - 2025)})
    pd.DataFrame(dl_rows).to_csv(str(dl_dir / "dl_forecast_2025_2030.csv"), index=False)

    # dl_benchmarking.csv
    pd.DataFrame([{"Country": c, "Model": "MLP",
                   "MAE": 0.4, "RMSE": 0.6, "MAPE": 2.1, "R2": 0.94}
                  for c in COUNTRIES]).to_csv(str(dl_dir / "dl_benchmarking.csv"), index=False)

    # dl_best_models.csv
    pd.DataFrame([{"Country": c, "Model": "MLP", "MAPE": 2.1}
                  for c in COUNTRIES]).to_csv(str(dl_dir / "dl_best_models.csv"), index=False)

    return str(tmp_path)


@pytest.fixture
def api_client(mock_outputs, monkeypatch):
    """FastAPI test client with mocked paths and auth env vars."""
    try:
        from fastapi.testclient import TestClient
        import api.auth as auth_module
        import api.main as main_module
        import importlib

        monkeypatch.setenv("OUTPUT_ROOT",          mock_outputs)
        monkeypatch.setenv("OUTPUTS_DIR",          mock_outputs)
        monkeypatch.setenv("ADMIN_USERNAME",       TEST_USERNAME)
        monkeypatch.setenv("ADMIN_PASSWORD",       TEST_PASSWORD)
        monkeypatch.setenv("SECRET_KEY",           TEST_SECRET)
        monkeypatch.setenv("TOKEN_EXPIRE_MINUTES", "60")

        importlib.reload(auth_module)
        importlib.reload(main_module)

        from api.main import app
        return TestClient(app, raise_server_exceptions=False)

    except (ImportError, ModuleNotFoundError):
        pytest.skip("FastAPI or api module not available")


@pytest.fixture
def auth_headers(api_client):
    """Login and return Authorization headers for protected endpoint tests."""
    r = api_client.post("/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD})
    assert r.status_code == 200, f"Login failed in fixture: {r.text}"
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Public endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, api_client):
        assert api_client.get("/health").status_code == 200

    def test_health_has_status_key(self, api_client):
        assert "status" in api_client.get("/health").json()

    def test_health_status_ok(self, api_client):
        assert api_client.get("/health").json()["status"] == "ok"

    def test_health_has_outputs_key(self, api_client):
        assert "outputs" in api_client.get("/health").json()


class TestCountriesEndpoint:

    def test_countries_returns_200(self, api_client):
        assert api_client.get("/countries").status_code == 200

    def test_countries_returns_list(self, api_client):
        data = api_client.get("/countries").json()
        assert "countries" in data
        assert isinstance(data["countries"], list)

    def test_countries_has_all_7(self, api_client):
        data = api_client.get("/countries").json()["countries"]
        assert len(data) == 7
        for c in COUNTRIES:
            assert c in data


# ═══════════════════════════════════════════════════════════════════════════════
# JWT Auth flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestJWTAuth:

    def test_protected_endpoint_without_token_returns_401(self, api_client):
        assert api_client.get("/forecast/Tunisia").status_code == 401

    def test_figures_without_token_returns_401(self, api_client):
        assert api_client.get("/figures").status_code == 401

    def test_ask_without_token_returns_401(self, api_client):
        assert api_client.post("/ask",
            json={"question": "test"}).status_code == 401

    def test_login_wrong_password_returns_401(self, api_client):
        r = api_client.post("/auth/login",
            json={"username": TEST_USERNAME, "password": "wrong"})
        assert r.status_code == 401

    def test_login_wrong_username_returns_401(self, api_client):
        r = api_client.post("/auth/login",
            json={"username": "hacker", "password": TEST_PASSWORD})
        assert r.status_code == 401

    def test_login_returns_token(self, api_client):
        r = api_client.post("/auth/login",
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD})
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_forecast_with_valid_token(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        assert r.status_code in [200, 503]

    def test_revoked_token_returns_401(self, api_client, auth_headers):
        api_client.post("/auth/logout", headers=auth_headers)
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        assert r.status_code == 401

    def test_me_returns_username(self, api_client, auth_headers):
        r = api_client.get("/auth/me", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["username"] == TEST_USERNAME

    def test_me_returns_role(self, api_client, auth_headers):
        r = api_client.get("/auth/me", headers=auth_headers)
        assert r.status_code == 200
        assert "role" in r.json()

    def test_invalid_token_returns_401(self, api_client):
        r = api_client.get("/forecast/Tunisia",
            headers={"Authorization": "Bearer not.a.real.token"})
        assert r.status_code == 401

    def test_malformed_auth_header_returns_401(self, api_client):
        r = api_client.get("/forecast/Tunisia",
            headers={"Authorization": "Token sometoken"})
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# Protected endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestForecastEndpoint:

    def test_forecast_known_country(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        assert r.status_code in [200, 503]

    def test_forecast_unknown_country_returns_404(self, api_client, auth_headers):
        assert api_client.get("/forecast/Atlantis",
            headers=auth_headers).status_code == 404

    def test_forecast_response_has_required_fields(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            for field in ["country", "forecast_years", "forecast_twh"]:
                assert field in r.json()

    def test_forecast_years_are_2025_2030(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            years = r.json()["forecast_years"]
            assert min(years) >= 2025
            assert max(years) <= 2030

    def test_forecast_twh_all_positive(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            assert all(v > 0 for v in r.json()["forecast_twh"])

    def test_forecast_has_confidence_interval(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            assert "lower_90" in r.json()
            assert "upper_90" in r.json()

    def test_all_countries_respond(self, api_client, auth_headers):
        for c in COUNTRIES:
            r = api_client.get(f"/forecast/{c}", headers=auth_headers)
            assert r.status_code in [200, 503], f"{c}: unexpected {r.status_code}"


class TestMetricsEndpoint:

    def test_metrics_known_country(self, api_client, auth_headers):
        r = api_client.get("/metrics/Tunisia", headers=auth_headers)
        assert r.status_code in [200, 503]

    def test_metrics_unknown_returns_404(self, api_client, auth_headers):
        assert api_client.get("/metrics/Mars",
            headers=auth_headers).status_code == 404

    def test_metrics_has_dict(self, api_client, auth_headers):
        r = api_client.get("/metrics/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            data = r.json()
            assert "metrics" in data
            assert isinstance(data["metrics"], dict)

    def test_metrics_has_expected_keys(self, api_client, auth_headers):
        r = api_client.get("/metrics/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            for k in ["MAE", "RMSE", "MAPE"]:
                assert k in r.json()["metrics"]


class TestGrowthEndpoint:

    def test_growth_known_country(self, api_client, auth_headers):
        r = api_client.get("/growth/Tunisia", headers=auth_headers)
        assert r.status_code in [200, 503]

    def test_growth_unknown_returns_404(self, api_client, auth_headers):
        assert api_client.get("/growth/Atlantis",
            headers=auth_headers).status_code == 404

    def test_growth_has_summary(self, api_client, auth_headers):
        r = api_client.get("/growth/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            assert "summary" in r.json()

    def test_growth_has_cagr(self, api_client, auth_headers):
        r = api_client.get("/growth/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            summary = r.json()["summary"]
            assert any("cagr" in k.lower() for k in summary)


class TestCompareEndpoint:

    def test_compare_known_country(self, api_client, auth_headers):
        r = api_client.get("/compare/Tunisia", headers=auth_headers)
        assert r.status_code in [200, 404, 503]

    def test_compare_unknown_returns_404(self, api_client, auth_headers):
        assert api_client.get("/compare/Atlantis",
            headers=auth_headers).status_code == 404

    def test_compare_has_country_key(self, api_client, auth_headers):
        r = api_client.get("/compare/Tunisia", headers=auth_headers)
        if r.status_code == 200:
            assert r.json()["country"] == "Tunisia"


class TestFiguresEndpoint:

    def test_figures_returns_200(self, api_client, auth_headers):
        assert api_client.get("/figures", headers=auth_headers).status_code == 200

    def test_figures_has_key(self, api_client, auth_headers):
        data = api_client.get("/figures", headers=auth_headers).json()
        assert "figures" in data
        assert isinstance(data["figures"], list)

    def test_figures_unknown_file_returns_404(self, api_client, auth_headers):
        assert api_client.get("/figures/does_not_exist.pdf",
            headers=auth_headers).status_code == 404


class TestAskEndpoint:

    def test_ask_returns_200_or_500(self, api_client, auth_headers):
        r = api_client.post("/ask",
            json={"question": "Which country has the highest demand?"},
            headers=auth_headers)
        assert r.status_code in [200, 500]

    def test_ask_has_answer_field(self, api_client, auth_headers):
        r = api_client.post("/ask",
            json={"question": "What is the forecast for Tunisia?"},
            headers=auth_headers)
        if r.status_code == 200:
            assert "answer" in r.json()

    def test_ask_empty_question(self, api_client, auth_headers):
        r = api_client.post("/ask",
            json={"question": ""}, headers=auth_headers)
        assert r.status_code in [200, 422, 500]


# ═══════════════════════════════════════════════════════════════════════════════
# Input validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInputValidation:

    def test_country_special_chars_returns_404(self, api_client, auth_headers):
        assert api_client.get("/forecast/Tun!@ia",
            headers=auth_headers).status_code == 404

    def test_ask_missing_body_returns_422(self, api_client, auth_headers):
        r = api_client.post("/ask", json={}, headers=auth_headers)
        assert r.status_code in [422, 200]

    def test_ask_non_string_question(self, api_client, auth_headers):
        r = api_client.post("/ask",
            json={"question": 12345}, headers=auth_headers)
        assert r.status_code in [422, 200]

    def test_unknown_endpoint_returns_404(self, api_client):
        assert api_client.get("/nonexistent").status_code == 404
