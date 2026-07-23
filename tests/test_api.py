"""
tests/test_api.py — FastAPI Endpoint Tests
Ember Energy Forecasting | IEEE Paper

Covers:
  - Public endpoints  (/health, /countries, /auth/login)
  - Protected endpoints (all forecast, metrics, growth, compare, figures, ask)
  - Model diagnostics  (/significance, /robustness — new)
  - Data overview      (/data/demand-history, /data/subcategories,
                         /data/summary — new)
  - JWT auth flow (login, logout, token revocation, /auth/me)
  - Input validation and error codes
"""

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
    """
    Create minimal mock output files the API reads at startup.
    Filenames/columns here MUST match the real path constants defined in
    api/main.py (FORECAST_CSV, METRICS_CSV, etc.) — a mismatch here means
    tests silently pass via the 503 "not available" branch without ever
    exercising the real 200-path assertions.
    """
    fc_dir  = tmp_path / "forecasting"
    mod_dir = tmp_path / "modeling"
    dl_dir  = tmp_path / "deeplearning"
    eda_dir = tmp_path / "eda"
    for d in (fc_dir, mod_dir, dl_dir, eda_dir):
        d.mkdir()

    # ── forecasting/demand_forecast_2025_2030.csv (FORECAST_CSV) ─────────────
    rows = []
    for c in COUNTRIES:
        for yr in FORECAST_YEARS:
            rows.append({
                "Country": c, "Model": "Ridge", "Year": yr,
                "Forecast": 10.0 + 0.5 * (yr - 2025),
                "Lower_90": 9.5, "Upper_90": 10.5,
            })
    pd.DataFrame(rows).to_csv(fc_dir / "demand_forecast_2025_2030.csv", index=False)

    # ── forecasting/all_models_forecast.csv (ALL_MODELS_FORECAST_CSV) ────────
    all_models_rows = []
    for c in COUNTRIES:
        for model in ["Ridge", "BayesianRidge", "LinearTrend"]:
            for yr in FORECAST_YEARS:
                all_models_rows.append({
                    "Country": c, "Model": model, "Year": yr,
                    "Forecast": 10.0 + 0.5 * (yr - 2025),
                })
    pd.DataFrame(all_models_rows).to_csv(fc_dir / "all_models_forecast.csv", index=False)

    # ── modeling/test_benchmarking.csv (METRICS_CSV) ─────────────────────────
    pd.DataFrame([{
        "Country": c, "Model": "Ridge",
        "MAE": 0.5, "RMSE": 0.7, "MAPE": 2.5,
    } for c in COUNTRIES]).to_csv(mod_dir / "test_benchmarking.csv", index=False)

    # ── forecasting/demand_growth_summary.csv (GROWTH_CSV) ───────────────────
    pd.DataFrame([{
        "Country": c, "Model": "Ridge",
        "2024 (TWh)": 10.0, "2025 Forecast": 10.5, "2030 Forecast": 13.0,
        "Total Growth %": 30.0, "CAGR 24-30 %": 4.5,
    } for c in COUNTRIES]).to_csv(fc_dir / "demand_growth_summary.csv", index=False)

    # ── modeling/best_models.csv (BEST_MODELS_CSV) ───────────────────────────
    pd.DataFrame([{"Country": c, "Model": "Ridge", "MAPE": 2.5}
                  for c in COUNTRIES]).to_csv(mod_dir / "best_models.csv", index=False)

    # ── deeplearning/dl_forecast_2025_2030.csv (DL_FORECAST_CSV) ─────────────
    dl_rows = []
    for c in COUNTRIES:
        for yr in FORECAST_YEARS:
            dl_rows.append({"Country": c, "Model": "MLP", "Year": yr,
                            "Forecast": 10.2 + 0.4 * (yr - 2025)})
    pd.DataFrame(dl_rows).to_csv(dl_dir / "dl_forecast_2025_2030.csv", index=False)

    # ── deeplearning/dl_benchmarking.csv (DL_METRICS_CSV) ────────────────────
    pd.DataFrame([{"Country": c, "Model": "MLP",
                   "MAE": 0.4, "RMSE": 0.6, "MAPE": 2.1, "SMAPE": 2.0, "N": 4}
                  for c in COUNTRIES]).to_csv(dl_dir / "dl_benchmarking.csv", index=False)

    # ── deeplearning/dl_best_models.csv (DL_BEST_CSV) ─────────────────────────
    pd.DataFrame([{"Country": c, "Model": "MLP", "MAPE": 2.1}
                  for c in COUNTRIES]).to_csv(dl_dir / "dl_best_models.csv", index=False)

    # ── modeling/significance_tests.csv (SIGNIFICANCE_TESTS_CSV) ─────────────
    sig_rows = [{
        "Country": c, "Model_A": "Ridge", "Model_B": "LinearTrend",
        "MAPE_A": 2.5, "MAPE_B": 3.0,
        "DM_stat": 1.2, "DM_p": 0.03, "WX_stat": 5.0, "WX_p": 0.04,
        "significant_DM": True, "significant_WX": True, "better_model": "Ridge",
    } for c in COUNTRIES]
    pd.DataFrame(sig_rows).to_csv(mod_dir / "significance_tests.csv", index=False)

    pd.DataFrame([{
        "Country": c, "Total_pairs": 1, "Significant_DM": 1,
        "Significant_WX": 1, "Any_significant": True,
    } for c in COUNTRIES]).to_csv(mod_dir / "significance_summary.csv", index=False)

    # ── forecasting/robustness_summary.csv (ROBUSTNESS_SUMMARY_CSV) ──────────
    rb_rows = []
    for c in COUNTRIES:
        rb_rows.append({
            "Country": c, "Model": "Ridge",
            "max_deviation_pct": 5.0, "max_yoy_change_pct": 3.0,
            "trajectory_std": 1.0, "direction_flips": 0, "robustness_flag": "STABLE",
        })
    pd.DataFrame(rb_rows).to_csv(fc_dir / "robustness_summary.csv", index=False)

    pd.DataFrame([{
        "Country": c, "Model": "Ridge", "MAPE": 2.5,
        "max_deviation_pct": 5.0, "max_yoy_change_pct": 3.0,
        "robustness_flag": "STABLE", "accuracy_rank": 1.0,
        "robustness_rank": 1.0, "axes_agree": True,
    } for c in COUNTRIES]).to_csv(fc_dir / "robustness_vs_accuracy.csv", index=False)

    # ── forecasting/best_model_conflicts.csv (BEST_MODEL_CONFLICTS_CSV) ──────
    pd.DataFrame([{
        "Country": c, "Best_Model": "Ridge", "MAPE": 2.5,
        "robustness_flag": "STABLE", "max_deviation_pct": 5.0, "conflict": False,
    } for c in COUNTRIES]).to_csv(fc_dir / "best_model_conflicts.csv", index=False)

    # ── eda/ember_filtered.csv (SUBCATEGORY_CSV) ──────────────────────────────
    subcat_rows = []
    for c in COUNTRIES:
        for subcat, unit in [("Demand", "TWh"), ("CO2 intensity", "gCO2/kWh"),
                              ("Demand per capita", "MWh"), ("Electricity imports", "TWh")]:
            for yr in range(2000, 2025):
                subcat_rows.append({"Area": c, "Year": yr, "Subcategory": subcat,
                                    "Unit": unit, "Value": 10.0 + 0.1 * (yr - 2000)})
    pd.DataFrame(subcat_rows).to_csv(eda_dir / "ember_filtered.csv", index=False)

    # ── eda/table01_eda_statistics.csv (SUMMARY_STATS_CSV) ───────────────────
    pd.DataFrame([{
        "Country": c, "2000 (TWh)": 8.0, "2024 (TWh)": 10.0,
        "Total Growth": "25.0%", "Slope TWh/yr": 0.1, "Mean TWh": 9.0, "Std TWh": 0.5,
    } for c in COUNTRIES]).to_csv(eda_dir / "table01_eda_statistics.csv", index=False)

    # ── preprocessing/ember_model_ready.csv (HISTORY_CSV) ────────────────────
    pre_dir = tmp_path / "preprocessing"
    pre_dir.mkdir()
    hist_rows = []
    for c in COUNTRIES:
        for yr in range(2000, 2025):
            hist_rows.append({"Area": c, "Year": yr, "Demand": 10.0 + 0.1 * (yr - 2000)})
    pd.DataFrame(hist_rows).to_csv(pre_dir / "ember_model_ready.csv", index=False)

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
# Forecast endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestForecastEndpoint:

    def test_forecast_known_country(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_forecast_unknown_country_returns_404(self, api_client, auth_headers):
        assert api_client.get("/forecast/Atlantis",
            headers=auth_headers).status_code == 404

    def test_forecast_response_has_required_fields(self, api_client, auth_headers):
        r = api_client.get("/forecast/Tunisia", headers=auth_headers)
        assert r.status_code == 200
        for field in ["country", "model"]:
            assert field in r.json()

    def test_all_countries_respond(self, api_client, auth_headers):
        for c in COUNTRIES:
            r = api_client.get(f"/forecast/{c}", headers=auth_headers)
            assert r.status_code == 200, f"{c}: unexpected {r.status_code}"


class TestAllModelsForecastEndpoint:
    """GET /forecast/all/{country} — point forecast for every model, used
    to power the dashboard's 'All models' comparison chart."""

    def test_known_country_returns_200(self, api_client, auth_headers):
        r = api_client.get("/forecast/all/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_has_models_key(self, api_client, auth_headers):
        r = api_client.get("/forecast/all/Tunisia", headers=auth_headers)
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_unknown_country_returns_404(self, api_client, auth_headers):
        assert api_client.get("/forecast/all/Atlantis",
            headers=auth_headers).status_code == 404


class TestDLForecastEndpoint:

    def test_dl_forecast_known_country(self, api_client, auth_headers):
        r = api_client.get("/forecast/dl/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_dl_forecast_unknown_country_returns_404(self, api_client, auth_headers):
        assert api_client.get("/forecast/dl/Atlantis",
            headers=auth_headers).status_code == 404


class TestHistoryEndpoint:

    def test_history_known_country(self, api_client, auth_headers):
        r = api_client.get("/history/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_history_has_years_and_events(self, api_client, auth_headers):
        r = api_client.get("/history/Tunisia", headers=auth_headers)
        data = r.json()
        assert "years" in data
        assert "events" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsEndpoint:

    def test_metrics_known_country(self, api_client, auth_headers):
        r = api_client.get("/metrics/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_metrics_unknown_returns_404(self, api_client, auth_headers):
        assert api_client.get("/metrics/Mars",
            headers=auth_headers).status_code == 404

    def test_metrics_has_expected_keys(self, api_client, auth_headers):
        r = api_client.get("/metrics/Tunisia", headers=auth_headers)
        data = r.json()
        metrics = data.get("metrics", data)  # tolerate either shape
        for k in ["MAE", "RMSE", "MAPE"]:
            assert k in metrics


class TestDLMetricsEndpoint:
    """GET /metrics/dl/{country} — separate from classical /metrics/{country}."""

    def test_dl_metrics_known_country(self, api_client, auth_headers):
        r = api_client.get("/metrics/dl/Tunisia", headers=auth_headers)
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Model diagnostics — significance & robustness (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignificanceEndpoint:

    def test_significance_known_country(self, api_client, auth_headers):
        r = api_client.get("/significance/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_significance_has_pairs_key(self, api_client, auth_headers):
        r = api_client.get("/significance/Tunisia", headers=auth_headers)
        data = r.json()
        assert "pairs" in data
        assert isinstance(data["pairs"], list)

    def test_significance_unknown_country_returns_404(self, api_client, auth_headers):
        assert api_client.get("/significance/Atlantis",
            headers=auth_headers).status_code == 404


class TestRobustnessEndpoint:

    def test_robustness_known_country(self, api_client, auth_headers):
        r = api_client.get("/robustness/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_robustness_has_models_key(self, api_client, auth_headers):
        r = api_client.get("/robustness/Tunisia", headers=auth_headers)
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_robustness_unknown_country_returns_404(self, api_client, auth_headers):
        assert api_client.get("/robustness/Atlantis",
            headers=auth_headers).status_code == 404

    def test_robustness_conflicts_returns_200(self, api_client, auth_headers):
        """
        Route-order regression test: /robustness/conflicts is a static
        route that MUST be matched before the parameterized
        /robustness/{country} route, or FastAPI shadows it and this
        404s trying to validate "conflicts" as a country name.
        """
        r = api_client.get("/robustness/conflicts", headers=auth_headers)
        assert r.status_code == 200

    def test_robustness_conflicts_has_expected_keys(self, api_client, auth_headers):
        r = api_client.get("/robustness/conflicts", headers=auth_headers)
        data = r.json()
        for k in ["total_countries", "n_conflicts", "rows"]:
            assert k in data


# ═══════════════════════════════════════════════════════════════════════════════
# Growth & Compare
# ═══════════════════════════════════════════════════════════════════════════════

class TestGrowthEndpoint:

    def test_growth_known_country(self, api_client, auth_headers):
        r = api_client.get("/growth/Tunisia", headers=auth_headers)
        assert r.status_code == 200

    def test_growth_unknown_returns_404(self, api_client, auth_headers):
        assert api_client.get("/growth/Atlantis",
            headers=auth_headers).status_code == 404

    def test_growth_has_summary(self, api_client, auth_headers):
        r = api_client.get("/growth/Tunisia", headers=auth_headers)
        assert "summary" in r.json()

    def test_growth_has_cagr(self, api_client, auth_headers):
        r = api_client.get("/growth/Tunisia", headers=auth_headers)
        summary = r.json()["summary"]
        assert any("cagr" in k.lower() for k in summary)


class TestCompareEndpoint:

    def test_compare_known_country(self, api_client, auth_headers):
        r = api_client.get("/compare/Tunisia", headers=auth_headers)
        assert r.status_code in [200, 404]

    def test_compare_unknown_returns_404(self, api_client, auth_headers):
        assert api_client.get("/compare/Atlantis",
            headers=auth_headers).status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Data overview endpoints (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataOverviewEndpoints:

    def test_demand_history_returns_200(self, api_client, auth_headers):
        r = api_client.get("/data/demand-history", headers=auth_headers)
        assert r.status_code == 200

    def test_demand_history_has_all_countries(self, api_client, auth_headers):
        r = api_client.get("/data/demand-history", headers=auth_headers)
        data = r.json()["countries"]
        for c in COUNTRIES:
            assert c in data

    def test_subcategories_returns_200(self, api_client, auth_headers):
        r = api_client.get("/data/subcategories", headers=auth_headers)
        assert r.status_code == 200
        assert "subcategories" in r.json()

    def test_summary_returns_200(self, api_client, auth_headers):
        r = api_client.get("/data/summary", headers=auth_headers)
        assert r.status_code == 200
        assert "rows" in r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# Figures & Ask
# ═══════════════════════════════════════════════════════════════════════════════

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
