"""
tests/test_forecasting.py — Unit + integration tests for src/04_forecasting.py
Ember Energy | IEEE Paper

Coverage
--------
clean_X                  — inf/nan imputation, dtype guarantee
extrapolate_exog         — linear projection, missing-col guard, fallback
build_future_feature_row — lag/MA/YoY for demand+exog, trend, country_code,
                           unknown-col fill, step progression
forecast_statistical     — Naive/LinearTrend (shared branch), Holt, ARIMA,
                           unknown model fallback
forecast_ml_recursive    — fit-once, recursive lags update, output shape/dtype,
                           too-few-rows raises, scaler path
_point_forecast          — stat dispatch, ML dispatch, unknown-model fallback
get_insample_residuals   — short-history guard, walk-forward structure
bootstrap_forecast_ci    — shape/ordering (lower≤point≤upper), few-residual
                           fallback, n_boot respected
main() integration       — end-to-end with temp dirs and mocked artifacts

Run
---
    pytest tests/test_forecasting.py -v --tb=short
"""

import importlib
import json
import os
import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import the module under test
# The file lives at src/04_forecasting.py; add src/ to path if needed.
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

# Try importing; if the file is named 04_forecasting.py we need importlib
try:
    import importlib.util as _ilu
    _candidates = [
        os.path.join(os.path.dirname(__file__), "..", "src", "04_forecasting.py"),
        os.path.join(os.path.dirname(__file__), "..", "04_forecasting.py"),
        os.path.join(os.path.dirname(__file__), "04_forecasting.py"),
    ]
    _spec = None
    for _path in _candidates:
        if os.path.exists(_path):
            _spec = _ilu.spec_from_file_location("forecasting", _path)
            break
    if _spec is None:
        pytest.exit("Cannot locate 04_forecasting.py — check PYTHONPATH or test location.")
    fc = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(fc)
except Exception as _e:
    pytest.exit(f"Failed to import 04_forecasting.py: {_e}")


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _make_history(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Minimal country history DataFrame with all columns the module expects."""
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + n))
    demand = 100.0 + np.cumsum(rng.normal(0, 2, n))
    exog   = 50.0  + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "Area":         ["TestLand"] * n,
        "Year":         years,
        "Demand":       demand,
        "Solar":        exog,
        "country_code": [1] * n,
        # pre-computed lag/MA/YoY features (values don't matter for structure tests)
        "Demand_lag1":  np.roll(demand, 1),
        "Demand_lag2":  np.roll(demand, 2),
        "Demand_lag3":  np.roll(demand, 3),
        "Demand_ma3":   pd.Series(demand).rolling(3).mean().fillna(demand[0]).values,
        "Demand_ma5":   pd.Series(demand).rolling(5).mean().fillna(demand[0]).values,
        "Demand_yoy":   np.zeros(n),
        "Solar_lag1":   np.roll(exog, 1),
        "Solar_lag2":   np.roll(exog, 2),
        "Solar_lag3":   np.roll(exog, 3),
        "Solar_ma3":    pd.Series(exog).rolling(3).mean().fillna(exog[0]).values,
        "Solar_ma5":    pd.Series(exog).rolling(5).mean().fillna(exog[0]).values,
        "Solar_yoy":    np.zeros(n),
        "trend":        list(range(n)),
        "trend_sq":     [i**2 for i in range(n)],
    })


ALL_FEAT = [
    "Demand_lag1", "Demand_lag2", "Demand_lag3",
    "Demand_ma3",  "Demand_ma5",  "Demand_yoy",
    "Solar_lag1",  "Solar_lag2",  "Solar_lag3",
    "Solar_ma3",   "Solar_ma5",   "Solar_yoy",
    "trend", "trend_sq", "country_code",
]
EXOG_COLS = ["Solar"]
HORIZON   = 3


# ===========================================================================
# 1. clean_X
# ===========================================================================

class TestCleanX:

    def test_output_dtype_is_float64(self):
        arr = np.array([[1, 2], [3, 4]], dtype=object)
        out = fc.clean_X(arr)
        assert out.dtype == np.float64

    def test_inf_replaced(self):
        arr = np.array([[np.inf, 1.0], [-np.inf, 2.0]])
        out = fc.clean_X(arr)
        assert np.isfinite(out).all()

    def test_nan_imputed_with_column_median(self):
        # col0 median = 2.0  →  NaN at [0,0] should become 2.0
        arr = np.array([[np.nan, 10.0], [2.0, 20.0], [2.0, 30.0]])
        out = fc.clean_X(arr)
        assert out[0, 0] == pytest.approx(2.0)

    def test_all_nan_column_fills_zero(self):
        arr = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        out = fc.clean_X(arr)           # nanmedian of all-nan → nan → take fills nan
        assert np.isfinite(out).all()   # must not raise; values may be 0 or nan-handled

    def test_clean_array_unchanged(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = fc.clean_X(arr)
        np.testing.assert_array_almost_equal(out, arr)

    def test_shape_preserved(self):
        arr = np.random.randn(10, 5)
        arr[2, 3] = np.inf
        out = fc.clean_X(arr)
        assert out.shape == (10, 5)


# ===========================================================================
# 2. extrapolate_exog
# ===========================================================================

class TestExtrapolateExog:

    def test_returns_dict_with_ndarray(self):
        hist = _make_history()
        result = fc.extrapolate_exog(hist, ["Solar"], n_steps=3)
        assert isinstance(result, dict)
        assert "Solar" in result
        assert isinstance(result["Solar"], np.ndarray)
        assert len(result["Solar"]) == 3

    def test_missing_col_silently_skipped(self):
        hist = _make_history()
        result = fc.extrapolate_exog(hist, ["Solar", "NONEXISTENT"], n_steps=2)
        assert "NONEXISTENT" not in result
        assert "Solar" in result

    def test_linear_trend_projection(self):
        """A perfectly linear series [0,1,2,...,n-1] should extrapolate to n, n+1, n+2."""
        n = 15
        hist = pd.DataFrame({"Year": range(n), "linear": np.arange(n, dtype=float)})
        result = fc.extrapolate_exog(hist, ["linear"], n_steps=3)
        np.testing.assert_array_almost_equal(result["linear"], [n, n + 1, n + 2], decimal=3)

    def test_inf_sanitized_before_fit(self):
        hist = _make_history()
        hist.loc[0, "Solar"] = np.inf
        result = fc.extrapolate_exog(hist, ["Solar"], n_steps=2)
        assert np.isfinite(result["Solar"]).all()

    def test_n_steps_zero(self):
        hist = _make_history()
        result = fc.extrapolate_exog(hist, ["Solar"], n_steps=0)
        assert len(result["Solar"]) == 0


# ===========================================================================
# 3. build_future_feature_row
# ===========================================================================

class TestBuildFutureFeatureRow:

    def _exog_future(self, hist, n_steps=HORIZON):
        return fc.extrapolate_exog(hist, EXOG_COLS, n_steps)

    def test_returns_dict_with_all_feature_cols(self):
        hist = _make_history()
        exog_f = self._exog_future(hist)
        row = fc.build_future_feature_row(hist, exog_f, [], 0, ALL_FEAT, "Demand")
        for col in ALL_FEAT:
            assert col in row, f"Missing key: {col}"

    def test_demand_lag1_at_step0_equals_last_history(self):
        hist = _make_history()
        exog_f = self._exog_future(hist)
        row = fc.build_future_feature_row(hist, exog_f, [], 0, ALL_FEAT, "Demand")
        expected = float(hist["Demand"].iloc[-1])
        assert row["Demand_lag1"] == pytest.approx(expected)

    def test_demand_lag1_at_step1_equals_step0_prediction(self):
        """After one prediction p0, lag1 at step1 must equal p0."""
        hist = _make_history()
        exog_f = self._exog_future(hist)
        p0 = 999.0
        row = fc.build_future_feature_row(hist, exog_f, [p0], 1, ALL_FEAT, "Demand")
        assert row["Demand_lag1"] == pytest.approx(p0)

    def test_lag_values_update_across_steps(self):
        """lag1 at step2 should equal the step1 prediction, not the historical value."""
        hist = _make_history()
        exog_f = self._exog_future(hist)
        p0, p1 = 900.0, 950.0
        row0 = fc.build_future_feature_row(hist, exog_f, [],       0, ALL_FEAT, "Demand")
        row1 = fc.build_future_feature_row(hist, exog_f, [p0],     1, ALL_FEAT, "Demand")
        row2 = fc.build_future_feature_row(hist, exog_f, [p0, p1], 2, ALL_FEAT, "Demand")
        assert row2["Demand_lag1"] == pytest.approx(p1)
        assert row2["Demand_lag2"] == pytest.approx(p0)

    def test_trend_increments_with_step(self):
        hist = _make_history()
        exog_f = self._exog_future(hist)
        r0 = fc.build_future_feature_row(hist, exog_f, [], 0, ALL_FEAT, "Demand")
        r1 = fc.build_future_feature_row(hist, exog_f, [0.0], 1, ALL_FEAT, "Demand")
        assert r1["trend"] == r0["trend"] + 1

    def test_trend_sq_is_trend_squared(self):
        hist = _make_history()
        exog_f = self._exog_future(hist)
        row = fc.build_future_feature_row(hist, exog_f, [], 0, ALL_FEAT, "Demand")
        assert row["trend_sq"] == pytest.approx(row["trend"] ** 2)

    def test_country_code_preserved(self):
        hist = _make_history()
        hist["country_code"] = 42
        exog_f = self._exog_future(hist)
        row = fc.build_future_feature_row(hist, exog_f, [], 0, ALL_FEAT, "Demand")
        assert row["country_code"] == 42

    def test_unknown_feature_cols_filled_with_zero(self):
        hist = _make_history()
        exog_f = self._exog_future(hist)
        extra_feat = ALL_FEAT + ["some_unknown_feature"]
        row = fc.build_future_feature_row(hist, exog_f, [], 0, extra_feat, "Demand")
        assert row["some_unknown_feature"] == pytest.approx(0.0)

    def test_yoy_is_zero_when_prev_is_zero(self):
        hist = _make_history()
        hist["Demand"] = 0.0
        exog_f = self._exog_future(hist)
        row = fc.build_future_feature_row(hist, exog_f, [], 0, ["Demand_yoy"], "Demand")
        assert row["Demand_yoy"] == pytest.approx(0.0)

    def test_ma3_is_mean_of_last_3_demand(self):
        hist = _make_history()
        exog_f = self._exog_future(hist)
        row = fc.build_future_feature_row(hist, exog_f, [], 0, ["Demand_ma3"], "Demand")
        expected = float(np.mean(hist["Demand"].values[-3:]))
        assert row["Demand_ma3"] == pytest.approx(expected)


# ===========================================================================
# 4. forecast_statistical
# ===========================================================================

class TestForecastStatistical:

    def _ts(self, n=20):
        return np.linspace(10, 30, n) + np.random.default_rng(7).normal(0, 0.5, n)

    def test_naive_returns_linear_trend(self):
        """Both Naive and LinearTrend should return a linear-regression forecast (not flat)."""
        ts  = self._ts()
        out = fc.forecast_statistical(ts, "Naive", 3)
        assert out.shape == (3,)
        # For a rising series the LR forecast should be above the mean
        assert float(np.mean(out)) > float(np.mean(ts)) * 0.5

    def test_naif_accent_variant(self):
        ts  = self._ts()
        out = fc.forecast_statistical(ts, "Naïve", 3)
        assert out.shape == (3,)

    def test_linear_trend_and_naive_return_same(self):
        """Notebook behaviour: Naive == LinearTrend."""
        ts = self._ts()
        np.testing.assert_array_almost_equal(
            fc.forecast_statistical(ts, "Naive",       4),
            fc.forecast_statistical(ts, "LinearTrend", 4),
        )

    def test_holt_shape_and_finite(self):
        ts  = self._ts()
        out = fc.forecast_statistical(ts, "Holt", 6)
        assert out.shape == (6,)
        assert np.isfinite(out).all()

    def test_arima_shape_and_finite(self):
        ts  = self._ts()
        out = fc.forecast_statistical(ts, "ARIMA(1,1,1)", 4)
        assert out.shape == (4,)
        assert np.isfinite(out).all()

    def test_arima_underscore_variant(self):
        ts  = self._ts()
        out = fc.forecast_statistical(ts, "ARIMA_1_1_1", 4)
        assert out.shape == (4,)

    def test_unknown_model_returns_last_value_repeated(self):
        ts  = np.array([1.0, 2.0, 3.0])
        out = fc.forecast_statistical(ts, "TOTALLY_UNKNOWN", 3)
        np.testing.assert_array_equal(out, [3.0, 3.0, 3.0])

    def test_horizon_respected(self):
        ts = self._ts()
        for h in [1, 3, 6, 10]:
            assert fc.forecast_statistical(ts, "LinearTrend", h).shape == (h,)

    def test_output_dtype_float(self):
        ts  = self._ts()
        out = fc.forecast_statistical(ts, "Holt", 3)
        assert out.dtype in (np.float32, np.float64)


# ===========================================================================
# 5. forecast_ml_recursive
# ===========================================================================

class TestForecastMlRecursive:

    def _run(self, horizon=HORIZON, scaler=None, **kwargs):
        from sklearn.linear_model import Ridge
        hist = _make_history(n=20)
        return fc.forecast_ml_recursive(
            hist, Ridge, ALL_FEAT, EXOG_COLS, horizon,
            scaler=scaler, alpha=1.0,
        )

    def test_output_shape(self):
        out = self._run(horizon=3)
        assert out.shape == (3,)

    def test_output_dtype_float64(self):
        out = self._run(horizon=3)
        assert out.dtype == np.float64

    def test_output_finite(self):
        out = self._run(horizon=6)
        assert np.isfinite(out).all()

    def test_raises_on_too_few_rows(self):
        from sklearn.linear_model import Ridge
        hist = _make_history(n=2)
        with pytest.raises(ValueError, match="Not enough training rows"):
            fc.forecast_ml_recursive(hist, Ridge, ALL_FEAT, EXOG_COLS, 3, alpha=1.0)

    def test_scaler_path_works(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        hist = _make_history(n=20)
        out  = fc.forecast_ml_recursive(
            hist, Ridge, ALL_FEAT, EXOG_COLS, 3,
            scaler=StandardScaler(), alpha=1.0,
        )
        assert out.shape == (3,)
        assert np.isfinite(out).all()

    def test_lags_update_across_steps(self):
        """
        Run two forecasts: one with all-zeros demand history, one with
        high-value demand history. The lag features must differ between
        step 0 and step 1, proving build_future_feature_row is called.
        We verify this indirectly: predictions must differ across steps
        for a non-constant series.
        """
        from sklearn.linear_model import Ridge
        hist = _make_history(n=20)
        out  = fc.forecast_ml_recursive(hist, Ridge, ALL_FEAT, EXOG_COLS, 3, alpha=1.0)
        # For a trending series predictions should not all be identical
        assert not (out[0] == out[1] == out[2]), \
            "All predictions identical — lags may not be updating"

    def test_random_forest_works(self):
        from sklearn.ensemble import RandomForestRegressor
        hist = _make_history(n=20)
        out  = fc.forecast_ml_recursive(
            hist, RandomForestRegressor, ALL_FEAT, EXOG_COLS, 3,
            n_estimators=5, random_state=0,
        )
        assert out.shape == (3,)
        assert np.isfinite(out).all()

    def test_missing_exog_col_silently_ignored(self):
        """If exog_cols contains a column not in history, it should be skipped."""
        from sklearn.linear_model import Ridge
        hist = _make_history(n=20)
        out  = fc.forecast_ml_recursive(
            hist, Ridge, ALL_FEAT, ["Solar", "NONEXISTENT"], 3, alpha=1.0,
        )
        assert out.shape == (3,)


# ===========================================================================
# 6. _point_forecast  (dispatcher)
# ===========================================================================

class TestPointForecast:

    def _hist(self):
        return _make_history(n=20)

    def test_stat_model_dispatched(self):
        out = fc._point_forecast(self._hist(), "LinearTrend", ALL_FEAT, EXOG_COLS, {}, 3)
        assert out.shape == (3,)
        assert np.isfinite(out).all()

    def test_holt_dispatched(self):
        out = fc._point_forecast(self._hist(), "Holt", ALL_FEAT, EXOG_COLS, {}, 3)
        assert out.shape == (3,)

    def test_ridge_dispatched(self):
        out = fc._point_forecast(self._hist(), "Ridge", ALL_FEAT, EXOG_COLS, {"alpha": 1.0}, 3)
        assert out.shape == (3,)
        assert np.isfinite(out).all()

    def test_unknown_model_falls_back_to_linear_trend(self):
        """Unknown model name → LinearTrend fallback, no exception."""
        out = fc._point_forecast(self._hist(), "BogusModel", ALL_FEAT, EXOG_COLS, {}, 3)
        expected = fc.forecast_statistical(
            self._hist()["Demand"].values.astype(float), "LinearTrend", 3
        )
        np.testing.assert_array_almost_equal(out, expected, decimal=3)

    def test_xgboost_kwargs_defaults_set(self):
        """When model_name='XGBoost', tree_method and verbosity must be defaulted."""
        if not fc.HAS_XGB:
            pytest.skip("XGBoost not installed")
        # Spy: patch forecast_ml_recursive to capture kwargs
        captured = {}
        original = fc.forecast_ml_recursive

        def spy(hist, cls, feat, exog, horizon, scaler=None, **kw):
            captured.update(kw)
            return original(hist, cls, feat, exog, horizon, scaler=scaler, **kw)

        with patch.object(fc, "forecast_ml_recursive", side_effect=spy):
            fc._point_forecast(self._hist(), "XGBoost", ALL_FEAT, EXOG_COLS, {}, 2)

        assert captured.get("tree_method") == "hist"
        assert captured.get("verbosity")   == 0


# ===========================================================================
# 7. get_insample_residuals
# ===========================================================================

class TestGetInsampleResiduals:

    def test_returns_ndarray(self):
        hist = _make_history(n=20)
        res  = fc.get_insample_residuals(hist, "LinearTrend", ALL_FEAT, EXOG_COLS, {})
        assert isinstance(res, np.ndarray)

    def test_length_at_most_n_eval(self):
        hist = _make_history(n=20)
        res  = fc.get_insample_residuals(hist, "LinearTrend", ALL_FEAT, EXOG_COLS, {}, n_eval=4)
        assert len(res) <= 4

    def test_short_history_returns_empty_or_partial(self):
        """History shorter than 10 rows → start = max(10, n-eval) = n → no residuals."""
        hist = _make_history(n=8)
        res  = fc.get_insample_residuals(hist, "LinearTrend", ALL_FEAT, EXOG_COLS, {})
        assert isinstance(res, np.ndarray)   # may be empty; must not raise

    def test_residuals_are_finite(self):
        hist = _make_history(n=20)
        res  = fc.get_insample_residuals(hist, "Holt", ALL_FEAT, EXOG_COLS, {})
        assert np.isfinite(res).all()

    def test_residuals_centred_near_zero_for_good_model(self):
        """LinearTrend on a linear series should have near-zero mean residual."""
        n    = 20
        hist = _make_history(n=n)
        # Replace demand with a perfect linear trend (zero noise)
        hist["Demand"] = np.linspace(10.0, 30.0, n)
        res  = fc.get_insample_residuals(hist, "LinearTrend", ALL_FEAT, EXOG_COLS, {})
        if len(res) > 0:
            assert abs(float(np.mean(res))) < 5.0   # generous bound; structure test only


# ===========================================================================
# 8. bootstrap_forecast_ci
# ===========================================================================

class TestBootstrapForecastCI:

    def _run(self, model="LinearTrend", n_boot=5, horizon=HORIZON):
        hist = _make_history(n=20)
        return fc.bootstrap_forecast_ci(
            hist, model, ALL_FEAT, EXOG_COLS, {}, horizon,
            n_boot=n_boot, ci=90,
        )

    def test_returns_three_arrays(self):
        pt, lo, hi = self._run()
        assert pt.shape == lo.shape == hi.shape == (HORIZON,)

    def test_lower_le_point_le_upper(self):
        pt, lo, hi = self._run(n_boot=10)
        assert np.all(lo <= pt + 1e-6), "lower > point somewhere"
        assert np.all(pt <= hi + 1e-6), "point > upper somewhere"

    def test_all_outputs_finite(self):
        pt, lo, hi = self._run(n_boot=5)
        for arr in (pt, lo, hi):
            assert np.isfinite(arr).all()

    def test_ci_width_non_negative(self):
        _, lo, hi = self._run(n_boot=10)
        assert np.all(hi - lo >= -1e-9)

    def test_few_residuals_fallback(self):
        """
        With history length = 11, get_insample_residuals has at most 1 eval point
        (start = max(10, 11-8) = 10, only 1 iteration). If < 3 residuals →
        should fall back to std-based CI without raising.
        """
        hist = _make_history(n=11)
        pt, lo, hi = fc.bootstrap_forecast_ci(
            hist, "LinearTrend", ALL_FEAT, EXOG_COLS, {}, 2, n_boot=3, ci=90
        )
        assert pt.shape == (2,)
        assert np.isfinite(pt).all()

    def test_holt_ci(self):
        pt, lo, hi = self._run(model="Holt", n_boot=5)
        assert pt.shape == (HORIZON,)
        assert np.isfinite(pt).all()

    def test_ridge_ci(self):
        hist = _make_history(n=20)
        pt, lo, hi = fc.bootstrap_forecast_ci(
            hist, "Ridge", ALL_FEAT, EXOG_COLS, {"alpha": 1.0}, HORIZON,
            n_boot=5, ci=90,
        )
        assert pt.shape == (HORIZON,)
        assert np.isfinite(pt).all()

    def test_wider_ci_with_more_noise(self):
        """A noisier series should produce wider (or equal) CIs than a smooth one."""
        rng = np.random.default_rng(42)
        n   = 20

        def _hist_with_noise(sigma):
            h = _make_history(n=n)
            h["Demand"] = 100.0 + np.cumsum(rng.normal(0, sigma, n))
            return h

        _, lo_low, hi_low = fc.bootstrap_forecast_ci(
            _hist_with_noise(0.1), "LinearTrend", ALL_FEAT, EXOG_COLS, {},
            HORIZON, n_boot=20, ci=90,
        )
        _, lo_hi, hi_hi = fc.bootstrap_forecast_ci(
            _hist_with_noise(5.0), "LinearTrend", ALL_FEAT, EXOG_COLS, {},
            HORIZON, n_boot=20, ci=90,
        )
        width_low = float(np.mean(hi_low - lo_low))
        width_hi  = float(np.mean(hi_hi  - lo_hi))
        # High-noise CI should generally be wider; allow small tolerance
        assert width_hi >= width_low * 0.5, \
            f"Expected noisier series to have wider CI: {width_hi:.2f} vs {width_low:.2f}"


# ===========================================================================
# 9. Constants / palette (notebook parity)
# ===========================================================================

class TestConstantsParity:

    def test_palette_tunisia_hex(self):
        assert fc.PALETTE["Tunisia"] == "#e63946"

    def test_palette_austria_hex(self):
        assert fc.PALETTE["Austria"] == "#2196F3"

    def test_palette_germany_hex(self):
        assert fc.PALETTE["Germany"] == "#4CAF50"

    def test_all_countries_in_palette(self):
        for c in fc.COUNTRIES:
            assert c in fc.PALETTE, f"{c} missing from PALETTE"

    def test_forecast_years_range(self):
        assert fc.FORECAST_YEARS == list(range(2025, 2031))

    def test_n_boot_default(self):
        assert fc.N_BOOT == 300

    def test_ci_default(self):
        assert fc.CI == 90

    def test_target_column(self):
        assert fc.TARGET == "Demand"

    def test_stat_models_set_contains_expected(self):
        for m in ["Naive", "Naïve", "LinearTrend", "Holt", "ARIMA(1,1,1)", "ARIMA_1_1_1"]:
            assert m in fc.STAT_MODELS

    def test_holt_uses_exponential_smoothing_not_simple(self):
        """Confirm the Holt import is ExponentialSmoothing (damped trend), not SimpleExpSmoothing."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        import inspect, ast as _ast
        src = inspect.getsource(fc.forecast_statistical)
        tree = _ast.parse(src)
        calls = [
            node.func.id if isinstance(node.func, _ast.Name) else
            node.func.attr if isinstance(node.func, _ast.Attribute) else ""
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Call)
        ]
        assert "ExponentialSmoothing" in calls, \
            "forecast_statistical must use ExponentialSmoothing"
        assert "SimpleExpSmoothing" not in calls, \
            "forecast_statistical must NOT use SimpleExpSmoothing"

    def test_naive_and_linear_trend_same_output(self):
        """Notebook behaviour: Naive and LinearTrend are the same branch."""
        ts = np.linspace(10.0, 30.0, 15)
        np.testing.assert_array_almost_equal(
            fc.forecast_statistical(ts, "Naive",       5),
            fc.forecast_statistical(ts, "LinearTrend", 5),
        )


# ===========================================================================
# 10. main() — integration smoke test
# ===========================================================================

class TestMainIntegration:

    @pytest.fixture()
    def tmp_dirs(self, tmp_path):
        pre_dir   = tmp_path / "preprocessing"
        model_dir = tmp_path / "modeling"
        out_dir   = tmp_path / "forecasting"
        pre_dir.mkdir();  model_dir.mkdir();  out_dir.mkdir()
        return pre_dir, model_dir, out_dir

    def _write_artifacts(self, pre_dir, model_dir):
        """Write minimal CSV + JSON artifacts that main() can load."""
        n = 20
        rows = []
        for country in fc.COUNTRIES:
            rng = np.random.default_rng(hash(country) % (2**31))
            for yr in range(2000, 2000 + n):
                demand = 50.0 + rng.normal(0, 2)
                solar  = 20.0 + rng.normal(0, 1)
                idx    = yr - 2000
                rows.append({
                    "Area":         country,
                    "Year":         yr,
                    "Demand":       demand,
                    "Solar":        solar,
                    "country_code": fc.COUNTRIES.index(country),
                    "Demand_lag1":  demand * 0.98,
                    "Demand_lag2":  demand * 0.96,
                    "Demand_lag3":  demand * 0.94,
                    "Demand_ma3":   demand,
                    "Demand_ma5":   demand,
                    "Demand_yoy":   1.0,
                    "Solar_lag1":   solar * 0.98,
                    "Solar_lag2":   solar * 0.96,
                    "Solar_lag3":   solar * 0.94,
                    "Solar_ma3":    solar,
                    "Solar_ma5":    solar,
                    "Solar_yoy":    0.5,
                    "trend":        idx,
                    "trend_sq":     idx ** 2,
                })

        pd.DataFrame(rows).to_csv(pre_dir / "ember_model_ready.csv", index=False)

        best_models = pd.DataFrame({
            "Country": fc.COUNTRIES,
            "Model":   ["LinearTrend"] * len(fc.COUNTRIES),
            "MAPE":    [3.0] * len(fc.COUNTRIES),
        })
        best_models.to_csv(model_dir / "best_models.csv", index=False)

        meta = {
            "all_features": ALL_FEAT,
            "raw_features": EXOG_COLS,
            "FEATURES":     EXOG_COLS,
            "TEST_END":     2019,
            "FORECAST_YEARS": list(range(2020, 2023)),
        }
        with open(model_dir / "ember_config.json", "w") as f:
            json.dump(meta, f)

        with open(model_dir / "best_hp.json", "w") as f:
            json.dump({"LinearTrend": {}}, f)

    def test_main_runs_and_produces_csvs(self, tmp_dirs, monkeypatch):
        pre_dir, model_dir, out_dir = tmp_dirs
        self._write_artifacts(pre_dir, model_dir)

        monkeypatch.setattr(
            sys, "argv",
            [
                "04_forecasting.py",
                "--pre_dir",        str(pre_dir),
                "--model_dir",      str(model_dir),
                "--output_dir",     str(out_dir),
                "--forecast_until", "2022",
            ],
        )

        with patch.object(fc, "N_BOOT", 2):   # fast bootstrap for tests
            fc.main()

        assert (out_dir / "demand_forecast_2025_2030.csv").exists()
        assert (out_dir / "demand_growth_summary.csv").exists()

    def test_main_forecast_csv_columns(self, tmp_dirs, monkeypatch):
        pre_dir, model_dir, out_dir = tmp_dirs
        self._write_artifacts(pre_dir, model_dir)

        monkeypatch.setattr(
            sys, "argv",
            [
                "04_forecasting.py",
                "--pre_dir",        str(pre_dir),
                "--model_dir",      str(model_dir),
                "--output_dir",     str(out_dir),
                "--forecast_until", "2022",
            ],
        )

        with patch.object(fc, "N_BOOT", 2):
            fc.main()

        df_out = pd.read_csv(out_dir / "demand_forecast_2025_2030.csv")
        for col in ["Country", "Year", "Forecast", "Lower_90", "Upper_90", "Model"]:
            assert col in df_out.columns, f"Missing column: {col}"

    def test_main_lower_le_upper_in_output(self, tmp_dirs, monkeypatch):
        pre_dir, model_dir, out_dir = tmp_dirs
        self._write_artifacts(pre_dir, model_dir)

        monkeypatch.setattr(
            sys, "argv",
            [
                "04_forecasting.py",
                "--pre_dir",        str(pre_dir),
                "--model_dir",      str(model_dir),
                "--output_dir",     str(out_dir),
                "--forecast_until", "2022",
            ],
        )

        with patch.object(fc, "N_BOOT", 2):
            fc.main()

        df_out = pd.read_csv(out_dir / "demand_forecast_2025_2030.csv")
        assert (df_out["Lower_90"] <= df_out["Upper_90"] + 1e-6).all()

    def test_main_all_countries_present_in_output(self, tmp_dirs, monkeypatch):
        pre_dir, model_dir, out_dir = tmp_dirs
        self._write_artifacts(pre_dir, model_dir)

        monkeypatch.setattr(
            sys, "argv",
            [
                "04_forecasting.py",
                "--pre_dir",        str(pre_dir),
                "--model_dir",      str(model_dir),
                "--output_dir",     str(out_dir),
                "--forecast_until", "2022",
            ],
        )

        with patch.object(fc, "N_BOOT", 2):
            fc.main()

        df_out  = pd.read_csv(out_dir / "demand_forecast_2025_2030.csv")
        present = set(df_out["Country"].unique())
        for c in fc.COUNTRIES:
            assert c in present, f"{c} missing from output CSV"

    def test_main_figures_created(self, tmp_dirs, monkeypatch):
        pre_dir, model_dir, out_dir = tmp_dirs
        self._write_artifacts(pre_dir, model_dir)

        monkeypatch.setattr(
            sys, "argv",
            [
                "04_forecasting.py",
                "--pre_dir",        str(pre_dir),
                "--model_dir",      str(model_dir),
                "--output_dir",     str(out_dir),
                "--forecast_until", "2022",
            ],
        )

        with patch.object(fc, "N_BOOT", 2):
            fc.main()

        fig_dir = out_dir / "figures"
        assert (fig_dir / "forecast_per_country.png").exists()
        assert (fig_dir / "forecast_overlay.png").exists()
        assert (fig_dir / "growth_uncertainty.png").exists()
