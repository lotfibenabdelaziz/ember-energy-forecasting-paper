"""
tests/test_deeplearning.py — Deep Learning Module Tests
Ember Energy | IEEE Paper

Covers:
  - src/dataset.py   : DemandDataset, make_loaders, clean_arr
  - src/model.py     : MLP, TCN, NBeats, TFT architectures
  - src/train.py     : train_model (smoke test)
  - src/evaluate.py  : mape, rmse, smape, walk_forward_dl
  - src/forecast.py  : recursive_forecast, bootstrap_ci, growth_summary
  - CLI integration  : dl_benchmarking.csv, dl_best_models.csv schema
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

COUNTRIES      = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET         = "Demand"
SEQ_LEN        = 5
N_FEATURES     = 10
BATCH_SIZE     = 4
FORECAST_YEARS = list(range(2025, 2031))
TRAIN_END      = 2016
VAL_END        = 2020
TEST_END       = 2024


# ── Shared fixture ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synth_country_df():
    """Single-country synthetic DataFrame with required features."""
    np.random.seed(42)
    years = list(range(2000, 2025))
    rows  = []
    for yr in years:
        demand = 10 + 0.5 * (yr - 2000) + np.random.normal(0, 0.2)
        rows.append({
            "Area":   "Tunisia",
            "Year":   yr,
            "Demand": demand,
            **{f"feat_{i}": np.random.normal(5, 1) for i in range(N_FEATURES - 1)},
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def feature_cols(synth_country_df):
    return [c for c in synth_country_df.columns if c not in ["Area", "Year", TARGET]]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestDemandDataset:

    def test_dataset_length(self, synth_country_df, feature_cols):
        from src.dataset import DemandDataset
        ds = DemandDataset(synth_country_df, feature_cols, TARGET, SEQ_LEN)
        assert len(ds) == max(0, len(synth_country_df) - SEQ_LEN)

    def test_dataset_item_shapes(self, synth_country_df, feature_cols):
        from src.dataset import DemandDataset
        ds  = DemandDataset(synth_country_df, feature_cols, TARGET, SEQ_LEN)
        X, y = ds[0]
        assert X.shape == (SEQ_LEN, len(feature_cols))
        assert y.shape == ()

    def test_dataset_no_nan_in_X(self, synth_country_df, feature_cols):
        from src.dataset import DemandDataset
        ds  = DemandDataset(synth_country_df, feature_cols, TARGET, SEQ_LEN)
        X, _ = ds[0]
        assert not torch.isnan(X).any()

    def test_dataset_scaler_fitted(self, synth_country_df, feature_cols):
        from src.dataset import DemandDataset
        ds = DemandDataset(synth_country_df, feature_cols, TARGET, SEQ_LEN)
        assert ds.scaler_X is not None
        assert ds.scaler_y is not None

    def test_make_loaders_returns_four(self, synth_country_df, feature_cols):
        from src.dataset import make_loaders
        tr, val, sX, sy = make_loaders(
            synth_country_df, feature_cols, TARGET, SEQ_LEN,
            TRAIN_END, VAL_END, BATCH_SIZE,
        )
        assert tr  is not None
        assert val is not None
        assert sX  is not None
        assert sy  is not None

    def test_make_loaders_no_single_batch(self, synth_country_df, feature_cols):
        """Ensure no batch of size 1 (BatchNorm requirement)."""
        from src.dataset import make_loaders
        tr, _, _, _ = make_loaders(
            synth_country_df, feature_cols, TARGET, SEQ_LEN,
            TRAIN_END, VAL_END, BATCH_SIZE,
        )
        for X_b, y_b in tr:
            if len(X_b) == 1:
                pytest.fail("Found batch of size 1 — will crash BatchNorm")

    def test_clean_arr_removes_inf(self):
        from src.dataset import clean_arr
        arr = np.array([[1.0, np.inf], [2.0, 3.0]])
        out = clean_arr(arr)
        assert np.isfinite(out).all()

    def test_clean_arr_removes_nan(self):
        from src.dataset import clean_arr
        arr = np.array([[np.nan, 2.0], [3.0, 4.0]])
        out = clean_arr(arr)
        assert not np.isnan(out).any()


# ═══════════════════════════════════════════════════════════════════════════════
# Model architectures
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLPForecaster:

    def test_output_shape(self):
        from src.model import MLPForecaster
        m = MLPForecaster(seq_len=SEQ_LEN, n_features=N_FEATURES)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert m(x).shape == (BATCH_SIZE, 1)

    def test_output_finite(self):
        from src.model import MLPForecaster
        m = MLPForecaster(seq_len=SEQ_LEN, n_features=N_FEATURES)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert torch.isfinite(m(x)).all()

    def test_batch_size_2_works(self):
        """Minimum batch size for BatchNorm."""
        from src.model import MLPForecaster
        m = MLPForecaster(seq_len=SEQ_LEN, n_features=N_FEATURES)
        m.eval()
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        assert m(x).shape == (2, 1)


class TestTCNForecaster:

    def test_output_shape(self):
        from src.model import TCNForecaster
        m = TCNForecaster(n_features=N_FEATURES)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert m(x).shape == (BATCH_SIZE, 1)

    def test_causal_dilations(self):
        """TCN with dilations=(1,2,4) should work on any batch size >= 2."""
        from src.model import TCNForecaster
        m = TCNForecaster(n_features=N_FEATURES, dilations=(1, 2, 4))
        m.eval()
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        out = m(x)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()

    def test_output_finite(self):
        from src.model import TCNForecaster
        m = TCNForecaster(n_features=N_FEATURES)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert torch.isfinite(m(x)).all()


class TestNBeatsForecaster:

    def test_output_shape(self):
        from src.model import NBeatsForecaster
        m = NBeatsForecaster(seq_len=SEQ_LEN, horizon=1, n_features=N_FEATURES)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        out = m(x)
        assert out.shape[0] == BATCH_SIZE

    def test_output_finite(self):
        from src.model import NBeatsForecaster
        m = NBeatsForecaster(seq_len=SEQ_LEN, n_features=N_FEATURES)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert torch.isfinite(m(x)).all()


class TestTFTForecaster:

    def test_output_shape(self):
        from src.model import TFTForecaster
        m = TFTForecaster(n_features=N_FEATURES, seq_len=SEQ_LEN)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert m(x).shape == (BATCH_SIZE, 1)

    def test_output_finite(self):
        from src.model import TFTForecaster
        m = TFTForecaster(n_features=N_FEATURES, seq_len=SEQ_LEN)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        assert torch.isfinite(m(x)).all()

    def test_attention_weights_stored(self):
        from src.model import TFTForecaster
        m = TFTForecaster(n_features=N_FEATURES, seq_len=SEQ_LEN)
        m.eval()
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        m(x)
        assert m.get_attention_weights() is not None


class TestModelRegistry:

    def test_registry_has_all_models(self):
        from src.model import build_model_registry
        reg = build_model_registry(SEQ_LEN, N_FEATURES)
        for name in ["MLP", "TCN", "N-BEATS", "TFT"]:
            assert name in reg

    def test_registry_models_instantiate(self):
        from src.model import build_model_registry
        reg = build_model_registry(SEQ_LEN, N_FEATURES)
        for name, entry in reg.items():
            m = entry["cls"](**entry["kwargs"])
            assert m is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainModel:

    def test_train_returns_loss_lists(self, synth_country_df, feature_cols):
        from src.dataset import make_loaders
        from src.model import MLPForecaster
        from src.train import train_model
        tr, val, _, _ = make_loaders(
            synth_country_df, feature_cols, TARGET, SEQ_LEN,
            TRAIN_END, VAL_END, BATCH_SIZE,
        )
        model = MLPForecaster(seq_len=SEQ_LEN, n_features=len(feature_cols))
        tr_l, val_l, ep = train_model(model, tr, val, epochs=5, patience=3, lr=1e-3)
        assert len(tr_l) > 0
        assert len(val_l) > 0
        assert ep >= 1

    def test_train_loss_decreases(self, synth_country_df, feature_cols):
        from src.dataset import make_loaders
        from src.model import MLPForecaster
        from src.train import train_model
        tr, val, _, _ = make_loaders(
            synth_country_df, feature_cols, TARGET, SEQ_LEN,
            TRAIN_END, VAL_END, BATCH_SIZE,
        )
        model = MLPForecaster(seq_len=SEQ_LEN, n_features=len(feature_cols))
        tr_l, _, _ = train_model(model, tr, val, epochs=20, patience=10, lr=1e-3)
        # Loss should generally decrease — first epoch higher than last
        assert tr_l[0] >= tr_l[-1] or len(tr_l) <= 3

    def test_train_tcn_no_batchnorm_crash(self, synth_country_df, feature_cols):
        """TCN with small dataset should not crash with BatchNorm."""
        from src.dataset import make_loaders
        from src.model import TCNForecaster
        from src.train import train_model
        tr, val, _, _ = make_loaders(
            synth_country_df, feature_cols, TARGET, SEQ_LEN,
            TRAIN_END, VAL_END, BATCH_SIZE,
        )
        model = TCNForecaster(n_features=len(feature_cols))
        tr_l, val_l, _ = train_model(model, tr, val, epochs=3, patience=3, lr=1e-3)
        assert len(tr_l) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestDLMetrics:

    def test_mape_perfect(self):
        from src.evaluate import mape
        a = np.array([10.0, 20.0, 30.0])
        assert mape(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_known(self):
        from src.evaluate import rmse
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert rmse(a, b) == pytest.approx(np.sqrt(12.5), abs=1e-4)

    # def test_smape_symmetric(self):
    #     from src.evaluate import smape
    #     a = np.array([100.0])
    #     b = np.array([150.0])
    #     c = np.array([50.0])
    #     # sMAPE should be same for over- and under-prediction of same magnitude
    #     assert abs(smape(a, b) - smape(a, c)) < 5.0
    def test_smape_symmetric(self):
        from src.evaluate import smape
        a = np.array([100.0])
        b = np.array([150.0])
        # True sMAPE symmetry: swapping actual and predicted gives same result
        assert abs(smape(a, b) - smape(b, a)) < 1e-6

    def test_mae_zero(self):
        from src.evaluate import mae
        a = np.array([10.0, 20.0])
        assert mae(a, a) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Recursive forecast
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecursiveForecast:

    def test_forecast_length(self, synth_country_df, feature_cols):
        from src.dataset import DemandDataset, clean_arr
        from src.model import MLPForecaster
        from src.train import train_model
        from src.forecast import recursive_forecast
        from src.dataset import make_loaders

        device = torch.device("cpu")
        tr, val, sX, sy = make_loaders(
            synth_country_df, feature_cols, TARGET, SEQ_LEN,
            TRAIN_END, VAL_END, BATCH_SIZE,
        )
        model = MLPForecaster(seq_len=SEQ_LEN, n_features=len(feature_cols))
        train_model(model, tr, val, epochs=3, patience=3, lr=1e-3, device=device)

        fc = recursive_forecast(
            model, synth_country_df, feature_cols, TARGET, SEQ_LEN,
            sX, sy, FORECAST_YEARS, device,
        )
        assert len(fc) == len(FORECAST_YEARS)
        assert "Year"     in fc.columns
        assert "Forecast" in fc.columns

    def test_forecast_values_positive(self, synth_country_df, feature_cols):
        from src.dataset import make_loaders
        from src.model import MLPForecaster
        from src.train import train_model
        from src.forecast import recursive_forecast

        device = torch.device("cpu")
        tr, val, sX, sy = make_loaders(
            synth_country_df, feature_cols, TARGET, SEQ_LEN,
            TRAIN_END, VAL_END, BATCH_SIZE,
        )
        model = MLPForecaster(seq_len=SEQ_LEN, n_features=len(feature_cols))
        train_model(model, tr, val, epochs=3, patience=3, lr=1e-3, device=device)

        fc = recursive_forecast(
            model, synth_country_df, feature_cols, TARGET, SEQ_LEN,
            sX, sy, FORECAST_YEARS, device,
        )
        assert (fc["Forecast"] > 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepLearningCLI:

    @pytest.fixture(autouse=True)
    def check_prereq(self):
        if not os.path.exists("outputs/deeplearning/dl_benchmarking.csv"):
            pytest.skip("Run make run-deeplearning first")

    def test_benchmarking_exists(self):
        assert os.path.exists("outputs/deeplearning/dl_benchmarking.csv")

    def test_best_models_exists(self):
        assert os.path.exists("outputs/deeplearning/dl_best_models.csv")

    def test_forecast_exists(self):
        assert os.path.exists("outputs/deeplearning/dl_forecast_2025_2030.csv")

    def test_benchmarking_schema(self):
        df = pd.read_csv("outputs/deeplearning/dl_benchmarking.csv")
        for col in ["Country", "Model", "MAE", "RMSE", "MAPE"]:
            assert col in df.columns

    def test_forecast_schema(self):
        df = pd.read_csv("outputs/deeplearning/dl_forecast_2025_2030.csv")
        for col in ["Country", "Model", "Year", "Forecast"]:
            assert col in df.columns

    def test_forecast_years_correct(self):
        df = pd.read_csv("outputs/deeplearning/dl_forecast_2025_2030.csv")
        assert set(df["Year"].unique()) == set(FORECAST_YEARS)

    def test_mape_reasonable(self):
        df = pd.read_csv("outputs/deeplearning/dl_benchmarking.csv")
        assert df["MAPE"].max() < 100
        assert df["MAPE"].min() >= 0
