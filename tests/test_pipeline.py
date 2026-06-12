"""
Smoke tests for the Fraud MLOps Pipeline.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure src is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    import yaml
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def features_df(config):
    path = ROOT_DIR / config["data"]["processed_path"]
    if not path.exists():
        pytest.skip(f"Processed features not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def model(config):
    import joblib
    model_path = ROOT_DIR / "models" / "xgb_fraud_model.pkl"
    if not model_path.exists():
        pytest.skip(f"Trained model not found: {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_loads(self, config):
        assert config is not None

    def test_required_keys(self, config):
        assert "data" in config
        assert "model" in config
        assert "mlflow" in config
        assert "drift" in config
        assert "serving" in config

    def test_model_hyperparams(self, config):
        m = config["model"]
        assert m["n_estimators"] > 0
        assert 1 <= m["max_depth"] <= 20
        assert 0 < m["learning_rate"] < 1


# ---------------------------------------------------------------------------
# Data ingestion tests
# ---------------------------------------------------------------------------

class TestDataIngestion:
    def test_raw_transaction_file_exists(self, config):
        path = ROOT_DIR / config["data"]["raw_transaction_path"]
        assert path.exists(), f"Raw transaction file missing: {path}"

    def test_raw_identity_file_exists(self, config):
        path = ROOT_DIR / config["data"]["raw_identity_path"]
        assert path.exists(), f"Raw identity file missing: {path}"

    def test_merged_parquet_exists(self, config):
        path = ROOT_DIR / config["data"]["merged_path"]
        assert path.exists(), f"Merged parquet missing: {path}"

    def test_merged_parquet_has_rows(self, config):
        path = ROOT_DIR / config["data"]["merged_path"]
        if not path.exists():
            pytest.skip("Merged parquet not yet created")
        df = pd.read_parquet(path)
        assert len(df) > 0, "Merged parquet is empty"


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_features_parquet_exists(self, config):
        path = ROOT_DIR / config["data"]["processed_path"]
        assert path.exists(), f"Features parquet missing: {path}"

    def test_target_column_present(self, features_df):
        assert "isFraud" in features_df.columns

    def test_no_inf_values(self, features_df):
        numeric = features_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values found in features"

    def test_engineered_columns_present(self, features_df):
        expected = ["hour", "day", "TransactionAmt_log", "card1_mean", "card1_std"]
        for col in expected:
            assert col in features_df.columns, f"Expected column '{col}' missing"

    def test_class_imbalance_exists(self, features_df):
        """Fraud rate should be low (realistic imbalanced dataset)."""
        fraud_rate = features_df["isFraud"].mean()
        assert 0.001 < fraud_rate < 0.5, f"Unexpected fraud rate: {fraud_rate}"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_model_loads(self, model):
        assert model is not None

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba"), "Model missing predict_proba method"

    def test_model_prediction_shape(self, model, features_df):
        X = features_df.drop(columns=["isFraud"]).iloc[:10]
        proba = model.predict_proba(X)
        assert proba.shape == (10, 2), f"Unexpected proba shape: {proba.shape}"

    def test_model_probabilities_in_range(self, model, features_df):
        X = features_df.drop(columns=["isFraud"]).iloc[:100]
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities out of [0, 1]"

    def test_model_predictions_are_binary(self, model, features_df):
        X = features_df.drop(columns=["isFraud"]).iloc[:100]
        preds = model.predict(X)
        unique = set(preds.tolist())
        assert unique.issubset({0, 1}), f"Non-binary predictions: {unique}"


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_optimal_threshold_file_exists_or_skipped(self):
        path = ROOT_DIR / "models" / "optimal_threshold.txt"
        if not path.exists():
            pytest.skip("optimal_threshold.txt not yet generated (run training first)")
        threshold = float(path.read_text().strip())
        assert 0.0 < threshold < 1.0, f"Invalid threshold: {threshold}"


# ---------------------------------------------------------------------------
# Drift module tests
# ---------------------------------------------------------------------------

class TestDriftModule:
    def test_psi_zero_for_identical_distributions(self):
        from src.monitoring.drift import calculate_psi
        data = np.random.normal(0, 1, 1000)
        psi = calculate_psi(data, data)
        assert psi < 0.05, f"PSI should be near 0 for identical data, got {psi}"

    def test_psi_high_for_shifted_distributions(self):
        from src.monitoring.drift import calculate_psi
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)  # large shift
        psi = calculate_psi(ref, cur)
        assert psi > 0.2, f"PSI should be high for shifted distributions, got {psi}"

    def test_psi_handles_empty_series(self):
        from src.monitoring.drift import calculate_psi
        result = calculate_psi([], [])
        assert np.isnan(result), "PSI of empty series should be NaN"


# ---------------------------------------------------------------------------
# FastAPI tests
# ---------------------------------------------------------------------------

class TestServingApp:
    @pytest.fixture(scope="class")
    def client(self):
        """Start a test client for the FastAPI app."""
        model_path = ROOT_DIR / "models" / "xgb_fraud_model.pkl"
        features_path = ROOT_DIR / "data" / "processed" / "features.parquet"
        if not model_path.exists() or not features_path.exists():
            pytest.skip("Model or features not available for API tests")

        from fastapi.testclient import TestClient
        from src.serving.app import app
        with TestClient(app) as c:
            yield c

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Fraud Detection API" in response.json()["message"]

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_threshold_endpoint(self, client):
        response = client.get("/threshold")
        assert response.status_code == 200
        data = response.json()
        assert "threshold" in data
        assert 0 < data["threshold"] < 1

    def test_predict_empty_input(self, client):
        """Empty input should succeed (missing features filled with defaults)."""
        response = client.post("/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "prediction" in data
        assert data["prediction"] in [0, 1]

    def test_predict_returns_label(self, client):
        response = client.post("/predict", json={})
        data = response.json()
        assert data["label"] in ["FRAUD", "LEGITIMATE"]


