"""Tests for the FastAPI serving app.

Uses FastAPI's TestClient (in-process, no network). The ``trained_model`` fixture
guarantees a production model exists for the app to load.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlops.data.schema import FEATURE_NAMES


@pytest.fixture
def client(trained_model):
    # Import inside the fixture so the model is registered (fixture ran) before
    # the app's startup tries to load it.
    from mlops.serving.app import app

    with TestClient(app) as c:
        yield c


def _valid_features():
    return {
        "amount": 120.5, "age_days": 30, "n_txn_24h": 3,
        "avg_txn_amt": 88.0, "country_risk": 0.1, "device_change": 0,
    }


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] is not None


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json={"features": _valid_features()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_fraud"] in (0, 1)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["model_version"] is not None


def test_predict_high_risk_scores_higher_than_low_risk(client):
    low = client.post("/predict", json={"features": _valid_features()}).json()
    high_features = {
        "amount": 900.0, "age_days": 2, "n_txn_24h": 9,
        "avg_txn_amt": 60.0, "country_risk": 0.85, "device_change": 1,
    }
    high = client.post("/predict", json={"features": high_features}).json()
    # The model should assign the obviously-risky transaction a higher probability.
    assert high["fraud_probability"] > low["fraud_probability"]


def test_predict_rejects_missing_feature(client):
    bad = {k: v for k, v in _valid_features().items() if k != FEATURE_NAMES[0]}
    resp = client.post("/predict", json={"features": bad})
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_prometheus(client):
    # Make one prediction so counters are non-empty.
    client.post("/predict", json={"features": _valid_features()})
    text = client.get("/metrics").text
    assert "mlops_predictions_total" in text
    assert "mlops_prediction_latency_seconds" in text
