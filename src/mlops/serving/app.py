"""FastAPI serving application.

Endpoints:
* ``GET  /health``   - liveness/readiness (is a model loaded?)
* ``POST /predict``  - score one or more transactions
* ``POST /reload``   - hot-reload the production model (after a promote/rollback)
* ``GET  /metrics``  - Prometheus exposition
* ``GET  /``         - a tiny human-readable index

KEY PRODUCTION BEHAVIOURS DEMONSTRATED HERE
-------------------------------------------
1. **Load the model by ALIAS, not version.** We load
   ``models:/<name>@production`` once and cache it. A deploy (moving the alias)
   plus ``POST /reload`` swaps the model with zero code change — no redeploy.
2. **Log every prediction.** Each request's features + output are appended to a
   JSON-lines file. This is the raw material the monitoring layer turns into a
   "production window" for drift detection. In the cloud you'd publish to Kafka;
   the shape of the data is identical.
3. **Expose metrics.** Latency, throughput, error rate, and the confidence
   distribution are recorded on every request for Prometheus/Grafana.
4. **Validate input at the edge.** Pydantic + the shared feature schema reject
   malformed requests with a clear 4xx instead of a confusing 5xx deep in numpy.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from mlops.config import get_settings
from mlops.data.features import FeatureValidationError, build_feature_frame, frame_to_matrix
from mlops.data.schema import FEATURE_NAMES
from mlops.serving import metrics
from mlops.training import registry


# ---------------------------------------------------------------------------
# Request / response schemas (Pydantic) — the API contract.
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """One transaction to score. ``features`` maps feature name -> value."""

    features: dict[str, float] = Field(
        ...,
        examples=[
            {
                "amount": 120.5,
                "age_days": 30,
                "n_txn_24h": 3,
                "avg_txn_amt": 88.0,
                "country_risk": 0.1,
                "device_change": 0,
            }
        ],
    )


class PredictResponse(BaseModel):
    is_fraud: int                # 0/1 hard prediction at the 0.5 threshold
    fraud_probability: float     # calibrated-ish probability of the positive class
    model_version: int | None    # which registry version produced this
    latency_ms: float


# ---------------------------------------------------------------------------
# A tiny model holder that loads lazily and can hot-reload.
# ---------------------------------------------------------------------------
class ModelHolder:
    """Caches the production model + its version; reload() re-reads the alias."""

    def __init__(self):
        self.model = None
        self.version: int | None = None

    def load(self) -> None:
        settings = get_settings()
        registry.configure_mlflow()
        self.version = registry.get_alias_version(settings.model_name, registry.PRODUCTION_ALIAS)
        if self.version is None:
            raise RuntimeError(
                "No production model is registered. Run `python -m scripts.run_pipeline` first."
            )
        self.model = registry.load_model(settings.model_name)
        metrics.CURRENT_MODEL_VERSION.set(self.version)

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.load()


_holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Try to load the model at startup, but don't crash if none exists yet.

    Loading at startup means the first real request isn't penalised by a cold
    load. We swallow the "no model" error so the app can still boot and report
    an unhealthy status via /health (nicer than a crash loop in Kubernetes).
    """
    try:
        _holder.load()
    except Exception as exc:  # noqa: BLE001
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="Self-Healing MLOps — Fraud Model Serving",
    version="0.1.0",
    lifespan=lifespan,
)


def _log_prediction(features: dict, is_fraud: int, proba: float, version: int | None) -> None:
    """Append one prediction to the JSON-lines log the monitor reads.

    We write the raw feature values so the drift detector can reconstruct the
    production distribution exactly. One JSON object per line = trivially
    streamable and appendable without locking the whole file.
    """
    settings = get_settings()
    settings.ensure_dirs()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "is_fraud": is_fraud,
        "fraud_probability": proba,
        **{name: features[name] for name in FEATURE_NAMES},
    }
    with settings.prediction_log.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


@app.get("/")
def index() -> dict:
    """Human-friendly index describing the API."""
    return {
        "service": "self-healing-mlops fraud serving",
        "model_loaded": _holder.model is not None,
        "model_version": _holder.version,
        "endpoints": ["/health", "/predict (POST)", "/reload (POST)", "/metrics"],
    }


@app.get("/health")
def health() -> dict:
    """Readiness probe. 200 with model_loaded=false still means 'alive'."""
    return {
        "status": "ok" if _holder.model is not None else "no_model",
        "model_version": _holder.version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score a single transaction and record metrics + a prediction-log entry."""
    start = time.perf_counter()
    try:
        _holder.ensure_loaded()
    except Exception as exc:  # noqa: BLE001
        metrics.PREDICTION_ERRORS_TOTAL.labels(reason="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Validate + assemble features through the shared, schema-enforcing path.
    try:
        frame = build_feature_frame(request.features)
    except FeatureValidationError as exc:
        metrics.PREDICTION_ERRORS_TOTAL.labels(reason="bad_input").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    matrix = frame_to_matrix(frame)
    proba = float(_holder.model.predict_proba(matrix)[0, 1])
    is_fraud = int(proba >= 0.5)

    # --- record everything ---
    latency = time.perf_counter() - start
    metrics.PREDICTION_LATENCY.observe(latency)
    metrics.PREDICTION_CONFIDENCE.observe(proba)
    metrics.PREDICTIONS_TOTAL.labels(
        model_version=str(_holder.version), predicted_class=str(is_fraud)
    ).inc()
    _log_prediction(request.features, is_fraud, proba, _holder.version)

    return PredictResponse(
        is_fraud=is_fraud,
        fraud_probability=proba,
        model_version=_holder.version,
        latency_ms=round(latency * 1000, 3),
    )


@app.post("/reload")
def reload_model() -> dict:
    """Hot-reload the production model after a promote or rollback.

    The self-healing loop calls this so a freshly-promoted model starts serving
    without restarting the process.
    """
    try:
        _holder.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "reloaded", "model_version": _holder.version}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Expose metrics in Prometheus text format for scraping."""
    payload, content_type = metrics.render_metrics()
    return Response(content=payload, media_type=content_type)
