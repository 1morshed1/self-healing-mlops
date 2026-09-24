"""Serving layer: a FastAPI app that loads the production model from the MLflow
registry, answers ``POST /predict``, logs every prediction for monitoring, and
exposes Prometheus metrics at ``/metrics``.

Run it with:
    uvicorn mlops.serving.app:app --reload
"""
