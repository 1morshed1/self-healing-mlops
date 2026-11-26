"""Prometheus metrics for the serving layer.

WHY PROMETHEUS
--------------
"Is the service healthy right now?" is answered by operational metrics, not by
the ML monitoring we built earlier. Prometheus is the de-facto standard: your
app exposes a plain-text ``/metrics`` endpoint, Prometheus *scrapes* it every
few seconds, stores the time series, and Grafana graphs it. This is the
observability half of the system (the drift detector is the ML half).

THE FOUR METRIC TYPES (worth knowing cold)
------------------------------------------
* **Counter** — monotonically increases (total predictions, total errors). You
  graph its *rate* (e.g. requests/sec) in Prometheus with ``rate()``.
* **Gauge** — a value that goes up and down (current model version, in-flight
  requests).
* **Histogram** — samples observations into buckets (latency, confidence). Lets
  you compute quantiles like p99 latency across instances.
* **Summary** — like a histogram but computes quantiles client-side (we don't
  use it here; histograms are more flexible for aggregation).

Choosing the right type per signal matters,
so each metric below documents *why* it's the type it is.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Total predictions served, labelled by the model version and the predicted
# class. Counter: it only ever grows; rate() gives throughput. Labels let us
# slice "fraud vs legit predictions per model version" in Grafana.
PREDICTIONS_TOTAL = Counter(
    "mlops_predictions_total",
    "Total number of predictions served.",
    ["model_version", "predicted_class"],
)

# Errors (bad input, model failure). Counter for the same reason; a rising
# error rate is a classic alerting signal.
PREDICTION_ERRORS_TOTAL = Counter(
    "mlops_prediction_errors_total",
    "Total number of failed prediction requests.",
    ["reason"],
)

# End-to-end request latency. Histogram so Prometheus can compute p50/p95/p99
# across many pods. Buckets are chosen around a fast tabular model (sub-100ms).
PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds",
    "Prediction request latency in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Distribution of predicted fraud probability. Histogram so a *shift* in the
# confidence distribution (degradation signal) is visible on a dashboard even
# before labels arrive — this ties the ops metrics back to ML health.
PREDICTION_CONFIDENCE = Histogram(
    "mlops_prediction_fraud_probability",
    "Distribution of predicted fraud probabilities.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Which model version is currently serving. Gauge because it changes on deploy
# and can go up (promote) or notionally down (rollback). Surfacing this makes
# "when did the model change?" answerable on the same timeline as its metrics.
CURRENT_MODEL_VERSION = Gauge(
    "mlops_current_model_version",
    "The model registry version currently loaded by this serving instance.",
)


def render_metrics() -> tuple[bytes, str]:
    """Return the Prometheus exposition payload + its content type.

    The ``/metrics`` route calls this. ``generate_latest()`` serializes every
    registered metric into the text format Prometheus expects.
    """
    return generate_latest(), CONTENT_TYPE_LATEST
