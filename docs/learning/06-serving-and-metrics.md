# 06 — Serving (FastAPI) & observability (Prometheus)

The serving layer is where the model meets the real world. Three production
behaviours matter more than the FastAPI boilerplate.

## 1. Load by alias, reload without redeploy

The app loads `models:/self_healing_classifier@production` — the **alias**, not
a pinned version — and caches it in a `ModelHolder`. When the self-healing loop
promotes a new version (or rolls back), it just calls `POST /reload` and the
holder re-reads the alias. **No code change, no image rebuild, no restart.**
That's the whole point of the registry-alias indirection from note 03, now paying
off at serving time.

We also load at **startup** (via FastAPI's `lifespan`) so the first real request
isn't slowed by a cold model load — but we *don't crash* if no model exists yet.
The app boots and reports `"no_model"` on `/health`. In Kubernetes a crash-on-boot
turns into a CrashLoopBackOff; a clean unhealthy status is far easier to operate.

## 2. Log every prediction — this is the monitoring feedstock

Every `/predict` appends one JSON line to `logs/predictions.jsonl` containing the
raw feature values, the prediction, and the model version. The monitoring layer
reads exactly this file to reconstruct the **production distribution** for drift
detection. The loop is now closed:

```
/predict  →  predictions.jsonl  →  DriftDetector  →  DiagnosticAgent  →  retrain
```

JSON-lines (one object per line) is the right format here: appendable without
rewriting the file, streamable, and trivially parseable. In a cloud deployment
you'd publish these records to **Kafka** instead of a file — but the *record
shape* is identical, so the monitoring code wouldn't change.

## 3. Prometheus metrics — pick the right instrument

`metrics.py` exposes four signals, and the *type* of each is a deliberate choice:

| Metric | Type | Why this type |
|---|---|---|
| `predictions_total{version,class}` | Counter | only grows; graph `rate()` for throughput; labels slice fraud-vs-legit per version |
| `prediction_errors_total{reason}` | Counter | rising error rate is a classic alert |
| `prediction_latency_seconds` | Histogram | lets Prometheus compute p95/p99 across pods |
| `prediction_fraud_probability` | Histogram | a *shift* in this distribution is a degradation signal visible before labels |
| `current_model_version` | Gauge | goes up on promote, down on rollback; puts "when did the model change?" on the same timeline as its metrics |

Counter vs Gauge vs Histogram is easy to get wrong. The one-line rule:
**Counter** = cumulative event count (rate it), **Gauge** = current value (up or
down), **Histogram** = distribution of observations (quantiles). Prometheus
*scrapes* the `/metrics` endpoint on a schedule — the app never pushes.

Notice that `prediction_fraud_probability` deliberately bridges ops and ML: it's
an operational metric (scraped by Prometheus) that *also* carries an ML-health
signal (a confidence shift toward 0.5). That overlap is what lets a single
Grafana dashboard tell both the "is it up?" and "is it still smart?" stories.

## Input validation at the edge

`/predict` funnels the request through `build_feature_frame()` — the same
schema-enforcing function training uses. A malformed request (missing feature,
non-numeric value) returns a clear **422** with a message naming the problem,
instead of a confusing 500 from deep inside numpy. Validating at the boundary is
both a security and a debuggability practice.

## What we verified

```
health   → {status: ok, model_version: 1}
low-risk → is_fraud 0, prob 0.19
high-risk→ is_fraud 1, prob 0.85     ← the model genuinely discriminates
bad input→ 422 "missing required feature(s): ['age_days', ...]"
/metrics → exposes predictions_total, latency, confidence, model version
```

Run it yourself: `uvicorn mlops.serving.app:app --reload`, then POST to
`/predict`. Open `http://localhost:8000/docs` for the auto-generated Swagger UI —
a free benefit of describing the contract with Pydantic models.
