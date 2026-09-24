# 08 — The observability stack (Docker Compose, Prometheus, Grafana)

This phase packages everything into a one-command stack and adds the dashboards.
It's the "I can operate this, not just run a script" evidence.

## The stack, in one `docker compose up`

```
mlflow ──┬─▶ trainer (one-shot: trains + registers the model, then exits)
         └─▶ serving (FastAPI /predict + /metrics)
                        │
                 prometheus (scrapes serving:/metrics every 5s)
                        │
                    grafana (dashboards over prometheus)
```

Five services, wired by `depends_on` **conditions** — the detail that makes it
actually work:

- `serving` waits for `mlflow` to be **healthy** (a real healthcheck hits
  `/health`) and for `trainer` to have **completed successfully**. So serving
  never starts before a model exists to serve. This ordering is a common source
  of flaky demos ("connection refused" on first boot); the conditions fix it.
- `trainer` has `restart: "no"` — it's a job, not a long-running service. It
  runs once, registers the model, and exits 0.

Persistence: `mlflow-data` (registry + artifacts) and `prediction-logs` are named
volumes, so a `docker compose restart` doesn't lose your trained model or the
prediction history.

## Prometheus: pull, don't push

Prometheus **scrapes** — it periodically GETs each target's `/metrics` and stores
the samples. Our `prometheus.yml` just declares one job pointing at
`serving:8000` (Docker's internal DNS resolves the compose service name to the
container IP). Nothing pushes; the app is passive. That pull model is why adding
a second serving replica needs zero app changes — you just add another target.

Useful PromQL the dashboard uses (worth being able to read):

```promql
# throughput: per-second request rate over the last minute
sum(rate(mlops_predictions_total[1m]))

# p99 latency from the histogram buckets
histogram_quantile(0.99, sum(rate(mlops_prediction_latency_seconds_bucket[5m])) by (le))

# predicted-fraud fraction
sum(rate(mlops_predictions_total{predicted_class="1"}[5m]))
  / sum(rate(mlops_predictions_total[5m]))
```

`rate()` on a counter gives per-second change; `histogram_quantile()` turns
bucket counters into a quantile. These two functions cover most of what you'll
ever write against app metrics.

## Grafana: dashboards as code

The datasource **and** the dashboard are **provisioned from files** in
`monitoring/grafana/`, not clicked together in the UI. Why this matters:

- The dashboard lives in git (`dashboards/mlops_overview.json`) and is reviewed
  and versioned like code. A dashboard clicked together by hand dies with the
  container.
- Grafana comes up with Prometheus already registered (fixed `uid: prometheus`)
  and the dashboard already loaded — zero manual setup.

The `mlops_overview.json` dashboard has: current model version, total
predictions, predicted-fraud rate, error count (stat tiles); throughput, fraud
fraction, latency quantiles, the **predicted-probability distribution** (the
degradation signal from note 06), and error rate by reason (time series).

## What's a demo shortcut vs. production

Called out honestly, because interviewers ask:

- **Single-stage Dockerfile, root user.** Fine for a demo; production wants a
  multi-stage build (build wheels, copy into a slim runtime) and a non-root
  `USER`. The Dockerfile comments say so.
- **SQLite-backed MLflow.** Zero-setup and registry-capable, but single-writer.
  Production swaps in Postgres + S3 artifact storage — same code, different
  `MLFLOW_TRACKING_URI`.
- **Grafana admin/admin.** Obviously a demo credential.
- **The drift heatmap / LLM-diagnosis table** from the project plan live on the
  ML-monitoring side, not in Prometheus. To surface them in Grafana you'd export
  per-feature PSI as gauges (`mlops_feature_psi{feature=...}`) from the
  monitoring job and add a heatmap panel — a natural next extension, noted here
  so the gap is intentional, not forgotten.

## Try it

```bash
docker compose up --build
# API docs:    http://localhost:8000/docs
# MLflow:      http://localhost:5000
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000  (admin/admin)
```

Then generate traffic (`python -m scripts.simulate_traffic --n 500`) and watch
the Grafana panels move.
