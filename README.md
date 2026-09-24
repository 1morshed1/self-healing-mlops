# Self-Healing MLOps Pipeline

[![CI](https://github.com/1morshed1/self-healing-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/1morshed1/self-healing-mlops/actions/workflows/ci.yml)

An end-to-end machine-learning system that **trains** a model, **serves** it
behind a FastAPI endpoint, **monitors** it in production for data drift and
accuracy degradation, and **heals itself** — an LLM diagnostic agent reads the
alert context, decides what to do (retrain / rollback / investigate), and
triggers the fix.

> This is a *portfolio / learning* project. It runs fully on a laptop with no
> cloud account, yet it mirrors how real production ML is operated. Every
> module is heavily commented, and `docs/learning/` contains a numbered set of
> notes explaining the concepts and the design decisions behind each phase.

---

## The idea in one picture

```
        ┌─────────────┐   train + register   ┌──────────────────┐
        │  Training   │ ───────────────────▶ │  MLflow Registry │
        │  pipeline   │   (quality gate)     │  (model + stats) │
        └─────────────┘                      └────────┬─────────┘
                                                       │ load "Production" model
                                                       ▼
   requests ─▶ ┌──────────────────┐  log every prediction  ┌──────────────┐
               │  FastAPI /predict │ ─────────────────────▶ │ predictions  │
               │  + /metrics       │                        │  .jsonl      │
               └──────────────────┘                         └──────┬───────┘
                                                                    │ production window
                                                                    ▼
                              ┌───────────────────────────────────────────┐
                              │  Monitoring:  DriftDetector (PSI + KS)     │
                              │               PerformanceMonitor           │
                              └───────────────────┬───────────────────────┘
                                                  │ drift / degradation alert
                                                  ▼
                              ┌───────────────────────────────────────────┐
                              │  LLM Diagnostic Agent                      │
                              │  → root cause + severity + ACTION          │
                              │  → RETRAIN / ROLLBACK / INVESTIGATE / IGNORE│
                              └───────────────────┬───────────────────────┘
                                                  │ auto-remediate
                                                  ▼
                                     back to the Training pipeline
```

The **self-healing loop** (`scripts/self_heal.py`) wires these together:
generate a drifted production window → detect drift → ask the agent → act.

---

## Why this project exists (the CV angle)

Most ML portfolios stop at "I trained a model in a notebook." This project
proves the *engineering* concerns that senior ML roles actually pay for:

| Skill demonstrated | Where to look |
|---|---|
| Statistical drift detection (PSI, KS test) | `src/mlops/monitoring/drift.py` |
| Model registry + quality gates | `src/mlops/training/` + MLflow |
| Production serving + observability | `src/mlops/serving/` (FastAPI + Prometheus) |
| Automated retraining / rollback | `src/mlops/orchestration/` + `scripts/self_heal.py` |
| Airflow orchestration | `src/mlops/orchestration/dags/training_dag.py` |
| Dashboards (Prometheus + Grafana) | `monitoring/` + `docker-compose.yml` |
| **LLM-powered diagnostics** (the unique angle) | `src/mlops/diagnostics/agent.py` |

---

## Quickstart

```bash
# 1. Install (editable + dev tools). A virtualenv is recommended.
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Train a model — writes to the MLflow registry + artifacts/
python -m scripts.run_pipeline

# 3. Serve it
uvicorn mlops.serving.app:app --reload
# then in another shell:
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"features": {"amount": 120.5, "age_days": 30, "n_txn_24h": 3, "avg_txn_amt": 88.0, "country_risk": 0.1, "device_change": 0}}'

# 4. Watch it heal — simulate drift, detect it, and let the agent decide
python -m scripts.self_heal

# 5. Run the tests
pytest
```

No `ANTHROPIC_API_KEY`? The diagnostic agent automatically falls back to a
deterministic rule-based diagnosis, so every step above still works offline.

The full observability stack (MLflow server + Prometheus + Grafana) comes up
with:

```bash
docker compose up
```

See **[docs/runbook.md](docs/runbook.md)** for operating instructions and
**[docs/architecture.md](docs/architecture.md)** for the design deep-dive.

---

## Learning notes

The `docs/learning/` folder is written *for the person building this* — it
explains not just what the code does but *why each decision was made*:

- `00-overview.md` — the mental model and how the pieces fit
- `01-config-and-scaffolding.md` — 12-factor config, `src/` layout, editable installs
- `02-data-and-drift-simulation.md` — faking realistic drift; covariate vs concept drift
- `03-training-and-mlflow.md` — the quality gate, MLflow registry, aliases
- `04-drift-detection.md` — **PSI vs KS test**, the statistics (read this before interviews)
- `05-llm-diagnostic-agent.md` — the unique angle: decide/act split, prompt-as-policy, fallback
- `06-serving-and-metrics.md` — FastAPI serving patterns + Prometheus metric types
- `07-orchestration-and-self-healing.md` — the closed loop + Airflow DAGs
- `08-observability-stack.md` — Docker Compose, Prometheus, Grafana-as-code
- `09-testing-and-quality.md` — test isolation, testing an LLM offline

Also see **[docs/architecture.md](docs/architecture.md)** (design map) and
**[docs/runbook.md](docs/runbook.md)** (how to operate it).

---

## Repository layout

```
src/mlops/
  config.py            # single source of truth for all settings
  data/                # dataset generation + feature engineering
  training/            # train, evaluate (quality gate), MLflow registry
  monitoring/          # drift detection (PSI/KS) + performance monitor
  diagnostics/         # LLM agent + prompts + auto-remediation
  serving/             # FastAPI app + Prometheus metrics
  orchestration/       # Airflow DAG + self-healing control loop
scripts/               # runnable entrypoints (train, serve-helpers, self-heal)
monitoring/            # prometheus + grafana provisioning
tests/                 # pytest suite
docs/                  # architecture, runbook, and learning notes
```

## License

MIT — see `LICENSE`.
