# Runbook — operating the self-healing MLOps pipeline

Practical, copy-pasteable instructions. Two ways to run it: **local Python** (fast
iteration) or **Docker Compose** (the full stack with dashboards).

## Prerequisites

- Python 3.10+ (local path), or Docker + Docker Compose (stack path).
- Optional: an `ANTHROPIC_API_KEY` for real LLM diagnoses. Without it the agent
  uses its rule-based fallback and everything still works.

## Option A — local Python

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: make install

# 2. Train + register the initial model
python -m scripts.run_pipeline    # or: make train

# 3. Serve it
uvicorn mlops.serving.app:app --reload      # or: make serve
#   → http://localhost:8000/docs  (interactive API)

# 4. Send it a prediction
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "features": {"amount":120.5,"age_days":30,"n_txn_24h":3,
               "avg_txn_amt":88.0,"country_risk":0.1,"device_change":0}}'

# 5. Run a self-healing cycle (advisory — just shows what it would do)
python -m scripts.self_heal --simulate medium
#   Let it actually heal:
python -m scripts.self_heal --simulate high --auto

# 6. Tests
pytest                            # or: make test
```

### Closing the loop with real traffic

```bash
# with the server running, in another shell:
python -m scripts.simulate_traffic --n 500 --drift medium
python -m scripts.self_heal --from-log --auto
```

## Option B — Docker Compose (full stack)

```bash
docker compose up --build         # or: make compose-up
```

Services:

| URL | What |
|---|---|
| http://localhost:8000/docs | Serving API (Swagger UI) |
| http://localhost:5000 | MLflow UI (runs, metrics, registry) |
| http://localhost:9090 | Prometheus |
| http://localhost:3000 | Grafana (admin/admin) — "MLOps → Serving Overview" |

The `trainer` service trains and registers the model once, then `serving` starts.
Generate traffic and watch the Grafana panels move:

```bash
docker compose exec serving python -m scripts.simulate_traffic --n 500 --drift medium
```

Tear down (removing volumes): `docker compose down -v` (or `make compose-down`).

## Configuration

All settings have safe defaults; override via environment variables or a `.env`
file (`cp .env.example .env`). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | (unset) | enables real LLM diagnoses |
| `MLOPS_LLM_MODEL` | `claude-opus-5` | which Claude model the agent calls |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | tracking + registry store |
| `MLOPS_ACCURACY_GATE` | `0.80` | min accuracy to promote a model |
| `MLOPS_PSI_THRESHOLD` | `0.2` | PSI above this = drift |
| `MLOPS_KS_PVALUE_THRESHOLD` | `0.01` | KS p-value below this = drift |
| `MLOPS_ALERT_WEBHOOK_URL` | (unset) | Slack/Discord webhook for alerts |

## Common operations

**Retrain manually** → `python -m scripts.run_pipeline`. The new version is
registered and promoted only if it clears the gate and isn't worse than the
current production model.

**Roll back the production model** →
```python
from mlops.training import registry
registry.rollback(registry.get_settings().model_name)  # moves alias back one version
```
then `curl -X POST localhost:8000/reload` so the server picks it up.

**Hot-swap the serving model after a promote/rollback** → `POST /reload`. No
restart needed (the app loads by alias).

**Inspect the registry** → open the MLflow UI (Compose) or:
```python
from mlops.training import registry
name = registry.get_settings().model_name
print(registry.list_versions(name), registry.get_alias_version(name, "production"))
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` returns `no_model` | No model registered yet → run `scripts.run_pipeline`. |
| `503` from `/predict` | Same — no production model, or the tracking store is unreachable. |
| Agent always says `source=rule-based` | No `ANTHROPIC_API_KEY`, or the API call failed (check logs). Expected offline. |
| Retrain runs but "NOT promoted" | Working as intended — the gate found the candidate no better than production. |
| Grafana panels empty | No traffic yet → run `simulate_traffic`; Prometheus scrapes every 5s. |
| `mlflow.exceptions...registry` errors | You pointed `MLFLOW_TRACKING_URI` at a `file:` store; the registry needs a DB (sqlite/postgres). |

## Reproducibility note

For a fully pinned, reproducible environment, freeze the resolved versions:

```bash
pip freeze > requirements.lock
```

and install from the lockfile in CI/production. The committed `requirements.txt`
uses `>=` lower bounds to stay installable over time; the lockfile is what you'd
deploy.
