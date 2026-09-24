# 00 — Overview: the mental model

> These learning notes are written for *you* — the person building this — so
> that six months from now, in an interview, you can explain not just *what*
> the code does but *why* each decision was made. Read them in order.

## The problem we're solving

A model that scored 92% accuracy in training does **not** stay at 92% forever.
The world changes: customer behavior shifts, an upstream data source changes
units, a new fraud pattern appears. The model's inputs start to look different
from what it was trained on — this is **data drift** — and its predictions
quietly get worse. Nobody notices until a business metric tanks.

"MLOps" is the discipline of operating models so this *doesn't* happen silently.
The core loop is:

```
train → deploy → monitor → detect problem → decide → remediate → (retrain) → …
```

Most tutorials only build the first two boxes. This project builds the whole
loop, and adds a twist: the "decide" step is done by an **LLM agent** that
reads the monitoring context and recommends an action in plain language.

## The four hard questions (and where we answer them)

1. **"Is the model still healthy?"**
   → Monitoring. We compare the live input distribution to the training
   distribution using two statistical tests: **PSI** and the **KS test**
   (`monitoring/drift.py`). We also track prediction confidence and, when
   ground-truth labels arrive, accuracy (`monitoring/performance.py`).

2. **"If not, why, and how bad?"**
   → Diagnostics. Raw numbers ("feature `amount` PSI = 0.34") are not an
   answer a human can act on at 3am. The LLM agent turns the numbers into a
   root-cause narrative and a severity (`diagnostics/agent.py`).

3. **"What should we do about it?"**
   → The agent picks one of `RETRAIN / ROLLBACK / INVESTIGATE / IGNORE`. This
   is a *policy* decision, and encoding it as a prompt makes the policy
   auditable and easy to change.

4. **"Do it, safely."**
   → Remediation. RETRAIN re-runs the training pipeline (which has a quality
   gate so a *worse* model can never be promoted). ROLLBACK flips the registry
   back to the previous version. Both are reversible and logged.

## The pieces and how they connect

| Layer | Module | Responsibility |
|---|---|---|
| Data | `mlops.data` | Make a realistic dataset; simulate drift on demand |
| Training | `mlops.training` | Fit model, evaluate, gate, register in MLflow |
| Serving | `mlops.serving` | `/predict` API, log inputs, expose Prometheus metrics |
| Monitoring | `mlops.monitoring` | PSI/KS drift + accuracy/confidence tracking |
| Diagnostics | `mlops.diagnostics` | LLM root-cause + action decision |
| Orchestration | `mlops.orchestration` | Airflow DAG + the self-healing control loop |

The connective tissue is deliberately simple and file-based so you can *see*
everything:
- Trained models + a snapshot of the **reference (training) distribution** are
  saved as artifacts and registered in MLflow.
- The serving layer appends every prediction to `logs/predictions.jsonl`.
- Monitoring reads that file to build the "production window" it compares
  against the reference snapshot.

In a real cloud deployment you'd swap files for S3 + Kafka + a Postgres
metadata store, but the *logic* is identical — and keeping it file-based means
the whole thing runs on your laptop with no infrastructure.

## How to read the rest of these notes

Each phase has its own note (`01-…`, `02-…`). They build on each other. If you
only read one thing before an interview, read this file plus
`04-drift-detection.md` (the statistics) and `05-llm-diagnostic-agent.md` (the
unique angle).
