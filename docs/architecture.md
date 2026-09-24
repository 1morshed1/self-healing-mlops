# Architecture

This document is the design deep-dive. For the *why* behind each decision, see
the numbered notes in `docs/learning/`; this file is the map.

## System diagram

```
                          ┌──────────────────────────────────────────┐
                          │           MLflow (registry + tracking)     │
                          │  versions, aliases, metrics, reference     │
                          │  snapshot per model version                │
                          └──────▲───────────────────────┬────────────┘
             train + register    │                       │ load @production
                                 │                       ▼
   ┌──────────────┐   gate    ┌──┴───────────┐     ┌──────────────────┐
   │  Training    │──────────▶│  register +  │     │  Serving API      │
   │  pipeline    │  pass?    │  promote     │     │  POST /predict    │
   │ (RF + eval)  │           └──────────────┘     │  GET  /metrics    │
   └──────▲───────┘                                └────────┬─────────┘
          │ RETRAIN                                         │ append
          │                                                 ▼
   ┌──────┴────────┐   action   ┌─────────────┐   window   ┌──────────────┐
   │  Remediator   │◀───────────│ Diagnostic  │◀───────────│ predictions  │
   │ retrain/rollbk│            │ Agent (LLM  │            │  .jsonl      │
   └───────────────┘            │ + fallback) │            └──────┬───────┘
                                └──────▲──────┘                   │
                                       │ AlertContext             ▼
                                ┌──────┴──────────────────────────────────┐
                                │ Monitoring: DriftDetector (PSI/KS) +     │
                                │ PerformanceMonitor (confidence/accuracy) │
                                └──────────────────────────────────────────┘

   Observability (parallel):  Prometheus scrapes Serving /metrics → Grafana
```

## Modules

| Module | Responsibility | Key types |
|---|---|---|
| `mlops.config` | one source of truth for all settings | `Settings`, `get_settings()` |
| `mlops.data.schema` | the shared feature contract | `FEATURE_NAMES` |
| `mlops.data.generate` | synthetic data + drift simulation | `generate_dataset`, `DriftSpec` |
| `mlops.data.features` | validated feature assembly | `build_feature_frame` |
| `mlops.training.evaluate` | metrics + quality gate | `evaluate`, `EvaluationResult` |
| `mlops.training.registry` | MLflow registry wrapper | `promote`, `rollback`, `load_model` |
| `mlops.training.train` | end-to-end training job | `train_model`, `TrainingOutcome` |
| `mlops.monitoring.drift` | PSI + KS drift detection | `DriftDetector`, `DriftReport` |
| `mlops.monitoring.performance` | confidence + accuracy tracking | `PerformanceMonitor` |
| `mlops.diagnostics.agent` | LLM root-cause + action (with fallback) | `DiagnosticAgent`, `Diagnosis` |
| `mlops.diagnostics.remediation` | safely execute the action | `Remediator` |
| `mlops.serving.app` | FastAPI prediction API | `app` |
| `mlops.serving.metrics` | Prometheus instrumentation | metric objects |
| `mlops.orchestration.self_healing` | the control loop | `SelfHealingMonitor` |
| `mlops.orchestration.dags` | Airflow training + monitoring DAGs | — |

## Data flow (the closed loop)

1. **Train** → `train_model` fits a RandomForest, evaluates it, and — if it
   clears the quality gate and isn't worse than the incumbent — registers it and
   points the `production` alias at it. It also logs a **reference sample** (the
   training feature distribution) as an artifact tied to that version.
2. **Serve** → the API loads `models:/…@production`, scores requests, and appends
   each `{features, prediction, version}` to `predictions.jsonl`.
3. **Monitor** → `SelfHealingMonitor` builds a production window (from the log or
   simulated), runs `DriftDetector` (input drift) and `PerformanceMonitor`
   (output health) against the model's reference sample.
4. **Diagnose** → if anything is off, the `DiagnosticAgent` produces a
   `Diagnosis` (root cause + severity + action) via Claude, or a rule-based
   fallback.
5. **Remediate** → the `Remediator` executes the action. `RETRAIN` re-enters
   step 1 (gate-protected); `ROLLBACK` moves the alias back a version.
6. **Notify** → every cycle alerts a human (webhook or log).

## Key invariants (the things that keep it safe)

- **One feature schema** (`FEATURE_NAMES`) shared by training, serving, and
  monitoring → no train/serve skew.
- **The quality gate guards every promotion**, manual or automated → a worse
  model can never reach production, even via auto-retrain.
- **The agent decides; the remediator acts** → all side effects in one auditable
  place, with an advisory/auto switch.
- **The agent never hard-fails** → rule-based fallback keeps the loop working
  offline; LLM output is validated/clamped to a safe vocabulary.
- **Reference distribution is versioned with the model** → the drift baseline
  updates automatically on every retrain.

## Local vs. production mapping

| Concern | This repo (laptop) | Production equivalent |
|---|---|---|
| Prediction log | `predictions.jsonl` file | Kafka topic |
| Tracking/registry store | SQLite | Postgres + S3 artifacts |
| Orchestration | run the script / Airflow locally | Airflow on K8s / Astronomer |
| Serving | uvicorn container | K8s Deployment + HPA, canary rollout |
| Secrets | `.env` | a secrets manager |

The *code* is the same across both columns — only configuration changes.
