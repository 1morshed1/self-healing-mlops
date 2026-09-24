# Learning notes

These notes are the teaching companion to the code. They're written to be read
**in order**, and each one explains the *decisions* behind a phase of the build —
the reasoning you'd want at hand in an interview, not just what the code does.

| # | Note | What you'll learn |
|---|---|---|
| 00 | [Overview](00-overview.md) | the mental model: train → serve → monitor → diagnose → heal |
| 01 | [Config & scaffolding](01-config-and-scaffolding.md) | 12-factor config, `src/` layout, editable installs |
| 02 | [Data & drift simulation](02-data-and-drift-simulation.md) | generating learnable data; covariate vs concept drift |
| 03 | [Training & MLflow](03-training-and-mlflow.md) | the quality gate; registry versions & aliases |
| 04 | [Drift detection](04-drift-detection.md) | **PSI vs KS test** — the statistics, precisely |
| 05 | [LLM diagnostic agent](05-llm-diagnostic-agent.md) | decide/act split, prompt-as-policy, safe fallback |
| 06 | [Serving & metrics](06-serving-and-metrics.md) | FastAPI patterns; Counter/Gauge/Histogram |
| 07 | [Orchestration & self-healing](07-orchestration-and-self-healing.md) | the closed loop; Airflow TaskFlow DAGs |
| 08 | [Observability stack](08-observability-stack.md) | Compose, Prometheus (pull), Grafana-as-code |
| 09 | [Testing & quality](09-testing-and-quality.md) | test isolation; testing an LLM offline |

If you only read three before an interview: **00** (the story), **04** (the
statistics), and **05** (the unique angle).
