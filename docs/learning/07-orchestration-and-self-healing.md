# 07 — Orchestration & the self-healing loop

This phase connects every previous layer into one closed loop and puts it on a
schedule. Two artifacts: the reusable `SelfHealingMonitor` and the Airflow DAGs.

## The closed loop

```
/predict → predictions.jsonl → DriftDetector ┐
                                              ├→ AlertContext → DiagnosticAgent
model.predict_proba → PerformanceMonitor ─────┘                     │
                                                                    ▼
              Remediator (RETRAIN via quality gate / ROLLBACK) ◀────┘
                                        │
                                        ▼
                                  notify a human
```

`SelfHealingMonitor.run_cycle(window)` performs exactly these steps. There is
**one** implementation of the healing logic; the demo CLI and the Airflow DAG
both call it. Don't duplicate control logic across "the script" and "the DAG" —
that's how they drift out of sync.

## Two ways to build the production window

- **Real path** (`window_from_log`): read the last N lines of
  `logs/predictions.jsonl` — literally what the live service saw. This is what
  runs in production.
- **Demo path** (`window_from_simulation`): generate a labelled window with
  injected drift, so you can *summon* any scenario on demand without waiting for
  the world to change.

The CLI prefers the log (`--from-log`) and falls back to simulation, so the demo
works whether or not you've sent real traffic through the API.

## We watched the whole loop heal itself

```
--simulate high --auto
  DRIFT [high] in 5/6 features (country_risk PSI 9.3)
  Diagnosis: RETRAIN, severity critical (accuracy dropped 0.195)
  Remediation: retrained v3 → passed gate (0.877 ≥ 0.877) → PROMOTED
```

And in advisory mode (`--simulate medium`, no `--auto`), the exact same
diagnosis is produced but remediation is **SKIPPED** — you see what the system
*would* do before letting it act. That advisory/auto switch is how you roll out
automation responsibly: run advisory for weeks, confirm the diagnoses are sane,
*then* flip `--auto` (or `AUTO_REMEDIATE = True` in the DAG).

## Why Airflow, and the two DAGs

`python -m scripts.run_pipeline` trains once. Airflow turns training and
monitoring into *managed* processes:

| Airflow gives you | Why it matters |
|---|---|
| **Scheduling** (`@weekly`, `@hourly`) | retrain and monitor without a human running commands |
| **Observability** | every step is a task with logs, duration, status in a UI |
| **Retries with backoff** | a flaky ingest step retries instead of failing the run |
| **Dependency graph** | a step starts only when its upstreams succeed |

Two DAGs:
- `training_dag.py` — `ingest → engineer_features → train_and_evaluate → report`,
  the pipeline as a scheduled workflow.
- `monitoring_dag.py` — one `monitor_cycle` task that runs the
  `SelfHealingMonitor` hourly. This is the DAG that makes the system self-healing
  in production.

Two Airflow habits worth knowing (both in the code):
1. **TaskFlow API** (`@dag`/`@task`): plain functions become tasks; return values
   pass between them via XCom. Much cleaner than the old `PythonOperator` style.
2. **Import heavy modules *inside* the task, not at file top.** Airflow re-parses
   every DAG file constantly to build the scheduler view; a heavy top-level
   `import mlflow` would slow parsing for the whole cluster. Keep the top of a DAG
   file light.

Airflow is an **optional dependency** here (it's a big install and the rest of
the project doesn't need it). The DAG files import `airflow` at top level because
only the Airflow scheduler ever loads them; to run them, `pip install
apache-airflow` and point the scheduler's DAGs folder at
`src/mlops/orchestration/dags/`.

## Always notify

Every cycle ends with `notify.send_alert()` — to a Slack/Discord webhook if
`MLOPS_ALERT_WEBHOOK_URL` is set, else to the log. The rule: **automated
remediation must never be silent.** "The system retrained itself and nobody
knew" is how teams stop trusting automation. The notification is not optional
decoration; it's what makes auto-remediation socially acceptable to operate.
