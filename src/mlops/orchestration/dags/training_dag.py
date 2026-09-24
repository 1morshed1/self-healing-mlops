"""Airflow DAG: the training pipeline as an orchestrated, scheduled workflow.

WHY AIRFLOW (and what it adds over `python -m scripts.run_pipeline`)
-------------------------------------------------------------------
The script trains a model once. Airflow makes training a *managed* process:
* **Scheduled** — retrain nightly/weekly without a human running a command.
* **Observable** — each step is a task with logs, duration, and status in the UI.
* **Retriable** — a flaky step retries with backoff instead of failing the run.
* **Dependency-aware** — steps run in a DAG (directed acyclic graph); a step
  only starts once its upstreams succeed.

We model the pipeline as four tasks that mirror the architecture diagram:
    ingest -> engineer_features -> train_and_evaluate -> register_and_promote

We use the modern **TaskFlow API** (`@dag` / `@task` decorators): plain Python
functions become tasks, and their return values pass between tasks via XCom.

NOTE: this file imports `airflow`, an optional dependency. Install it only to
run the orchestration layer (`pip install apache-airflow`), then point the
Airflow scheduler's DAGs folder at this directory.
"""

from __future__ import annotations

from datetime import datetime, timedelta

# These imports resolve only inside an Airflow environment. They're at module
# top-level because Airflow imports this file to discover the DAG.
from airflow.decorators import dag, task

# Default arguments applied to every task in the DAG.
_DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,                       # transient failures retry automatically
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="mlops_training_pipeline",
    description="Train, evaluate (quality gate), and register the fraud model.",
    schedule="@weekly",                 # retrain on a cadence; also triggerable on demand
    start_date=datetime(2024, 1, 1),
    catchup=False,                      # don't backfill missed runs
    default_args=_DEFAULT_ARGS,
    tags=["mlops", "training"],
)
def training_pipeline():
    """The training DAG. Each @task below is a node in the graph."""

    @task
    def ingest(n_rows: int = 8000) -> dict:
        """Data ingestion step. Here we synthesize; in production this reads
        from S3/a warehouse. Returns metadata passed downstream via XCom."""
        # Import inside the task so DAG *parsing* stays fast and dependency-light
        # (Airflow re-parses DAG files constantly; heavy imports at top slow it).
        return {"n_rows": n_rows}

    @task
    def engineer_features(meta: dict) -> dict:
        """Feature engineering step. Our features are model-ready, so this is a
        pass-through that documents where transforms would live."""
        return meta

    @task
    def train_and_evaluate(meta: dict) -> dict:
        """Train + evaluate against the quality gate. Returns the outcome."""
        from mlops.training.train import train_model

        outcome = train_model(n_rows=meta["n_rows"])
        return {
            "run_id": outcome.run_id,
            "version": outcome.registered_version,
            "passed_gate": outcome.evaluation.passed_gate,
            "roc_auc": outcome.evaluation.roc_auc,
            "promoted": outcome.promoted,
            "note": outcome.promotion_note,
        }

    @task
    def report(outcome: dict) -> None:
        """Final step: surface the result. train_model already registers and
        promotes (guarded by the gate); this makes the decision visible in the
        Airflow logs and could fan out to Slack."""
        from mlops.orchestration import notify

        notify.send_alert(
            "Training pipeline finished",
            f"version={outcome['version']} roc_auc={outcome['roc_auc']:.3f} "
            f"promoted={outcome['promoted']} — {outcome['note']}",
        )

    # Wire the DAG: each call's output feeds the next task's input.
    report(train_and_evaluate(engineer_features(ingest())))


# Airflow discovers the DAG by finding the object this call returns.
training_pipeline_dag = training_pipeline()
