"""Airflow DAG: the self-healing monitoring loop, on a schedule.

This is the DAG that makes the system "self-healing" in production. It runs
frequently (hourly here), pulls the recent production window, and executes one
full monitor -> diagnose -> remediate -> notify cycle via ``SelfHealingMonitor``.

Because remediation (retrain/rollback) can be triggered automatically, this DAG
carries ``auto_remediate`` — flip it on only once you trust the loop. Until then
it runs in advisory mode: it diagnoses and alerts, but a human pulls the trigger.

The single monitoring task delegates to the same ``SelfHealingMonitor`` the demo
CLI uses, so there is exactly one implementation of the healing logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task

_DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Master switch for automated remediation. Keep False in production until the
# advisory diagnoses have proven trustworthy over time.
AUTO_REMEDIATE = False


@dag(
    dag_id="mlops_self_healing_monitor",
    description="Detect drift, diagnose with the LLM agent, and (optionally) heal.",
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["mlops", "monitoring", "self-healing"],
)
def self_healing_monitor():
    @task
    def monitor_cycle() -> dict:
        """Run one full self-healing cycle over the live prediction window."""
        from mlops.data.generate import make_drift_spec
        from mlops.orchestration.self_healing import SelfHealingMonitor

        monitor = SelfHealingMonitor(auto_remediate=AUTO_REMEDIATE)

        # Prefer the real serving log; if empty (no traffic yet), fall back to a
        # simulated window so the DAG still exercises the full path in a demo.
        window = monitor.window_from_log()
        if window is None:
            window = monitor.window_from_simulation(make_drift_spec("medium"))

        result = monitor.run_cycle(window)
        return {
            "max_severity": result.drift.get("max_severity"),
            "acted": result.acted,
            "action": result.diagnosis.action if result.diagnosis else "none",
        }

    monitor_cycle()


self_healing_monitor_dag = self_healing_monitor()
