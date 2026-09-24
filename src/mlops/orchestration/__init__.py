"""Orchestration: the self-healing control loop + Airflow DAG definitions.

* ``self_healing.SelfHealingMonitor`` - one reusable object that runs a full
  monitor -> diagnose -> remediate -> notify cycle. This is the "brain" the demo
  script and the Airflow monitoring DAG both call.
* ``notify`` - send alerts/diagnoses to a Slack/Discord webhook (or just log).
* ``dags/`` - Airflow DAGs (training + monitoring). Airflow is an optional dep,
  so these modules import it lazily and won't break the rest of the package.
"""
