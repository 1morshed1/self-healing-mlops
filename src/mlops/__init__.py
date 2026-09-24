"""Self-healing MLOps pipeline.

A compact, runnable demonstration of production ML engineering concerns:

* ``mlops.data``          - synthetic dataset generation + drift simulation
* ``mlops.training``      - model training, evaluation gate, MLflow registry
* ``mlops.monitoring``    - statistical data-drift + performance monitoring
* ``mlops.diagnostics``   - LLM-powered root-cause agent + auto-remediation
* ``mlops.serving``       - FastAPI prediction API with Prometheus metrics
* ``mlops.orchestration`` - Airflow DAG + the self-healing control loop

The package is importable as ``mlops`` once ``src`` is on the path (the
project installs itself in editable mode via pyproject.toml, and the tests /
scripts add ``src`` to sys.path as a fallback).
"""

__version__ = "0.1.0"
