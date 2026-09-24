"""Central configuration for the self-healing MLOps pipeline.

WHY THIS FILE EXISTS
--------------------
Every non-trivial system needs *one* place that knows how it is wired: where
the model registry lives, what accuracy gate promotes a model, which drift
thresholds trigger an alert. Scattering these values across modules is how you
end up with a service that behaves one way in tests and another in production.

We follow the "12-factor app" principle of **config in the environment**:
* Defaults live here in code (so the project runs with zero setup).
* Any value can be overridden by an environment variable (so ops can retune
  thresholds without a code change and secrets never get committed).

`Settings` is a frozen dataclass built once at import time via
``get_settings()``. Import it like::

    from mlops.config import get_settings
    settings = get_settings()
    print(settings.accuracy_gate)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Load a local .env file if one exists. python-dotenv is optional at runtime;
# if it isn't installed we simply rely on the real process environment.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not required
    pass


# The repository root, resolved relative to this file. Everything the pipeline
# writes (models, reference data, logs) is anchored here so paths work no
# matter what directory you launch commands from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_str(name: str, default: str) -> str:
    """Read an environment variable as a string, falling back to ``default``."""
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    """Read an environment variable as a float, tolerating bad input."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        # A malformed override should not crash startup; log-and-default is
        # friendlier than a stack trace during a 3am incident.
        return default


def _get_int(name: str, default: int) -> int:
    """Read an environment variable as an int, tolerating bad input."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of every tunable knob in the system.

    Frozen (immutable) on purpose: config should be read many times and never
    mutated at runtime. If you need different values in a test, build a new
    ``Settings`` instance rather than patching a global.
    """

    # --- Paths --------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    # Where generated datasets, trained models, and reference snapshots go.
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "generated")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    # --- MLflow (experiment tracking + model registry) ----------------------
    # NOTE: the MLflow *Model Registry* requires a database-backed tracking
    # store (it does not work with the plain `file:` store). We therefore
    # default to a local SQLite file, which needs no server yet gives us the
    # full registry (versions + aliases). Point this at a real server
    # (http://localhost:5000) or Postgres for a team/production setup.
    mlflow_tracking_uri: str = field(
        default_factory=lambda: _get_str(
            "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
        )
    )
    # Where MLflow stores model/artifact files (used when creating the experiment).
    mlflow_artifact_root: str = field(
        default_factory=lambda: _get_str(
            "MLFLOW_ARTIFACT_ROOT", f"{PROJECT_ROOT / 'mlartifacts'}"
        )
    )
    experiment_name: str = "self_healing_mlops"
    model_name: str = field(
        default_factory=lambda: _get_str("MLOPS_MODEL_NAME", "self_healing_classifier")
    )

    # --- Training quality gate ---------------------------------------------
    # The evaluate step refuses to promote a model whose test accuracy is
    # below this. This is the "don't ship a worse model" safety rail.
    accuracy_gate: float = field(
        default_factory=lambda: _get_float("MLOPS_ACCURACY_GATE", 0.80)
    )
    random_seed: int = 42  # fixed so training is reproducible

    # --- Serving -----------------------------------------------------------
    serve_host: str = field(default_factory=lambda: _get_str("MLOPS_SERVE_HOST", "0.0.0.0"))
    serve_port: int = field(default_factory=lambda: _get_int("MLOPS_SERVE_PORT", 8000))
    # File the serving layer appends every prediction to (JSON lines). The
    # monitoring layer reads this to build the "production" window for drift.
    prediction_log: Path = field(
        default_factory=lambda: PROJECT_ROOT / "logs" / "predictions.jsonl"
    )

    # --- Drift detection thresholds ----------------------------------------
    psi_threshold: float = field(
        default_factory=lambda: _get_float("MLOPS_PSI_THRESHOLD", 0.2)
    )
    ks_pvalue_threshold: float = field(
        default_factory=lambda: _get_float("MLOPS_KS_PVALUE_THRESHOLD", 0.01)
    )

    # --- LLM diagnostic agent ----------------------------------------------
    anthropic_api_key: str = field(default_factory=lambda: _get_str("ANTHROPIC_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _get_str("MLOPS_LLM_MODEL", "claude-opus-5"))

    # --- Notifications -----------------------------------------------------
    alert_webhook_url: str = field(default_factory=lambda: _get_str("MLOPS_ALERT_WEBHOOK_URL", ""))

    def ensure_dirs(self) -> None:
        """Create the runtime directories if they don't exist.

        Called by entrypoints (training, serving) before they write anything,
        so a fresh checkout "just works" without a manual mkdir step.
        """
        for directory in (self.artifacts_dir, self.data_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    ``lru_cache`` means the environment is read exactly once. Tests that need
    to override a value can call ``get_settings.cache_clear()`` after setting
    an env var, or construct ``Settings(...)`` directly.
    """
    return Settings()
