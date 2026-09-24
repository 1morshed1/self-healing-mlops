"""Shared pytest fixtures + test isolation.

TEST ISOLATION STRATEGY
-----------------------
The pipeline writes real state (an MLflow SQLite DB, model artifacts). Tests must
NOT touch your development ``mlflow.db`` or leave junk in the repo. So before any
project code is imported, we point MLflow at a throwaway temp directory via
environment variables, then clear the cached settings so they're re-read.

Because ``get_settings()`` is ``lru_cache``d and reads the environment on first
call, setting these env vars at conftest import time (which pytest runs before
collecting test modules) guarantees every test sees the isolated config.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# --- Redirect all runtime state into a temp dir BEFORE importing project code -
_TMP = Path(tempfile.mkdtemp(prefix="mlops-tests-"))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP / 'mlflow.db'}"
os.environ["MLFLOW_ARTIFACT_ROOT"] = str(_TMP / "mlartifacts")
# Keep the model name/threshold defaults; ensure no real API key leaks in so the
# agent tests exercise the rule-based path unless they inject a client.
os.environ.pop("ANTHROPIC_API_KEY", None)

from mlops.config import get_settings  # noqa: E402  (must come after env setup)

# Re-read settings now that the env is patched.
get_settings.cache_clear()


@pytest.fixture(scope="session")
def trained_model():
    """Train + register one small model for the whole test session.

    Session-scoped because training, while fast, isn't instant — and many tests
    (registry, API, self-healing) just need *a* production model to exist.
    Returns the TrainingOutcome.
    """
    from mlops.training.train import train_model

    outcome = train_model(n_rows=3000)
    assert outcome.registered_version is not None
    return outcome


@pytest.fixture
def settings():
    """The isolated settings object."""
    return get_settings()
