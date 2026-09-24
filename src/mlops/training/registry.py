"""A thin, well-documented wrapper over the MLflow Model Registry.

WHY WRAP MLFLOW AT ALL
----------------------
MLflow's API is powerful but low-level, and it has evolved (model "stages" like
Production/Staging are deprecated in favour of **aliases**). The rest of our
code — serving, orchestration, the self-healing loop — should not have to know
those details or track API deprecations. So everything registry-related lives
here behind a small, intention-revealing API:

    configure_mlflow()                       # call once at process start
    promote(name, version, alias="production")
    get_alias_version(name, "production")
    load_model(name, alias="production")
    rollback(name)                           # flip production to previous version
    load_reference_sample(name)              # training-time feature snapshot

WHAT AN "ALIAS" IS
------------------
An alias is a movable pointer to a specific model version, e.g.
``production`` → version 7. Serving loads ``models:/<name>@production`` and
therefore always gets whatever version is currently promoted, with no code
change on deploy. Rollback is just moving the pointer back to version 6.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops.config import get_settings

# The alias serving reads. Named constant so it can't be mistyped across files.
PRODUCTION_ALIAS = "production"

_configured = False


def configure_mlflow() -> None:
    """Point MLflow at our tracking store and ensure the experiment exists.

    Idempotent: safe to call from every entrypoint. The first call sets the
    tracking URI and creates the experiment (with an explicit artifact root so
    models land in a predictable, git-ignored directory).
    """
    global _configured
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        client.create_experiment(
            settings.experiment_name,
            artifact_location=settings.mlflow_artifact_root,
        )
    mlflow.set_experiment(settings.experiment_name)
    _configured = True


def get_client() -> MlflowClient:
    """Return an MlflowClient, configuring MLflow first if needed."""
    if not _configured:
        configure_mlflow()
    return MlflowClient()


def _model_uri(name: str, *, alias: str | None = None, version: str | None = None) -> str:
    """Build a ``models:/`` URI for either an alias or an explicit version."""
    if alias is not None:
        return f"models:/{name}@{alias}"
    if version is not None:
        return f"models:/{name}/{version}"
    raise ValueError("Provide either alias or version.")


def list_versions(name: str) -> list[int]:
    """Return all registered version numbers for ``name``, sorted ascending."""
    client = get_client()
    try:
        versions = client.search_model_versions(f"name = '{name}'")
    except Exception:
        return []
    return sorted(int(v.version) for v in versions)


def get_alias_version(name: str, alias: str = PRODUCTION_ALIAS) -> int | None:
    """Return the version an alias points to, or None if the alias is unset."""
    client = get_client()
    try:
        mv = client.get_model_version_by_alias(name, alias)
        return int(mv.version)
    except Exception:
        # No such alias / no such model yet — a normal "cold start" state.
        return None


def promote(name: str, version: int, alias: str = PRODUCTION_ALIAS) -> None:
    """Point ``alias`` at ``version`` (this is what "deploy" means here)."""
    client = get_client()
    client.set_registered_model_alias(name, alias, str(version))


def load_model(name: str, *, alias: str = PRODUCTION_ALIAS, version: int | None = None):
    """Load a fitted model from the registry.

    Defaults to whatever version the ``production`` alias points at, so callers
    (the serving app) don't hard-code a version. Pass ``version`` to load a
    specific one (used by canary comparisons / rollback checks).
    """
    uri = _model_uri(name, version=str(version)) if version else _model_uri(name, alias=alias)
    return mlflow.sklearn.load_model(uri)


def rollback(name: str, alias: str = PRODUCTION_ALIAS) -> tuple[int | None, int | None]:
    """Move ``alias`` back to the highest version below the current one.

    Returns ``(old_version, new_version)``. If there is nothing to roll back to
    (only one version, or none), ``new_version`` is None and the alias is left
    unchanged. This is the ROLLBACK remediation the diagnostic agent can invoke.
    """
    current = get_alias_version(name, alias)
    versions = list_versions(name)
    if current is None or len(versions) < 2:
        return current, None

    # Candidates are all versions strictly older than the current production one.
    older = [v for v in versions if v < current]
    if not older:
        return current, None

    target = max(older)
    promote(name, target, alias)
    return current, target


def load_reference_sample(name: str, version: int | None = None) -> pd.DataFrame:
    """Download the training-time feature snapshot logged with a model version.

    The drift detector needs the distribution the model was *trained* on to
    compare production data against. We log that as a ``reference_sample.parquet``
    artifact in the training run, and tie it to the model version via the run.

    Defaults to the current production version. Raises if no snapshot is found
    (which would mean the model was registered without one — a bug we guard in
    train.py).
    """
    client = get_client()
    version = version if version is not None else get_alias_version(name, PRODUCTION_ALIAS)
    if version is None:
        raise RuntimeError(f"No production version of {name!r} to load a reference sample from.")

    mv = client.get_model_version(name, str(version))
    run_id = mv.run_id
    with tempfile.TemporaryDirectory() as tmp:
        local_path = client.download_artifacts(run_id, "reference_sample.parquet", tmp)
        return pd.read_parquet(local_path)
