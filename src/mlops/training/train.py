"""The training pipeline: data -> fit -> evaluate -> gate -> register -> promote.

This is the box labelled "Training pipeline" in the architecture diagram, and
it is what the self-healing loop re-runs when it decides to RETRAIN.

DESIGN NOTES
------------
* **Reproducible**: fixed seed, logged params. Re-running gives the same model.
* **Honest logging**: EVERY run is logged to MLflow (params, metrics, the model
  itself, and the reference-distribution snapshot) — even runs that fail the
  gate. You want the record of the bad model that *didn't* ship as much as the
  good one that did.
* **Safe promotion**: a run is only promoted to the ``production`` alias if it
  clears the quality gate AND is not worse than the incumbent. This is the rule
  that keeps automated retraining from degrading the service.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlops.config import get_settings
from mlops.data.generate import DriftSpec, generate_dataset
from mlops.data.schema import FEATURE_NAMES, TARGET_NAME
from mlops.training import registry
from mlops.training.evaluate import EvaluationResult, evaluate


@dataclass
class TrainingOutcome:
    """Everything a caller (or the LLM agent) needs to know about a training run."""

    run_id: str
    evaluation: EvaluationResult
    registered_version: int | None      # None if the gate rejected it
    promoted: bool                      # did it become the new production model?
    promotion_note: str


def _build_model(seed: int) -> RandomForestClassifier:
    """Construct the classifier.

    RandomForest is a deliberate choice for a tabular fraud problem: it handles
    non-linear feature interactions, needs no feature scaling, gives calibrated-
    enough probabilities for a demo, and exposes feature importances (handy
    context for the diagnostic agent). ``class_weight="balanced"`` counters the
    ~4% class imbalance so the model actually learns to flag the rare positives.
    """
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def train_model(
    *,
    n_rows: int = 8000,
    data_seed: int | None = None,
    drift: DriftSpec | None = None,
    register: bool = True,
) -> TrainingOutcome:
    """Run one end-to-end training job.

    Parameters
    ----------
    n_rows:
        Size of the dataset to generate for this run.
    data_seed:
        Seed for the training data. Defaults to the configured global seed for
        the initial model; the self-healing loop passes a fresh seed (and a
        DriftSpec) to simulate "retrain on recent, drifted data".
    drift:
        If given, the training data itself is drawn from the drifted world.
        This models retraining on *recent* production data that has moved.
    register:
        If False, train + evaluate but skip MLflow registration (used in fast
        unit tests).
    """
    settings = get_settings()
    settings.ensure_dirs()
    seed = settings.random_seed if data_seed is None else data_seed

    # --- 1. Data -----------------------------------------------------------
    dataset = generate_dataset(n_rows, seed=seed, drift=drift)
    x = dataset[FEATURE_NAMES].to_numpy(dtype=float)
    y = dataset[TARGET_NAME].to_numpy(dtype=int)

    # Stratify so the tiny positive class is represented in both splits.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=settings.random_seed, stratify=y
    )

    # --- 2. Fit ------------------------------------------------------------
    model = _build_model(settings.random_seed)
    model.fit(x_train, y_train)

    # --- 3. Evaluate + gate ------------------------------------------------
    evaluation = evaluate(model, x_test, y_test, accuracy_gate=settings.accuracy_gate)

    if not register:
        return TrainingOutcome(
            run_id="(unregistered)",
            evaluation=evaluation,
            registered_version=None,
            promoted=False,
            promotion_note="register=False",
        )

    # --- 4. Log to MLflow (always) -----------------------------------------
    registry.configure_mlflow()
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "n_rows": n_rows,
                "data_seed": seed,
                "drifted_training_data": drift is not None and not drift.is_noop(),
                "model_type": "RandomForestClassifier",
                "n_estimators": 250,
                "max_depth": 12,
            }
        )
        mlflow.log_metrics(evaluation.as_metrics_dict())
        mlflow.set_tag("passed_gate", str(evaluation.passed_gate))
        mlflow.set_tag("gate_reason", evaluation.gate_reason)

        # Log the model. registered_model_name creates a new version in the
        # registry automatically (this is the modern one-call path).
        #
        # serialization_format=cloudpickle: newer MLflow defaults to the
        # `skops` format, which refuses to save sklearn tree internals as
        # "untrusted". cloudpickle is the long-standing default and is safe for
        # models we produce and load ourselves within this trusted pipeline.
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=settings.model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # Log the reference-distribution snapshot the drift detector will need.
        # We use the TRAIN split (what the model actually learned from). A
        # sample keeps the KS test fast without changing its conclusion.
        _log_reference_sample(x_train, y_train)

    # --- 5. Decide promotion ----------------------------------------------
    version = max(registry.list_versions(settings.model_name))
    promoted, note = _maybe_promote(settings.model_name, version, evaluation)

    return TrainingOutcome(
        run_id=run_id,
        evaluation=evaluation,
        registered_version=version,
        promoted=promoted,
        promotion_note=note,
    )


def _log_reference_sample(x_train, y_train, max_rows: int = 3000) -> None:
    """Write the training features as an MLflow artifact for drift comparison."""
    frame = pd.DataFrame(x_train, columns=FEATURE_NAMES)
    frame[TARGET_NAME] = y_train
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=0).reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "reference_sample.parquet"
        frame.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


def _maybe_promote(name: str, version: int, evaluation: EvaluationResult) -> tuple[bool, str]:
    """Promote the new version to production only if it's safe to do so.

    Rules:
    1. Must clear the quality gate.
    2. If there's already a production model, the candidate must not be clearly
       worse (we compare ROC-AUC, the threshold-independent quality measure).
       The first model ever always gets promoted if it passes the gate.
    """
    if not evaluation.passed_gate:
        return False, f"Not promoted — failed gate. {evaluation.gate_reason}"

    current = registry.get_alias_version(name, registry.PRODUCTION_ALIAS)
    if current is None:
        registry.promote(name, version)
        return True, f"Promoted v{version} to production (first model)."

    # Compare against the incumbent's logged ROC-AUC.
    client = registry.get_client()
    incumbent_mv = client.get_model_version(name, str(current))
    incumbent_run = client.get_run(incumbent_mv.run_id)
    incumbent_auc = incumbent_run.data.metrics.get("roc_auc", 0.0)

    # Allow a tiny tolerance so noise doesn't block a genuinely-equal retrain.
    if evaluation.roc_auc + 1e-4 >= incumbent_auc:
        registry.promote(name, version)
        return True, (
            f"Promoted v{version} (roc_auc {evaluation.roc_auc:.3f} "
            f">= incumbent v{current} {incumbent_auc:.3f})."
        )
    return False, (
        f"Not promoted — v{version} roc_auc {evaluation.roc_auc:.3f} "
        f"< incumbent v{current} {incumbent_auc:.3f}. Kept v{current}."
    )
