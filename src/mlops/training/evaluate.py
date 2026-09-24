"""Model evaluation + the quality gate.

WHY A SEPARATE "GATE" CONCEPT
-----------------------------
The single most important safety rail in an automated retraining system is:
**never promote a model that is worse than what's already serving.** Without it,
a self-healing loop can heal itself into a disaster — retrain on bad data, ship
a broken model, retrain again on even worse data.

So evaluation returns not just numbers but a *decision*: does this candidate
clear the bar to be promoted? The training pipeline (and later the Airflow DAG)
refuses to register/promote a model that fails the gate.

WHY NOT JUST ACCURACY
---------------------
Our fraud data is ~4% positive. A model that predicts "never fraud" scores 96%
accuracy and is useless. So we compute precision, recall, F1, and ROC-AUC too,
and the gate can consider them. Accuracy is the headline number people expect,
but ROC-AUC (ranking quality, threshold-independent) is the honest one for
imbalanced problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """The outcome of evaluating a trained model on a held-out test set."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    # The gate decision + a human-readable reason (surfaced to logs and to the
    # LLM agent so it understands *why* a retrain did or didn't ship).
    passed_gate: bool = False
    gate_reason: str = ""

    def as_metrics_dict(self) -> dict[str, float]:
        """Metrics only (no gate fields) — convenient for MLflow logging."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def evaluate(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    accuracy_gate: float,
    min_roc_auc: float = 0.7,
) -> EvaluationResult:
    """Score ``model`` on the test set and decide whether it clears the gate.

    Parameters
    ----------
    model:
        A fitted scikit-learn classifier exposing ``predict`` and
        ``predict_proba``.
    x_test, y_test:
        Held-out features/labels the model has never seen.
    accuracy_gate:
        Minimum test accuracy required to promote (from config).
    min_roc_auc:
        Minimum ranking quality required. Guards the imbalanced-data trap
        where high accuracy hides a model that can't actually separate classes.
    """
    y_pred = model.predict(x_test)
    # Probability of the positive (fraud) class for ROC-AUC.
    y_proba = model.predict_proba(x_test)[:, 1]

    # zero_division=0: if the model predicts no positives at all, treat
    # precision/recall as 0 rather than raising — a legitimate (bad) outcome.
    result = EvaluationResult(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
    )

    # --- The gate ----------------------------------------------------------
    reasons = []
    if result.accuracy < accuracy_gate:
        reasons.append(f"accuracy {result.accuracy:.3f} < gate {accuracy_gate:.3f}")
    if result.roc_auc < min_roc_auc:
        reasons.append(f"roc_auc {result.roc_auc:.3f} < min {min_roc_auc:.3f}")

    if reasons:
        result.passed_gate = False
        result.gate_reason = "REJECTED: " + "; ".join(reasons)
    else:
        result.passed_gate = True
        result.gate_reason = (
            f"PASSED: accuracy {result.accuracy:.3f} >= {accuracy_gate:.3f}, "
            f"roc_auc {result.roc_auc:.3f} >= {min_roc_auc:.3f}"
        )
    return result
