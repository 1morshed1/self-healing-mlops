"""Performance monitoring: are the model's OUTPUTS degrading?

Drift detection watches the *inputs*. This watches the *outputs*, in two modes:

1. **Confidence monitoring (always available).** Even without ground-truth
   labels, a healthy model tends to make confident predictions (probabilities
   near 0 or 1). When inputs drift into unfamiliar territory, predictions
   cluster near 0.5 — the model is "unsure." A drop in average confidence is an
   early, label-free warning sign.

2. **Accuracy/recall monitoring (when labels arrive).** In fraud you eventually
   learn the truth (a chargeback confirms fraud) — but *late*. When those
   delayed labels arrive we can compute real accuracy/recall on that window and
   compare it to the training-time baseline. A significant drop is the lagging
   confirmation that drift actually hurt the model.

Together: confidence is the smoke alarm, accuracy is the thermometer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, recall_score


@dataclass
class PerformanceReport:
    """Output-health summary for one production window."""

    mean_confidence: float          # avg distance of P(class) from 0.5, scaled
    low_confidence_rate: float      # fraction of predictions in the "unsure" band
    baseline_confidence: float      # training-time mean confidence for comparison
    confidence_drop: float          # baseline - current (positive = degraded)
    # Present only when ground-truth labels were supplied for this window:
    accuracy: float | None = None
    recall: float | None = None
    baseline_accuracy: float | None = None
    accuracy_drop: float | None = None
    degraded: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        d = {
            "mean_confidence": round(self.mean_confidence, 4),
            "low_confidence_rate": round(self.low_confidence_rate, 4),
            "baseline_confidence": round(self.baseline_confidence, 4),
            "confidence_drop": round(self.confidence_drop, 4),
            "degraded": self.degraded,
            "reason": self.reason,
        }
        if self.accuracy is not None:
            d.update(
                accuracy=round(self.accuracy, 4),
                recall=round(self.recall, 4) if self.recall is not None else None,
                baseline_accuracy=round(self.baseline_accuracy, 4),
                accuracy_drop=round(self.accuracy_drop, 4),
            )
        return d


def _confidence(proba: np.ndarray) -> np.ndarray:
    """Map a positive-class probability to a 0-1 confidence score.

    A prediction of 0.5 is maximally uncertain (confidence 0); 0.0 or 1.0 is
    maximally confident (confidence 1). ``2 * |p - 0.5|`` does exactly that.
    """
    return 2.0 * np.abs(np.asarray(proba, dtype=float) - 0.5)


class PerformanceMonitor:
    """Tracks confidence and (when labels exist) accuracy against a baseline."""

    def __init__(
        self,
        *,
        baseline_confidence: float,
        baseline_accuracy: float | None = None,
        confidence_drop_threshold: float = 0.1,
        accuracy_drop_threshold: float = 0.05,
        low_confidence_band: float = 0.6,
    ):
        # Baselines captured at training time (the model's behaviour on a
        # healthy, in-distribution test set).
        self.baseline_confidence = baseline_confidence
        self.baseline_accuracy = baseline_accuracy
        self.confidence_drop_threshold = confidence_drop_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        # Predictions whose probability is within this band around 0.5 count as
        # "low confidence" (e.g. band 0.6 → p in [0.2, 0.8]).
        self.low_confidence_band = low_confidence_band

    def evaluate(
        self,
        proba: np.ndarray,
        *,
        y_true: np.ndarray | None = None,
    ) -> PerformanceReport:
        """Assess one window of predicted probabilities (+ optional true labels)."""
        proba = np.asarray(proba, dtype=float)
        conf = _confidence(proba)
        mean_conf = float(np.mean(conf))
        half_band = self.low_confidence_band / 2.0
        low_conf_rate = float(np.mean(np.abs(proba - 0.5) < half_band))
        conf_drop = self.baseline_confidence - mean_conf

        report = PerformanceReport(
            mean_confidence=mean_conf,
            low_confidence_rate=low_conf_rate,
            baseline_confidence=self.baseline_confidence,
            confidence_drop=conf_drop,
        )

        reasons = []
        if conf_drop >= self.confidence_drop_threshold:
            reasons.append(
                f"confidence dropped {conf_drop:.3f} "
                f"(>= {self.confidence_drop_threshold:.3f})"
            )

        # --- Label-based metrics, only if truth is available ----------------
        if y_true is not None:
            y_true = np.asarray(y_true, dtype=int)
            y_pred = (proba >= 0.5).astype(int)
            report.accuracy = float(accuracy_score(y_true, y_pred))
            report.recall = float(recall_score(y_true, y_pred, zero_division=0))
            if self.baseline_accuracy is not None:
                report.baseline_accuracy = self.baseline_accuracy
                report.accuracy_drop = self.baseline_accuracy - report.accuracy
                if report.accuracy_drop >= self.accuracy_drop_threshold:
                    reasons.append(
                        f"accuracy dropped {report.accuracy_drop:.3f} "
                        f"(>= {self.accuracy_drop_threshold:.3f})"
                    )

        report.degraded = bool(reasons)
        report.reason = "; ".join(reasons) if reasons else "within normal bounds"
        return report
