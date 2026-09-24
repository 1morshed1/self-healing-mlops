"""Monitoring layer: is the model still healthy in production?

Two independent signals:

* ``drift.DriftDetector`` - has the *input* distribution moved away from what
  the model was trained on? (PSI + Kolmogorov-Smirnov test, per feature.) This
  works with NO ground-truth labels, which is the realistic case — you rarely
  know the true label at prediction time.
* ``performance.PerformanceMonitor`` - are the *outputs* degrading? Tracks
  prediction-confidence shifts always, and true accuracy/recall once labels
  arrive (delayed feedback).

Drift is the leading indicator (you see it first); performance drop is the
lagging confirmation.
"""

from mlops.monitoring.drift import DriftDetector, DriftReport, DriftResult

__all__ = ["DriftDetector", "DriftReport", "DriftResult"]
