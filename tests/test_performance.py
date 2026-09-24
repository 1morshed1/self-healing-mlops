"""Tests for the performance monitor."""

from __future__ import annotations

import numpy as np

from mlops.monitoring.performance import PerformanceMonitor, _confidence


def test_confidence_scoring():
    # 0.5 = maximally unsure (0.0); 0.0 or 1.0 = maximally sure (1.0).
    conf = _confidence(np.array([0.5, 0.0, 1.0, 0.75]))
    assert conf[0] == 0.0
    assert conf[1] == 1.0
    assert conf[2] == 1.0
    assert abs(conf[3] - 0.5) < 1e-9


def test_healthy_window_not_degraded():
    monitor = PerformanceMonitor(baseline_confidence=0.8, baseline_accuracy=0.9)
    # Confident, correct predictions.
    proba = np.array([0.95, 0.02, 0.97, 0.01] * 25)
    y_true = np.array([1, 0, 1, 0] * 25)
    report = monitor.evaluate(proba, y_true=y_true)
    assert not report.degraded
    assert report.accuracy == 1.0


def test_accuracy_drop_flags_degraded():
    monitor = PerformanceMonitor(baseline_confidence=0.8, baseline_accuracy=0.9)
    # Confident but WRONG half the time -> accuracy ~0.5, big drop from 0.9.
    proba = np.array([0.95, 0.97] * 50)
    y_true = np.array([1, 0] * 50)  # second class is always wrong
    report = monitor.evaluate(proba, y_true=y_true)
    assert report.degraded
    assert report.accuracy_drop > 0.05


def test_confidence_drop_flags_without_labels():
    monitor = PerformanceMonitor(baseline_confidence=0.9)
    # All predictions near 0.5 -> low confidence, big drop from 0.9 baseline.
    proba = np.full(100, 0.52)
    report = monitor.evaluate(proba)  # no labels
    assert report.degraded
    assert report.accuracy is None  # nothing to compute without labels
