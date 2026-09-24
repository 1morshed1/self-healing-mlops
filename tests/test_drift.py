"""Tests for the drift detector — the statistical core."""

from __future__ import annotations

import numpy as np

from mlops.data.generate import generate_dataset, make_drift_spec
from mlops.data.schema import FEATURE_NAMES
from mlops.monitoring.drift import DriftDetector, calculate_psi


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    # Same distribution -> PSI ~ 0 (allow a small sampling wobble).
    assert calculate_psi(x, x) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    small = rng.normal(0.3, 1, 5000)
    large = rng.normal(2.0, 1, 5000)
    # A bigger mean shift must produce a bigger PSI.
    assert calculate_psi(ref, small) < calculate_psi(ref, large)


def test_detector_quiet_on_clean_data():
    ref = generate_dataset(3000, seed=1)
    detector = DriftDetector(ref, features=FEATURE_NAMES)
    clean = generate_dataset(1500, seed=999)  # same process, no drift
    report = detector.check(clean)
    # No feature should be flagged on same-distribution data.
    assert not report.any_drift
    assert report.max_severity == "none"


def test_detector_fires_on_high_drift():
    ref = generate_dataset(3000, seed=1)
    detector = DriftDetector(ref, features=FEATURE_NAMES)
    drifted = generate_dataset(1500, seed=2, drift=make_drift_spec("high"))
    report = detector.check(drifted)
    assert report.any_drift
    assert report.max_severity == "high"
    # amount and country_risk are directly perturbed by the high preset.
    drifted_names = {r.feature for r in report.drifted_features}
    assert "amount" in drifted_names
    assert "country_risk" in drifted_names


def test_report_serialization_roundtrip():
    ref = generate_dataset(2000, seed=1)
    detector = DriftDetector(ref, features=FEATURE_NAMES)
    report = detector.check(generate_dataset(1000, seed=5, drift=make_drift_spec("medium")))
    d = report.as_dict()
    assert set(d) == {"any_drift", "max_severity", "n_drifted", "n_features", "features"}
    assert d["n_features"] == len(FEATURE_NAMES)
