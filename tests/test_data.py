"""Tests for data generation + feature engineering."""

from __future__ import annotations

import pytest

from mlops.data.features import FeatureValidationError, build_feature_frame, frame_to_matrix
from mlops.data.generate import generate_dataset, make_drift_spec
from mlops.data.schema import FEATURE_NAMES, TARGET_NAME


def test_dataset_shape_and_schema():
    df = generate_dataset(1000, seed=1)
    assert len(df) == 1000
    # Columns must be features (in order) followed by the target.
    assert list(df.columns) == [*FEATURE_NAMES, TARGET_NAME]


def test_dataset_is_reproducible():
    a = generate_dataset(500, seed=42)
    b = generate_dataset(500, seed=42)
    # Same seed -> identical data (reproducibility is a hard requirement).
    assert a.equals(b)


def test_fraud_is_rare_but_present():
    df = generate_dataset(5000, seed=3)
    rate = df[TARGET_NAME].mean()
    # Realistic imbalance: rare but not absent.
    assert 0.005 < rate < 0.15


def test_drift_actually_shifts_distribution():
    ref = generate_dataset(3000, seed=1)
    drifted = generate_dataset(3000, seed=2, drift=make_drift_spec("high"))
    # High drift should visibly raise the mean transaction amount and device rate.
    assert drifted["amount"].mean() > ref["amount"].mean() + 20
    assert drifted["device_change"].mean() > ref["device_change"].mean()


def test_make_drift_spec_rejects_unknown():
    with pytest.raises(ValueError):
        make_drift_spec("catastrophic")


def test_build_feature_frame_happy_path():
    record = {n: 1.0 for n in FEATURE_NAMES}
    frame = build_feature_frame(record)
    assert list(frame.columns) == FEATURE_NAMES
    assert frame_to_matrix(frame).shape == (1, len(FEATURE_NAMES))


def test_build_feature_frame_rejects_missing_feature():
    record = {n: 1.0 for n in FEATURE_NAMES}
    del record[FEATURE_NAMES[0]]
    with pytest.raises(FeatureValidationError):
        build_feature_frame(record)


def test_build_feature_frame_rejects_non_numeric():
    record = {n: 1.0 for n in FEATURE_NAMES}
    record[FEATURE_NAMES[0]] = "not-a-number"
    with pytest.raises(FeatureValidationError):
        build_feature_frame(record)
