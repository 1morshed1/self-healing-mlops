"""Tests for the MLflow registry wrapper (versions, aliases, reference sample)."""

from __future__ import annotations

from mlops.config import get_settings
from mlops.training import registry


def test_production_alias_and_model_load(trained_model):
    name = get_settings().model_name
    version = registry.get_alias_version(name, registry.PRODUCTION_ALIAS)
    assert version is not None
    # The model is loadable via the alias and can predict.
    model = registry.load_model(name)
    assert hasattr(model, "predict_proba")


def test_reference_sample_has_features(trained_model):
    name = get_settings().model_name
    ref = registry.load_reference_sample(name)
    from mlops.data.schema import FEATURE_NAMES

    for feature in FEATURE_NAMES:
        assert feature in ref.columns
    assert len(ref) > 0


def test_rollback_with_single_version_is_safe(trained_model):
    # Behaviour must be well-defined even when there's nothing to roll back to.
    name = get_settings().model_name
    old, new = registry.rollback(name)
    # Either there's a prior version (new set) or there isn't (new is None) —
    # crucially, it must not raise and must not leave the alias unset.
    assert registry.get_alias_version(name, registry.PRODUCTION_ALIAS) is not None
