"""Tests for the Remediator — dispatch + the advisory-mode safety gate."""

from __future__ import annotations

from mlops.diagnostics.agent import Diagnosis
from mlops.diagnostics.remediation import Remediator


def _diag(action):
    return Diagnosis(
        root_cause="x", severity="high", action=action,
        reasoning="y", confidence=0.8, source="rule-based",
    )


def test_ignore_is_a_noop():
    result = Remediator(auto_remediate=True).remediate(_diag("IGNORE"))
    assert not result.executed
    assert "nothing to do" in result.detail.lower()


def test_investigate_never_acts():
    result = Remediator(auto_remediate=True).remediate(_diag("INVESTIGATE"))
    assert not result.executed
    assert "human" in result.detail.lower()


def test_advisory_mode_skips_retrain():
    # With auto_remediate off, RETRAIN is diagnosed but NOT executed.
    result = Remediator(auto_remediate=False).remediate(_diag("RETRAIN"))
    assert not result.executed
    assert "advisory" in result.detail.lower()


def test_advisory_mode_skips_rollback():
    result = Remediator(auto_remediate=False).remediate(_diag("ROLLBACK"))
    assert not result.executed


def test_auto_retrain_executes_through_quality_gate(trained_model):
    # In auto mode RETRAIN really runs the training pipeline. The gate still
    # applies, so a new version is registered; promotion depends on quality.
    result = Remediator(auto_remediate=True).remediate(_diag("RETRAIN"))
    assert result.executed
    assert result.new_model_version is not None
