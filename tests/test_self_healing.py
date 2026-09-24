"""Integration test for the full self-healing cycle (advisory mode)."""

from __future__ import annotations

from mlops.data.generate import make_drift_spec
from mlops.orchestration.self_healing import SelfHealingMonitor


def test_healthy_window_produces_no_action(trained_model):
    monitor = SelfHealingMonitor(auto_remediate=False)
    window = monitor.window_from_simulation(make_drift_spec("none"), seed=123)
    result = monitor.run_cycle(window)
    # Clean data -> no diagnosis, no remediation.
    assert result.diagnosis is None
    assert not result.acted


def test_high_drift_triggers_diagnosis_but_advisory_skips_action(trained_model):
    monitor = SelfHealingMonitor(auto_remediate=False)
    window = monitor.window_from_simulation(make_drift_spec("high"), seed=7)
    result = monitor.run_cycle(window)
    # Drift detected and diagnosed...
    assert result.diagnosis is not None
    assert result.drift["max_severity"] == "high"
    # ...but advisory mode must not perform a side effect.
    assert not result.acted
