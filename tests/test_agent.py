"""Tests for the diagnostic agent: rule-based policy, LLM path, and fallback.

The LLM path is tested WITHOUT network by injecting a fake client that mimics the
Anthropic SDK's response shape. This lets us assert the agent parses/validates
model output correctly and falls back safely on bad output — deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlops.diagnostics.agent import AlertContext, DiagnosticAgent


# --- helpers to build contexts --------------------------------------------
def _ctx(any_drift, max_sev, n_drifted, acc_drop=None, degraded=False):
    features = [
        {"feature": "amount", "psi": 0.5, "ks_pvalue": 0.0, "drifted": True,
         "severity": max_sev, "reference_mean": 90, "production_mean": 160}
    ] if any_drift else []
    drift = {
        "any_drift": any_drift, "max_severity": max_sev,
        "n_drifted": n_drifted, "n_features": 6, "features": features,
    }
    perf = {
        "mean_confidence": 0.5, "baseline_confidence": 0.8, "confidence_drop": 0.3,
        "low_confidence_rate": 0.4, "degraded": degraded, "reason": "test",
    }
    if acc_drop is not None:
        perf.update(accuracy=0.9 - acc_drop, baseline_accuracy=0.9, accuracy_drop=acc_drop)
    return AlertContext(
        model_name="m", model_version=1, drift=drift, performance=perf, window_size=100
    )


# --- rule-based policy -----------------------------------------------------
def test_rule_based_ignores_when_healthy():
    agent = DiagnosticAgent()  # no API key -> rule-based
    d = agent.diagnose(_ctx(any_drift=False, max_sev="none", n_drifted=0))
    assert d.action == "IGNORE"
    assert d.source == "rule-based"


def test_rule_based_retrains_on_drift_plus_accuracy_drop():
    agent = DiagnosticAgent()
    d = agent.diagnose(
        _ctx(any_drift=True, max_sev="high", n_drifted=4, acc_drop=0.2, degraded=True)
    )
    assert d.action == "RETRAIN"
    assert d.severity == "critical"  # >= 0.15 drop


def test_rule_based_investigates_drift_without_confirmed_accuracy_loss():
    agent = DiagnosticAgent()
    # High drift, but no accuracy label available -> escalate, don't auto-act.
    d = agent.diagnose(_ctx(any_drift=True, max_sev="high", n_drifted=4))
    assert d.action == "INVESTIGATE"


# --- LLM path with an injected fake client ---------------------------------
@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Msg:
    content: list


class _FakeMessages:
    def __init__(self, payload: str):
        self._payload = payload

    def create(self, **kwargs):
        return _Msg(content=[_Block(type="text", text=self._payload)])


class _FakeClient:
    def __init__(self, payload: str):
        self.messages = _FakeMessages(payload)


def test_llm_path_parses_valid_json():
    payload = (
        '{"root_cause": "device-spoofing wave", "severity": "high", '
        '"action": "RETRAIN", "reasoning": "drift + accuracy loss", "confidence": 0.82}'
    )
    agent = DiagnosticAgent(client=_FakeClient(payload))
    d = agent.diagnose(_ctx(any_drift=True, max_sev="high", n_drifted=3, acc_drop=0.1))
    assert d.source == "llm"
    assert d.action == "RETRAIN"
    assert d.severity == "high"
    assert 0.8 <= d.confidence <= 0.85


def test_llm_out_of_vocab_action_is_clamped_to_safe():
    # A misbehaving model returns an action we don't allow -> clamp to INVESTIGATE.
    payload = '{"root_cause": "x", "severity": "banana", "action": "DELETE_PROD", "confidence": 5}'
    agent = DiagnosticAgent(client=_FakeClient(payload))
    d = agent.diagnose(_ctx(any_drift=True, max_sev="high", n_drifted=3))
    assert d.action == "INVESTIGATE"     # unknown action -> safe default
    assert d.severity == "medium"        # unknown severity -> default
    assert d.confidence == 1.0           # clamped into [0, 1]


def test_llm_garbage_falls_back_to_rules():
    # Non-JSON output -> extraction fails -> rule-based fallback (never crashes).
    agent = DiagnosticAgent(client=_FakeClient("I could not produce JSON, sorry."))
    d = agent.diagnose(_ctx(any_drift=False, max_sev="none", n_drifted=0))
    assert d.source == "rule-based"
    assert d.action == "IGNORE"
