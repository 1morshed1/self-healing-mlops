"""The LLM diagnostic agent + a deterministic rule-based fallback.

DESIGN PRINCIPLE: DECIDE, DON'T ACT
-----------------------------------
The agent's only job is to produce a ``Diagnosis`` (root cause + severity +
recommended action). It does NOT retrain or roll back anything itself. Executing
the action is the ``Remediator``'s job (remediation.py). This separation means:
* the agent is a pure, easily-tested function of its input context,
* you can run it in "advisory" mode (log the recommendation, act manually), and
* the dangerous side effects live in one small, auditable place.

GRACEFUL DEGRADATION
--------------------
If no API key is set, or the API call fails for any reason, the agent falls back
to a transparent rule-based diagnosis. The rest of the system can therefore rely
on ``diagnose()`` always returning a valid ``Diagnosis`` — the demo, the tests,
and CI all work with no network. ``Diagnosis.source`` records which path
produced it ("llm" vs "rule-based") for honest logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mlops.config import Settings, get_settings
from mlops.diagnostics import prompts

logger = logging.getLogger(__name__)


@dataclass
class AlertContext:
    """Everything the agent needs to reason about one alert.

    ``drift`` and ``performance`` are the ``.as_dict()`` outputs of DriftReport
    and PerformanceReport respectively — plain dicts, so the agent has no import
    dependency on the monitoring classes and is trivial to construct in tests.
    """

    model_name: str
    model_version: int | None
    drift: dict
    performance: dict | None = None
    window_size: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Diagnosis:
    """The agent's verdict."""

    root_cause: str
    severity: str          # one of prompts.ALLOWED_SEVERITIES
    action: str            # one of prompts.ALLOWED_ACTIONS
    reasoning: str
    confidence: float
    source: str            # "llm" or "rule-based"

    def as_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "severity": self.severity,
            "action": self.action,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }

    def summary_line(self) -> str:
        return (
            f"[{self.source}] action={self.action} severity={self.severity} "
            f"(conf {self.confidence:.2f}): {self.root_cause}"
        )


class DiagnosticAgent:
    """Produces a Diagnosis from an AlertContext, via Claude or a rule fallback."""

    def __init__(self, settings: Settings | None = None, client=None):
        self.settings = settings or get_settings()
        # Allow a pre-built client to be injected (used in tests to simulate
        # the API without network). Otherwise we lazily create one only if a
        # key is present.
        self._client = client

    # -- public API ---------------------------------------------------------
    def diagnose(self, context: AlertContext) -> Diagnosis:
        """Return a Diagnosis, preferring the LLM but never failing outright."""
        if self._can_use_llm():
            try:
                return self._diagnose_with_llm(context)
            except Exception as exc:  # noqa: BLE001 - fall back on ANY failure
                logger.warning("LLM diagnosis failed (%s); using rule-based fallback.", exc)
        return self._diagnose_with_rules(context)

    # -- LLM path -----------------------------------------------------------
    def _can_use_llm(self) -> bool:
        return bool(self._client) or bool(self.settings.anthropic_api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily so the dep is optional at runtime

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def _diagnose_with_llm(self, context: AlertContext) -> Diagnosis:
        client = self._get_client()
        message = client.messages.create(
            model=self.settings.llm_model,
            max_tokens=1024,
            system=prompts.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompts.build_user_prompt(context)}],
        )
        # Concatenate all text blocks (thinking blocks, if any, are skipped).
        text = "".join(block.text for block in message.content if block.type == "text")
        data = prompts.extract_json(text)
        return self._validate(data, source="llm")

    def _validate(self, data: dict, *, source: str) -> Diagnosis:
        """Coerce a parsed dict into a valid Diagnosis, clamping bad fields.

        Even a good model occasionally returns an out-of-vocabulary action. We
        clamp to the allowed sets rather than trust the raw output, because this
        value drives an automated side effect. Anything unrecognized becomes the
        *safe* choice: INVESTIGATE (keep a human in the loop).
        """
        action = str(data.get("action", "")).upper()
        if action not in prompts.ALLOWED_ACTIONS:
            action = "INVESTIGATE"
        severity = str(data.get("severity", "")).lower()
        if severity not in prompts.ALLOWED_SEVERITIES:
            severity = "medium"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))
        return Diagnosis(
            root_cause=str(data.get("root_cause", "(none provided)")),
            severity=severity,
            action=action,
            reasoning=str(data.get("reasoning", "(none provided)")),
            confidence=confidence,
            source=source,
        )

    # -- Rule-based fallback ------------------------------------------------
    def _diagnose_with_rules(self, context: AlertContext) -> Diagnosis:
        """A transparent, deterministic policy that mirrors the prompt's intent.

        This is not a toy: it's a defensible default policy, and it doubles as
        documentation of what "good" looks like for the LLM to emulate.

        Decision logic:
        * No drift and no degradation           -> IGNORE
        * Accuracy dropped materially + drift    -> RETRAIN (the classic case)
        * Broad/high drift but no accuracy proof -> INVESTIGATE (don't act blind)
        * Only confidence dipped / minor drift   -> INVESTIGATE or IGNORE
        """
        drift = context.drift or {}
        perf = context.performance or {}
        any_drift = drift.get("any_drift", False)
        max_sev = drift.get("max_severity", "none")
        n_drifted = drift.get("n_drifted", 0)
        acc_drop = perf.get("accuracy_drop")
        degraded = perf.get("degraded", False)

        drifted_names = [f["feature"] for f in drift.get("features", []) if f.get("drifted")]
        cause_bits = []
        if drifted_names:
            cause_bits.append(
                f"{n_drifted} feature(s) drifted ({', '.join(drifted_names)}), "
                f"max severity {max_sev}"
            )
        if acc_drop is not None:
            cause_bits.append(f"accuracy dropped {acc_drop:.3f} from baseline")
        elif perf.get("confidence_drop"):
            cause_bits.append(f"confidence dropped {perf.get('confidence_drop'):.3f}")
        root_cause = "; ".join(cause_bits) if cause_bits else "No significant anomaly detected."

        # --- the policy ---
        if not any_drift and not degraded:
            return Diagnosis(
                root_cause="Inputs and outputs are within normal bounds.",
                severity="low",
                action="IGNORE",
                reasoning="No drift and no performance degradation; no action warranted.",
                confidence=0.9,
                source="rule-based",
            )

        # We have proof of harm: accuracy actually fell and inputs drifted.
        if acc_drop is not None and acc_drop >= 0.05 and any_drift:
            severity = "critical" if acc_drop >= 0.15 else "high"
            return Diagnosis(
                root_cause=root_cause,
                severity=severity,
                action="RETRAIN",
                reasoning=(
                    "Input drift coincides with a measured accuracy drop, so the "
                    "model is genuinely degraded on the current regime. Retrain on "
                    "recent data; the quality gate will block a worse candidate."
                ),
                confidence=0.8,
                source="rule-based",
            )

        # Strong drift but no label-confirmed accuracy loss yet: don't auto-act.
        if max_sev == "high" or n_drifted >= 3:
            return Diagnosis(
                root_cause=root_cause,
                severity="high",
                action="INVESTIGATE",
                reasoning=(
                    "Substantial input drift, but ground-truth accuracy loss is not "
                    "yet confirmed. Escalate to a human before an automated change "
                    "to avoid retraining on a transient shift."
                ),
                confidence=0.65,
                source="rule-based",
            )

        # Minor drift / only a confidence dip.
        return Diagnosis(
            root_cause=root_cause,
            severity="medium" if degraded else "low",
            action="INVESTIGATE" if degraded else "IGNORE",
            reasoning=(
                "Drift is mild and accuracy is not confirmed to have dropped. "
                "Monitor closely; act only if degradation is confirmed."
            ),
            confidence=0.6,
            source="rule-based",
        )
