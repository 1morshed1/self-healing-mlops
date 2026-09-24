"""Remediation: safely execute the action a Diagnosis recommends.

This is the ONLY place in the system that performs a dangerous, state-changing
action (retrain-and-promote, or roll back the production model). Concentrating
side effects here — instead of letting the agent act directly — makes the blast
radius small and auditable.

SAFETY FEATURES
---------------
* **Dry-run by default off, but gated.** ``auto_remediate`` must be explicitly
  enabled; otherwise the remediator logs what it *would* do and stops. This lets
  you run the whole loop in "advisory" mode in production until you trust it.
* **The quality gate still applies.** RETRAIN calls the normal training pipeline,
  which refuses to promote a worse model. So even an over-eager RETRAIN can't
  degrade the service.
* **Every action returns a structured RemediationResult** for logging/alerting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mlops.config import Settings, get_settings
from mlops.data.generate import DriftSpec
from mlops.diagnostics.agent import Diagnosis
from mlops.training import registry

logger = logging.getLogger(__name__)


@dataclass
class RemediationResult:
    """What actually happened when we tried to act on a diagnosis."""

    action: str
    executed: bool          # did we perform a real side effect?
    detail: str             # human-readable outcome
    new_model_version: int | None = None

    def summary_line(self) -> str:
        state = "EXECUTED" if self.executed else "SKIPPED"
        return f"{state} {self.action}: {self.detail}"


class Remediator:
    """Maps a Diagnosis.action to a safe, logged side effect."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        auto_remediate: bool = False,
        retrain_drift: DriftSpec | None = None,
        retrain_seed: int | None = None,
    ):
        self.settings = settings or get_settings()
        # When False, RETRAIN/ROLLBACK are logged but NOT executed (advisory
        # mode). Flip to True to let the system truly heal itself.
        self.auto_remediate = auto_remediate
        # If the loop is simulating "retrain on recent, drifted data", pass the
        # same DriftSpec here so the retraining data reflects the new regime.
        self.retrain_drift = retrain_drift
        self.retrain_seed = retrain_seed

    def remediate(self, diagnosis: Diagnosis) -> RemediationResult:
        """Dispatch on the diagnosis action."""
        action = diagnosis.action
        if action == "IGNORE":
            return RemediationResult(action, False, "Within bounds — nothing to do.")
        if action == "INVESTIGATE":
            # Investigation is a human task; we surface it loudly but don't act.
            return RemediationResult(
                action, False, "Escalated for human investigation (no automated change)."
            )
        if action == "RETRAIN":
            return self._retrain()
        if action == "ROLLBACK":
            return self._rollback()
        return RemediationResult(action, False, f"Unknown action {action!r}; no-op.")

    # -- actions ------------------------------------------------------------
    def _retrain(self) -> RemediationResult:
        if not self.auto_remediate:
            return RemediationResult(
                "RETRAIN", False,
                "auto_remediate is off — would retrain on recent data. (advisory mode)",
            )
        # Imported here to avoid a heavy import at module load and to keep the
        # dependency direction clean (remediation -> training, not vice versa).
        from mlops.training.train import train_model

        logger.info("Remediation: retraining on recent data...")
        outcome = train_model(drift=self.retrain_drift, data_seed=self.retrain_seed)
        if outcome.promoted:
            return RemediationResult(
                "RETRAIN", True,
                f"Retrained and promoted v{outcome.registered_version}. "
                f"{outcome.promotion_note}",
                new_model_version=outcome.registered_version,
            )
        return RemediationResult(
            "RETRAIN", True,
            f"Retrained v{outcome.registered_version} but NOT promoted "
            f"(quality gate protected production). {outcome.promotion_note}",
            new_model_version=outcome.registered_version,
        )

    def _rollback(self) -> RemediationResult:
        if not self.auto_remediate:
            return RemediationResult(
                "ROLLBACK", False,
                "auto_remediate is off — would roll back to the previous version. (advisory)",
            )
        old, new = registry.rollback(self.settings.model_name)
        if new is None:
            return RemediationResult(
                "ROLLBACK", False,
                f"No earlier version to roll back to (current v{old}). Left unchanged.",
            )
        return RemediationResult(
            "ROLLBACK", True,
            f"Rolled production back from v{old} to v{new}.",
            new_model_version=new,
        )
