"""The self-healing control loop — the piece that ties every layer together.

ONE CYCLE =
    1. Load the production model + its reference (training) distribution.
    2. Build a "production window" of recent data — either from the real
       prediction log, or (for the demo) freshly generated with injected drift.
    3. Run drift detection (PSI/KS) and performance monitoring.
    4. If anything is off, hand the full context to the DiagnosticAgent.
    5. Execute the recommended action via the Remediator (advisory or auto).
    6. Notify a human with the diagnosis + what was done.

This object is intentionally reusable: the demo CLI (`scripts/self_heal.py`) and
the Airflow monitoring DAG both just call ``SelfHealingMonitor.run_cycle()``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlops.config import Settings, get_settings
from mlops.data.generate import DriftSpec, generate_dataset
from mlops.data.schema import FEATURE_NAMES, TARGET_NAME
from mlops.diagnostics.agent import AlertContext, Diagnosis, DiagnosticAgent
from mlops.diagnostics.remediation import RemediationResult, Remediator
from mlops.monitoring.drift import DriftDetector
from mlops.monitoring.performance import PerformanceMonitor, _confidence
from mlops.orchestration import notify
from mlops.training import registry

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """The full record of one monitoring cycle — returned for logging/tests."""

    drift: dict
    performance: dict | None
    diagnosis: Diagnosis | None
    remediation: RemediationResult | None
    acted: bool

    def summary(self) -> str:
        lines = [f"Drift: {self.drift.get('max_severity')} "
                 f"({self.drift.get('n_drifted')}/{self.drift.get('n_features')} features)"]
        if self.diagnosis:
            lines.append(f"Diagnosis: {self.diagnosis.summary_line()}")
        if self.remediation:
            lines.append(f"Remediation: {self.remediation.summary_line()}")
        if self.diagnosis is None:
            lines.append("No anomaly — model healthy.")
        return "\n".join(lines)


class SelfHealingMonitor:
    """Runs monitor -> diagnose -> remediate -> notify for one production window."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        auto_remediate: bool = False,
        baseline_accuracy: float = 0.90,
    ):
        self.settings = settings or get_settings()
        self.auto_remediate = auto_remediate
        self.baseline_accuracy = baseline_accuracy
        self.agent = DiagnosticAgent(self.settings)

        # Load the production model + the reference distribution it was trained
        # on. These are fixed for the life of the monitor instance.
        registry.configure_mlflow()
        self.model_version = registry.get_alias_version(
            self.settings.model_name, registry.PRODUCTION_ALIAS
        )
        if self.model_version is None:
            raise RuntimeError("No production model registered. Train one first.")
        self.model = registry.load_model(self.settings.model_name)
        self.reference = registry.load_reference_sample(self.settings.model_name)

        self.detector = DriftDetector(
            self.reference,
            features=FEATURE_NAMES,
            psi_threshold=self.settings.psi_threshold,
            ks_pvalue_threshold=self.settings.ks_pvalue_threshold,
        )
        # Baseline confidence = the model's decisiveness on its own reference set.
        # Pass a plain array (the model was fitted on arrays) to avoid sklearn's
        # "fitted without feature names" warning and keep train/serve identical.
        ref_matrix = self.reference[FEATURE_NAMES].to_numpy(dtype=float)
        base_conf = float(np.mean(_confidence(self.model.predict_proba(ref_matrix)[:, 1])))
        self.perf_monitor = PerformanceMonitor(
            baseline_confidence=base_conf,
            baseline_accuracy=self.baseline_accuracy,
        )

    # -- building the production window -------------------------------------
    def window_from_log(self, limit: int = 2000) -> pd.DataFrame | None:
        """Load the most recent predictions from the serving log, if available.

        This is the REAL path: the window is whatever the live service actually
        saw. Returns None if the log is missing/empty (then callers fall back to
        the simulated window for the demo).
        """
        path = self.settings.prediction_log
        if not path.exists():
            return None
        rows = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            return None
        frame = pd.DataFrame(rows[-limit:])
        return frame

    def window_from_simulation(
        self, drift: DriftSpec, n: int = 1500, seed: int = 0
    ) -> pd.DataFrame:
        """Generate a labelled production window with injected drift (demo path)."""
        return generate_dataset(n, seed=seed, drift=drift)

    # -- the cycle ----------------------------------------------------------
    def run_cycle(self, window: pd.DataFrame) -> CycleResult:
        """Run drift + performance + diagnosis + remediation on one window."""
        # 1. Drift on the input features.
        drift_report = self.detector.check(window[FEATURE_NAMES])
        drift = drift_report.as_dict()
        logger.info(drift_report.summary_line())

        # 2. Performance: predicted probabilities, and accuracy if labels exist.
        proba = self.model.predict_proba(window[FEATURE_NAMES].to_numpy(dtype=float))[:, 1]
        y_true = window[TARGET_NAME].to_numpy() if TARGET_NAME in window.columns else None
        perf_report = self.perf_monitor.evaluate(proba, y_true=y_true)
        perf = perf_report.as_dict()

        # 3. If nothing is wrong, stop early — the healthy fast path.
        if not drift_report.any_drift and not perf_report.degraded:
            notify.send_alert(
                "MLOps monitor: healthy",
                f"Model v{self.model_version} nominal. {drift_report.summary_line()}",
            )
            return CycleResult(drift, perf, None, None, acted=False)

        # 4. Diagnose.
        context = AlertContext(
            model_name=self.settings.model_name,
            model_version=self.model_version,
            drift=drift,
            performance=perf,
            window_size=len(window),
        )
        diagnosis = self.agent.diagnose(context)
        logger.info(diagnosis.summary_line())

        # 5. Remediate. If we're simulating drift, retraining should use fresh
        #    (differently-seeded) data to mimic "recent production data".
        remediator = Remediator(
            self.settings,
            auto_remediate=self.auto_remediate,
            retrain_seed=None,
        )
        remediation = remediator.remediate(diagnosis)
        logger.info(remediation.summary_line())

        # 6. Always notify.
        notify.send_alert(
            f"MLOps monitor: {diagnosis.action} ({diagnosis.severity})",
            f"Model v{self.model_version}\n"
            f"Root cause: {diagnosis.root_cause}\n"
            f"Reasoning: {diagnosis.reasoning}\n"
            f"Action taken: {remediation.summary_line()}",
        )

        return CycleResult(drift, perf, diagnosis, remediation, acted=remediation.executed)
