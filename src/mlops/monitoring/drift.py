"""Data-drift detection: PSI + Kolmogorov-Smirnov, per feature.

THE CORE QUESTION
-----------------
"Does today's production data look like the data the model was trained on?"
If not, the model is extrapolating into territory it never learned, and its
predictions can't be trusted even though nothing has "crashed."

We use TWO complementary statistical tests because each catches things the
other misses:

1. **PSI (Population Stability Index)** — an effect-size measure. It bins the
   feature and asks "how much probability mass moved between bins?" It gives an
   interpretable magnitude with industry-standard thresholds:
       PSI < 0.1   → no significant shift
       0.1–0.2     → moderate shift, investigate
       > 0.2       → major shift, likely action needed
   PSI does NOT depend on sample size, which makes it stable and comparable
   across features and over time (great for a dashboard heatmap).

2. **KS test (Kolmogorov-Smirnov)** — a hypothesis test. It measures the
   largest gap between the two cumulative distributions and returns a p-value.
   A tiny p-value means "these samples almost certainly come from different
   distributions." KS is sensitive and catches subtle shape changes, but its
   p-value shrinks as samples grow, so on big data it flags trivially. That's
   exactly why we pair it with PSI's effect size — significance AND magnitude.

A feature is flagged as drifted if EITHER test crosses its threshold. Requiring
both would miss real drift; requiring either is the sensitive, safe default for
a system whose job is to notice problems early.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# Below this many distinct values we treat a feature as categorical/discrete
# (e.g. device_change is 0/1) and compare category proportions instead of
# quantile-binning it, which would be degenerate.
_DISCRETE_MAX_CARDINALITY = 12


@dataclass
class DriftResult:
    """Drift statistics for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool
    severity: str  # "none" | "low" | "medium" | "high"
    # Small distribution summary, useful context for the LLM agent's reasoning.
    reference_mean: float = 0.0
    production_mean: float = 0.0

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "drifted": self.drifted,
            "severity": self.severity,
            "reference_mean": round(self.reference_mean, 4),
            "production_mean": round(self.production_mean, 4),
        }


@dataclass
class DriftReport:
    """Aggregated drift across all features for one production window."""

    results: list[DriftResult] = field(default_factory=list)

    @property
    def any_drift(self) -> bool:
        """True if at least one feature drifted — the trigger for the agent."""
        return any(r.drifted for r in self.results)

    @property
    def drifted_features(self) -> list[DriftResult]:
        return [r for r in self.results if r.drifted]

    @property
    def max_severity(self) -> str:
        """The worst severity across features (drives alert urgency)."""
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        if not self.results:
            return "none"
        return max((r.severity for r in self.results), key=lambda s: order[s])

    def as_dict(self) -> dict:
        return {
            "any_drift": self.any_drift,
            "max_severity": self.max_severity,
            "n_drifted": len(self.drifted_features),
            "n_features": len(self.results),
            "features": [r.as_dict() for r in self.results],
        }

    def summary_line(self) -> str:
        """One-line human summary for logs."""
        if not self.any_drift:
            return f"No drift ({len(self.results)} features checked)."
        names = ", ".join(f"{r.feature}(psi={r.psi:.2f})" for r in self.drifted_features)
        return f"DRIFT [{self.max_severity}] in {len(self.drifted_features)} feature(s): {names}"


def calculate_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a production sample.

    Method:
    1. Define bin edges from the REFERENCE data's quantiles (equal-frequency
       bins). Quantile bins are more robust than equal-width bins when the data
       is skewed — every reference bin holds ~the same count, so the comparison
       isn't dominated by one fat bin.
    2. Compute the proportion of each sample falling in each bin.
    3. PSI = Σ (prod%% − ref%%) · ln(prod%% / ref%%)

    A tiny epsilon is added to every proportion so empty bins don't produce
    ln(0) = −inf. PSI is symmetric-ish and always ≥ 0; larger = more shift.
    """
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)

    # Quantile edges from the reference. np.unique guards against duplicate
    # edges when the data is highly concentrated (which would make bins vanish).
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        # Not enough distinct values to bin meaningfully (near-constant feature).
        return 0.0
    # Make the outer edges infinite so production values outside the reference
    # range still land in the first/last bin instead of being dropped.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    prod_counts, _ = np.histogram(production, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    prod_pct = prod_counts / prod_counts.sum() + eps

    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def _psi_categorical(reference: np.ndarray, production: np.ndarray) -> float:
    """PSI for a discrete feature, comparing per-category proportions."""
    categories = np.unique(np.concatenate([reference, production]))
    eps = 1e-6
    psi = 0.0
    for cat in categories:
        ref_pct = np.mean(reference == cat) + eps
        prod_pct = np.mean(production == cat) + eps
        psi += (prod_pct - ref_pct) * np.log(prod_pct / ref_pct)
    return float(psi)


class DriftDetector:
    """Compares a production feature window against a fixed reference sample.

    Construct it once with the reference (training) distribution and thresholds,
    then call ``check(production_df)`` for each new window of live data.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        *,
        features: list[str],
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.01,
    ):
        self.reference = reference
        self.features = features
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold

    @staticmethod
    def _severity(psi: float) -> str:
        """Map a PSI value to the industry-standard severity bands."""
        if psi >= 0.25:
            return "high"
        if psi >= 0.1:
            return "medium"
        if psi >= 0.05:
            return "low"
        return "none"

    def _check_feature(self, feature: str, prod: pd.Series) -> DriftResult:
        ref_values = self.reference[feature].to_numpy(dtype=float)
        prod_values = prod.to_numpy(dtype=float)

        is_discrete = self.reference[feature].nunique() <= _DISCRETE_MAX_CARDINALITY
        psi = (
            _psi_categorical(ref_values, prod_values)
            if is_discrete
            else calculate_psi(ref_values, prod_values)
        )

        # KS test compares the full empirical distributions. It's meaningful for
        # continuous features; for a 0/1 discrete feature it still runs but PSI
        # is the more honest signal, so we lean on PSI's severity there.
        ks_stat, ks_pvalue = stats.ks_2samp(ref_values, prod_values)

        drifted = psi >= self.psi_threshold or ks_pvalue < self.ks_pvalue_threshold
        severity = self._severity(psi)
        # If KS screams drift but PSI is mild, still register at least "low" so
        # the feature isn't reported as perfectly stable.
        if drifted and severity == "none":
            severity = "low"

        return DriftResult(
            feature=feature,
            psi=psi,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            drifted=bool(drifted),
            severity=severity,
            reference_mean=float(np.mean(ref_values)),
            production_mean=float(np.mean(prod_values)),
        )

    def check(self, production: pd.DataFrame) -> DriftReport:
        """Run drift tests for every configured feature over ``production``."""
        results = [self._check_feature(f, production[f]) for f in self.features]
        return DriftReport(results=results)
