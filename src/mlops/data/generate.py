"""Synthetic dataset generation + realistic drift simulation.

WHY SYNTHETIC DATA
------------------
To demonstrate a *self-healing* system we need to be able to summon drift on
command — we can't wait weeks for a real distribution to shift. So we generate
data from a known process, which gives us two superpowers:

1. There is genuine, learnable signal (fraud probability is a real function of
   the features), so the trained model is meaningful rather than random.
2. We can produce a "drifted" version whose feature distributions have shifted
   in specific, controllable ways, then check that our monitoring actually
   catches it. This is effectively an integration test for the whole loop.

THE DATA-GENERATING PROCESS
---------------------------
Each row is a transaction. Fraud is more likely when the amount is unusually
high relative to the account's average, the account is young, there are many
recent transactions, the country is risky, or the device changed. We push
those signals through a logistic function to get a fraud probability, then
sample the label. This is a classic "latent linear score → sigmoid → Bernoulli"
generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlops.data.schema import FEATURE_NAMES, TARGET_NAME


@dataclass
class DriftSpec:
    """Describes how to perturb the *production* distribution relative to the
    reference (training) distribution.

    Each field nudges one feature. Defaults are "no drift". The self-healing
    demo dials these up to simulate, e.g., inflation (amounts creep up) plus a
    device-spoofing fraud wave (more device changes), which is exactly the kind
    of correlated shift that degrades a fraud model in the real world.
    """

    amount_shift: float = 0.0        # add this to the mean transaction amount
    amount_scale: float = 1.0        # multiply amount spread by this
    country_risk_shift: float = 0.0  # add this to mean country risk (clamped 0-1)
    device_change_rate_delta: float = 0.0  # add this to P(device changed)
    n_txn_shift: float = 0.0         # add this to mean transactions/24h

    def is_noop(self) -> bool:
        """True when this spec describes no drift at all."""
        return (
            self.amount_shift == 0.0
            and self.amount_scale == 1.0
            and self.country_risk_shift == 0.0
            and self.device_change_rate_delta == 0.0
            and self.n_txn_shift == 0.0
        )


# Baseline distribution parameters for the *reference* (training) population.
# Keeping these as module constants makes the "normal world" explicit and lets
# drift be expressed as a delta from it.
_BASE = {
    "amount_mean": 90.0,
    "amount_sd": 40.0,
    "age_days_mean": 400.0,
    "n_txn_24h_lambda": 3.0,      # Poisson rate
    "country_risk_mean": 0.15,
    "device_change_rate": 0.08,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function, maps any real number to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(
    n_rows: int = 5000,
    *,
    seed: int = 42,
    drift: DriftSpec | None = None,
) -> pd.DataFrame:
    """Generate a labelled transactions dataset.

    Parameters
    ----------
    n_rows:
        Number of transactions to synthesize.
    seed:
        RNG seed. Fixed by default so training is reproducible; the drift
        simulator varies it so production windows differ run to run.
    drift:
        If given, shifts the feature distributions to simulate drift. If None
        (default) you get the clean reference distribution.

    Returns
    -------
    A DataFrame with the feature columns plus the ``is_fraud`` target column.
    """
    rng = np.random.default_rng(seed)
    drift = drift or DriftSpec()

    n = int(n_rows)

    # --- Draw each feature from its (possibly drifted) distribution ---------
    # amount: normal, truncated at 0. Drift can shift the mean (inflation) and
    # widen the spread (more extreme transactions).
    amount = rng.normal(
        _BASE["amount_mean"] + drift.amount_shift,
        _BASE["amount_sd"] * drift.amount_scale,
        n,
    ).clip(min=0.0)

    # age_days: exponential-ish; most accounts young, a long tail of old ones.
    age_days = rng.gamma(shape=2.0, scale=_BASE["age_days_mean"] / 2.0, size=n).clip(min=1)

    # n_txn_24h: Poisson count. Drift can raise the rate (a bot burst).
    n_txn_24h = rng.poisson(max(0.1, _BASE["n_txn_24h_lambda"] + drift.n_txn_shift), n)

    # avg_txn_amt: correlated with amount but noisier — an account's baseline.
    avg_txn_amt = (amount * 0.7 + rng.normal(_BASE["amount_mean"], 20.0, n)).clip(min=1.0)

    # country_risk: beta-like score in [0, 1]; drift shifts the mean upward.
    country_risk = (
        rng.beta(2.0, 10.0, n) + drift.country_risk_shift
    ).clip(0.0, 1.0)

    # device_change: Bernoulli; drift raises the rate (device-spoofing wave).
    p_device = np.clip(_BASE["device_change_rate"] + drift.device_change_rate_delta, 0.0, 1.0)
    device_change = rng.binomial(1, p_device, n)

    # --- Latent fraud score: a linear combination of standardized signals ---
    # The coefficients encode the "true" relationship the model must learn.
    # We standardize each term roughly so no single feature dominates by scale.
    amount_ratio = amount / (avg_txn_amt + 1.0)  # spending far above your norm
    score = (
        -3.2                                   # intercept: fraud is rare
        + 1.1 * (amount_ratio - 1.0)           # spending well above baseline
        + 0.9 * (n_txn_24h / 5.0)              # rapid-fire transactions
        + 2.0 * country_risk                   # risky geography
        + 1.3 * device_change                  # unfamiliar device
        - 0.6 * (age_days / 400.0)             # older accounts safer
    )
    fraud_prob = _sigmoid(score)
    is_fraud = rng.binomial(1, fraud_prob)

    frame = pd.DataFrame(
        {
            "amount": amount,
            "age_days": age_days,
            "n_txn_24h": n_txn_24h,
            "avg_txn_amt": avg_txn_amt,
            "country_risk": country_risk,
            "device_change": device_change,
            TARGET_NAME: is_fraud,
        }
    )
    # Guarantee the column order matches the schema (features first, target last).
    return frame[[*FEATURE_NAMES, TARGET_NAME]]


def make_drift_spec(severity: str = "medium") -> DriftSpec:
    """Return a ready-made DriftSpec for the self-healing demo.

    Named severities keep the demo scripts readable ("simulate a medium drift")
    and give the LLM agent a spectrum of situations to reason about.
    """
    presets = {
        "none": DriftSpec(),
        "low": DriftSpec(amount_shift=8.0, device_change_rate_delta=0.03),
        "medium": DriftSpec(
            amount_shift=35.0,
            amount_scale=1.4,
            country_risk_shift=0.08,
            device_change_rate_delta=0.10,
        ),
        "high": DriftSpec(
            amount_shift=80.0,
            amount_scale=1.9,
            country_risk_shift=0.20,
            device_change_rate_delta=0.22,
            n_txn_shift=2.5,
        ),
    }
    if severity not in presets:
        raise ValueError(f"Unknown severity {severity!r}; choose from {sorted(presets)}")
    return presets[severity]
