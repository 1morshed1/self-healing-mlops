"""The feature contract for the whole pipeline.

WHY A SHARED SCHEMA MODULE
--------------------------
Training, serving, and monitoring must all agree on *exactly* which features
exist, in what order, and what they mean. If they drift apart you get the
single most common production ML bug: **training/serving skew**, where the
model is fed columns in a different order or with different preprocessing than
it was trained on, and predictions silently become garbage.

Defining the schema in one place and importing it everywhere makes that class
of bug structurally impossible: there is exactly one list of feature names.

The problem: predict whether a financial transaction is fraudulent.
"""

from __future__ import annotations

# The raw input features the API accepts and the model consumes, in a fixed
# canonical order. ALWAYS build model input in this order (see features.py).
FEATURE_NAMES: list[str] = [
    "amount",          # transaction amount in currency units (float, >= 0)
    "age_days",        # age of the account in days (int, >= 0)
    "n_txn_24h",       # number of transactions by this account in last 24h (int)
    "avg_txn_amt",     # account's historical average transaction amount (float)
    "country_risk",    # risk score of the transaction's country, 0.0-1.0 (float)
    "device_change",   # 1 if the device differs from the account's usual, else 0
]

# The label the model predicts.
TARGET_NAME: str = "is_fraud"

# Human-readable descriptions, reused by the LLM diagnostic agent so its
# root-cause analysis can talk about features in business terms rather than
# opaque column names.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "amount": "Transaction amount in currency units.",
    "age_days": "Age of the customer account in days (older = more trusted).",
    "n_txn_24h": "Number of transactions from this account in the last 24 hours.",
    "avg_txn_amt": "The account's historical average transaction amount.",
    "country_risk": "Risk score (0-1) of the country the transaction originates from.",
    "device_change": "1 if the transaction came from an unfamiliar device, else 0.",
}


def feature_frame_columns() -> list[str]:
    """Return the canonical feature column order.

    A tiny helper, but calling this instead of re-typing the list at every use
    site means there is genuinely one source of truth.
    """
    return list(FEATURE_NAMES)
