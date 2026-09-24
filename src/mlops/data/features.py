"""Feature engineering: turn raw inputs into the exact matrix the model expects.

WHY A DEDICATED MODULE (and why it's so small here)
---------------------------------------------------
Feature engineering is the #1 place training/serving skew creeps in. The safe
pattern is: *the same function transforms data at training time and at serving
time.* Training calls it on the dataset; the API calls it on each request. If
they ever diverge, the model sees inputs it never trained on.

In this project the raw features are already model-ready (all numeric), so the
"engineering" is mostly about **enforcing the schema**: correct columns, correct
order, correct dtypes, no missing values. That sounds trivial until a caller
sends `{"amount": "120.5"}` (a string) or omits a field — then you want a clear
error at the boundary, not a confusing failure deep inside the model.

If you later add derived features (ratios, log transforms, encodings), they go
HERE and are automatically shared by training and serving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops.data.schema import FEATURE_NAMES


class FeatureValidationError(ValueError):
    """Raised when incoming raw features don't satisfy the schema contract."""


def build_feature_frame(records: list[dict] | dict) -> pd.DataFrame:
    """Validate + assemble raw feature dict(s) into a canonical DataFrame.

    Accepts either a single record (dict) or a list of records. This is the
    single entry point used by BOTH the serving API (one request) and any batch
    scoring path.

    Guarantees on the returned frame:
    * columns are exactly ``FEATURE_NAMES`` in canonical order,
    * all values are numeric floats,
    * no NaNs.

    Raises ``FeatureValidationError`` with an actionable message otherwise.
    """
    if isinstance(records, dict):
        records = [records]
    if not records:
        raise FeatureValidationError("No records supplied.")

    rows = []
    for i, record in enumerate(records):
        missing = [f for f in FEATURE_NAMES if f not in record]
        if missing:
            raise FeatureValidationError(
                f"Record {i} is missing required feature(s): {missing}. "
                f"Expected all of: {FEATURE_NAMES}"
            )
        row = {}
        for feature in FEATURE_NAMES:
            value = record[feature]
            try:
                row[feature] = float(value)
            except (TypeError, ValueError) as exc:
                raise FeatureValidationError(
                    f"Record {i} feature {feature!r} = {value!r} is not numeric."
                ) from exc
        rows.append(row)

    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)

    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        raise FeatureValidationError(f"NaN values in feature(s): {bad}")

    return frame


def frame_to_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the feature columns as a 2-D float array in canonical order.

    scikit-learn wants a plain array. Selecting ``FEATURE_NAMES`` explicitly
    (rather than ``frame.values``) means an unordered or extra-columned frame
    still produces model input in the exact order the model was trained on.
    """
    return frame[FEATURE_NAMES].to_numpy(dtype=float)
