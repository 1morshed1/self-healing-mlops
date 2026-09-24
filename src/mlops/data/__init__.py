"""Data layer: the schema, synthetic dataset generation, drift simulation, and
feature engineering.

We use a **fraud-detection** style problem because it is intuitive, tabular
(so drift is easy to reason about per feature), and has genuine signal a model
can learn. See ``schema.py`` for the feature contract shared across the whole
pipeline (training, serving, and monitoring must all agree on it).
"""

from mlops.data.schema import FEATURE_NAMES, TARGET_NAME, feature_frame_columns

__all__ = ["FEATURE_NAMES", "TARGET_NAME", "feature_frame_columns"]
