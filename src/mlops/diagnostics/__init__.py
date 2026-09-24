"""Diagnostics: the LLM-powered root-cause agent — this project's unique angle.

When monitoring raises a drift/degradation alert, raw numbers ("feature X PSI =
0.34") are not something a human can act on quickly. The ``DiagnosticAgent``
turns that context into:

* a plain-language **root cause**,
* a **severity** (low/medium/high/critical),
* a recommended **action** (RETRAIN / ROLLBACK / INVESTIGATE / IGNORE), and
* the **reasoning** behind it.

It calls Claude when an API key is configured, and **transparently falls back
to a deterministic rule-based diagnosis** when it isn't — so the whole system
works offline and in CI. ``Remediator`` then safely executes the chosen action.
"""

from mlops.diagnostics.agent import AlertContext, Diagnosis, DiagnosticAgent
from mlops.diagnostics.remediation import Remediator

__all__ = ["AlertContext", "Diagnosis", "DiagnosticAgent", "Remediator"]
