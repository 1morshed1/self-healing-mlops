"""Prompt construction for the diagnostic agent.

WHY PROMPTS LIVE IN THEIR OWN MODULE
------------------------------------
The prompt *is* the policy. "When should we auto-retrain vs. page a human?" is
a business decision, and here it's encoded as English in a prompt rather than
buried in if/else branches. Keeping prompts in one file makes that policy easy
to read, review, and version — you can diff a prompt change like any code
change. This is a core "LLM engineering" habit worth demonstrating.

We ask for **strict JSON** so the response is machine-parseable and can drive
automated remediation. The schema is described explicitly and we show the exact
allowed values for `action` and `severity`.
"""

from __future__ import annotations

import json

from mlops.data.schema import FEATURE_DESCRIPTIONS

# The four actions the agent may recommend. Kept as a constant so the prompt,
# the parser, and the remediator can't disagree about the vocabulary.
ALLOWED_ACTIONS = ["RETRAIN", "ROLLBACK", "INVESTIGATE", "IGNORE"]
ALLOWED_SEVERITIES = ["low", "medium", "high", "critical"]


SYSTEM_PROMPT = f"""You are an MLOps diagnostic agent responsible for keeping a \
production fraud-detection model healthy. You receive monitoring context when a \
drift or performance alert fires, and you decide what to do.

You must reason about ROOT CAUSE, not just restate the numbers. Consider:
- Which input features drifted, by how much (PSI), and what that implies about \
the real world (e.g. a jump in transaction amounts may be inflation or a new \
product; a spike in device changes may be an account-takeover fraud wave).
- Whether model confidence and accuracy actually degraded, or only the inputs \
moved (input drift without accuracy loss can be benign).
- Severity and the cost of each action. Retraining is powerful but spends \
compute and risks fitting to a transient blip. Rollback is safe but only helps \
if a previous model handled this regime better. Investigate keeps a human in \
the loop for ambiguous or novel situations.

Choose exactly ONE action from: {ALLOWED_ACTIONS}
  - RETRAIN: input drift is real and broad and accuracy has degraded — refit on \
recent data (a quality gate will still block a worse model).
  - ROLLBACK: the current model version is clearly worse than its predecessor \
for this data regime; revert while investigating.
  - INVESTIGATE: the situation is ambiguous, novel, or severe enough to need a \
human before any automated change.
  - IGNORE: drift is minor and within acceptable bounds; no action needed.

Respond with ONLY a JSON object, no prose before or after, of the form:
{{
  "root_cause": "<2-3 sentence plain-language explanation>",
  "severity": one of {ALLOWED_SEVERITIES},
  "action": one of {ALLOWED_ACTIONS},
  "reasoning": "<why this action, 2-3 sentences>",
  "confidence": <float 0.0-1.0>
}}"""


def _format_drift(drift: dict) -> str:
    """Render the drift report as a compact, readable block for the prompt."""
    if not drift or not drift.get("features"):
        return "No drift report available."
    lines = [
        f"Overall: any_drift={drift.get('any_drift')}, "
        f"max_severity={drift.get('max_severity')}, "
        f"{drift.get('n_drifted')}/{drift.get('n_features')} features drifted."
    ]
    for feat in drift["features"]:
        desc = FEATURE_DESCRIPTIONS.get(feat["feature"], "")
        flag = "  <-- DRIFTED" if feat["drifted"] else ""
        lines.append(
            f"  - {feat['feature']}: PSI={feat['psi']}, KS_p={feat['ks_pvalue']}, "
            f"ref_mean={feat['reference_mean']} -> prod_mean={feat['production_mean']} "
            f"[{feat['severity']}]{flag}  ({desc})"
        )
    return "\n".join(lines)


def _format_performance(perf: dict | None) -> str:
    """Render the performance report block for the prompt."""
    if not perf:
        return "No performance report available (labels may not have arrived yet)."
    lines = [
        f"Mean confidence: {perf.get('mean_confidence')} "
        f"(baseline {perf.get('baseline_confidence')}, "
        f"drop {perf.get('confidence_drop')}).",
        f"Low-confidence rate: {perf.get('low_confidence_rate')}.",
    ]
    if perf.get("accuracy") is not None:
        lines.append(
            f"Accuracy: {perf.get('accuracy')} "
            f"(baseline {perf.get('baseline_accuracy')}, "
            f"drop {perf.get('accuracy_drop')}); recall: {perf.get('recall')}."
        )
    lines.append(f"Degraded: {perf.get('degraded')} — {perf.get('reason')}.")
    return "\n".join(lines)


def build_user_prompt(context: AlertContext) -> str:  # noqa: F821 (type in agent.py)
    """Assemble the full context block the agent reasons over."""
    return f"""A monitoring alert has fired for a production model.

Model: {context.model_name} (version {context.model_version})
Time: {context.timestamp}
Production window size: {context.window_size} predictions

=== DATA DRIFT ===
{_format_drift(context.drift)}

=== MODEL PERFORMANCE ===
{_format_performance(context.performance)}

Diagnose the root cause and recommend an action as specified. \
Return only the JSON object."""


def extract_json(text: str) -> dict:
    """Best-effort extraction of the first JSON object from an LLM response.

    Models usually return clean JSON when asked, but occasionally wrap it in
    prose or a ```json fence. We slice from the first '{' to the last '}' so the
    parser is resilient to that without a heavy dependency. Raises
    ``json.JSONDecodeError`` / ``ValueError`` if nothing parseable is found, and
    the caller treats that as "LLM failed -> use rule-based fallback".
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start : end + 1])
