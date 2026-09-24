# 05 — The LLM diagnostic agent (the unique angle)

This is what makes the project more than a generic MLOps tutorial. When drift is
detected, instead of firing a raw "feature X PSI = 0.34" alert, we feed the full
monitoring context to an LLM agent that produces a **human-readable root cause**
and a **recommended action**, then a separate component safely executes it.

## Three design decisions worth defending in an interview

### 1. Decide vs. act are separated

`DiagnosticAgent.diagnose()` is a **pure function**: context in, `Diagnosis`
out. It never retrains or rolls back anything. `Remediator.remediate()` is the
*only* place that performs a side effect. Why split them?

- The agent becomes trivially testable (no mocking of training infra).
- You can run in **advisory mode** — log the recommendation, let a human act —
  which is how you'd actually roll this out in production before trusting it.
- All the dangerous, state-changing code lives in one small, auditable file.

This "planner/executor" split is a standard, respected agent-design pattern.

### 2. The prompt *is* the policy

"When do we auto-retrain vs. page a human?" is a business decision. We encode it
as English in `prompts.py`, not as buried if/else. That makes the policy
reviewable and versionable — a prompt change diffs like a code change. The system
prompt explicitly enumerates the four actions and *when each applies*, and asks
for strict JSON so the output can drive automation.

### 3. Graceful degradation to rules

No `ANTHROPIC_API_KEY`? The API down? `diagnose()` **never fails** — it falls
back to a deterministic rule-based policy that mirrors the prompt's intent, and
records `source="rule-based"` so logs stay honest. This is why the whole demo,
the tests, and CI run offline. Two lessons here:
- An LLM in a control loop must have a **safe fallback**; you can't let a
  network blip break remediation.
- Writing the rule-based version *first* forces you to actually specify the
  policy — and it doubles as the reference behaviour the LLM should emulate.

We also **validate and clamp** the LLM's output: an out-of-vocabulary `action`
becomes `INVESTIGATE` (the safe, human-in-the-loop choice), because this value
triggers a real side effect and we don't blindly trust free-text generation.

## The safety rail we actually watched fire

The self-healing loop could, in principle, heal itself into a disaster: retrain
on a transient blip, ship a worse model, repeat. We ran exactly that scenario:

```
RETRAIN executed → trained v2 on drifted data → v2 ROC-AUC 0.862
Quality gate: 0.862 < incumbent v1 0.877 → NOT promoted. Production stays v1.
```

The remediator did the "brave" thing (retrain) but the **quality gate refused to
promote the worse model**, so production was never harmed. This is the single
most important property of a self-healing system: *every automated path is
guarded by the same gate that guards manual deploys.* Note also the realistic
subtlety — retraining on *drifted* data didn't beat the clean baseline, which is
often true; the fix is better labeled production data, and the gate makes trying
it safe.

## The decision spectrum (rule-based, offline)

Against the real trained model across injected drift severities:

```
none   → IGNORE      (inputs & outputs normal)
low    → IGNORE      (mild, no confirmed harm)
medium → RETRAIN     (drift + measured accuracy drop)
high   → RETRAIN     (severity critical)
```

The escalation is monotonic and explainable — exactly what you want a human on
call to be able to sanity-check at a glance.

## How the LLM path differs (when a key is set)

With `ANTHROPIC_API_KEY` set, the same context goes to Claude (`claude-opus-5`
by default, configurable via `MLOPS_LLM_MODEL`). The model can produce richer,
more nuanced root causes ("a simultaneous jump in transaction amounts and
device-change rate is consistent with an account-takeover fraud wave rather than
benign inflation"), and can distinguish benign input drift from genuine
degradation. But because of the fallback and the validation clamp, the *system's
safety* never depends on the LLM being available or well-behaved.
