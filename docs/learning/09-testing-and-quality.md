# 09 — Testing & quality

38 tests, all green, running offline in ~18s. This note explains the testing
*strategy*, which matters more than the count.

## Test isolation: never touch real state

`conftest.py` redirects MLflow to a **throwaway temp directory** (via env vars
set *before* any project code is imported) and clears the cached settings. So the
suite never reads or writes your development `mlflow.db` or leaves junk in the
repo. This is a general rule: **tests must not depend on or mutate developer or
production state.** Getting the ordering right (env vars → clear cache → import)
is the subtle part, and it's why the env setup sits at the very top of conftest.

## The testing pyramid, applied

- **Unit tests (fast, many):** PSI math, confidence scoring, feature validation,
  the agent's rule policy, remediator dispatch. Pure functions, no I/O.
- **Integration tests (fewer):** the registry (real MLflow), the FastAPI app
  (real model via TestClient), the full self-healing cycle. These share one
  session-scoped `trained_model` fixture so the model is trained *once*.

Most tests are unit; the slow, stateful ones are minimized and share setup. That
shape keeps the suite fast enough to run on every save.

## Testing an LLM without a network

The agent's LLM path is tested with an **injected fake client** that mimics the
Anthropic SDK's response shape (`message.content` = list of blocks with `.type`
and `.text`). This lets us assert, deterministically and offline:

- valid JSON is parsed into the right `Diagnosis`,
- an out-of-vocabulary action (`"DELETE_PROD"`) is **clamped to `INVESTIGATE`**,
  a bad severity to `medium`, and `confidence: 5` into `[0, 1]`,
- non-JSON garbage triggers the **rule-based fallback** instead of crashing.

This is how you test code that *calls* an LLM: inject the boundary, feed it the
exact good and bad outputs you care about, and assert your handling. You are
testing *your* logic, not the model.

## Testing behaviour, not implementation

Good assertions describe *what the system guarantees*, e.g.:
- "a high-risk transaction scores higher than a low-risk one" (not "probability
  == 0.8547"),
- "drift on clean data is not flagged" and "high drift is flagged",
- "advisory mode never performs a side effect,"
- "retrain runs through the quality gate."

These survive refactors and pin the properties that actually matter. Asserting
exact floats would make the suite brittle and teach you nothing when it breaks.

## What each test file guards

| File | Guards |
|---|---|
| `test_data.py` | reproducibility, schema order, realistic imbalance, drift really shifts, input validation |
| `test_drift.py` | PSI = 0 on identical data, PSI grows with shift, detector quiet on clean / fires on drift |
| `test_performance.py` | confidence scoring, accuracy-drop and label-free confidence-drop detection |
| `test_agent.py` | rule policy branches, LLM parse, safety clamping, garbage → fallback |
| `test_remediation.py` | IGNORE/INVESTIGATE no-op, advisory skip, auto retrain runs |
| `test_registry.py` | alias load, reference sample, safe single-version rollback |
| `test_api.py` | health, predict, risk ordering, 422 on bad input, metrics exposed |
| `test_self_healing.py` | full cycle: healthy → no action; high drift → diagnosis, advisory skips |

## Running it

```bash
pytest            # or: make test
ruff check .      # or: make lint
```

The whole thing runs with no API key and no Docker — a reviewer can verify the
system works in under a minute, which is worth a lot when someone is deciding
whether to take your project seriously.
