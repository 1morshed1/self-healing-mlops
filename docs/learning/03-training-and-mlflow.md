# 03 — Training pipeline, quality gate & MLflow registry

This phase turns "fit a model" into "operate a model." Three ideas matter:
the **quality gate**, the **model registry with aliases**, and **logging every
run honestly**.

## The quality gate — the most important safety rail

`evaluate.py` returns not just metrics but a **decision**: `passed_gate` +
`gate_reason`. `train.py` refuses to promote a model that fails it.

Why this is the linchpin of a *self-healing* system: automation without a gate
is dangerous. If the loop retrains on bad/drifted data and blindly ships the
result, it can degrade the service and then retrain on even worse data — a
doom loop. The gate is the circuit breaker: **a candidate must clear an
absolute bar (accuracy ≥ 0.80, ROC-AUC ≥ 0.70) AND not be worse than the model
already in production** before it's promoted.

We saw both branches work:
- The first, under-tuned model scored AUC 0.648 → **REJECTED**, not promoted.
- After tuning the signal, AUC 0.877 → **PASSED**, promoted to production v1.

That rejection is a feature, not a bug — the system correctly refused to ship a
weak model.

## Why not judge on accuracy alone (the imbalanced-data trap)

Fraud is ~3% of rows. A model that predicts "never fraud" is 97% accurate and
completely useless. So the gate also requires **ROC-AUC**, which measures
*ranking* quality (can the model give frauds higher scores than non-frauds?)
independent of the decision threshold. We also report precision/recall/F1.
Note our recall (~0.62) is far more informative than the headline accuracy: it
says we catch ~62% of fraud, which is the number a fraud team actually cares
about. `class_weight="balanced"` is what makes the model bother to learn the
rare class at all.

## The MLflow Model Registry + aliases

`registry.py` wraps MLflow behind a small API. Two things worth internalizing:

**1. The registry needs a database backend.** The plain `file:` tracking store
can log runs but *cannot* host the model registry. That's why the default
tracking URI is `sqlite:///mlflow.db` — zero setup, full registry. In a team
you'd swap SQLite for Postgres and run an MLflow server; the code is identical.

**2. Aliases, not stages.** MLflow deprecated the old `Production`/`Staging`
stages in favour of **aliases** — movable named pointers to versions:

```
production  ──▶  version 7
```

The serving app loads `models:/self_healing_classifier@production` and always
gets whatever version is currently promoted. **Deploying** = moving the alias
to a new version. **Rollback** = moving it back. No serving-code change, no
redeploy. `rollback()` implements exactly this: it finds the highest version
below the current production one and repoints the alias.

## Tying a model to its training distribution

Drift detection needs to compare production data against *the distribution the
model was trained on*. So every training run logs a `reference_sample.parquet`
artifact (a 3000-row sample of the training features). `load_reference_sample()`
walks model version → run → artifact to fetch it. This means each model version
carries its own reference snapshot — when you retrain, the drift baseline
updates automatically. That coupling (model + its reference data as one unit) is
a genuinely important production pattern people often get wrong.

## One serialization gotcha worth remembering

Newer MLflow defaults to the `skops` format for sklearn models, which refuses
to serialize decision-tree internals as "untrusted types." We pin
`serialization_format=cloudpickle` — the long-standing default, safe for models
we produce and load within our own trusted pipeline. If you ever load a model
from an *untrusted* third party, that skops warning is real and you should heed
it; for your own artifacts it's just noise.

## What runs now

`python -m scripts.run_pipeline` → trains, evaluates, gates, logs to MLflow,
registers a version, and promotes it if it's safe. The registry API
(`list_versions`, `get_alias_version`, `load_model`, `rollback`,
`load_reference_sample`) is the interface every later phase builds on.
