# 01 — Scaffolding & configuration

This note covers the boring-but-load-bearing decisions that make the rest of
the project pleasant to work on.

## Decision 1: the `src/` layout

The package lives at `src/mlops/`, not `mlops/` in the repo root. Why bother?

Because with a root-level package, `import mlops` works simply because your
shell happens to be in the repo directory — Python puts the current directory
on `sys.path`. Your tests then pass locally but the *installed* package might
be broken (a missing file in `MANIFEST`, a bad import) and you'd never know.

The `src/` layout forces you to actually install the package
(`pip install -e .`) before you can import it. Now you're testing the same
thing a user installs. This is the modern Python packaging consensus.

## Decision 2: config in the environment (12-factor)

`src/mlops/config.py` defines a frozen `Settings` dataclass. Every value has:
1. a **sensible default in code** (so `git clone && run` works with zero setup), and
2. an **environment-variable override** (so ops can retune without a code change,
   and secrets like `ANTHROPIC_API_KEY` never get committed).

This is [factor III of the 12-factor app](https://12factor.net/config). The key
insight: *config that varies between deploys* (dev vs prod thresholds, API keys,
hostnames) belongs in the environment, not in code.

Two small robustness touches worth calling out:
- `Settings` is **frozen** (`@dataclass(frozen=True)`) — config is read-only at
  runtime. If a test needs a different value it builds a fresh instance instead
  of mutating a global, which prevents spooky test-order-dependent bugs.
- The `_get_float`/`_get_int` helpers **tolerate malformed input** and fall back
  to the default instead of crashing. A typo in an env var shouldn't take down
  the service on boot.

`get_settings()` is wrapped in `@lru_cache`, so the environment is read exactly
once and everyone shares one immutable snapshot.

## Decision 3: two requirements files

- `requirements.txt` — runtime deps, each with a comment explaining *why* it's
  there. Pinned with `>=` lower bounds so the project stays installable.
- `requirements-dev.txt` — testing/linting, and it `-r requirements.txt` so one
  install command gets everything.

Airflow is deliberately **excluded** from the core deps and left commented in
the dev file. It's a huge dependency tree, and the training/serving/monitoring
code doesn't need it to run or be tested. You only install Airflow if you're
actually working on the orchestration DAG. Keeping optional heavyweight deps
optional is a real production instinct — it keeps CI fast and images small.

## Decision 4: git-ignore everything *generated*

`mlruns/`, `artifacts/`, `logs/`, `*.joblib`, `.env` — all ignored. The rule of
thumb: **if a command regenerates it, don't commit it.** Committing a trained
model or a 200MB `mlruns/` folder is a classic beginner mistake that bloats the
repo and creates merge conflicts on binary files. The code that *produces* the
artifact is the source of truth; the artifact itself is disposable.

## What "runs with zero setup" bought us

Because every default is safe (local file-based MLflow store, rule-based agent
fallback when there's no API key), a reviewer can clone the repo and get a
working demo in two commands. Lowering the friction to *see it work* is worth a
disproportionate amount when someone is evaluating your project in 10 minutes.
