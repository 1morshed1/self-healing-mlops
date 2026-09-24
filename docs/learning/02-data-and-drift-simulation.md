# 02 — Data generation & drift simulation

## Why generate data instead of downloading a dataset?

To *demonstrate self-healing*, we need to trigger drift on demand. A static
downloaded dataset can't drift. By generating from a known process we get two
things a real dataset can't give us cheaply:

1. **Learnable signal.** Fraud is a real function of the features, so the model
   learns something meaningful (not noise).
2. **Controllable drift.** We can produce a "production" batch whose
   distributions have shifted in specific ways and verify the monitor catches
   it. This makes the whole pipeline self-testing.

## The data-generating process (worth understanding)

Each transaction's features are drawn from plausible distributions:

| Feature | Distribution | Intuition |
|---|---|---|
| `amount` | Normal(90, 40), clipped ≥0 | most transactions modest, some large |
| `age_days` | Gamma | many young accounts, long tail of old ones |
| `n_txn_24h` | Poisson(3) | counts of events per day |
| `avg_txn_amt` | correlated w/ amount + noise | the account's personal baseline |
| `country_risk` | Beta(2,10) → skewed toward 0 | most countries low risk |
| `device_change` | Bernoulli(0.08) | occasionally a new device |

Then the **label** comes from a latent-score model:

```
score  = intercept
       + 1.1 * (amount / avg_txn_amt - 1)   # spending above your own norm
       + 0.9 * (n_txn_24h / 5)              # rapid-fire activity
       + 2.0 * country_risk
       + 1.3 * device_change
       - 0.6 * (age_days / 400)             # older accounts safer
fraud_prob = sigmoid(score)
is_fraud   = Bernoulli(fraud_prob)
```

This "linear score → sigmoid → Bernoulli" pattern is exactly how logistic
regression *assumes* the world works, which is why a decent classifier can
recover the relationship. The intercept is set low so fraud is **rare (~4%)** —
a realistic **class imbalance**, which matters later for why we don't judge the
model on accuracy alone (see the training note).

## Simulating drift the honest way

`DriftSpec` expresses drift as a **delta from the reference world**, one knob
per feature:

- `amount_shift` / `amount_scale` → inflation or a change in transaction mix
- `country_risk_shift` → expansion into riskier markets
- `device_change_rate_delta` → a device-spoofing fraud wave
- `n_txn_shift` → bot-driven activity bursts

The `make_drift_spec("low"|"medium"|"high")` presets bundle correlated shifts,
because **real drift is rarely one feature in isolation** — inflation and a
fraud wave happen together. Feeding the agent correlated, realistic drift makes
its root-cause reasoning a fairer test.

A subtle but important point: notice the drift changes the **feature
distributions but not the labelling function**. That models *covariate shift*
(a.k.a. data drift): the inputs move into regions the model saw little of, so
its calibration degrades even though the underlying "what is fraud" rule is
unchanged. (The other kind, *concept drift*, is when the rule itself changes;
you could model that by also perturbing the score coefficients.)

## The schema contract (the quiet hero)

`schema.py` defines `FEATURE_NAMES` once. Training, serving, and monitoring all
import it. This is the structural defense against **training/serving skew** —
the bug where the model is fed columns in a different order or shape than it was
trained on. `features.build_feature_frame()` is the single validated entry point
that both the training path and the API request path funnel through, so they
*cannot* diverge.

## What we verified

Running the generator: reference `amount` mean ≈ 89, drifted-high mean ≈ 167;
`device_change` rate 8% → 32%. The drift is real and large — good, because now
when the monitor flags it we know it's detecting a true signal, not noise.
