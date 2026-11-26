# 04 — Drift detection (the statistics that matter)

This is the most important note. Being able to explain PSI and
the KS test — *and why you use both* — is what separates "I detect drift with a
library" from "I understand drift."

## Two questions, two tests

| | PSI | KS test |
|---|---|---|
| Type | Effect **size** (magnitude) | Hypothesis **test** (significance) |
| Answers | "How *big* is the shift?" | "Is the shift *real* or noise?" |
| Output | a number, with fixed bands | a p-value |
| Sample-size sensitivity | **insensitive** (stable) | **sensitive** (flags on big N) |
| Best for | dashboards, trend-over-time | catching subtle shape changes |

We flag a feature if **either** crosses its threshold. Using PSI alone can miss
a subtle-but-real shift; using KS alone floods you with "significant" trivia on
large samples. Together you get *significant AND meaningful*.

## PSI, precisely

```
1. bin edges = quantiles of the REFERENCE data     (equal-frequency bins)
2. ref%[i]   = fraction of reference in bin i
   prod%[i]  = fraction of production in bin i
3. PSI = Σ_i (prod%[i] − ref%[i]) · ln(prod%[i] / ref%[i])
```

Standard interpretation bands (used for our `severity`):

```
PSI < 0.1   → stable
0.1 – 0.2   → moderate shift (investigate)
> 0.2       → major shift (act)
```

Three implementation details that are easy to get wrong (and that we handle):

- **Quantile (equal-frequency) bins, from the reference.** Equal-*width* bins
  break on skewed data — one bin swallows everything. Quantile bins give each
  reference bin ~equal mass, so the comparison is fair across the range.
- **Infinite outer edges.** Production values outside the reference's observed
  range (e.g. amounts higher than anything in training — exactly what inflation
  produces) must land in the end bins, not get dropped. We set the outer edges
  to ±∞.
- **Epsilon on every proportion.** An empty bin makes `ln(0) = −∞`. Adding
  `1e-6` keeps PSI finite.

## The KS test, precisely

The two-sample Kolmogorov-Smirnov statistic is the **largest vertical gap
between the two empirical CDFs**. Its p-value tests the null hypothesis "both
samples come from the same distribution." A p-value below `0.01` means we reject
that null: the distributions differ. `scipy.stats.ks_2samp` gives both the
statistic and the p-value in one call.

## Discrete features need special handling

`device_change` is 0/1. Quantile-binning a two-value feature is meaningless, so
we detect low-cardinality features (≤ 12 distinct values) and compute PSI over
**category proportions** instead. Forgetting this is a classic bug that makes a
binary flag look either always-stable or always-drifted.

## The output side: performance monitoring

`PerformanceMonitor` watches the model's predictions, not its inputs:

- **Confidence** = `2 · |p − 0.5|`. Healthy models are decisive (p near 0/1);
  under drift, predictions crowd toward 0.5 ("I don't know"). A drop in mean
  confidence is a **label-free** early warning — critical because true labels
  arrive late in fraud (you learn the truth only when a chargeback lands).
- **Accuracy/recall** are computed only once delayed labels arrive, and compared
  to the training baseline. This is the *lagging confirmation*.

## What we observed (the system works)

Against the real trained model:

```
CLEAN : No drift (6 features checked)          ← no false alarms
LOW   : drift in 2 features (small, KS-caught)
MEDIUM: drift [high] in 4 features
HIGH  : drift [high] in 5 features (country_risk PSI 8.4!)
PERF  : confidence −0.266, accuracy 0.909 → 0.697 (−0.212)
```

The severity gradient tracks the injected drift, clean data stays quiet, and
the accuracy drop confirms the drift genuinely hurt the model. That last point
matters: it proves our drift signal isn't crying wolf — when PSI screams, the
model really did get worse. This report is exactly the context we hand to the
LLM diagnostic agent next.
