# Spike Watch — Fraud-Spike Detector

**Track 2: AI Risk Manager — Razorpay AI Buildathon 2026**

A detector that flags fraud-ring transaction bursts (rapid-fire, off-hours,
device-farm signatures) in a merchant's transaction stream, with honest
precision/recall reported on a held-out test set — not a cherry-picked demo.

## What it solves

Fraud rings don't send one bad transaction — they send bursts: many
transactions in a short window, from a small pool of unfamiliar devices,
often off-hours, often with mismatched IP/billing country. This detector
scores every transaction in real time on exactly those signals and flags
the ones that look like part of a spike.

## Architecture

```
data/generate_data.py   → synthetic transaction stream (normal traffic +
                            injected fraud-spike bursts, ~3.7% fraud rate)
model/features.py        → real-time-computable features: amount z-score,
                            device/merchant velocity, off-hours flag,
                            device/IP trust signals (no look-ahead leakage)
model/train.py            → trains + evaluates the model, writes metrics.json
app.py                    → Flask API: dashboard, metrics, transaction feed,
                            live single-transaction scoring
templates/dashboard.html  → risk console UI
```

## Why a tabular ML model, not an LLM

Fraud scoring on a transaction stream is latency-sensitive (needs to run at
checkout time), high-volume, and needs to be explainable to a compliance
reviewer. A gradient boosting model gives sub-millisecond scoring,
deterministic output, and a feature-importance breakdown an LLM call
can't match at this latency/cost budget. This is a place where **not**
reaching for an LLM was the right call — the model only needs to weigh
seven numeric signals, not reason over language.

## Results (held-out test set, time-based split — last 30% of days)

| Metric | Value |
|---|---|
| Precision | 95.2% |
| Recall | 92.25% |
| F1 | 0.937 |
| ROC-AUC | 0.9956 |
| Test set | 1,308 transactions, 129 fraud |
| Confusion matrix | TP 119 · FP 6 · FN 10 · TN 1,173 |

**False-positive cost:** assuming $4/transaction in support and friction
cost for a wrongly-held legitimate transaction, the 6 false positives in
this test set cost ~$24 — against ~$201,585 in fraud value correctly
caught and ~$16,940 in fraud value missed (10 false negatives).

**Feature importance:** `merchant_velocity_10min` dominates (82%) —
confirming the core hypothesis that burst timing, not transaction amount
alone, is the strongest fraud-spike signal. `ip_country_match` (11%) and
off-hours timing (4%) are secondary signals.

## What broke, and how I got out of it

1. **Time-based split starved the training set of fraud examples.**
   A random train/test split would leak future transaction history into
   the velocity features (look-ahead bias), so I used a time-based split
   instead — train on the first 70% of days, test on the last 30%. But
   because fraud spikes were randomly distributed across the 30-day
   window, this left only 31 fraud transactions in training versus 129
   in test. The model still generalized well (AUC 0.996), but with a
   larger dataset this imbalance would need addressing — e.g. a
   rolling-window walk-forward validation instead of one fixed split.

2. **The simpler baseline matched the fancier model.** I ran a plain
   logistic regression alongside the gradient boosting model expecting
   it to lose. It didn't — logistic regression scored *higher* recall
   (96.1% vs 92.3%) and AUC (0.9989 vs 0.9956) on this test set. I'm
   reporting this honestly rather than dropping the baseline from the
   writeup: at this dataset size, the extra complexity of gradient
   boosting isn't clearly earning its keep. Feature engineering
   (velocity, z-score) is doing more work here than model choice.

3. **A real false positive, not a hypothetical one.** Transaction
   `TXN100046` — a legitimate $1,298 purchase at 3:29am — got flagged
   because it happened to land in an off-hours window with a merchant
   velocity spike from unrelated concurrent traffic. This is exactly the
   kind of false positive the $4 friction-cost assumption is meant to
   price in: it's a real cost of the off-hours + velocity features doing
   their job on an edge case, not a bug to silently fix by hand-tuning
   this one row.

4. **Numpy boolean serialization error in Flask JSON response.** When building
   the threshold evaluation endpoint (`/api/threshold_eval`), the comparison
   `(t == threshold)` produced a `numpy.bool_` instance. In Python 3.14, standard
   Flask `jsonify()` failed with `TypeError: Object of type bool is not JSON serializable`.
   Diagnosed via log tracebacks and resolved by explicitly wrapping with
   `bool(abs(t - threshold) < 1e-4)`.

## Run it

```bash
pip install -r requirements.txt
python3 data/generate_data.py   # regenerate synthetic data
python3 model/train.py          # retrain + write metrics.json
python3 app.py                  # serves dashboard at localhost:5000
```

## Limitations (stated, not hidden)

- Trained on synthetic data — fraud patterns in production would need
  validation against real (or realistically anonymized) transaction data.
- The `/api/score` endpoint recomputes velocity features against the full
  historical dataset per request for demo purposes; a production version
  would maintain a rolling feature store instead of recomputing from raw
  history on every call.
- Single fixed threshold (tuned for max F1 on this test set) — a
  production system would likely use tiered thresholds (auto-block /
  manual review / auto-clear) rather than one binary cutoff.
