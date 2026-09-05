# RiskShield AI — Merchant Risk Intelligence & Investigation Center

**Track 2: AI Risk Manager — Razorpay AI Buildathon 2026**

A real-time merchant risk intelligence prototype and fraud-spike detector. RiskShield AI flags fraud-ring transaction bursts (rapid-fire velocity, off-hours timing, device-farm signatures) in a merchant's transaction stream with honest precision and recall reported on an untouched chronological final test split — not a cherry-picked demo or overfitted threshold.

---

## Product Capabilities & Workflow Lifecycle

Built with a 1440px high-density fintech console layout supporting dual Dark and Light visual themes:

$$\text{DETECT} \longrightarrow \text{INVESTIGATE} \longrightarrow \text{EXPLAIN} \longrightarrow \text{DECIDE} \longrightarrow \text{MEASURE IMPACT} \longrightarrow \text{VALIDATE} \longrightarrow \text{TEST RESILIENCE} \longrightarrow \text{AUDIT}$$

1. **Operational Command Center Header**: Real-time KPI summary displaying dynamic split metrics and live system telemetry status (`MODEL: ONLINE`, `SCORING API: HEALTHY`, `AUDIT STORE: HEALTHY`, `FALLBACK: ARMED`).
2. **Synthetic Live Transaction Stream Simulator**: Demonstrates live transaction ingestion (`POST /api/live_stream/next`), feature engineering, ML scoring, active policy gate thresholding, and SQLite audit logging.
3. **Interactive Risk Posture System**: Dynamic policy configuration supporting **STRICT** (0.0080 cutoff), **BALANCED** (0.0160 cutoff), and **FRICTIONLESS** (0.0320 cutoff) postures that dynamically adjust scoring logic and decision provenance.
4. **Merchant Risk Stream & Priority Queue**: High-density transaction table with multi-column sorting (Score, Amount, Timestamp), merchant filtering, instant search, pagination, and dynamic triage rankings (`CRITICAL`, `HIGH`, `REVIEW`).
5. **Model Feature Importance & Transaction Evidence**: Distinguishes global Gradient Boosting feature weights (`merchant_velocity_10min` 82.1%, `ip_country_match` 10.7%, `is_off_hours` 4.9%) from transaction-specific evidence.
6. **False-Positive Economics Balance Sheet**: Financial trade-off simulator pricing friction cost ($4.00 per FP) against fraud value caught ($180,831.88) and unrecovered missed fraud ($19,769.19).
7. **Model Evaluation & Threshold Explorer**: Interactive operating threshold evaluator (`/api/threshold_eval`) comparing candidate thresholds (0.002 to 0.40) on the untouched final test split.
8. **Fraud Spike & Burst Telemetry**: Detects rolling 10-minute velocity surge anomalies (e.g. Merchant M9 burst signature: 32 txns / 10 min vs 1.2 baseline).
9. **Failure Lab & System Health Resilience**: Operational testing suite for simulated edge cases:
   - *Model Unavailable*: Safe fallback to deterministic heuristic rule (`Amount > $5,000` & `Unknown Device`).
   - *Duplicate Event*: Idempotency validation on repeated transaction IDs via SQLite lookup.
   - *Borderline Score Triage*: Escalates transactions within margin ($\pm 0.008$) of active threshold to Manual Compliance Review.
10. **SQLite Persistent Audit Trail**: Persistent, append-only decision audit store stored in `data/audit_trail.db`.

---

## Evaluation Methodology (Chronological 60 / 20 / 20 Split)

To prevent time-series data leakage, the transaction dataset is ordered chronologically by timestamp and partitioned as follows:

- **60% Chronological Train** (2,616 transactions, 31 fraud): Used exclusively for training `GradientBoostingClassifier` and `LogisticRegression` models.
- **20% Chronological Validation** (872 transactions, 18 fraud): Used exclusively for tuning the decision threshold by maximizing F1 score (`BALANCED` threshold = `0.0160`).
- **20% Chronological Final Test** (872 transactions, 111 fraud): Untouched test split reserved strictly for final performance measurement.

> [!NOTE]
> Time-based validation set used for threshold selection; final chronological test set used for performance measurement.

### Final Performance Results (Untouched Final Test Set)

| Metric | Value |
|---|---|
| Precision | 95.3% (95.28%) |
| Recall | 91.0% (90.99%) |
| F1 Score | 0.9309 |
| ROC-AUC | 0.9928 |
| False Positive Rate (FPR) | 0.66% (0.0066) |
| Review Rate | 12.2% (12.16%) |
| Final Test Split Size | 872 transactions (111 fraud) |
| Confusion Matrix | TP 101 · FP 5 · FN 10 · TN 756 |
| Fraud Value Caught | $180,831.88 |
| Unrecovered Missed Fraud | $19,769.19 |
| False Positive Friction Cost | $20.00 (5 FP × $4.00) |
| Net Financial Value Saved | **+$161,042.69** |

### Baseline Model Comparison

| Model Architecture | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| **Baseline Logistic Regression** | 95.6% | 97.3% | 0.9643 | 0.9994 |
| **Deployed Gradient Boosting** | 95.3% | 91.0% | 0.9309 | 0.9928 |

*Key Finding:* A simpler Logistic Regression baseline matched or slightly outperformed Gradient Boosting on this test set, demonstrating that domain feature engineering (velocity bursts, z-scores) carries more weight than raw model depth. Gradient Boosting was the initial architecture choice built into the production prototype and LR's performance advantage was only discovered during final benchmark evaluation, making Logistic Regression the prime candidate for the next model iteration.

---

## Synthetic Evaluation Data Disclosure

Evaluation data is synthetic and contains injected fraud-burst scenarios designed to test risk detection behavior. Results demonstrate system behavior on the evaluation dataset and should not be interpreted as production fraud performance.

---

## API Architecture & Endpoints

- `GET /` : Serves primary RiskShield AI console (`templates/dashboard.html`).
- `GET /api/metrics` : Returns model metrics, 60/20/20 split metadata, active policy mode, and financial assumptions.
- `GET /api/policy` : Returns active policy posture and threshold cutoffs.
- `POST /api/policy` : Updates active policy mode (`STRICT`, `BALANCED`, `FRICTIONLESS`).
- `GET /api/system_health` : Returns live system telemetry health indicators.
- `GET /api/transactions` : Returns transactions scored under current active policy threshold.
- `POST /api/score` : Scores single transaction with local risk evidence and SQLite audit logging.
- `POST /api/live_stream/next` : Generates and scores synthetic live transaction event.
- `POST /api/live_stream/toggle` : Toggles synthetic live stream generator.
- `GET /api/threshold_eval` : Evaluates precision, recall, F1, review rate, and net savings across candidate thresholds.
- `GET /api/audit_trail` : Returns persistent decision audit records from SQLite.
- `POST /api/failure_lab/fallback` : Simulates ML model timeout with deterministic heuristic fallback.
- `POST /api/failure_lab/reset` : Restores ML model server status to `ONLINE`.
- `POST /api/failure_lab/duplicate` : Simulates duplicate transaction ID idempotency check.
- `POST /api/failure_lab/borderline` : Simulates borderline transaction scoring near active threshold.

---

## Quick Start & Local Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Retrain model on 60/20/20 split and generate metrics
python model/train.py

# 3. Launch RiskShield AI application
python app.py

# 4. Open in browser
# http://127.0.0.1:5000
```

---

## System Architecture & Integration Points

- Prototype architecture with clear production integration points.
- Real-time scoring computes rolling velocity against transaction history; production deployment would integrate a streaming feature store (e.g., Redis/Feast).
- Audit events persist locally in SQLite (`data/audit_trail.db`).
- Interactive theme toggle persists preference (`dark`/`light`) in local storage.
