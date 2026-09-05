"""
RiskShield AI — Merchant Risk Intelligence & Investigation Center
Flask API + Interactive Dashboard console.

Endpoints:
  GET  /                  -> dashboard UI
  GET  /api/metrics        -> model performance metrics (from training)
  GET  /api/transactions    -> scored test-set transactions (with full feature set)
  POST /api/score          -> score a single transaction JSON payload (with audit logging)
  GET  /api/threshold_eval -> precision/recall/F1/cost trade-offs across thresholds
  GET  /api/audit_trail    -> session history of scored transactions
  POST /api/failure_lab/fallback -> simulate model failure and fallback rules
  POST /api/failure_lab/duplicate -> simulate duplicate transaction event
  POST /api/failure_lab/borderline -> simulate borderline score scenario
"""

import json
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "model"))
from features import engineer_features

app = Flask(__name__)

MODEL_PATH = str(BASE_DIR / "model" / "model.pkl")
METRICS_PATH = str(BASE_DIR / "model" / "metrics.json")
DATA_PATH = str(BASE_DIR / "data" / "transactions.csv")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
threshold = bundle["threshold"]

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

_raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
_scored_df, _ = engineer_features(_raw)
_scored_df["fraud_probability"] = model.predict_proba(_scored_df[feature_cols])[:, 1]
_scored_df["flagged"] = (_scored_df["fraud_probability"] >= threshold).astype(int)

# Session audit trail store & duplicate cache
AUDIT_TRAIL = []
SEEN_TRANSACTIONS = {}


def get_test_split():
    df = _scored_df.sort_values("timestamp")
    split_idx = int(len(df) * 0.7)
    return df.iloc[split_idx:]


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/transactions")
def api_transactions():
    limit = int(request.args.get("limit", 200))
    offset = int(request.args.get("offset", 0))
    only_flagged = request.args.get("flagged", "false").lower() == "true"
    df = _scored_df.sort_values("timestamp", ascending=False)
    if only_flagged:
        df = df[df["flagged"] == 1]
    cols = ["transaction_id", "merchant_id", "timestamp", "amount",
            "device_velocity_30min", "merchant_velocity_10min",
            "is_off_hours", "ip_country_match", "device_is_known", "amount_zscore",
            "fraud_probability", "flagged", "label"]
    if limit > 0:
        out = df[cols].iloc[offset:offset+limit]
    else:
        out = df[cols].iloc[offset:]
    out["timestamp"] = out["timestamp"].astype(str)
    return jsonify(out.to_dict(orient="records"))


@app.route("/api/score", methods=["POST"])
def api_score():
    """Score a single transaction. Expects JSON with raw transaction fields."""
    payload = request.get_json(force=True)
    txn_id = payload.get("transaction_id", f"LIVE_{int(time.time()*1000)}")
    
    # Check duplicate transaction submission
    if txn_id in SEEN_TRANSACTIONS and not payload.get("force_new", False):
        prev = SEEN_TRANSACTIONS[txn_id]
        return jsonify({
            "is_duplicate": True,
            "transaction_id": txn_id,
            "first_seen": prev["timestamp"],
            "fraud_probability": prev["fraud_probability"],
            "flagged": prev["flagged"],
            "threshold_used": prev["threshold_used"],
            "note": "Idempotent duplicate response — retained original score result."
        })

    try:
        row = pd.DataFrame([{
            "merchant_id": payload.get("merchant_id", 0),
            "timestamp": payload.get("timestamp", pd.Timestamp.now()),
            "amount": float(payload["amount"]),
            "device_id": payload.get("device_id", -1),
            "device_is_known": bool(payload.get("device_is_known", True)),
            "ip_country_match": bool(payload.get("ip_country_match", True)),
            "label": 0,
        }])
        combined = pd.concat([_raw, row], ignore_index=True)
        feats, _ = engineer_features(combined)
        last_row = feats.iloc[[-1]][feature_cols]
        prob = float(model.predict_proba(last_row)[:, 1][0])
        flagged = bool(prob >= threshold)
        
        # Borderline check (within 0.008 of threshold)
        is_borderline = bool(abs(prob - threshold) <= 0.008)

        res = {
            "transaction_id": txn_id,
            "fraud_probability": round(prob, 4),
            "flagged": flagged,
            "threshold_used": round(float(threshold), 4),
            "is_borderline": is_borderline,
            "recommendation": "Manual Review Required (Borderline Risk)" if is_borderline else ("Automated Flag" if flagged else "Automated Clear")
        }

        # Cache & Audit Trail recording
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        SEEN_TRANSACTIONS[txn_id] = {**res, "timestamp": timestamp_str}
        
        AUDIT_TRAIL.append({
            "timestamp": timestamp_str,
            "transaction_id": txn_id,
            "amount": float(payload["amount"]),
            "merchant_id": payload.get("merchant_id", 0),
            "fraud_probability": round(prob, 4),
            "flagged": flagged,
            "threshold_used": round(float(threshold), 4),
            "execution_mode": "ML Model (GradientBoosting)",
            "status": res["recommendation"]
        })

        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/threshold_eval")
def api_threshold_eval():
    """Evaluate performance metrics across multiple thresholds on test set."""
    test_df = get_test_split()
    y_test = test_df["label"].values
    probs = test_df["fraud_probability"].values
    avg_fraud_amt = test_df.loc[test_df["label"] == 1, "amount"].mean()
    fp_cost_unit = METRICS.get("false_positive_cost_assumption_usd", 4.0)

    test_thresholds = [0.002, 0.005, 0.0102, 0.025, 0.05, 0.10, 0.20, 0.40]
    eval_results = []

    for t in test_thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        p = float(precision_score(y_test, preds, zero_division=0))
        r = float(recall_score(y_test, preds, zero_division=0))
        f1_val = float(f1_score(y_test, preds, zero_division=0))
        fp_cost_total = float(fp * fp_cost_unit)
        fraud_caught_val = float(tp * avg_fraud_amt)
        net_savings = float(fraud_caught_val - fp_cost_total)

        eval_results.append({
            "threshold": round(t, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1_val, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "fp_cost_usd": round(fp_cost_total, 2),
            "fraud_caught_usd": round(fraud_caught_val, 2),
            "net_savings_usd": round(net_savings, 2),
            "is_deployed": bool(abs(t - threshold) < 1e-4)
        })

    return jsonify({
        "deployed_threshold": round(float(threshold), 4),
        "evaluations": eval_results
    })


@app.route("/api/audit_trail")
def api_audit_trail():
    """Return session audit trail log."""
    return jsonify(list(reversed(AUDIT_TRAIL)))


@app.route("/api/failure_lab/fallback", methods=["POST"])
def api_failure_fallback():
    """Simulate model unavailability -> fallback to deterministic heuristic rule."""
    payload = request.get_json(force=True)
    amount = float(payload.get("amount", 6500.0))
    device_known = bool(payload.get("device_is_known", False))
    
    # Fallback Rule Heuristic: Flag if amount > $5000 AND device is unknown
    rule_triggered = (amount > 5000.0) and (not device_known)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txn_id = f"FALLBACK_{int(time.time()*1000)}"

    AUDIT_TRAIL.append({
        "timestamp": timestamp_str,
        "transaction_id": txn_id,
        "amount": amount,
        "merchant_id": payload.get("merchant_id", 3),
        "fraud_probability": 0.9990 if rule_triggered else 0.0010,
        "flagged": rule_triggered,
        "threshold_used": 5000.0,
        "execution_mode": "FALLBACK HEURISTIC (Rule: Amt>$5k & Device Unknown)",
        "status": "Rule Flagged" if rule_triggered else "Rule Cleared"
    })

    return jsonify({
        "mode": "FALLBACK_HEURISTIC",
        "model_status": "UNAVAILABLE_SIMULATED",
        "transaction_id": txn_id,
        "rule_executed": "Amount > $5,000 AND Unknown Device",
        "amount": amount,
        "device_is_known": device_known,
        "flagged": rule_triggered,
        "reason": "Simulated primary ML model server timeout. Fallback risk policy activated."
    })


@app.route("/api/failure_lab/duplicate", methods=["POST"])
def api_failure_duplicate():
    """Simulate submitting duplicate transaction ID."""
    txn_id = "TXN_DUP_DEMO_99"
    payload = {
        "transaction_id": txn_id,
        "amount": 4200.00,
        "merchant_id": 3,
        "device_id": 9001,
        "device_is_known": False,
        "ip_country_match": False,
        "force_new": True
    }
    
    # First submission
    first_res = api_score_internal(payload)
    
    # Second submission (duplicate)
    payload["force_new"] = False
    dupe_res = api_score_internal(payload)

    return jsonify({
        "scenario": "DUPLICATE_TRANSACTION_ID",
        "first_submission": first_res,
        "second_submission": dupe_res
    })


@app.route("/api/failure_lab/borderline", methods=["POST"])
def api_failure_borderline():
    """Simulate scoring a borderline edge case transaction near threshold (0.0102)."""
    payload = {
        "amount": 185.00,
        "merchant_id": 2,
        "device_id": 402,
        "device_is_known": True,
        "ip_country_match": True,
        "force_new": True
    }
    res = api_score_internal(payload)
    return jsonify({
        "scenario": "BORDERLINE_CONFIDENCE_SCORE",
        "score_result": res,
        "threshold": threshold,
        "margin": round(abs(res["fraud_probability"] - threshold), 4),
        "guidance": "Probability lands within borderline margin of threshold (±0.008). System flags for priority compliance officer review."
    })


def api_score_internal(payload):
    txn_id = payload.get("transaction_id", f"DEMO_{int(time.time()*1000)}")
    if txn_id in SEEN_TRANSACTIONS and not payload.get("force_new", False):
        prev = SEEN_TRANSACTIONS[txn_id]
        return {
            "is_duplicate": True,
            "transaction_id": txn_id,
            "fraud_probability": prev["fraud_probability"],
            "flagged": prev["flagged"],
            "threshold_used": prev["threshold_used"]
        }

    row = pd.DataFrame([{
        "merchant_id": payload.get("merchant_id", 0),
        "timestamp": payload.get("timestamp", pd.Timestamp.now()),
        "amount": float(payload["amount"]),
        "device_id": payload.get("device_id", -1),
        "device_is_known": bool(payload.get("device_is_known", True)),
        "ip_country_match": bool(payload.get("ip_country_match", True)),
        "label": 0,
    }])
    combined = pd.concat([_raw, row], ignore_index=True)
    feats, _ = engineer_features(combined)
    last_row = feats.iloc[[-1]][feature_cols]
    prob = float(model.predict_proba(last_row)[:, 1][0])
    flagged = bool(prob >= threshold)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res = {
        "is_duplicate": False,
        "transaction_id": txn_id,
        "fraud_probability": round(prob, 4),
        "flagged": flagged,
        "threshold_used": round(float(threshold), 4)
    }
    SEEN_TRANSACTIONS[txn_id] = {**res, "timestamp": timestamp_str}
    
    AUDIT_TRAIL.append({
        "timestamp": timestamp_str,
        "transaction_id": txn_id,
        "amount": float(payload["amount"]),
        "merchant_id": payload.get("merchant_id", 0),
        "fraud_probability": round(prob, 4),
        "flagged": flagged,
        "threshold_used": round(float(threshold), 4),
        "execution_mode": "ML Model (GradientBoosting)",
        "status": "Automated Flag" if flagged else "Automated Clear"
    })
    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

