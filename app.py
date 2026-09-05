"""
RiskShield AI — Merchant Risk Intelligence & Investigation Center
Flask API + Interactive Dashboard console.

Endpoints:
  GET  /                          -> dashboard UI
  GET  /api/metrics                -> model performance metrics (from training)
  GET  /api/transactions           -> scored test-set transactions (with full feature set)
  POST /api/score                  -> score single transaction JSON payload
  GET  /api/policy                 -> active risk posture policy (STRICT, BALANCED, FRICTIONLESS)
  POST /api/policy                 -> update active risk posture policy mode
  GET  /api/system_health          -> operational telemetry health state
  GET  /api/live_stream/status     -> synthetic live stream status & buffer
  POST /api/live_stream/next       -> generate & score next synthetic live stream transaction
  POST /api/live_stream/toggle     -> toggle synthetic live stream state
  GET  /api/threshold_eval         -> precision/recall/F1/cost trade-offs across thresholds
  GET  /api/audit_trail            -> SQLite persistent decision audit trail
  POST /api/failure_lab/fallback   -> simulate model failure and fallback rules
  POST /api/failure_lab/reset      -> reset model failure status to ONLINE
  POST /api/failure_lab/duplicate  -> simulate duplicate transaction event
  POST /api/failure_lab/borderline -> simulate borderline score scenario
"""

import json
import os
import sys
import time
import sqlite3
import random
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
DB_PATH = str(BASE_DIR / "data" / "audit_trail.db")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
base_threshold = bundle["threshold"]
POLICY_THRESHOLDS = bundle.get("policy_thresholds", {
    "STRICT": round(base_threshold * 0.5, 4),
    "BALANCED": round(base_threshold, 4),
    "FRICTIONLESS": round(min(0.5, base_threshold * 2.0), 4)
})

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

_raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
_scored_df, _ = engineer_features(_raw)

ACTIVE_POLICY_MODE = "BALANCED"
MODEL_STATUS = "ONLINE"
FALLBACK_STATUS = "ARMED"
LIVE_STREAM_ACTIVE = False
LIVE_STREAM_BUFFER = []


def get_active_threshold():
    return POLICY_THRESHOLDS.get(ACTIVE_POLICY_MODE, base_threshold)


# --- SQLite Database Initialization ---
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            transaction_id TEXT UNIQUE NOT NULL,
            merchant_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            fraud_probability REAL NOT NULL,
            threshold REAL NOT NULL,
            decision TEXT NOT NULL,
            policy_mode TEXT NOT NULL DEFAULT 'BALANCED',
            execution_mode TEXT NOT NULL,
            status_reason TEXT NOT NULL,
            is_duplicate INTEGER DEFAULT 0,
            risk_signals TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


def db_get_transaction(txn_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_events WHERE transaction_id = ?", (txn_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def db_insert_audit_event(event_dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO audit_events 
            (timestamp, transaction_id, merchant_id, amount, fraud_probability, threshold, decision, policy_mode, execution_mode, status_reason, is_duplicate, risk_signals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_dict["timestamp"],
            event_dict["transaction_id"],
            event_dict["merchant_id"],
            event_dict["amount"],
            event_dict["fraud_probability"],
            event_dict["threshold"],
            event_dict["decision"],
            event_dict.get("policy_mode", ACTIVE_POLICY_MODE),
            event_dict["execution_mode"],
            event_dict["status_reason"],
            1 if event_dict.get("is_duplicate") else 0,
            json.dumps(event_dict.get("risk_signals", {}))
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


def db_get_audit_trail(limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_events ORDER BY event_id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("risk_signals"):
            try:
                d["risk_signals"] = json.loads(d["risk_signals"])
            except Exception:
                pass
        d["flagged"] = bool(d["decision"] in ["FLAGGED", "Rule Flagged", "Automated Flag"])
        out.append(d)
    return out


def compute_scored_df(curr_threshold):
    df = _scored_df.copy()
    df["fraud_probability"] = model.predict_proba(df[feature_cols])[:, 1]
    df["flagged"] = (df["fraud_probability"] >= curr_threshold).astype(int)
    return df


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/metrics")
def api_metrics():
    out_metrics = dict(METRICS)
    out_metrics["active_policy_mode"] = ACTIVE_POLICY_MODE
    out_metrics["active_threshold"] = get_active_threshold()
    return jsonify(out_metrics)


@app.route("/api/policy", methods=["GET", "POST"])
def api_policy():
    global ACTIVE_POLICY_MODE
    if request.method == "POST":
        payload = request.get_json(force=True)
        mode = payload.get("mode", "BALANCED").upper()
        if mode in POLICY_THRESHOLDS:
            ACTIVE_POLICY_MODE = mode
            return jsonify({
                "status": "success",
                "active_policy_mode": ACTIVE_POLICY_MODE,
                "active_threshold": get_active_threshold(),
                "policy_thresholds": POLICY_THRESHOLDS
            })
        return jsonify({"error": "Invalid policy mode"}), 400

    return jsonify({
        "active_policy_mode": ACTIVE_POLICY_MODE,
        "active_threshold": get_active_threshold(),
        "policy_thresholds": POLICY_THRESHOLDS
    })


@app.route("/api/system_health")
def api_system_health():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_events")
        count = cursor.fetchone()[0]
        conn.close()
        audit_status = "HEALTHY"
    except Exception:
        count = 0
        audit_status = "DEGRADED"

    return jsonify({
        "model_status": MODEL_STATUS,
        "scoring_api": "HEALTHY",
        "audit_store": audit_status,
        "audit_event_count": count,
        "fallback_status": FALLBACK_STATUS,
        "active_policy_mode": ACTIVE_POLICY_MODE,
        "active_threshold": get_active_threshold()
    })


@app.route("/api/live_stream/next", methods=["POST"])
def api_live_stream_next():
    is_m9_burst = (random.random() < 0.20)
    now_ts = datetime.now()

    if is_m9_burst:
        merchant_id = 9
        amount = round(random.uniform(450.0, 2400.0), 2)
        dev_known = random.random() < 0.05
        ip_match = random.random() < 0.20
        device_id = random.choice([9901, 9902, 9903])
    else:
        merchant_id = random.randint(0, 11)
        amount = round(random.lognormvariate(5.5, 0.6), 2)
        dev_known = random.random() < 0.92
        ip_match = random.random() < 0.97
        device_id = random.randint(100, 800)

    txn_id = f"SIM_{int(time.time()*1000)}_{random.randint(100,999)}"
    payload = {
        "transaction_id": txn_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "device_id": device_id,
        "device_is_known": dev_known,
        "ip_country_match": ip_match,
        "timestamp": now_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "force_new": True
    }

    res = api_score_internal(payload, source="SYNTHETIC_LIVE_STREAM")
    LIVE_STREAM_BUFFER.insert(0, res)
    if len(LIVE_STREAM_BUFFER) > 50:
        LIVE_STREAM_BUFFER.pop()

    return jsonify({
        "status": "success",
        "live_transaction": res,
        "buffer_size": len(LIVE_STREAM_BUFFER)
    })


@app.route("/api/live_stream/toggle", methods=["POST"])
def api_live_stream_toggle():
    global LIVE_STREAM_ACTIVE
    payload = request.get_json(silent=True) or {}
    if "active" in payload:
        LIVE_STREAM_ACTIVE = bool(payload["active"])
    else:
        LIVE_STREAM_ACTIVE = not LIVE_STREAM_ACTIVE
    return jsonify({
        "live_stream_active": LIVE_STREAM_ACTIVE
    })


@app.route("/api/live_stream/status")
def api_live_stream_status():
    return jsonify({
        "live_stream_active": LIVE_STREAM_ACTIVE,
        "recent_transactions": LIVE_STREAM_BUFFER[:10]
    })


@app.route("/api/transactions")
def api_transactions():
    limit = int(request.args.get("limit", 200))
    offset = int(request.args.get("offset", 0))
    only_flagged = request.args.get("flagged", "false").lower() == "true"
    df = compute_scored_df(get_active_threshold()).sort_values("timestamp", ascending=False)
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
    payload = request.get_json(force=True)
    res = api_score_internal(payload)
    return jsonify(res)


@app.route("/api/threshold_eval")
def api_threshold_eval():
    df = compute_scored_df(get_active_threshold())
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    y_test = test_df["label"].values
    probs = test_df["fraud_probability"].values
    avg_fraud_amt = test_df.loc[test_df["label"] == 1, "amount"].mean()
    fp_cost_unit = METRICS.get("false_positive_cost_assumption_usd", 4.0)

    test_thresholds = [0.002, 0.005, 0.008, 0.016, 0.032, 0.05, 0.10, 0.20, 0.40]
    eval_results = []
    curr_active = get_active_threshold()

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
            "is_deployed": bool(abs(t - curr_active) < 1e-4)
        })

    return jsonify({
        "deployed_threshold": round(float(curr_active), 4),
        "active_policy_mode": ACTIVE_POLICY_MODE,
        "evaluations": eval_results
    })


@app.route("/api/audit_trail")
def api_audit_trail():
    limit = int(request.args.get("limit", 200))
    trail = db_get_audit_trail(limit=limit)
    return jsonify(trail)


@app.route("/api/failure_lab/fallback", methods=["POST"])
def api_failure_fallback():
    global MODEL_STATUS, FALLBACK_STATUS
    MODEL_STATUS = "UNAVAILABLE"
    FALLBACK_STATUS = "ACTIVE"
    payload = request.get_json(silent=True) or {}
    amount = float(payload.get("amount", 6500.0))
    device_known = bool(payload.get("device_is_known", False))
    
    rule_triggered = (amount > 5000.0) and (not device_known)
    decision = "FLAGGED" if rule_triggered else "CLEAR"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txn_id = f"FALLBACK_{int(time.time()*1000)}"

    db_insert_audit_event({
        "timestamp": timestamp_str,
        "transaction_id": txn_id,
        "amount": amount,
        "merchant_id": payload.get("merchant_id", 3),
        "fraud_probability": 0.9990 if rule_triggered else 0.0010,
        "threshold": 5000.0,
        "decision": decision,
        "policy_mode": ACTIVE_POLICY_MODE,
        "execution_mode": "FALLBACK HEURISTIC (Rule: Amt>$5k & Device Unknown)",
        "status_reason": f"Fallback rule decision: {decision}"
    })

    return jsonify({
        "mode": "FALLBACK_HEURISTIC",
        "model_status": "UNAVAILABLE",
        "fallback_status": "ACTIVE",
        "transaction_id": txn_id,
        "rule_executed": "Amount > $5,000 AND Unknown Device",
        "amount": amount,
        "device_is_known": device_known,
        "flagged": rule_triggered,
        "reason": "Simulated primary ML model server timeout. Fallback risk policy activated."
    })


@app.route("/api/failure_lab/reset", methods=["POST"])
def api_failure_reset():
    global MODEL_STATUS, FALLBACK_STATUS
    MODEL_STATUS = "ONLINE"
    FALLBACK_STATUS = "ARMED"
    return jsonify({
        "status": "success",
        "model_status": MODEL_STATUS,
        "fallback_status": FALLBACK_STATUS,
        "message": "Model scoring service restored to ONLINE status."
    })


@app.route("/api/failure_lab/duplicate", methods=["POST"])
def api_failure_duplicate():
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
    
    first_res = api_score_internal(payload)
    payload["force_new"] = False
    dupe_res = api_score_internal(payload)

    return jsonify({
        "scenario": "DUPLICATE_TRANSACTION_ID",
        "first_submission": first_res,
        "second_submission": dupe_res,
        "idempotency_validated": bool(dupe_res.get("is_duplicate"))
    })


@app.route("/api/failure_lab/borderline", methods=["POST"])
def api_failure_borderline():
    curr_thresh = get_active_threshold()
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
        "active_threshold": curr_thresh,
        "margin": round(abs(res["fraud_probability"] - curr_thresh), 4),
        "guidance": f"Probability lands within borderline margin of threshold (±0.008). System flags for compliance review."
    })


def api_score_internal(payload, source="MANUAL"):
    global MODEL_STATUS, FALLBACK_STATUS

    txn_id = payload.get("transaction_id", f"DEMO_{int(time.time()*1000)}")
    curr_thresh = get_active_threshold()

    if MODEL_STATUS == "UNAVAILABLE":
        amt = float(payload.get("amount", 0.0))
        device_known = bool(payload.get("device_is_known", False))
        rule_triggered = (amt > 5000.0) and (not device_known)
        decision = "FLAGGED" if rule_triggered else "CLEAR"
        exec_mode = "FALLBACK HEURISTIC (Rule: Amt>$5k & Device Unknown)"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        res = {
            "is_duplicate": False,
            "transaction_id": txn_id,
            "amount": amt,
            "merchant_id": payload.get("merchant_id", 0),
            "fraud_probability": 0.9990 if rule_triggered else 0.0010,
            "flagged": rule_triggered,
            "decision": decision,
            "threshold_used": 5000.0,
            "policy_mode": ACTIVE_POLICY_MODE,
            "execution_mode": exec_mode
        }
        db_insert_audit_event({
            "timestamp": timestamp_str,
            "transaction_id": txn_id,
            "merchant_id": payload.get("merchant_id", 0),
            "amount": amt,
            "fraud_probability": res["fraud_probability"],
            "threshold": 5000.0,
            "decision": decision,
            "policy_mode": ACTIVE_POLICY_MODE,
            "execution_mode": exec_mode,
            "status_reason": f"Fallback rule decision: {decision}"
        })
        return res

    if not payload.get("force_new", False):
        prev = db_get_transaction(txn_id)
        if prev:
            return {
                "is_duplicate": True,
                "transaction_id": txn_id,
                "first_seen": prev["timestamp"],
                "fraud_probability": prev["fraud_probability"],
                "flagged": bool(prev["decision"] in ["FLAGGED", "Automated Flag"]),
                "decision": prev["decision"],
                "threshold_used": prev["threshold"],
                "policy_mode": prev["policy_mode"],
                "note": "Idempotent duplicate response — retained original score result."
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
    flagged = bool(prob >= curr_thresh)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res = {
        "is_duplicate": False,
        "transaction_id": txn_id,
        "merchant_id": payload.get("merchant_id", 0),
        "amount": float(payload["amount"]),
        "fraud_probability": round(prob, 4),
        "flagged": flagged,
        "decision": "FLAGGED" if flagged else "CLEAR",
        "threshold_used": round(float(curr_thresh), 4),
        "policy_mode": ACTIVE_POLICY_MODE,
        "source": source
    }
    
    db_insert_audit_event({
        "timestamp": timestamp_str,
        "transaction_id": txn_id,
        "amount": float(payload["amount"]),
        "merchant_id": payload.get("merchant_id", 0),
        "fraud_probability": round(prob, 4),
        "threshold": round(float(curr_thresh), 4),
        "decision": "FLAGGED" if flagged else "CLEAR",
        "policy_mode": ACTIVE_POLICY_MODE,
        "execution_mode": f"ML Model (GradientBoosting - Policy: {ACTIVE_POLICY_MODE})",
        "status_reason": "Automated Flag" if flagged else "Automated Clear"
    })
    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
