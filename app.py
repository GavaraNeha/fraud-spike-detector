"""
Fraud-Spike Detector — Flask API + dashboard.

Endpoints:
  GET  /                  -> dashboard UI
  GET  /api/metrics        -> model performance metrics (from training)
  GET  /api/transactions    -> scored test-set transactions (for the dashboard table)
  POST /api/score          -> score a single transaction JSON payload
"""

import json
import os
import sys
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "model"))
from features import engineer_features

app = Flask(__name__)

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")

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


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/transactions")
def api_transactions():
    limit = int(request.args.get("limit", 200))
    only_flagged = request.args.get("flagged", "false").lower() == "true"
    df = _scored_df.sort_values("timestamp", ascending=False)
    if only_flagged:
        df = df[df["flagged"] == 1]
    cols = ["transaction_id", "merchant_id", "timestamp", "amount",
            "device_velocity_30min", "merchant_velocity_10min",
            "is_off_hours", "ip_country_match", "fraud_probability",
            "flagged", "label"]
    out = df[cols].head(limit)
    out["timestamp"] = out["timestamp"].astype(str)
    return jsonify(out.to_dict(orient="records"))


@app.route("/api/score", methods=["POST"])
def api_score():
    """Score a single transaction. Expects JSON with the raw transaction fields.
    Note: real-time velocity features require transaction history, so this
    endpoint scores against a simplified feature set for demo purposes —
    documented limitation, not hidden."""
    payload = request.get_json(force=True)
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
        return jsonify({
            "fraud_probability": round(prob, 4),
            "flagged": flagged,
            "threshold_used": round(float(threshold), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
