"""
Trains the fraud-spike detector using a 60% Train / 20% Validation / 20% Final Test
chronological split and reports honest, untouched final test metrics.

Design choices & evaluation methodology:
  - Chronological Split: 60% Train, 20% Validation, 20% Final Test based on timestamp order.
  - Model Training: GradientBoostingClassifier trained exclusively on the 60% Train split.
  - Threshold Selection: Decision threshold tuned exclusively on the 20% Validation split
    by maximizing F1 score.
  - Final Evaluation: Performance metrics (Precision, Recall, F1, ROC-AUC, FPR, Review Rate,
    and Financial Savings) measured exclusively on the untouched 20% Final Test split.
  - Baseline Model: Plain Logistic Regression baseline trained on Train and evaluated on Final Test.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "model"))
from features import engineer_features

DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

# Business assumption: $4.00 friction cost per false positive transaction
FP_COST = 4.0


def chronological_split(df, train_frac=0.6, val_frac=0.2):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    raw_sorted = raw.sort_values("timestamp").reset_index(drop=True)
    train_end_idx = int(len(raw_sorted) * 0.6)
    train_cutoff = raw_sorted.iloc[:train_end_idx]["timestamp"].max()

    df, feature_cols = engineer_features(raw_sorted, train_cutoff=train_cutoff)

    train_df, val_df, test_df = chronological_split(df, 0.6, 0.2)

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    print(f"Train:      {len(X_train)} rows ({y_train.sum()} fraud) [{train_df['timestamp'].min()} to {train_df['timestamp'].max()}]")
    print(f"Validation: {len(X_val)} rows ({y_val.sum()} fraud) [{val_df['timestamp'].min()} to {val_df['timestamp'].max()}]")
    print(f"Final Test: {len(X_test)} rows ({y_test.sum()} fraud) [{test_df['timestamp'].min()} to {test_df['timestamp'].max()}]")

    # --- Baseline: plain logistic regression on Train ---
    baseline = LogisticRegression(class_weight="balanced", max_iter=1000)
    baseline.fit(X_train, y_train)
    baseline_probs_test = baseline.predict_proba(X_test)[:, 1]
    baseline_preds_test = (baseline_probs_test >= 0.5).astype(int)

    # --- Main model: gradient boosting on Train ---
    sample_weight = np.where(y_train == 1, (y_train == 0).sum() / max(1, (y_train == 1).sum()), 1.0)
    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # --- Threshold selection: pick threshold that maximizes F1 on VALIDATION set ---
    val_probs = model.predict_proba(X_val)[:, 1]
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val, val_probs)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_idx = np.argmax(f1_arr[:-1]) if len(thresh_arr) > 0 else 0
    balanced_threshold = float(thresh_arr[best_idx]) if len(thresh_arr) > 0 else 0.5

    # Define policy mode thresholds
    strict_threshold = round(float(max(0.001, balanced_threshold * 0.5)), 4)
    frictionless_threshold = round(float(min(0.5, balanced_threshold * 2.0)), 4)
    balanced_threshold = round(float(balanced_threshold), 4)

    print(f"Selected Threshold on Validation (BALANCED): {balanced_threshold}")
    print(f"Policy Cutoffs -> STRICT: {strict_threshold}, BALANCED: {balanced_threshold}, FRICTIONLESS: {frictionless_threshold}")

    # --- Final Evaluation: untouched FINAL TEST set ---
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= balanced_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    review_rate = float((tp + fp) / len(y_test))

    # Exact financial calculations on final test set
    fraud_caught_value = float(test_df.loc[(y_test == 1) & (test_preds == 1), "amount"].sum())
    fraud_missed_value = float(test_df.loc[(y_test == 1) & (test_preds == 0), "amount"].sum())
    fp_cost_total = float(fp * FP_COST)
    net_savings = float(fraud_caught_value - fraud_missed_value - fp_cost_total)

    # Baseline logistic regression on final test set
    baseline_precision = precision_score(y_test, baseline_preds_test, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_preds_test, zero_division=0)
    baseline_f1 = f1_score(y_test, baseline_preds_test, zero_division=0)
    baseline_auc = roc_auc_score(y_test, baseline_probs_test)

    metrics = {
        "model": "GradientBoostingClassifier",
        "threshold": balanced_threshold,
        "train_cutoff": str(train_cutoff),
        "policy_thresholds": {
            "STRICT": strict_threshold,
            "BALANCED": balanced_threshold,
            "FRICTIONLESS": frictionless_threshold
        },
        "split_metadata": {
            "methodology": "Chronological 60% Train / 20% Validation / 20% Final Test",
            "threshold_selection": "Time-based validation set used for threshold selection; final chronological test set used for performance measurement.",
            "train": {
                "count": int(len(X_train)),
                "fraud_count": int(y_train.sum()),
                "start": str(train_df["timestamp"].min()),
                "end": str(train_df["timestamp"].max())
            },
            "validation": {
                "count": int(len(X_val)),
                "fraud_count": int(y_val.sum()),
                "start": str(val_df["timestamp"].min()),
                "end": str(val_df["timestamp"].max())
            },
            "final_test": {
                "count": int(len(X_test)),
                "fraud_count": int(y_test.sum()),
                "start": str(test_df["timestamp"].min()),
                "end": str(test_df["timestamp"].max())
            }
        },
        "test_set_size": int(len(y_test)),
        "test_set_fraud_count": int(y_test.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "fpr": round(float(fpr), 4),
        "review_rate": round(float(review_rate), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_cost_assumption_usd": FP_COST,
        "estimated_fp_cost_total_usd": round(float(fp_cost_total), 2),
        "estimated_fraud_value_caught_usd": round(float(fraud_caught_value), 2),
        "estimated_fraud_value_missed_usd": round(float(fraud_missed_value), 2),
        "net_financial_savings_usd": round(float(net_savings), 2),
        "baseline_logistic_regression": {
            "precision": round(float(baseline_precision), 4),
            "recall": round(float(baseline_recall), 4),
            "f1": round(float(baseline_f1), 4),
            "roc_auc": round(float(baseline_auc), 4),
        },
        "feature_importances": dict(sorted(
            zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]),
            key=lambda x: -x[1]
        )),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "threshold": balanced_threshold,
        "policy_thresholds": metrics["policy_thresholds"],
        "train_cutoff": str(train_cutoff)
    }, MODEL_PATH)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
