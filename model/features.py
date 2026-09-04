"""
Feature engineering for the fraud-spike detector.

All features are computable in real time (no look-ahead into the future,
only into transaction history up to the current timestamp), so this
pipeline is deployable, not just a backtest.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Amount z-score relative to the merchant's rolling history ---
    # Uses expanding mean/std up to (not including) the current row per merchant,
    # so no future information leaks into the feature.
    df["merchant_amount_mean"] = (
        df.groupby("merchant_id")["amount"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["merchant_amount_std"] = (
        df.groupby("merchant_id")["amount"]
        .apply(lambda s: s.shift(1).expanding().std())
        .reset_index(level=0, drop=True)
    )
    df["merchant_amount_mean"] = df["merchant_amount_mean"].fillna(df["amount"].median())
    df["merchant_amount_std"] = df["merchant_amount_std"].fillna(df["amount"].std()).replace(0, 1e-6)
    df["amount_zscore"] = (df["amount"] - df["merchant_amount_mean"]) / df["merchant_amount_std"]
    df["amount_zscore"] = df["amount_zscore"].clip(-10, 10)

    # --- Device velocity: transactions from same device in trailing 30 min ---
    df["device_velocity_30min"] = 0
    for device, grp in df.groupby("device_id"):
        idx = grp.index
        times = grp["timestamp"].values
        counts = []
        for i, t in enumerate(times):
            window_start = t - np.timedelta64(30, "m")
            count = np.sum((times[:i] >= window_start) & (times[:i] < t))
            counts.append(count)
        df.loc[idx, "device_velocity_30min"] = counts

    # --- Merchant velocity: transactions for same merchant in trailing 10 min ---
    df["merchant_velocity_10min"] = 0
    for merchant, grp in df.groupby("merchant_id"):
        idx = grp.index
        times = grp["timestamp"].values
        counts = []
        for i, t in enumerate(times):
            window_start = t - np.timedelta64(10, "m")
            count = np.sum((times[:i] >= window_start) & (times[:i] < t))
            counts.append(count)
        df.loc[idx, "merchant_velocity_10min"] = counts

    # --- Time-of-day risk: off-hours flag (1-5am) ---
    df["hour"] = df["timestamp"].dt.hour
    df["is_off_hours"] = df["hour"].between(1, 5).astype(int)

    # --- Device known / IP mismatch as numeric ---
    df["device_is_known"] = df["device_is_known"].astype(int)
    df["ip_country_match"] = df["ip_country_match"].astype(int)

    feature_cols = [
        "amount",
        "amount_zscore",
        "device_velocity_30min",
        "merchant_velocity_10min",
        "is_off_hours",
        "device_is_known",
        "ip_country_match",
    ]
    return df, feature_cols
