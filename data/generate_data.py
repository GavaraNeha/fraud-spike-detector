"""
Synthetic transaction data generator for the Fraud-Spike Detector.

Simulates a merchant's transaction stream with normal traffic plus
injected "fraud spikes" — short bursts of anomalous transactions that
share the velocity/device/amount signatures real fraud rings exhibit:
  - Rapid-fire transactions from the same device/IP in a short window
  - Amount z-score outliers relative to that merchant's normal ticket size
  - New/unrecognized device + high amount combos
  - Off-hours bursts (2am-5am local) at volumes inconsistent with normal traffic

This is intentionally NOT hand-crafted per-row — it's generated from
distributions + injected spike events, so the model has to actually
learn the pattern rather than memorize a lookup table.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_NORMAL = 4200
N_MERCHANTS = 12
N_DEVICES_POOL = 800
START = datetime(2026, 8, 1)
DAYS = 30


def gen_normal_transactions(n):
    merchant_ids = rng.integers(0, N_MERCHANTS, size=n)
    # each merchant has its own "typical ticket size" -> lognormal per-merchant
    merchant_mu = {m: rng.uniform(5.5, 7.5) for m in range(N_MERCHANTS)}  # log-scale
    amounts = np.array([
        rng.lognormal(mean=merchant_mu[m], sigma=0.5) for m in merchant_ids
    ])

    # timestamps spread across DAYS, weighted toward normal business hours
    day_offsets = rng.uniform(0, DAYS, size=n)
    hour_weights_normal = np.concatenate([
        np.full(6, 0.3),   # 00-06: quiet
        np.full(12, 1.5),  # 06-18: business hours
        np.full(6, 0.6),   # 18-24: evening
    ])
    hour_weights_normal = hour_weights_normal / hour_weights_normal.sum()
    hours = rng.choice(24, size=n, p=hour_weights_normal)
    minutes = rng.integers(0, 60, size=n)
    timestamps = [START + timedelta(days=float(d), hours=int(h), minutes=int(m))
                  for d, h, m in zip(day_offsets, hours, minutes)]

    device_ids = rng.integers(0, N_DEVICES_POOL, size=n)
    # 92% of normal transactions come from a device the customer has used before
    device_is_known = rng.random(n) < 0.92

    ip_country_match = rng.random(n) < 0.97  # billing country == IP country

    df = pd.DataFrame({
        "transaction_id": [f"TXN{100000+i}" for i in range(n)],
        "merchant_id": merchant_ids,
        "timestamp": timestamps,
        "amount": amounts.round(2),
        "device_id": device_ids,
        "device_is_known": device_is_known,
        "ip_country_match": ip_country_match,
        "label": 0,
    })
    return df


def gen_fraud_spikes():
    """
    Inject ~6-9 distinct spike events. Each spike is a short burst
    (5-40 transactions within a tight time window) from a small pool
    of devices, at elevated amounts, mostly unknown devices, mostly
    off-hours, mostly IP/country mismatch.
    """
    spikes = []
    n_spikes = rng.integers(6, 10)
    for _ in range(n_spikes):
        merchant_id = rng.integers(0, N_MERCHANTS)
        burst_size = rng.integers(5, 40)
        # tight window: 2-25 minutes
        window_start = START + timedelta(
            days=float(rng.uniform(0, DAYS)),
            hours=int(rng.choice([1, 2, 3, 4, 23])),  # skew off-hours
        )
        window_minutes = rng.uniform(2, 25)

        # small device pool reused across the burst (device farm signature)
        device_pool = rng.integers(0, N_DEVICES_POOL, size=max(1, burst_size // 8))

        amount_multiplier = rng.uniform(2.2, 5.0)  # elevated vs normal ticket
        base_amount = rng.lognormal(mean=6.0, sigma=0.4) * amount_multiplier

        for i in range(burst_size):
            offset_min = rng.uniform(0, window_minutes)
            ts = window_start + timedelta(minutes=float(offset_min))
            amt = base_amount * rng.uniform(0.7, 1.3)
            device = int(rng.choice(device_pool))
            spikes.append({
                "transaction_id": None,  # filled later
                "merchant_id": int(merchant_id),
                "timestamp": ts,
                "amount": round(float(amt), 2),
                "device_id": device,
                "device_is_known": bool(rng.random() < 0.08),  # rarely known
                "ip_country_match": bool(rng.random() < 0.25),  # usually mismatched
                "label": 1,
            })
    df = pd.DataFrame(spikes)
    df["transaction_id"] = [f"TXNF{200000+i}" for i in range(len(df))]
    return df


def build_dataset():
    normal = gen_normal_transactions(N_NORMAL)
    fraud = gen_fraud_spikes()
    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = "/home/claude/fraud-spike-detector/data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} transactions ({df['label'].sum()} fraud, "
          f"{len(df) - df['label'].sum()} normal) to {out_path}")
    print(df['label'].value_counts(normalize=True))
