"""Sends one known-fraud and one normal transaction from the dataset to the running API.

Run it in a SECOND terminal while the API is running (see the steps in the chat).
"""
import os
import sys

import httpx
import pandas as pd

URL = "http://127.0.0.1:8000"
key = os.environ.get("FRAUD_API_KEY")
if not key:
    sys.exit("Set FRAUD_API_KEY in this window first (same value as in the server window).")
H = {"x-api-key": key}

df = pd.read_csv("data/creditcard.csv").drop_duplicates().reset_index(drop=True)
samples = [("known fraud", df[df["Class"] == 1].iloc[0]), ("normal", df[df["Class"] == 0].iloc[0])]
rows = []
for name, r in samples:
    d = r.drop("Class").to_dict()
    d["transaction_id"] = name
    rows.append(d)

resp = httpx.post(f"{URL}/predict/batch", json={"transactions": rows}, headers=H, timeout=60)
resp.raise_for_status()
print("Predictions:")
for p in resp.json()["predictions"]:
    print(f'  {p["transaction_id"]:12s} probability={p["fraud_probability"]:.4f}  '
          f'tier={p["tier"]:6s} action={p["action"]}')

ex = httpx.post(f"{URL}/explain", params={"top_k": 5}, json=rows[0], headers=H, timeout=60)
ex.raise_for_status()
print("\nTop 5 reasons for the known fraud (XGBoost half, log-odds; positive = toward fraud):")
for c in ex.json()["contributions"]:
    print(f'  {c["feature"]:7s} value={c["value"]:10.3f}  contribution={c["contribution"]:+.2f}')

print("\nQueue stats:", httpx.get(f"{URL}/stats", headers=H, timeout=60).json())
