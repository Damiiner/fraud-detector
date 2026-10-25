"""Replays held-out TEST transactions into the running API, like a live feed.

Run in a second terminal while the API is running:

    python simulate.py                              # 200 transactions, about 0.1 s apart
    python simulate.py --n 500 --delay 0.05 --fraud-rate 0.03
    python simulate.py --auto-review                # a pretend analyst labels the review queue

Only transactions from the notebooks' TEST split are used, so the model has never seen them.
The feed is ENRICHED with fraud (default 5%, the real rate is 0.17%) so a small run shows
interesting cases. Rates and costs from a simulated feed are not real-world numbers.
"""
import argparse
import os
import sys
import time

import httpx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

URL = "http://127.0.0.1:8000"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200, help="number of transactions to send")
ap.add_argument("--delay", type=float, default=0.1, help="seconds between transactions")
ap.add_argument("--fraud-rate", type=float, default=0.05, help="share of fraud in the feed (0 to 1)")
ap.add_argument("--auto-review", action="store_true",
                help="after sending, label every medium/high transaction with its true label")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
if not 0 <= args.fraud_rate <= 1 or args.n < 1:
    sys.exit("--n must be at least 1 and --fraud-rate must be between 0 and 1")

key = os.environ.get("FRAUD_API_KEY")
if not key:
    sys.exit("Set FRAUD_API_KEY in this window first (same value as in the server window).")
H = {"x-api-key": key}

# Rebuild the notebooks' TEST split (same dedup, same seeds) so no training rows are replayed.
df = pd.read_csv("data/creditcard.csv").drop_duplicates().reset_index(drop=True)
X, y = df.drop(columns="Class"), df["Class"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
if int(y_te.sum()) != 95:
    print(f"WARNING: expected 95 frauds in the test split but found {int(y_te.sum())}; "
          "the data may differ from the notebooks.")

rng = np.random.default_rng(args.seed)
fraud_ids, legit_ids = y_te.index[y_te == 1].to_numpy(), y_te.index[y_te == 0].to_numpy()
n_fraud = min(round(args.n * args.fraud_rate), len(fraud_ids))
chosen = np.concatenate([rng.choice(fraud_ids, n_fraud, replace=False),
                         rng.choice(legit_ids, args.n - n_fraud, replace=False)])
rng.shuffle(chosen)

results = []
try:
    with httpx.Client(timeout=60) as client:
        for i, idx in enumerate(chosen, 1):
            row = X.loc[idx, FEATURES].astype(float).to_dict()
            row["transaction_id"] = f"sim-{idx}"
            r = client.post(f"{URL}/predict", json=row, headers=H)
            if r.status_code == 401:
                sys.exit("401: the API key here differs from the key in the server window.")
            r.raise_for_status()
            p = r.json()
            actual = int(y.loc[idx])
            results.append({"id": p["id"], "tier": p["tier"], "actual": actual})
            print(f'#{i:<4d} ${row["Amount"]:>9.2f}  p={p["fraud_probability"]:.4f}  '
                  f'{p["tier"]:6s} {p["action"]:9s} (actually {"FRAUD" if actual else "legit"})')
            time.sleep(args.delay)

        res = pd.DataFrame(results)
        if args.auto_review:
            queue = res[res["tier"] != "low"]
            for _, q in queue.iterrows():
                client.post(f"{URL}/transactions/{q['id']}/review", headers=H,
                            json={"label": "fraud" if q["actual"] else "legit"}).raise_for_status()
            print(f"\nA pretend analyst reviewed {len(queue)} medium/high transactions.")
except httpx.ConnectError:
    sys.exit("Could not connect. Is the API running in the other window (and finished starting)?")

summary = (res.assign(actual=res["actual"].map({1: "fraud", 0: "legit"}))
              .groupby(["tier", "actual"]).size().unstack(fill_value=0)
              .reindex(["low", "medium", "high"]).fillna(0).astype(int))
for col in ("fraud", "legit"):
    if col not in summary:
        summary[col] = 0
print("\nSent", len(res), "transactions. Outcome by tier:")
print(summary[["fraud", "legit"]].to_string())
n_fr = int(res["actual"].sum())
caught = int(((res["tier"] != "low") & (res["actual"] == 1)).sum())
print(f"\nFrauds caught (medium or high): {caught} of {n_fr}")