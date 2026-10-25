"""Loads the saved model artifact and scores raw transactions.

This mirrors section 7 of notebook 04, so the API gives the same numbers as the notebook.
"""
import json
import logging
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from tensorflow import keras
import tensorflow as tf
from xgboost import XGBClassifier

log = logging.getLogger("fraud.scoring")

EPS = 1e-6
ACTIONS = {"low": "approve", "medium": "review", "high": "escalate"}


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Scorer:
    def __init__(self, model_dir):
        d = Path(model_dir)
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"No model artifact found in '{d}'. Run notebook 04 first.")
        self.cfg = json.loads((d / "config.json").read_text())
        self.features = self.cfg["features"]
        self.version = self.cfg["version"]
        self.xgb = XGBClassifier()
        self.xgb.load_model(str(d / "xgb.json"))
        self.keras = keras.models.load_model(d / "keras.keras")
        self.scaler = joblib.load(d / "scaler.joblib")
        self.platt = joblib.load(d / "platt.joblib")
        self._lock = threading.Lock()   # one prediction at a time keeps inference simple and safe
        self._check_versions()

    def _check_versions(self):
        now = {"tensorflow": tf.__version__, "xgboost": xgb.__version__, "scikit-learn": sklearn.__version__}
        for lib, saved in self.cfg.get("libraries", {}).items():
            if now.get(lib) != saved:
                log.warning("Library mismatch for %s: artifact built with %s, running %s",
                            lib, saved, now.get(lib))

    def _prepare(self, raw: pd.DataFrame) -> np.ndarray:
        x = raw[self.features].astype(float).copy()
        x["Amount"] = np.log1p(x["Amount"])
        x[["Time", "Amount"]] = self.scaler.transform(x[["Time", "Amount"]])
        return x.to_numpy()

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """raw: columns Time, V1..V28, Amount (Amount in dollars)."""
        v = self._prepare(raw)
        amount = raw["Amount"].to_numpy(dtype=float)
        with self._lock:
            p_xgb = self.xgb.predict_proba(v)[:, 1]
            p_keras = np.asarray(self.keras(v, training=False)).ravel()
        s = (p_xgb + p_keras) / 2
        p = np.clip(self.platt.predict_proba(_logit(s))[:, 1], EPS, 1 - EPS)
        ratio = p * (amount + self.cfg["extra_per_miss"]) / self.cfg["review_cost"]
        tier = np.where(p >= self.cfg["high_risk_p"], "high", np.where(ratio >= 1, "medium", "low"))
        return pd.DataFrame({"fraud_probability": p, "expected_loss_ratio": ratio, "tier": tier},
                            index=raw.index)

    def explain(self, raw: pd.DataFrame, top_k: int = 5) -> dict:
        """TreeSHAP contributions from the XGBoost half of the ensemble (log-odds scale)."""
        v = self._prepare(raw)
        with self._lock:
            contribs = self.xgb.get_booster().predict(xgb.DMatrix(v), pred_contribs=True)[0]
            p_xgb = float(self.xgb.predict_proba(v)[0, 1])
        values = raw[self.features].iloc[0]
        rows = [{"feature": f, "value": float(values[f]), "contribution": float(c)}
                for f, c in zip(self.features, contribs[:-1])]
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return {"xgboost_probability": p_xgb, "base_value": float(contribs[-1]), "contributions": rows[:top_k]}
