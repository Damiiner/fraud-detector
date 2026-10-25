"""Fraud detection API.  Run from the project root:

    export FRAUD_API_KEY="a-long-random-string"      # Windows PowerShell: $env:FRAUD_API_KEY="..."
    uvicorn app.main:app --reload
"""
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.db import Database
from app.scoring import ACTIONS, Scorer

log = logging.getLogger("fraud.api")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
Tier = Literal["low", "medium", "high"]
Action = Literal["approve", "review", "escalate"]

# ---- request schema: Time, V1..V28, Amount (+ optional client reference) ----
_fields = {"Time": (float, Field(ge=0, allow_inf_nan=False))}
_fields.update({f"V{i}": (float, Field(allow_inf_nan=False)) for i in range(1, 29)})
_fields["Amount"] = (float, Field(ge=0, le=1_000_000, allow_inf_nan=False))
_fields["transaction_id"] = (Optional[str], Field(default=None, max_length=64))
Transaction = create_model("Transaction", __config__=ConfigDict(extra="forbid"), **_fields)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["fraud", "legit"]


class Prediction(BaseModel):
    id: int
    transaction_id: Optional[str] = None
    fraud_probability: float
    expected_loss_ratio: float
    tier: Tier
    action: Action
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class Contribution(BaseModel):
    feature: str
    value: float
    contribution: float


class Explanation(BaseModel):
    fraud_probability: float
    tier: Tier
    action: Action
    model_version: str
    xgboost_probability: float
    base_value: float
    contributions: list[Contribution]
    note: str = ("Contributions are TreeSHAP values for the XGBoost half of the ensemble, in log-odds. "
                 "Positive pushes toward fraud. V1-V28 are anonymized PCA components.")


class TransactionRow(BaseModel):
    id: int
    created_at: str
    transaction_id: Optional[str] = None
    model_version: str
    amount: float
    fraud_probability: float
    expected_loss_ratio: float
    tier: Tier
    review_label: Optional[Literal["fraud", "legit"]] = None
    reviewed_at: Optional[str] = None


class TierStats(BaseModel):
    n: int
    reviewed: int
    confirmed_fraud: int


class Stats(BaseModel):
    total: int
    by_tier: dict[str, TierStats]


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.environ.get("FRAUD_API_KEY", "")
    if len(key) < 16:
        raise RuntimeError("Set FRAUD_API_KEY (at least 16 characters) before starting the API.")
    app.state.api_key = key
    app.state.scorer = Scorer(os.environ.get("FRAUD_MODEL_DIR", "models/fraud_v1"))
    app.state.db = Database(os.environ.get("FRAUD_DB", "fraud.db"))
    log.info("Loaded model %s", app.state.scorer.version)
    yield


app = FastAPI(title="Fraud Detection API", version="1.0", lifespan=lifespan)


def require_key(request: Request, x_api_key: Optional[str] = Header(default=None)):
    expected = request.app.state.api_key.encode()
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key",
                            headers={"WWW-Authenticate": "ApiKey"})


def _raw_frame(txs) -> pd.DataFrame:
    return pd.DataFrame([{f: getattr(t, f) for f in FEATURES} for t in txs])


def _score_and_log(app: FastAPI, txs) -> list[Prediction]:
    scorer, db = app.state.scorer, app.state.db
    raw = _raw_frame(txs)
    res = scorer.score(raw)
    rows = []
    for t, (_, r) in zip(txs, res.iterrows()):
        rows.append({"transaction_id": t.transaction_id, "model_version": scorer.version,
                     "amount": float(t.Amount), "features": {f: getattr(t, f) for f in FEATURES},
                     "fraud_probability": float(r["fraud_probability"]),
                     "expected_loss_ratio": float(r["expected_loss_ratio"]), "tier": r["tier"]})
    ids = db.insert_predictions(rows)
    return [Prediction(id=i, transaction_id=row["transaction_id"], fraud_probability=row["fraud_probability"],
                       expected_loss_ratio=row["expected_loss_ratio"], tier=row["tier"],
                       action=ACTIONS[row["tier"]], model_version=scorer.version)
            for i, row in zip(ids, rows)]


@app.get("/health")
def health(request: Request):
    request.app.state.db.ping()
    scorer = request.app.state.scorer
    return {"status": "ok", "model_version": scorer.version,
            "rules": {k: scorer.cfg[k] for k in ("review_cost", "extra_per_miss", "high_risk_p")}}


DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"
# The page may only talk to this server; scripts may only come from this page and the pinned CDN files.
CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
       "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    if not DASHBOARD.exists():
        raise HTTPException(status_code=404, detail="app/static/dashboard.html is missing")
    return FileResponse(DASHBOARD, media_type="text/html",
                        headers={"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                                 "Cache-Control": "no-store"})


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_key)])
def predict(tx: Transaction, request: Request):
    return _score_and_log(request.app, [tx])[0]


@app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_key)])
def predict_batch(body: BatchRequest, request: Request):
    return BatchResponse(predictions=_score_and_log(request.app, body.transactions))


@app.post("/explain", response_model=Explanation, dependencies=[Depends(require_key)])
def explain(tx: Transaction, request: Request, top_k: int = Query(5, ge=1, le=30)):
    scorer = request.app.state.scorer
    raw = _raw_frame([tx])
    res = scorer.score(raw).iloc[0]
    return Explanation(fraud_probability=float(res["fraud_probability"]), tier=res["tier"],
                       action=ACTIONS[res["tier"]], model_version=scorer.version,
                       **scorer.explain(raw, top_k))


@app.get("/transactions", response_model=list[TransactionRow], dependencies=[Depends(require_key)])
def list_transactions(request: Request, tier: Optional[Tier] = None,
                      status: Optional[Literal["unreviewed", "fraud", "legit"]] = None,
                      limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return request.app.state.db.list_transactions(tier, status, limit, offset)


@app.post("/transactions/{tx_id}/review", response_model=TransactionRow, dependencies=[Depends(require_key)])
def review_transaction(tx_id: int, body: ReviewRequest, request: Request):
    row = request.app.state.db.set_review(tx_id, body.label)
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return row


@app.get("/stats", response_model=Stats, dependencies=[Depends(require_key)])
def stats(request: Request):
    return request.app.state.db.stats()
