import math
import os

import pytest
from fastapi.testclient import TestClient

KEY = "test-key-0123456789abcdef"
H = {"X-API-Key": KEY}


def make_tx(**over):
    tx = {"Time": 10000.0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 50.0}
    tx.update(over)
    return tx


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("FRAUD_API_KEY", KEY)
    mp.setenv("FRAUD_MODEL_DIR", os.environ.get("FRAUD_MODEL_DIR", "models/fraud_v1"))
    mp.setenv("FRAUD_DB", str(tmp_path_factory.mktemp("db") / "test.db"))
    from app.main import app
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health_is_open(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("method,path", [("post", "/predict"), ("post", "/explain"),
                                         ("get", "/transactions"), ("get", "/stats"),
                                         ("post", "/predict/batch"), ("post", "/transactions/1/review")])
def test_endpoints_require_key(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("bad", [
    {"Amount": -1.0}, {"Amount": 1e9}, {"Time": -5.0}, {"V3": "abc"}, {"unexpected": 1},
])
def test_validation_rejects_bad_input(client, bad):
    assert client.post("/predict", json=make_tx(**bad), headers=H).status_code == 422


def test_missing_field_rejected(client):
    tx = make_tx(); del tx["V14"]
    assert client.post("/predict", json=tx, headers=H).status_code == 422


def test_predict_returns_valid_result(client):
    r = client.post("/predict", json=make_tx(transaction_id="t-valid"), headers=H)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["fraud_probability"] <= 1 and j["tier"] in ("low", "medium", "high")
    assert j["action"] == {"low": "approve", "medium": "review", "high": "escalate"}[j["tier"]]
    assert j["transaction_id"] == "t-valid" and isinstance(j["id"], int)


def test_prediction_is_logged(client):
    j = client.post("/predict", json=make_tx(transaction_id="t-logged"), headers=H).json()
    rows = client.get("/transactions", params={"limit": 500}, headers=H).json()
    assert j["id"] in [r["id"] for r in rows]


def test_batch_matches_single(client):
    txs = [make_tx(Amount=a, V3=v) for a, v in [(5.0, -3.0), (500.0, 0.5), (20.0, -6.0)]]
    batch = client.post("/predict/batch", json={"transactions": txs}, headers=H).json()["predictions"]
    singles = [client.post("/predict", json=t, headers=H).json() for t in txs]
    for b, s in zip(batch, singles):
        assert math.isclose(b["fraud_probability"], s["fraud_probability"], abs_tol=1e-5)
        assert b["tier"] == s["tier"]


def test_batch_size_limits(client):
    assert client.post("/predict/batch", json={"transactions": []}, headers=H).status_code == 422
    too_many = {"transactions": [make_tx()] * 1001}
    assert client.post("/predict/batch", json=too_many, headers=H).status_code == 422


def test_review_flow_and_stats(client):
    before = client.get("/stats", headers=H).json()
    j = client.post("/predict", json=make_tx(transaction_id="t-review"), headers=H).json()
    r = client.post(f"/transactions/{j['id']}/review", json={"label": "fraud"}, headers=H)
    assert r.status_code == 200 and r.json()["review_label"] == "fraud" and r.json()["reviewed_at"]
    fraud_rows = client.get("/transactions", params={"status": "fraud", "limit": 500}, headers=H).json()
    assert j["id"] in [x["id"] for x in fraud_rows]
    unreviewed = client.get("/transactions", params={"status": "unreviewed", "limit": 500}, headers=H).json()
    assert j["id"] not in [x["id"] for x in unreviewed]
    after = client.get("/stats", headers=H).json()
    assert after["total"] == before["total"] + 1
    assert sum(t["confirmed_fraud"] for t in after["by_tier"].values()) == \
           sum(t["confirmed_fraud"] for t in before["by_tier"].values()) + 1


def test_review_rejects_bad_label_and_unknown_id(client):
    j = client.post("/predict", json=make_tx(), headers=H).json()
    assert client.post(f"/transactions/{j['id']}/review", json={"label": "maybe"}, headers=H).status_code == 422
    assert client.post("/transactions/99999999/review", json={"label": "legit"}, headers=H).status_code == 404


def test_list_filters_validate(client):
    assert client.get("/transactions", params={"tier": "extreme"}, headers=H).status_code == 422
    assert client.get("/transactions", params={"limit": 0}, headers=H).status_code == 422


def test_explain_contributions_reconstruct_xgboost_probability(client):
    r = client.post("/explain", params={"top_k": 30}, json=make_tx(V3=-4.0, V14=-5.0), headers=H)
    assert r.status_code == 200
    j = r.json()
    assert len(j["contributions"]) == 30
    margin = j["base_value"] + sum(c["contribution"] for c in j["contributions"])
    assert math.isclose(1 / (1 + math.exp(-margin)), j["xgboost_probability"], abs_tol=1e-4)
    top5 = client.post("/explain", json=make_tx(V3=-4.0, V14=-5.0), headers=H).json()["contributions"]
    assert len(top5) == 5
    assert abs(top5[0]["contribution"]) >= abs(top5[-1]["contribution"])
