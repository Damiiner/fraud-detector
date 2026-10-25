import os

import pytest
from fastapi.testclient import TestClient

KEY = "test-key-0123456789abcdef"


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


def test_dashboard_page_is_served_with_security_headers(client):
    r = client.get("/dashboard")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "connect-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert KEY not in r.text and "integrity=" in r.text      # no secret in the page; CDN scripts are hash-pinned


def test_health_exposes_decision_rules_for_the_dashboard(client):
    rules = client.get("/health").json()["rules"]
    assert set(rules) == {"review_cost", "extra_per_miss", "high_risk_p"}
