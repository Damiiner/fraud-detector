"""SQLite storage for predictions and analyst reviews (standard library only)."""
import json
import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    transaction_id TEXT,
    model_version TEXT NOT NULL,
    amount REAL NOT NULL,
    features_json TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    expected_loss_ratio REAL NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('low','medium','high')),
    review_label TEXT CHECK (review_label IN ('fraud','legit')),
    reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_tier ON predictions(tier, id);
"""

COLUMNS = ("id, created_at, transaction_id, model_version, amount, fraud_probability, "
           "expected_loss_ratio, tier, review_label, reviewed_at")


class Database:
    def __init__(self, path):
        self.path = str(path)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ping(self) -> bool:
        with self._conn() as c:
            c.execute("SELECT 1")
        return True

    def insert_predictions(self, rows: list[dict]) -> list[int]:
        ids = []
        with self._conn() as c:
            for r in rows:
                cur = c.execute(
                    "INSERT INTO predictions (transaction_id, model_version, amount, features_json, "
                    "fraud_probability, expected_loss_ratio, tier) VALUES (?,?,?,?,?,?,?)",
                    (r["transaction_id"], r["model_version"], r["amount"], json.dumps(r["features"]),
                     r["fraud_probability"], r["expected_loss_ratio"], r["tier"]))
                ids.append(cur.lastrowid)
        return ids

    def list_transactions(self, tier=None, status=None, limit=50, offset=0) -> list[dict]:
        where, params = [], []
        if tier:
            where.append("tier = ?"); params.append(tier)
        if status == "unreviewed":
            where.append("review_label IS NULL")
        elif status in ("fraud", "legit"):
            where.append("review_label = ?"); params.append(status)
        sql = f"SELECT {COLUMNS} FROM predictions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, (*params, limit, offset)).fetchall()]

    def set_review(self, tx_id: int, label: str):
        with self._conn() as c:
            cur = c.execute(
                "UPDATE predictions SET review_label = ?, "
                "reviewed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?", (label, tx_id))
            if cur.rowcount == 0:
                return None
            return dict(c.execute(f"SELECT {COLUMNS} FROM predictions WHERE id = ?", (tx_id,)).fetchone())

    def stats(self) -> dict:
        with self._conn() as c:
            rows = c.execute(
                "SELECT tier, COUNT(*) AS n, COALESCE(SUM(review_label IS NOT NULL), 0) AS reviewed, "
                "COALESCE(SUM(review_label = 'fraud'), 0) AS confirmed_fraud "
                "FROM predictions GROUP BY tier").fetchall()
        by_tier = {t: {"n": 0, "reviewed": 0, "confirmed_fraud": 0} for t in ("low", "medium", "high")}
        for r in rows:
            by_tier[r["tier"]] = {"n": r["n"], "reviewed": r["reviewed"], "confirmed_fraud": r["confirmed_fraud"]}
        return {"total": sum(v["n"] for v in by_tier.values()), "by_tier": by_tier}
