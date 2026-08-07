"""
Append-only, hash-chained audit trail backed by SQLite. Each record's
`record_hash` = sha256(canonical record fields + prev_hash), so mutating
any past record breaks every hash that follows it. The DB user we connect
as only ever runs INSERT/SELECT -- no UPDATE/DELETE statements exist
anywhere in this module, and we additionally revoke nothing-needed
privileges are irrelevant for local SQLite, so instead we enforce
append-only at the application layer and verify_chain() is the tamper
detector of record.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "SENTINELMESH_AUDIT_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "audit_trail.db"),
)

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

GENESIS_HASH = "0" * 64


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id TEXT UNIQUE NOT NULL,
                agent TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                score REAL NOT NULL,
                tier TEXT NOT NULL,
                action TEXT NOT NULL,
                reasons TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def _canonical_payload(
    audit_id, agent, entity_id, entity_type, score, tier, action, reasons,
    policy_version, timestamp, prev_hash,
) -> str:
    payload = {
        "audit_id": audit_id,
        "agent": agent,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "score": round(float(score), 6),
        "tier": tier,
        "action": action,
        "reasons": reasons,
        "policy_version": policy_version,
        "timestamp": timestamp,
        "prev_hash": prev_hash,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _last_hash(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT record_hash FROM audit_log ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else GENESIS_HASH


def write_audit(
    agent: str,
    entity_id: str,
    entity_type: str,
    score: float,
    tier: str,
    action: str,
    reasons: list[str],
    policy_version: str,
) -> str:
    with _lock:
        conn = _get_conn()
        prev_hash = _last_hash(conn)
        audit_id = f"AUD_{uuid.uuid4().hex[:16]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = _canonical_payload(
            audit_id, agent, entity_id, entity_type, score, tier, action,
            reasons, policy_version, timestamp, prev_hash,
        )
        record_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        conn.execute(
            """
            INSERT INTO audit_log
            (audit_id, agent, entity_id, entity_type, score, tier, action,
             reasons, policy_version, timestamp, prev_hash, record_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id, agent, entity_id, entity_type, float(score), tier,
                action, json.dumps(reasons), policy_version, timestamp,
                prev_hash, record_hash,
            ),
        )
        conn.commit()
        return audit_id


def get_audit(audit_id: str) -> dict | None:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            """
            SELECT audit_id, agent, entity_id, entity_type, score, tier,
                   action, reasons, policy_version, timestamp, prev_hash,
                   record_hash
            FROM audit_log WHERE audit_id = ?
            """,
            (audit_id,),
        ).fetchone()
        if not row:
            return None
        cols = [
            "audit_id", "agent", "entity_id", "entity_type", "score", "tier",
            "action", "reasons", "policy_version", "timestamp", "prev_hash",
            "record_hash",
        ]
        record = dict(zip(cols, row))
        record["reasons"] = json.loads(record["reasons"])
        return record


def list_audits(limit: int = 200) -> list[dict]:
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT audit_id, agent, entity_id, entity_type, score, tier,
                   action, reasons, policy_version, timestamp, prev_hash,
                   record_hash
            FROM audit_log ORDER BY seq DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        cols = [
            "audit_id", "agent", "entity_id", "entity_type", "score", "tier",
            "action", "reasons", "policy_version", "timestamp", "prev_hash",
            "record_hash",
        ]
        out = []
        for row in rows:
            record = dict(zip(cols, row))
            record["reasons"] = json.loads(record["reasons"])
            out.append(record)
        return out


def verify_chain() -> bool:
    """
    Walks the whole table in insertion order and confirms every record's
    stored hash matches a fresh recomputation, and that prev_hash correctly
    links to the previous record's stored hash. Returns False the moment
    anything doesn't match -- i.e. detects tampering anywhere in history.
    """
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT audit_id, agent, entity_id, entity_type, score, tier,
                   action, reasons, policy_version, timestamp, prev_hash,
                   record_hash
            FROM audit_log ORDER BY seq ASC
            """
        ).fetchall()

        expected_prev = GENESIS_HASH
        for row in rows:
            (audit_id, agent, entity_id, entity_type, score, tier, action,
             reasons_json, policy_version, timestamp, prev_hash,
             record_hash) = row

            if prev_hash != expected_prev:
                return False

            reasons = json.loads(reasons_json)
            payload = _canonical_payload(
                audit_id, agent, entity_id, entity_type, score, tier,
                action, reasons, policy_version, timestamp, prev_hash,
            )
            recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if recomputed != record_hash:
                return False

            expected_prev = record_hash

        return True


def reset_for_tests():
    """Test helper: drop and recreate the table + close cached connection."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        _get_conn()
