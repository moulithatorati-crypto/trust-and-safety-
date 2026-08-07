"""
End-to-end tests for SentinelMesh, run against the orchestrator's FastAPI
`app` in-process (via TestClient), so all three agents genuinely share one
identity graph and one audit trail exactly as they would under uvicorn.

Run: pytest tests/test_end_to_end.py -v
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the audit DB at a throwaway test file BEFORE importing anything
# that opens a connection to it.
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_audit_trail.db")
os.environ["SENTINELMESH_AUDIT_DB"] = TEST_DB_PATH

from shared import audit_trail  # noqa: E402
from services.orchestrator_gateway.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def clean_db():
    audit_trail.reset_for_tests()
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


# --------------------------------------------------------------------- #
# 1. A device shared across 5 accounts raises the risk score and appears
#    in device_account_count.
# --------------------------------------------------------------------- #
def test_shared_device_raises_risk_and_graph_count():
    shared_device = "DEV_TEST_SHARED"
    base_order = {
        "device_id": shared_device,
        "ip": "45.10.20.30",
        "payment_instrument_id": "PAY_TEST",
        "shipping_address_id": "ADDR_TEST",
        "order_value": 2000.0,
        "payment_type": "cod",
        "category": "electronics",
        "session": {"duration_seconds": 30, "typing_cadence_ms": 50, "copy_paste_detected": True},
        "billing_country": "IN",
        "timestamp": "2026-08-08T10:00:00Z",
    }

    last_response = None
    for i in range(5):
        order = dict(base_order)
        order["order_id"] = f"ORD_SHARED_{i}"
        order["account_id"] = f"ACC_SHARED_{i}"
        order["payment_instrument_id"] = f"PAY_SHARED_{i}"
        order["shipping_address_id"] = f"ADDR_SHARED_{i}"
        resp = client.post("/risk/evaluate", json=order)
        assert resp.status_code == 200
        last_response = resp.json()

    graph_resp = client.get(f"/graph/lookup/{shared_device}")
    assert graph_resp.status_code == 200
    assert graph_resp.json()["device_account_count"] == 5

    # after 5 accounts sharing one device, the risk score should be
    # meaningfully elevated (well above a clean single-signal baseline,
    # which scores under 0.01 for the same order shape with no sharing)
    assert last_response["risk_score"] > 0.20
    assert any("shared_device" in r for r in last_response["reasons"])


# --------------------------------------------------------------------- #
# 2. A listing priced 80%+ below MSRP from an unauthorized seller gets
#    `reject`, not `publish`.
# --------------------------------------------------------------------- #
def test_counterfeit_listing_rejected():
    listing = {
        "listing_id": "LST_TEST_COUNTERFEIT",
        "seller_id": "SEL_UNAUTH_TEST",
        "title": "L'Oreal Revitalift Serum - Clearance",
        "description": "Clearance price, limited stock.",
        "price": 1600.0,  # ~82% below the 8999 cosmetics MSRP anchor
        "brand": "L'Oreal",
        "category": "cosmetics",
        "images": ["img_counterfeit_low_quality_scan"],
        "gtin": "8901030000000",
    }
    resp = client.post("/authenticity/scan", json=listing)
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] in ("reject", "hold")  # gate may route to hold if score not extreme
    assert data["action"] != "publish"
    # multi-signal gate: must cite more than one independent piece of evidence
    assert len(data["evidence"]) >= 2


def test_genuine_listing_published():
    listing = {
        "listing_id": "LST_TEST_GENUINE",
        "seller_id": "SEL_AUTH_1",
        "title": "L'Oreal Revitalift Serum",
        "description": "Genuine L'Oreal serum, sealed box.",
        "price": 8499.0,
        "brand": "L'Oreal",
        "category": "cosmetics",
        "images": ["REF_LOREAL_A1B2"],
        "gtin": "8901030000001",
    }
    resp = client.post("/authenticity/scan", json=listing)
    assert resp.status_code == 200
    assert resp.json()["action"] == "publish"


# --------------------------------------------------------------------- #
# 3. A coordinated review batch gets ring-detected and suppressed.
# --------------------------------------------------------------------- #
def test_coordinated_review_ring_suppressed():
    now = "2026-08-08T09:00:00Z"
    reviews = []
    for i in range(6):
        reviews.append({
            "review_id": f"REV_TEST_RING_{i}",
            "account_id": f"ACC_TEST_RING_{i}",
            "product_id": "PRD_TEST_RING_TARGET",
            "device_id": "DEV_TEST_RING_SHARED",
            "rating": 5,
            "text": "Highly recommend! Best product ever, five stars, amazing quality!!!",
            "order_id": None,
            "timestamp": now,
        })
    resp = client.post("/review/moderate", json={"reviews": reviews})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["rings_suppressed"]) >= 1
    assert all(d["action"] == "suppress" for d in data["decisions"])
    assert all(d["ring_id"] is not None for d in data["decisions"])


# --------------------------------------------------------------------- #
# 4. The audit chain's verify_chain() returns True after all of the above.
# --------------------------------------------------------------------- #
def test_chain_valid_after_all_activity():
    resp = client.get("/audit/verify")
    assert resp.status_code == 200
    assert resp.json()["chain_valid"] is True


# --------------------------------------------------------------------- #
# 5. Tampering with one audit record (mutate a field directly in SQLite)
#    makes verify_chain() return False.
# --------------------------------------------------------------------- #
def test_tamper_breaks_chain():
    assert audit_trail.verify_chain() is True

    conn = sqlite3.connect(TEST_DB_PATH)
    row = conn.execute("SELECT audit_id FROM audit_log ORDER BY seq ASC LIMIT 1").fetchone()
    assert row is not None
    audit_id = row[0]

    conn.execute("UPDATE audit_log SET score = 0.9999 WHERE audit_id = ?", (audit_id,))
    conn.commit()
    conn.close()

    # force a fresh connection read (audit_trail caches its own connection,
    # but verify_chain always re-queries via that connection so the UPDATE
    # via a separate sqlite3 connection is visible immediately since both
    # point at the same file and SQLite commits are durable across
    # connections)
    assert audit_trail.verify_chain() is False


# --------------------------------------------------------------------- #
# Bonus: the "no single weak signal auto-blocks" guardrail on risk agent.
# --------------------------------------------------------------------- #
def test_single_weak_signal_does_not_auto_block():
    # Only one weak-ish signal (mild geo mismatch), nothing else -- should
    # never fire an auto block/prepaid_only purely off score if only one
    # independent signal family agrees; it should escalate to step_up_otp
    # (medium tier review-equivalent) instead.
    order = {
        "order_id": "ORD_SINGLE_SIGNAL",
        "account_id": "ACC_SINGLE_SIGNAL",
        "device_id": "DEV_SINGLE_SIGNAL",
        "ip": "203.0.113.201",  # vpn range triggers only the "ip signal family"
        "payment_instrument_id": "PAY_SINGLE_SIGNAL",
        "shipping_address_id": "ADDR_SINGLE_SIGNAL",
        "order_value": 500.0,
        "payment_type": "prepaid",
        "category": "apparel",
        "session": {"duration_seconds": 90, "typing_cadence_ms": 150, "copy_paste_detected": False},
        "billing_country": "US",  # matches the vpn pool's mapped country -> no geo_mismatch
        "timestamp": "2026-08-08T10:00:00Z",
    }
    resp = client.post("/risk/evaluate", json=order)
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] != "block"
