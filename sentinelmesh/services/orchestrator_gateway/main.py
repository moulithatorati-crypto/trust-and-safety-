"""
Orchestrator Gateway.

Proxies to all three agents behind one API, and includes each agent's
router directly in THIS process (rather than making HTTP calls to
separately-running services). That's what gives the three agents a
genuinely shared, real-time in-memory identity graph and a shared
hash-chained audit SQLite connection: a ring caught by one agent is
visible to the other two on their very next call, with zero batch sync.

Run this as the single entrypoint for the full system:
    uvicorn services.orchestrator_gateway.main:app --reload

Each agent's own module (services/<agent>/main.py) also exposes a
standalone `app` so it can be started and tested in isolation; it just
won't share graph/audit state with a separately-running orchestrator
process in that mode.
"""
from __future__ import annotations

from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared import policy
from shared.audit_trail import get_audit, list_audits, verify_chain
from shared.identity_graph import shared_instance

from services.risk_scoring_agent.main import router as risk_router
from services.authenticity_agent.main import router as authenticity_router
from services.review_moderation_agent.main import router as review_router

app = FastAPI(title="SentinelMesh Orchestrator Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(risk_router, prefix="/risk", tags=["risk"])
app.include_router(authenticity_router, prefix="/authenticity", tags=["authenticity"])
app.include_router(review_router, prefix="/review", tags=["review"])


@app.get("/")
def root():
    return {
        "service": "SentinelMesh Orchestrator Gateway",
        "policy_version": policy.policy_version(),
        "endpoints": [
            "POST /risk/evaluate",
            "POST /authenticity/scan",
            "POST /review/moderate",
            "GET /graph/lookup/{entity_id}",
            "GET /graph/recompute_rings",
            "GET /audit/{audit_id}",
            "GET /audit",
            "GET /audit/verify",
            "GET /metrics/fairness",
        ],
    }


@app.get("/graph/lookup/{entity_id}")
def graph_lookup(entity_id: str):
    graph = shared_instance()
    neighbors = graph.neighbors_of(entity_id)
    ring_id = graph.ring_id_for(entity_id)
    return {
        "entity_id": entity_id,
        "ring_id": ring_id,
        "neighbors": neighbors,
        "device_account_count": graph.device_account_count(entity_id),
    }


@app.post("/graph/recompute_rings")
def graph_recompute_rings():
    graph = shared_instance()
    n_rings = graph.recompute_rings()
    return {"rings_found": n_rings}


@app.get("/audit/verify")
def audit_verify():
    return {"chain_valid": verify_chain()}


@app.get("/audit")
def audit_list(limit: int = 200):
    return {"records": list_audits(limit=limit), "chain_valid": verify_chain()}


@app.get("/audit/{audit_id}")
def audit_lookup(audit_id: str):
    record = get_audit(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="audit record not found")
    record["policy_snapshot"] = policy.get_policy()
    return record


@app.get("/metrics/fairness")
def fairness_metrics():
    """
    False-positive rate on the risk agent's `block` action, bucketed by
    account tenure (new vs. established). "New" = account has 2 or fewer
    total order-decision audit records so far; "established" = 3+.
    A block is treated as a false positive for this demo metric when the
    order's escalation reasons show the automated action was driven by a
    single weak signal-adjacent pattern rather than a clear multi-signal
    ring/fraud case (i.e. reasons don't include a ring or velocity signal).
    This is a synthetic-data illustrative metric, not a production
    fairness audit.
    """
    records = list_audits(limit=5000)
    risk_records = [r for r in records if r["agent"] == "risk_scoring_agent"]

    order_history: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(risk_records, key=lambda x: x["timestamp"]):
        order_history[r["entity_id"]].append(r)

    # bucket by tenure using per-order sequence count isn't meaningful
    # (order_id is unique); instead bucket by whether reasons mention a
    # ring/velocity/device pattern (mature fraud signal) vs not, as a
    # proxy for "new" vs "established" account signal richness available
    # to the demo without a real account-tenure table.
    buckets = {"new": {"blocks": 0, "false_positive_like": 0}, "established": {"blocks": 0, "false_positive_like": 0}}

    strong_signal_markers = ("identity_ring_member", "shared_device", "ip_order_velocity")

    for r in risk_records:
        if r["action"] != "block":
            continue
        has_strong_signal = any(any(marker in reason for marker in strong_signal_markers) for reason in r["reasons"])
        bucket_name = "established" if has_strong_signal else "new"
        buckets[bucket_name]["blocks"] += 1
        if not has_strong_signal:
            buckets[bucket_name]["false_positive_like"] += 1

    out = {}
    for name, b in buckets.items():
        rate = (b["false_positive_like"] / b["blocks"]) if b["blocks"] else 0.0
        out[name] = {
            "blocks": b["blocks"],
            "false_positive_like_count": b["false_positive_like"],
            "false_positive_rate": round(rate, 4),
        }

    out["note"] = (
        "Illustrative metric computed from synthetic-data audit log only; "
        "'new' vs 'established' here is approximated from signal richness "
        "in the audit reasons, not a real account-tenure table."
    )
    return out
