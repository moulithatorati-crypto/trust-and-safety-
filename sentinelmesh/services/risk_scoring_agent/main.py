"""
Agent 1: Risk Scoring Agent.

POST /evaluate  (also exposed as POST /risk/evaluate on the orchestrator)

Combines graph signals (device_account_count, ip_order_count, ring
membership), IP intelligence, and behavioral drift into a risk score,
prefers a calibrated LogisticRegression trained by scripts/train_models.py,
and falls back to a transparent weighted rule-based score if the model
artifact is missing so the service never hard-fails.
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
from fastapi import FastAPI, APIRouter

from shared import policy
from shared.audit_trail import write_audit
from shared.behavioral_drift import drift_score
from shared.identity_graph import shared_instance
from shared.ip_intelligence import check_ip
from shared.schemas import OrderEvent, RiskDecision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "risk_model.joblib",
)

router = APIRouter()

_model = None
_model_loaded_from_disk = False


def _load_model():
    global _model, _model_loaded_from_disk
    if os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            _model_loaded_from_disk = True
        except Exception:
            _model = None
            _model_loaded_from_disk = False
    else:
        _model = None
        _model_loaded_from_disk = False


_load_model()

FEATURE_NAMES = [
    "order_value_norm",
    "device_account_count",
    "ip_order_count",
    "drift_score",
    "is_vpn",
    "is_datacenter",
    "geo_mismatch",
    "in_ring",
    "cod_and_high_value",
]

# High-order-value normalization reference (99th percentile-ish clamp).
_ORDER_VALUE_CAP = 25000.0


def _build_features(order: OrderEvent) -> dict:
    graph = shared_instance()

    dev_count = graph.device_account_count(order.device_id)
    ip_count = graph.ip_order_count(order.ip, window_minutes=10)
    drift = drift_score(order.account_id, order.session.model_dump())
    ip_info = check_ip(order.ip, order.billing_country)
    ring = graph.ring_id_for(order.account_id) or graph.ring_id_for(order.device_id)

    order_value_norm = min(order.order_value / _ORDER_VALUE_CAP, 1.0)
    cod_and_high_value = 1.0 if (order.payment_type == "cod" and order.order_value > 3000) else 0.0

    return {
        "order_value_norm": order_value_norm,
        "device_account_count": min(dev_count / 8.0, 1.0),
        "ip_order_count": min(ip_count / 10.0, 1.0),
        "drift_score": drift,
        "is_vpn": 1.0 if ip_info["is_vpn"] else 0.0,
        "is_datacenter": 1.0 if ip_info["is_datacenter"] else 0.0,
        "geo_mismatch": 1.0 if ip_info["geo_mismatch"] else 0.0,
        "in_ring": 1.0 if ring else 0.0,
        "cod_and_high_value": cod_and_high_value,
        # raw (non-normalized) values kept around for reason strings
        "_raw_device_count": dev_count,
        "_raw_ip_count": ip_count,
        "_raw_ring": ring,
        "_raw_ip_info": ip_info,
    }


# Weighted rule-based fallback -- used whenever the trained model file is
# absent, so the service never hard-fails just because train_models.py
# hasn't been run yet.
_FALLBACK_WEIGHTS = {
    "order_value_norm": 0.10,
    "device_account_count": 0.22,
    "ip_order_count": 0.18,
    "drift_score": 0.18,
    "is_vpn": 0.12,
    "is_datacenter": 0.08,
    "geo_mismatch": 0.06,
    "in_ring": 0.20,
    "cod_and_high_value": 0.08,
}


def _rule_based_score(features: dict) -> float:
    score = sum(_FALLBACK_WEIGHTS[k] * features[k] for k in FEATURE_NAMES)
    # normalize by max possible weighted sum so score stays in [0,1]
    max_possible = sum(_FALLBACK_WEIGHTS.values())
    return max(0.0, min(1.0, score / max_possible))


def _model_score(features: dict) -> float:
    x = np.array([[features[k] for k in FEATURE_NAMES]])
    proba = _model.predict_proba(x)[0]
    # class 1 == fraudulent/risky
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


def _reasons_for(features: dict, ip_info: dict) -> list[str]:
    reasons = []
    if features["_raw_device_count"] >= 3:
        reasons.append(f"shared_device_{features['_raw_device_count']}_accounts")
    if ip_info["is_vpn"]:
        reasons.append("vpn_detected")
    if ip_info["is_datacenter"]:
        reasons.append("datacenter_ip_detected")
    if ip_info["geo_mismatch"]:
        reasons.append(f"geo_mismatch:ip={ip_info['mapped_country']}")
    if features["_raw_ip_count"] >= 4:
        reasons.append(f"ip_order_velocity:{features['_raw_ip_count']}_in_10min")
    if features["drift_score"] >= 0.5:
        reasons.append(f"behavioral_drift:{round(features['drift_score'], 2)}")
    if features["_raw_ring"]:
        reasons.append(f"identity_ring_member:{features['_raw_ring']}")
    if features["cod_and_high_value"]:
        reasons.append("cod_high_value_order")
    if not reasons:
        reasons.append("no_significant_risk_signals")
    return reasons[:3] if len(reasons) >= 2 else (reasons + ["baseline_profile_normal"])[:3]


def _count_independent_signals(features: dict, ip_info: dict) -> int:
    """
    Counts how many INDEPENDENT signal families fired, for the
    'no auto-negative-action on one weak signal' gate. Families:
    device sharing, ip velocity/vpn/datacenter/geo, behavioral drift,
    ring membership, cod+high-value.
    """
    count = 0
    if features["_raw_device_count"] >= 3:
        count += 1
    if ip_info["is_vpn"] or ip_info["is_datacenter"] or ip_info["geo_mismatch"] or features["_raw_ip_count"] >= 4:
        count += 1
    if features["drift_score"] >= 0.5:
        count += 1
    if features["_raw_ring"]:
        count += 1
    if features["cod_and_high_value"]:
        count += 1
    return count


@router.post("/evaluate", response_model=RiskDecision)
def evaluate(order: OrderEvent):
    start = time.perf_counter()

    graph = shared_instance()
    graph.register_order(order.model_dump())

    features = _build_features(order)
    ip_info = features["_raw_ip_info"]

    if _model is not None:
        # Blend the calibrated model with the transparent rule-based score
        # rather than trusting the model in isolation: the model is fit on
        # a comparatively small synthetic-fraud sample, so on its own it
        # can under-weight a graph signal (e.g. device_account_count) that
        # a human reviewer would treat as decisive. Blending keeps the
        # model's learned signal while ensuring strong, explainable graph
        # evidence still moves the score even if the model under-calibrates
        # it.
        score = 0.55 * _model_score(features) + 0.45 * _rule_based_score(features)
    else:
        score = _rule_based_score(features)

    tier, action = policy.evaluate_tier("risk", score)

    # Guardrail: never auto-fire a negative action (prepaid_only / block)
    # on fewer than the policy-configured minimum number of agreeing
    # signals. If the ladder says "block" but only one weak signal fired,
    # downgrade to human_review-equivalent (step_up_otp) instead of
    # guessing.
    agreeing_signals = _count_independent_signals(features, ip_info)
    min_required = policy.risk_min_agreeing_signals()
    escalated_to_review = False
    if action in ("prepaid_only", "block") and agreeing_signals < min_required:
        action = "step_up_otp"
        tier = "medium"
        escalated_to_review = True

    reasons = _reasons_for(features, ip_info)
    if escalated_to_review:
        reasons = (reasons + ["escalated_single_signal_insufficient_for_auto_negative"])[:3]

    audit_id = write_audit(
        agent="risk_scoring_agent",
        entity_id=order.order_id,
        entity_type="Order",
        score=score,
        tier=tier,
        action=action,
        reasons=reasons,
        policy_version=policy.policy_version(),
    )

    # Recompute rings live so a device just registered on this order is
    # immediately reflected for the other two agents' next calls.
    graph.recompute_rings()

    latency_ms = (time.perf_counter() - start) * 1000.0

    return RiskDecision(
        order_id=order.order_id,
        risk_score=round(score, 4),
        tier=tier,
        action=action,
        reasons=reasons,
        audit_id=audit_id,
        latency_ms=round(latency_ms, 2),
    )


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model_loaded_from_disk,
        "policy_version": policy.policy_version(),
    }


app = FastAPI(title="SentinelMesh - Risk Scoring Agent")
app.include_router(router)
