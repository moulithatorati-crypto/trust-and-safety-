"""
Agent 3: Review Moderation Agent.

POST /moderate (also exposed as POST /review/moderate on the orchestrator)
Accepts a batch (list) of ReviewEvent.

Per review:
  - AI-generated-text likelihood via a TF-IDF + classifier trained in
    scripts/train_models.py (falls back to a heuristic if untrained)
  - purchase verification: does order_id exist & match account_id
    (checked against the identity graph / synthetic order log)
  - semantic relevance: TF-IDF cosine similarity between review text and
    product description

Ring detection: groups reviews by identity_graph.ring_id_for() for each
reviewer/device; if a ring's reviews arrived in a tight time window with
high rating uniformity, suppress the whole cluster and flag every
reviewer in it. Writes one audit record per review, plus one more for any
ring-level action.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, APIRouter

from shared import policy
from shared.audit_trail import write_audit
from shared.identity_graph import shared_instance
from shared.schemas import ReviewEvent, ReviewDecision, ReviewBatch

router = APIRouter()

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "review_authenticity_model.joblib")
PRODUCT_DESC_PATH = os.path.join(_BASE_DIR, "data", "product_descriptions.json")
ORDERS_PATH = os.path.join(_BASE_DIR, "data", "orders.json")

_model_bundle = None  # {"vectorizer":..., "clf":...}
_product_descriptions: dict[str, str] = {}
_valid_orders: dict[str, str] = {}  # order_id -> account_id


def _load_artifacts():
    global _model_bundle, _product_descriptions, _valid_orders
    if os.path.exists(MODEL_PATH):
        try:
            _model_bundle = joblib.load(MODEL_PATH)
        except Exception:
            _model_bundle = None
    if os.path.exists(PRODUCT_DESC_PATH):
        try:
            with open(PRODUCT_DESC_PATH) as f:
                _product_descriptions = json.load(f)
        except Exception:
            _product_descriptions = {}
    if os.path.exists(ORDERS_PATH):
        try:
            with open(ORDERS_PATH) as f:
                orders = json.load(f)
            _valid_orders = {o["order_id"]: o["account_id"] for o in orders}
        except Exception:
            _valid_orders = {}


_load_artifacts()


def _ai_generated_prob(text: str) -> float:
    if _model_bundle is not None:
        try:
            vec = _model_bundle["vectorizer"].transform([text])
            proba = _model_bundle["clf"].predict_proba(vec)[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception:
            pass
    # heuristic fallback: very short/templated/generic reviews with
    # repeated superlatives score higher
    generic_phrases = ["highly recommend", "best product ever", "five stars", "great quality", "value for money"]
    text_l = text.lower()
    hits = sum(1 for p in generic_phrases if p in text_l)
    length_penalty = 0.15 if len(text.split()) < 6 else 0.0
    return min(1.0, 0.15 * hits + length_penalty)


def _semantic_relevance(review_text: str, product_id: str) -> float:
    desc = _product_descriptions.get(product_id)
    if not desc:
        return 0.5
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform([review_text, desc])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.5


def _purchase_verified(review: ReviewEvent) -> bool:
    if not review.order_id:
        return False
    return _valid_orders.get(review.order_id) == review.account_id


def _score_review(review: ReviewEvent) -> tuple[float, list[str]]:
    ai_prob = _ai_generated_prob(review.text)
    verified = _purchase_verified(review)
    relevance = _semantic_relevance(review.text, review.product_id)

    # authenticity score: high = looks like a real, relevant, verified review
    score = 1.0
    score -= 0.45 * ai_prob
    score -= 0.0 if verified else 0.30
    score -= 0.25 * max(0.0, 1.0 - relevance) if relevance < 0.15 else 0.0
    score = max(0.0, min(1.0, score))

    reasons = [f"ai_generated_prob:{round(ai_prob, 2)}"]
    reasons.append("purchase_verified" if verified else "purchase_unverified")
    if relevance < 0.15:
        reasons.append(f"low_semantic_relevance:{round(relevance, 2)}")
    return score, reasons


@router.post("/moderate")
def moderate(batch: ReviewBatch):
    graph = shared_instance()

    for review in batch.reviews:
        graph.register_review(review.model_dump())
    graph.recompute_rings()

    thresholds = policy.review_thresholds()
    ring_cfg = policy.review_ring_thresholds()

    per_review_scores: dict[str, dict] = {}
    ring_members: dict[str, list[ReviewEvent]] = defaultdict(list)

    for review in batch.reviews:
        score, reasons = _score_review(review)
        ring_id = graph.ring_id_for(review.account_id) or graph.ring_id_for(review.device_id)
        per_review_scores[review.review_id] = {"score": score, "reasons": reasons, "ring_id": ring_id}
        if ring_id:
            ring_members[ring_id].append(review)

    # ---- ring-level evaluation --------------------------------------
    suppressed_rings: set[str] = set()
    ring_audit_ids: dict[str, str] = {}

    for ring_id, reviews in ring_members.items():
        if len(reviews) < ring_cfg["min_ring_size"]:
            continue

        timestamps = []
        for r in reviews:
            try:
                timestamps.append(datetime.fromisoformat(r.timestamp.replace("Z", "+00:00")))
            except Exception:
                pass
        if len(timestamps) >= 2:
            span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60.0
        else:
            span_minutes = 0.0

        ratings = [r.rating for r in reviews]
        most_common_count = max(ratings.count(r) for r in set(ratings))
        uniformity = most_common_count / len(ratings)

        tight_window = span_minutes <= ring_cfg["time_window_minutes"]
        uniform_ratings = uniformity >= ring_cfg["min_rating_uniformity"]

        if tight_window and uniform_ratings:
            suppressed_rings.add(ring_id)
            ring_reasons = [
                f"coordinated_ring:{len(reviews)}_reviews",
                f"tight_time_window:{round(span_minutes, 1)}min",
                f"rating_uniformity:{round(uniformity, 2)}",
            ]
            ring_audit_id = write_audit(
                agent="review_moderation_agent",
                entity_id=ring_id,
                entity_type="ReviewRing",
                score=1.0 - uniformity,  # lower = more clearly coordinated
                tier="suppressed",
                action="suppress",
                reasons=ring_reasons,
                policy_version=policy.policy_version(),
            )
            ring_audit_ids[ring_id] = ring_audit_id

    # ---- per-review decisions -----------------------------------------
    decisions: list[ReviewDecision] = []
    for review in batch.reviews:
        info = per_review_scores[review.review_id]
        score = info["score"]
        reasons = list(info["reasons"])
        ring_id = info["ring_id"]

        if ring_id and ring_id in suppressed_rings:
            action = "suppress"
            reasons.append("ring_member")
        elif score >= thresholds["auto_publish"]:
            action = "publish"
        elif score <= thresholds["auto_suppress"]:
            action = "suppress"
        else:
            action = "flag"  # human_review-equivalent for this 3-way action set

        audit_id = write_audit(
            agent="review_moderation_agent",
            entity_id=review.review_id,
            entity_type="Review",
            score=score,
            tier=action,
            action=action,
            reasons=reasons[:3],
            policy_version=policy.policy_version(),
        )

        decisions.append(ReviewDecision(
            review_id=review.review_id,
            authenticity_score=round(score, 4),
            action=action,
            ring_id=ring_id if (ring_id and ring_id in suppressed_rings) else None,
            reasons=reasons[:3],
            audit_id=audit_id,
        ))

    return {
        "decisions": [d.model_dump() for d in decisions],
        "rings_suppressed": list(suppressed_rings),
        "ring_audit_ids": ring_audit_ids,
    }


@router.get("/health")
def health():
    return {"status": "ok", "policy_version": policy.policy_version(), "model_loaded": _model_bundle is not None}


app = FastAPI(title="SentinelMesh - Review Moderation Agent")
app.include_router(router)
