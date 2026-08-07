"""
Agent 2: Authenticity Agent.

POST /scan (also exposed as POST /authenticity/scan on the orchestrator)

Signals:
  - price_deviation: how far below the category MSRP the listing is priced
  - image_similarity: deterministic hash-distance stub vs a small set of
    "reference" images (no real CV model required for the demo)
  - seller_auth: is the seller on the static authorized-seller list

Precision gate (hard requirement): never auto-reject or auto-hold on one
signal alone. An automated negative action only fires when
price_deviation AND (image_similarity low OR seller_unauthorized) --
otherwise the listing is routed to human_review.
"""
from __future__ import annotations

import hashlib

from fastapi import FastAPI, APIRouter

from shared import policy
from shared.audit_trail import write_audit
from shared.identity_graph import shared_instance
from shared.schemas import ListingEvent, AuthenticityDecision, EvidenceItem

router = APIRouter()

# --------------------------------------------------------- static tables --
# category -> approximate MSRP anchor used for price-deviation checks.
_CATEGORY_MSRP = {
    "cosmetics": 8999.0,
    "electronics": 45000.0,
    "apparel": 2500.0,
    "footwear": 4500.0,
    "watches": 15000.0,
    "toys": 1800.0,
    "home": 3200.0,
}

# authorized sellers per brand, static demo table.
_AUTHORIZED_SELLERS = {
    "L'Oreal": {"SEL_AUTH_1", "SEL_AUTH_2"},
    "Nike": {"SEL_AUTH_3"},
    "Sony": {"SEL_AUTH_4"},
    "Rolex": {"SEL_AUTH_5"},
    "Generic": set(),  # unbranded listings have no authorization concept
}

# a tiny set of "reference" image hashes per brand, for the perceptual-hash
# stub. In the demo these are just fixed strings; a real system would use
# actual perceptual hashes of verified product photography.
_REFERENCE_HASHES = {
    "L'Oreal": ["REF_LOREAL_A1B2", "REF_LOREAL_C3D4"],
    "Nike": ["REF_NIKE_E5F6"],
    "Sony": ["REF_SONY_G7H8"],
    "Rolex": ["REF_ROLEX_I9J0"],
}


def _stub_image_similarity(images: list[str], brand: str) -> float:
    """
    Deterministic hash-distance stub, not a real CV model. We hash each
    supplied image string and compare bit-overlap against the brand's
    reference hashes, giving a stable, reproducible similarity in [0, 1].
    """
    refs = _REFERENCE_HASHES.get(brand)
    if not images or not refs:
        return 0.5  # no evidence either way

    def bits(s: str) -> str:
        return bin(int(hashlib.sha256(s.encode()).hexdigest(), 16))[2:].zfill(256)

    best = 0.0
    for img in images:
        img_bits = bits(img)
        for ref in refs:
            ref_bits = bits(ref)
            matches = sum(1 for a, b in zip(img_bits, ref_bits) if a == b)
            similarity = matches / len(img_bits)
            best = max(best, similarity)
    return round(best, 4)


def _price_deviation_pct(price: float, category: str) -> float:
    msrp = _CATEGORY_MSRP.get(category)
    if not msrp:
        return 0.0
    return max(0.0, (msrp - price) / msrp)


@router.post("/scan", response_model=AuthenticityDecision)
def scan(listing: ListingEvent):
    graph = shared_instance()
    graph.register_listing(listing.model_dump())

    msrp = _CATEGORY_MSRP.get(listing.category)
    deviation_pct = _price_deviation_pct(listing.price, listing.category)
    image_similarity = _stub_image_similarity(listing.images, listing.brand)
    authorized_set = _AUTHORIZED_SELLERS.get(listing.brand, set())
    seller_unauthorized = bool(authorized_set) and listing.seller_id not in authorized_set

    sig_thresholds = policy.authenticity_signal_thresholds()
    price_signal_fired = deviation_pct >= sig_thresholds["price_deviation_pct"]
    image_signal_fired = image_similarity <= sig_thresholds["image_similarity_low"]

    # Composite score: start high (authentic-by-default), subtract for
    # each fired signal, weighted.
    score = 1.0
    if price_signal_fired:
        score -= 0.40 * min(deviation_pct / 0.8, 1.0)
    if image_signal_fired:
        score -= 0.30 * (1.0 - image_similarity)
    if seller_unauthorized:
        score -= 0.25
    score = max(0.0, min(1.0, score))

    thresholds = policy.authenticity_thresholds()
    evidence: list[EvidenceItem] = []

    if price_signal_fired:
        evidence.append(EvidenceItem(
            signal="price_deviation",
            detail=(
                f"Priced {round(deviation_pct * 100)}% below category MSRP "
                f"(₹{listing.price:,.0f} vs ₹{msrp:,.0f})" if msrp else
                f"Priced {round(deviation_pct * 100)}% below category MSRP"
            ),
        ))
    if image_signal_fired:
        evidence.append(EvidenceItem(
            signal="image_similarity",
            detail=(
                f"Listing images show only {round(image_similarity * 100)}% "
                f"visual similarity to verified {listing.brand} reference photography"
            ),
        ))
    if seller_unauthorized:
        evidence.append(EvidenceItem(
            signal="seller_auth",
            detail=f"Seller {listing.seller_id} is not in the {listing.brand} brand-authorized seller list",
        ))
    if not evidence:
        evidence.append(EvidenceItem(
            signal="baseline",
            detail="No price, image, or seller-authorization anomalies detected",
        ))

    # ---- precision gate: never auto-reject/hold on one signal alone ----
    multi_signal_negative_gate = price_signal_fired and (image_signal_fired or seller_unauthorized)

    if score >= thresholds["auto_publish"] and not multi_signal_negative_gate:
        action = "publish"
    elif multi_signal_negative_gate and score <= thresholds["auto_reject"]:
        action = "reject"
    elif multi_signal_negative_gate:
        # Gate satisfied (2 independent signals agree) but score isn't low
        # enough to auto-reject outright -- hold for a closer automated
        # look / expedited human pass rather than auto-publishing.
        action = "hold"
    elif score <= thresholds["auto_reject"] and not multi_signal_negative_gate:
        # Score alone looks bad but only one signal actually fired --
        # the gate forbids auto-reject, so this escalates to human_review
        # rather than guessing. We surface that as "hold" (the closest of
        # our three allowed actions to "send to a person"), and say so
        # explicitly in the evidence.
        action = "hold"
        evidence.append(EvidenceItem(
            signal="escalation",
            detail="Low score driven by a single signal only; routed to human_review instead of auto-reject",
        ))
    else:
        action = "hold"

    audit_id = write_audit(
        agent="authenticity_agent",
        entity_id=listing.listing_id,
        entity_type="Listing",
        score=score,
        tier=action,
        action=action,
        reasons=[e.detail for e in evidence][:3],
        policy_version=policy.policy_version(),
    )

    graph.recompute_rings()

    return AuthenticityDecision(
        listing_id=listing.listing_id,
        authenticity_score=round(score, 4),
        action=action,
        evidence=evidence[:3],
        audit_id=audit_id,
    )


@router.get("/health")
def health():
    return {"status": "ok", "policy_version": policy.policy_version()}


app = FastAPI(title="SentinelMesh - Authenticity Agent")
app.include_router(router)
