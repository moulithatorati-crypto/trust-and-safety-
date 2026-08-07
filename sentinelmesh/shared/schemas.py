"""
Pydantic models for every event / decision type exchanged between
SentinelMesh agents.
"""
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------- Risk ----
class SessionInfo(BaseModel):
    duration_seconds: float
    typing_cadence_ms: float
    copy_paste_detected: bool = False


class OrderEvent(BaseModel):
    order_id: str
    account_id: str
    device_id: str
    ip: str
    payment_instrument_id: str
    shipping_address_id: str
    order_value: float
    payment_type: str
    category: str
    session: SessionInfo
    billing_country: str
    timestamp: str


class RiskDecision(BaseModel):
    order_id: str
    risk_score: float
    tier: Literal["low", "medium", "high", "critical"]
    action: Literal["allow", "step_up_otp", "prepaid_only", "block"]
    reasons: List[str]
    audit_id: str
    latency_ms: float


# --------------------------------------------------------- Authenticity ---
class ListingEvent(BaseModel):
    listing_id: str
    seller_id: str
    title: str
    description: str
    price: float
    brand: str
    category: str
    images: List[str] = Field(default_factory=list)
    gtin: Optional[str] = None


class EvidenceItem(BaseModel):
    signal: str
    detail: str


class AuthenticityDecision(BaseModel):
    listing_id: str
    authenticity_score: float
    action: Literal["publish", "hold", "reject"]
    evidence: List[EvidenceItem]
    audit_id: str


# --------------------------------------------------------------- Review ---
class ReviewEvent(BaseModel):
    review_id: str
    account_id: str
    product_id: str
    device_id: str
    rating: int
    text: str
    order_id: Optional[str] = None
    timestamp: str


class ReviewDecision(BaseModel):
    review_id: str
    authenticity_score: float
    action: Literal["publish", "flag", "suppress"]
    ring_id: Optional[str] = None
    reasons: List[str]
    audit_id: str


class ReviewBatch(BaseModel):
    reviews: List[ReviewEvent]


# ----------------------------------------------------------------- Misc ---
class AuditRecord(BaseModel):
    audit_id: str
    agent: str
    entity_id: str
    entity_type: str
    score: float
    tier: str
    action: str
    reasons: List[str]
    policy_version: str
    timestamp: str
    prev_hash: str
    record_hash: str
