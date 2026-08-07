"""
Loads shared/decision_policy.yaml and exposes typed accessors. All
decision thresholds live in the YAML file -- nothing here is hardcoded,
and every agent snapshots `policy_version` into its audit records so a
past decision can always be replayed against the policy that produced it.
"""
from __future__ import annotations

import os
import threading
from typing import Tuple

import yaml

_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_policy.yaml")

_lock = threading.RLock()
_policy_cache: dict | None = None
_mtime_cache: float | None = None


def _load() -> dict:
    global _policy_cache, _mtime_cache
    with _lock:
        mtime = os.path.getmtime(_POLICY_PATH)
        if _policy_cache is None or mtime != _mtime_cache:
            with open(_POLICY_PATH, "r") as f:
                _policy_cache = yaml.safe_load(f)
            _mtime_cache = mtime
        return _policy_cache


def get_policy() -> dict:
    return _load()


def policy_version() -> str:
    return _load()["policy_version"]


def evaluate_tier(agent_name: str, score: float) -> Tuple[str, str]:
    """
    Maps a 0..1 score to (tier_name, action) using the `risk_tiers` ladder
    in the policy file. Currently used by the risk-scoring agent; the
    authenticity and review agents use their own threshold blocks below
    since they have a 3-way publish/hold(or human_review)/reject shape
    rather than a 4-tier ladder.
    """
    policy = _load()
    tiers = policy["risk_tiers"]
    score = max(0.0, min(1.0, score))
    for tier in tiers:
        if score <= tier["score_max"]:
            return tier["name"], tier["action"]
    last = tiers[-1]
    return last["name"], last["action"]


def risk_min_agreeing_signals() -> int:
    return int(_load().get("risk_min_agreeing_signals", 2))


def authenticity_thresholds() -> dict:
    return _load()["authenticity_thresholds"]


def authenticity_signal_thresholds() -> dict:
    return _load()["authenticity_signal_thresholds"]


def review_thresholds() -> dict:
    return _load()["review_thresholds"]


def review_ring_thresholds() -> dict:
    return _load()["review_ring_thresholds"]
