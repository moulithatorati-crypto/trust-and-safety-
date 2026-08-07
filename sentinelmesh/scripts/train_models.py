"""
Trains:
  1. LogisticRegression for the risk-scoring agent, on graph + IP +
     drift-derived features computed from data/orders.json.
  2. TF-IDF + LogisticRegression classifier for the review-moderation
     agent's AI-generated-text likelihood signal, on a synthetic
     human-vs-templated-fake text set.

Saves artifacts to models/. Prints precision/recall/AUC and an explicit
disclaimer that these are synthetic-data metrics, not production
validation.

Run: python scripts/train_models.py
(after scripts/generate_synthetic_data.py)
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.identity_graph import IdentityGraph
from shared.ip_intelligence import _is_in_ranges, _VPN_DATACENTER_RANGES, _DATACENTER_ONLY, _country_for_ip
from shared.behavioral_drift import BehavioralDriftTracker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "order_value_norm", "device_account_count", "ip_order_count", "drift_score",
    "is_vpn", "is_datacenter", "geo_mismatch", "in_ring", "cod_and_high_value",
]
_ORDER_VALUE_CAP = 25000.0


def train_risk_model():
    with open(os.path.join(DATA_DIR, "orders.json")) as f:
        orders = json.load(f)

    orders_sorted = sorted(orders, key=lambda o: o["timestamp"])

    graph = IdentityGraph()
    drift_tracker = BehavioralDriftTracker()

    X, y = [], []

    # first pass: register everything so device_account_count etc. reflect
    # the FULL dataset (as it would once the demo has been running a while)
    for o in orders_sorted:
        graph.register_order(o)
    graph.recompute_rings()

    for o in orders_sorted:
        dev_count = graph.device_account_count(o["device_id"])
        ip_count = graph.ip_order_count(o["ip"], window_minutes=10)
        drift = drift_tracker.drift_score(o["account_id"], o["session"])
        ring = graph.ring_id_for(o["account_id"]) or graph.ring_id_for(o["device_id"])

        is_vpn = _is_in_ranges(o["ip"], _VPN_DATACENTER_RANGES) and not _is_in_ranges(o["ip"], _DATACENTER_ONLY)
        is_dc = _is_in_ranges(o["ip"], _DATACENTER_ONLY)
        geo_mismatch = _country_for_ip(o["ip"]) != o["billing_country"]

        order_value_norm = min(o["order_value"] / _ORDER_VALUE_CAP, 1.0)
        cod_and_high_value = 1.0 if (o["payment_type"] == "cod" and o["order_value"] > 3000) else 0.0

        features = [
            order_value_norm,
            min(dev_count / 8.0, 1.0),
            min(ip_count / 10.0, 1.0),
            drift,
            1.0 if is_vpn else 0.0,
            1.0 if is_dc else 0.0,
            1.0 if geo_mismatch else 0.0,
            1.0 if ring else 0.0,
            cod_and_high_value,
        ]
        X.append(features)
        y.append(int(o.get("label_fraud", 0)))

    X = np.array(X)
    y = np.array(y)

    if len(set(y.tolist())) < 2:
        print("WARNING: risk training data has only one class; skipping model fit, "
              "risk_scoring_agent will use its rule-based fallback.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"[risk_model] precision={precision:.3f} recall={recall:.3f} auc={auc:.3f}")
    print("[risk_model] NOTE: these are synthetic-data metrics for demo purposes only, "
          "not a production validation result.")

    joblib.dump(clf, os.path.join(MODELS_DIR, "risk_model.joblib"))

    # Also fit an IsolationForest purely for illustrative anomaly scoring
    # exposed in case future agents want an unsupervised signal too.
    iso = IsolationForest(random_state=42, contamination=min(0.15, max(0.01, y.mean())))
    iso.fit(X_train)
    joblib.dump(iso, os.path.join(MODELS_DIR, "risk_isolation_forest.joblib"))


def train_review_model():
    with open(os.path.join(DATA_DIR, "reviews.json")) as f:
        reviews = json.load(f)

    texts = [r["text"] for r in reviews]
    labels = [int(r.get("label_fake", 0)) for r in reviews]

    if len(set(labels)) < 2:
        print("WARNING: review training data has only one class; skipping model fit, "
              "review_moderation_agent will use its heuristic fallback.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_proba = clf.predict_proba(X_test_vec)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"[review_model] precision={precision:.3f} recall={recall:.3f} auc={auc:.3f}")
    print("[review_model] NOTE: these are synthetic-data metrics for demo purposes only, "
          "not a production validation result.")

    joblib.dump({"vectorizer": vectorizer, "clf": clf}, os.path.join(MODELS_DIR, "review_authenticity_model.joblib"))


if __name__ == "__main__":
    train_risk_model()
    train_review_model()
    print("Done. Model artifacts written to", MODELS_DIR)
