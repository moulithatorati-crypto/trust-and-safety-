# SentinelMesh

A graph-native, real-time fraud intelligence prototype for an e-commerce
marketplace. Three agents — **risk scoring**, **listing authenticity**,
and **review moderation** — share one live in-memory identity graph and
one hash-chained, append-only audit ledger, instead of running as three
isolated classifiers on separate batch schedules.

## Why "graph-native"

All three agents call the *same* `IdentityGraph` singleton
(`shared/identity_graph.py`). A ring caught by one agent — say, five
accounts sharing one device on the risk agent — is visible to the review
agent's `ring_id_for()` lookup on its very next call, with no batch sync
step. This works because the orchestrator gateway `include_router()`s all
three agents' FastAPI routers **into one process**, rather than proxying
HTTP calls to three separately-running services. Each agent module also
exposes a standalone `app` you can run and test in isolation — it just
won't share graph state with a separately-running orchestrator process in
that mode.

## Design principles enforced in code

1. **Immutable, hash-chained audit trail.** Every automated decision is
   written via `shared/audit_trail.write_audit()`. Each record's
   `record_hash = sha256(canonical fields + prev_hash)`. `verify_chain()`
   walks the table and returns `False` the moment anything doesn't match —
   see `tests/test_end_to_end.py::test_tamper_breaks_chain`.
2. **Thresholds live in YAML, never hardcoded.** See
   `shared/decision_policy.yaml` + `shared/policy.py`. Every audit record
   snapshots the `policy_version` in effect.
3. **No single weak signal auto-fires a negative action.** The risk agent
   requires `risk_min_agreeing_signals` (default 2) independent signal
   families to agree before `prepaid_only`/`block` fires — otherwise it
   downgrades to `step_up_otp` and says so in the reasons. The
   authenticity agent's precision gate requires `price_deviation AND
   (image_similarity_low OR seller_unauthorized)` before `reject`/`hold`
   fires — a price anomaly alone routes to `hold` (human_review) instead.
4. **Every score ships with 2–3 human-readable reasons/evidence**, not
   just a number.
5. **All three agents read/write the same graph in real time** — see
   above.

## Repository layout

```
sentinelmesh/
  shared/            # schemas, identity graph, ip intel, drift, audit, policy
  services/
    risk_scoring_agent/main.py
    authenticity_agent/main.py
    review_moderation_agent/main.py
    orchestrator_gateway/main.py   # <- run this one
  scripts/
    generate_synthetic_data.py
    train_models.py
  dashboard/index.html            # single-file ops console, no build step
  tests/test_end_to_end.py
```

## Quickstart

```bash
pip install -r requirements.txt

python scripts/generate_synthetic_data.py
python scripts/train_models.py        # prints precision/recall/AUC (synthetic-data metrics only)

uvicorn services.orchestrator_gateway.main:app --reload
```

Then open `dashboard/index.html` directly in a browser (it talks to
`http://127.0.0.1:8000` by default, editable in the top-right field), or
hit the API directly:

```bash
curl -X POST http://127.0.0.1:8000/risk/evaluate -H "Content-Type: application/json" -d '{
  "order_id":"ORD_1","account_id":"ACC_1","device_id":"DEV_1","ip":"203.0.113.5",
  "payment_instrument_id":"PAY_1","shipping_address_id":"ADDR_1","order_value":2499.0,
  "payment_type":"cod","category":"electronics",
  "session":{"duration_seconds":42,"typing_cadence_ms":120,"copy_paste_detected":true},
  "billing_country":"IN","timestamp":"2026-08-08T10:00:00Z"}'
```

## Tests

```bash
pytest tests/test_end_to_end.py -v
```

Covers: shared-device risk escalation, counterfeit-listing rejection
(multi-signal gate), coordinated review-ring suppression, audit chain
integrity, and audit chain tamper detection.

## Endpoints (via the orchestrator, port 8000)

| Method | Path | Purpose |
|---|---|---|
| POST | `/risk/evaluate` | Score a checkout event |
| POST | `/authenticity/scan` | Score a marketplace listing |
| POST | `/review/moderate` | Score a batch of reviews + detect rings |
| GET | `/graph/lookup/{entity_id}` | Connected entities + ring membership |
| GET | `/graph/recompute_rings` | Force an on-demand ring recompute |
| GET | `/audit/{audit_id}` | Full audit record + policy snapshot in effect |
| GET | `/audit?limit=200` | Recent audit records + chain validity |
| GET | `/audit/verify` | Just the chain-validity boolean |
| GET | `/metrics/fairness` | Illustrative false-positive-rate metric by account signal richness |

## Notes on the ML components

- `risk_scoring_agent` prefers a calibrated `LogisticRegression` trained
  by `scripts/train_models.py` on graph + IP + drift features, and falls
  back to a transparent weighted rule-based score if `models/risk_model.joblib`
  is missing — the service never hard-fails just because training hasn't
  run yet.
- `review_moderation_agent`'s AI-generated-text signal uses a
  TF-IDF + `LogisticRegression` classifier trained on synthetic
  human-vs-templated-fake text, with a heuristic fallback (repeated
  generic superlatives, very short text) if untrained.
- `authenticity_agent`'s "image similarity" is a **deterministic
  hash-distance stub**, not a real CV model, per the spec — it compares a
  SHA-256 bit-overlap of the supplied image string(s) against a small set
  of fixed reference hashes per brand.
- All metrics printed by `train_models.py` and returned by
  `/metrics/fairness` are computed on synthetic data and are explicitly
  labeled as such — not a production validation result.

## Everything runs CPU-only, offline

No external network calls are made anywhere except `pip install`. VPN/
datacenter/geo lookups, MSRP tables, authorized-seller lists, and
reference image hashes are all static tables baked into `shared/` and
`services/authenticity_agent/main.py` for the demo.
