"""
IdentityGraph: a single, shared, in-memory NetworkX graph that all three
agents read from and write to. This is what makes SentinelMesh "graph
native" instead of three isolated classifiers -- a ring discovered by one
agent is visible to the other two on the very next call, because they all
share this one process-wide instance (see `shared_instance()` below).
"""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import networkx as nx

try:
    # networkx >= 2.6 ships this natively
    from networkx.algorithms.community import greedy_modularity_communities
    _HAVE_GREEDY_MODULARITY = True
except Exception:  # pragma: no cover
    _HAVE_GREEDY_MODULARITY = False


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


class IdentityGraph:
    """
    Node types (all stored as (type, id) tuples so ids never collide across
    types): Account, Device, IP, PaymentInstrument, ShippingAddress,
    Seller, Product, Review.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.g = nx.Graph()
        # ring_id -> frozenset already-computed at recompute_rings() time
        self._ring_of: dict[str, str] = {}
        self._next_ring_num = 1
        # order timestamps per ip, for velocity windows
        self._ip_order_times: dict[str, list[datetime]] = defaultdict(list)

    # ---------------------------------------------------------- helpers --
    def _node(self, ntype: str, nid: str) -> str:
        key = f"{ntype}:{nid}"
        if key not in self.g:
            self.g.add_node(key, ntype=ntype, nid=nid)
        return key

    def _edge(self, a: str, b: str, **kw):
        if self.g.has_edge(a, b):
            self.g[a][b]["weight"] = self.g[a][b].get("weight", 1) + 1
        else:
            self.g.add_edge(a, b, weight=1, **kw)

    # -------------------------------------------------------- registration
    def register_order(self, order: dict) -> None:
        with self._lock:
            acc = self._node("Account", order["account_id"])
            dev = self._node("Device", order["device_id"])
            ip = self._node("IP", order["ip"])
            pay = self._node("PaymentInstrument", order["payment_instrument_id"])
            addr = self._node("ShippingAddress", order["shipping_address_id"])

            self._edge(acc, dev)
            self._edge(acc, ip)
            self._edge(acc, pay)
            self._edge(acc, addr)
            self._edge(dev, ip)

            self._ip_order_times[order["ip"]].append(_parse_ts(order["timestamp"]))

    def register_listing(self, listing: dict) -> None:
        with self._lock:
            seller = self._node("Seller", listing["seller_id"])
            product = self._node("Product", listing["listing_id"])
            self._edge(seller, product)

    def register_review(self, review: dict) -> None:
        with self._lock:
            acc = self._node("Account", review["account_id"])
            dev = self._node("Device", review["device_id"])
            rev = self._node("Review", review["review_id"])
            prod = self._node("Product", review["product_id"])

            self._edge(acc, dev)
            self._edge(acc, rev)
            self._edge(rev, prod)
            self._edge(dev, rev)

    # -------------------------------------------------------------- reads
    def device_account_count(self, device_id: str) -> int:
        with self._lock:
            key = f"Device:{device_id}"
            if key not in self.g:
                return 0
            return sum(
                1 for n in self.g.neighbors(key) if n.startswith("Account:")
            )

    def ip_order_count(self, ip: str, window_minutes: int = 10) -> int:
        with self._lock:
            times = self._ip_order_times.get(ip, [])
            if not times:
                return 0
            now = times[-1]
            cutoff = now - timedelta(minutes=window_minutes)
            return sum(1 for t in times if t >= cutoff)

    def ring_id_for(self, entity_id: str) -> Optional[str]:
        with self._lock:
            for ntype in (
                "Account", "Device", "IP", "PaymentInstrument",
                "ShippingAddress", "Seller", "Product", "Review",
            ):
                key = f"{ntype}:{entity_id}"
                if key in self._ring_of:
                    return self._ring_of[key]
            return None

    def neighbors_of(self, entity_id: str) -> list[dict]:
        """Used by orchestrator /graph/lookup/{entity_id}."""
        with self._lock:
            out = []
            for ntype in (
                "Account", "Device", "IP", "PaymentInstrument",
                "ShippingAddress", "Seller", "Product", "Review",
            ):
                key = f"{ntype}:{entity_id}"
                if key in self.g:
                    for n in self.g.neighbors(key):
                        data = self.g.nodes[n]
                        out.append({"type": data["ntype"], "id": data["nid"]})
            return out

    # ----------------------------------------------------------- ring calc
    def recompute_rings(self, min_size: int = 3) -> int:
        """
        Label dense connected subgraphs as fraud rings. Uses greedy
        modularity communities (a simple, dependency-free approximation of
        Louvain) restricted to components with >= min_size nodes and more
        than one edge-dense cluster of connections -- i.e. components that
        look like a shared-identity cluster, not just an isolated
        account+device+ip triangle from one normal order.

        Callable on demand so the dashboard can show a ring appear live,
        not just on a timer.
        """
        with self._lock:
            self._ring_of.clear()
            self._next_ring_num = 1

            for component in nx.connected_components(self.g):
                if len(component) < min_size:
                    continue
                sub = self.g.subgraph(component)

                # A component built from a single order/review touches each
                # node type at most once except Account/Device. Treat a
                # component as a *candidate ring* only if some Device or IP
                # or Seller node fans out to >= 2 Accounts/Products, i.e.
                # there's real sharing going on.
                shared_hub = any(
                    sub.degree(n) >= 3
                    for n in sub.nodes
                    if sub.nodes[n]["ntype"] in ("Device", "IP", "Seller")
                )
                if not shared_hub:
                    continue

                # Small/medium shared-hub components (the common case for a
                # single ring of accounts fanning out from one device/IP/
                # seller) are kept as ONE ring rather than further split --
                # splitting a star-shaped cluster by modularity tends to
                # fragment it into several small pieces that no longer read
                # as "a ring". Only larger components, which more plausibly
                # contain multiple distinct rings merged by incidental
                # shared infrastructure, get decomposed via greedy
                # modularity communities.
                if _HAVE_GREEDY_MODULARITY and len(sub) > 40:
                    try:
                        communities = list(greedy_modularity_communities(sub))
                    except Exception:
                        communities = [set(sub.nodes)]
                else:
                    communities = [set(sub.nodes)]

                for comm in communities:
                    if len(comm) < min_size:
                        continue
                    ring_id = f"CLU_{self._next_ring_num}"
                    self._next_ring_num += 1
                    for node in comm:
                        self._ring_of[node] = ring_id

            return self._next_ring_num - 1


# ------------------------------------------------------------ singleton --
_instance: Optional[IdentityGraph] = None
_instance_lock = threading.Lock()


def shared_instance() -> IdentityGraph:
    """
    Returns a process-wide singleton. Each agent module exposes both a
    standalone FastAPI `app` (so it can be run and tested on its own) and a
    plain `router` (so the orchestrator gateway can `include_router()` all
    three inside ONE process). Running the system via the orchestrator is
    what gives the three agents a genuinely shared, real-time identity
    graph and a shared audit-trail SQLite connection -- a ring caught by
    one agent's write is visible to the other two on their very next call,
    with no batch sync step. Running an individual agent's `app` standalone
    (e.g. for isolated testing) still works, it just won't share graph
    state with the other agents' separate processes.
    """
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = IdentityGraph()
        return _instance
