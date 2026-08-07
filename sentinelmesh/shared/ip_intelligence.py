"""
Deterministic, static IP intelligence lookup for the demo. No external
network calls. Real deployments would swap this module for a MaxMind /
IP2Location / commercial VPN-detection feed behind the same function
signature.
"""
from __future__ import annotations

import ipaddress
from shared.identity_graph import shared_instance

# --------------------------------------------------------- static tables --
# CIDR ranges we treat as "known VPN / datacenter" for the demo.
_VPN_DATACENTER_RANGES = [
    ipaddress.ip_network("203.0.113.0/24"),   # TEST-NET-3, used as our "VPN pool"
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2, used as "datacenter pool"
    ipaddress.ip_network("192.0.2.0/24"),     # TEST-NET-1
]
_DATACENTER_ONLY = {ipaddress.ip_network("198.51.100.0/24")}

# Very small static "IP -> country" map so we can flag geo-mismatch without
# calling a real geo-IP service.
_IP_COUNTRY_PREFIXES = {
    "203.0.113.": "US",   # VPN pool advertises itself as US exit nodes
    "198.51.100.": "NL",  # datacenter pool
    "192.0.2.": "SG",
    "45.": "IN",
    "103.": "IN",
    "1.": "IN",
}


def _country_for_ip(ip: str) -> str:
    for prefix, country in _IP_COUNTRY_PREFIXES.items():
        if ip.startswith(prefix):
            return country
    return "IN"  # default assumption for unrecognized ranges in this demo


def _is_in_ranges(ip: str, ranges) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in ranges)


def check_ip(ip: str, billing_country: str) -> dict:
    """
    -> {"is_vpn": bool, "is_datacenter": bool, "geo_mismatch": bool,
        "velocity_10min": int}
    """
    is_datacenter = _is_in_ranges(ip, _DATACENTER_ONLY)
    is_vpn = _is_in_ranges(ip, _VPN_DATACENTER_RANGES) and not is_datacenter
    mapped_country = _country_for_ip(ip)
    geo_mismatch = mapped_country != billing_country

    graph = shared_instance()
    velocity = graph.ip_order_count(ip, window_minutes=10)

    return {
        "is_vpn": is_vpn,
        "is_datacenter": is_datacenter,
        "geo_mismatch": geo_mismatch,
        "velocity_10min": velocity,
        "mapped_country": mapped_country,
    }
