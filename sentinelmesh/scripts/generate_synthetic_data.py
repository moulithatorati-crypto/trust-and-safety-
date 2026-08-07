"""
Generates a few thousand correlated synthetic records (accounts, devices,
IPs, orders, listings, reviews) matching the SentinelMesh schemas, with
deliberately embedded fraud patterns:
  - one device shared across 5+ accounts
  - one IP with a burst of orders in a short window
  - one seller with a counterfeit listing (price far below MSRP + unauthorized)
  - one coordinated review ring (shared devices, tight timing, similar text)

Writes CSV/JSON under data/.

Run: python scripts/generate_synthetic_data.py
"""
from __future__ import annotations

import json
import os
import random
import string
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_ACCOUNTS = 600
N_ORDERS = 2500
N_LISTINGS = 300
N_REVIEWS = 1800

CATEGORIES = ["cosmetics", "electronics", "apparel", "footwear", "watches", "toys", "home"]
CATEGORY_MSRP = {
    "cosmetics": 8999.0, "electronics": 45000.0, "apparel": 2500.0,
    "footwear": 4500.0, "watches": 15000.0, "toys": 1800.0, "home": 3200.0,
}
BRANDS = ["L'Oreal", "Nike", "Sony", "Rolex", "Generic"]
AUTHORIZED_SELLERS = {
    "L'Oreal": {"SEL_AUTH_1", "SEL_AUTH_2"},
    "Nike": {"SEL_AUTH_3"},
    "Sony": {"SEL_AUTH_4"},
    "Rolex": {"SEL_AUTH_5"},
}
NORMAL_IP_PREFIXES = ["45.", "103.", "1."]
VPN_IPS = [f"203.0.113.{i}" for i in range(5, 60)]
DATACENTER_IPS = [f"198.51.100.{i}" for i in range(5, 60)]

GENUINE_REVIEW_TEMPLATES = [
    "Works exactly as described, been using it for two weeks now and no issues.",
    "Delivery was a bit slow but the product quality made up for it.",
    "Decent for the price, not amazing but does the job for daily use.",
    "My second time ordering this, still happy with the durability.",
    "Packaging was damaged but the item itself was fine, works well.",
    "Took a while to get used to it but now I really like it.",
]
FAKE_REVIEW_TEMPLATES = [
    "Highly recommend! Best product ever, five stars, amazing quality!!!",
    "Great quality, value for money, highly recommend to everyone!",
    "Five stars, best product ever, highly recommend, amazing!",
    "Amazing product, best quality, highly recommend, five stars!!!",
]


def rand_id(prefix, n):
    return f"{prefix}_{n}"


def rand_str(k=8):
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    now = datetime(2026, 8, 8, 10, 0, 0, tzinfo=timezone.utc)

    # ---------------- accounts / devices / ips (normal population) -------
    accounts = [rand_id("ACC", i) for i in range(1, N_ACCOUNTS + 1)]
    devices = [rand_id("DEV", i) for i in range(1, int(N_ACCOUNTS * 0.9) + 1)]
    payment_instruments = [rand_id("PAY", i) for i in range(1, N_ACCOUNTS + 1)]
    addresses = [rand_id("ADDR", i) for i in range(1, N_ACCOUNTS + 1)]

    account_device = {acc: random.choice(devices) for acc in accounts}
    account_payment = {acc: random.choice(payment_instruments) for acc in accounts}
    account_address = {acc: random.choice(addresses) for acc in accounts}
    account_ip = {
        acc: random.choice(NORMAL_IP_PREFIXES) + str(random.randint(2, 254))
        for acc in accounts
    }

    # -------- embedded fraud pattern #1: one device shared by 5+ accounts
    ring_device = "DEV_RING_1"
    ring_accounts = [f"ACC_RING_{i}" for i in range(1, 7)]  # 6 accounts, one device
    for acc in ring_accounts:
        account_device[acc] = ring_device
        account_payment[acc] = f"PAY_RING_{random.randint(1,2)}"
        account_address[acc] = f"ADDR_RING_{random.randint(1,2)}"
        account_ip[acc] = random.choice(NORMAL_IP_PREFIXES) + str(random.randint(2, 254))
    accounts += ring_accounts

    orders = []
    order_seq = 1

    # normal orders
    for _ in range(N_ORDERS - 40):
        acc = random.choice(accounts)
        category = random.choice(CATEGORIES)
        order = {
            "order_id": rand_id("ORD", order_seq),
            "account_id": acc,
            "device_id": account_device[acc],
            "ip": account_ip[acc],
            "payment_instrument_id": account_payment[acc],
            "shipping_address_id": account_address[acc],
            "order_value": round(random.uniform(300, CATEGORY_MSRP[category] * 0.9), 2),
            "payment_type": random.choice(["cod", "prepaid", "prepaid", "prepaid"]),
            "category": category,
            "session": {
                "duration_seconds": round(random.gauss(90, 25), 1),
                "typing_cadence_ms": round(random.gauss(150, 30), 1),
                "copy_paste_detected": random.random() < 0.05,
            },
            "billing_country": "IN",
            "timestamp": iso(now - timedelta(minutes=random.randint(0, 60 * 24 * 10))),
            "label_fraud": 0,
        }
        orders.append(order)
        order_seq += 1

    # -------- embedded fraud pattern #2: one IP with burst of orders -----
    burst_ip = "203.0.113.99"
    burst_time = now - timedelta(hours=2)
    for i in range(15):
        acc = f"ACC_BURST_{i}"
        accounts.append(acc)
        account_device[acc] = f"DEV_BURST_{i}"
        order = {
            "order_id": rand_id("ORD", order_seq),
            "account_id": acc,
            "device_id": f"DEV_BURST_{i}",
            "ip": burst_ip,
            "payment_instrument_id": f"PAY_BURST_{i}",
            "shipping_address_id": f"ADDR_BURST_{i}",
            "order_value": round(random.uniform(2000, 6000), 2),
            "payment_type": "cod",
            "category": random.choice(CATEGORIES),
            "session": {
                "duration_seconds": round(random.uniform(10, 25), 1),
                "typing_cadence_ms": round(random.uniform(30, 60), 1),
                "copy_paste_detected": True,
            },
            "billing_country": "IN",
            "timestamp": iso(burst_time + timedelta(seconds=i * 20)),
            "label_fraud": 1,
        }
        orders.append(order)
        order_seq += 1

    # -------- ring-device orders (fraud pattern #1 continued) -----------
    for i, acc in enumerate(ring_accounts):
        order = {
            "order_id": rand_id("ORD", order_seq),
            "account_id": acc,
            "device_id": account_device[acc],
            "ip": account_ip[acc],
            "payment_instrument_id": account_payment[acc],
            "shipping_address_id": account_address[acc],
            "order_value": round(random.uniform(1500, 4000), 2),
            "payment_type": "cod",
            "category": random.choice(CATEGORIES),
            "session": {
                "duration_seconds": round(random.uniform(20, 40), 1),
                "typing_cadence_ms": round(random.uniform(40, 70), 1),
                "copy_paste_detected": True,
            },
            "billing_country": "IN",
            "timestamp": iso(now - timedelta(hours=random.randint(1, 48))),
            "label_fraud": 1,
        }
        orders.append(order)
        order_seq += 1

    # a handful of extra vpn/datacenter/geo-mismatch fraud-flavored orders
    for _ in range(10):
        acc = random.choice(accounts)
        order = {
            "order_id": rand_id("ORD", order_seq),
            "account_id": acc,
            "device_id": account_device.get(acc, rand_id("DEV", 9000)),
            "ip": random.choice(VPN_IPS + DATACENTER_IPS),
            "payment_instrument_id": account_payment.get(acc, rand_id("PAY", 9000)),
            "shipping_address_id": account_address.get(acc, rand_id("ADDR", 9000)),
            "order_value": round(random.uniform(1000, 8000), 2),
            "payment_type": "cod",
            "category": random.choice(CATEGORIES),
            "session": {
                "duration_seconds": round(random.uniform(15, 50), 1),
                "typing_cadence_ms": round(random.uniform(40, 90), 1),
                "copy_paste_detected": random.random() < 0.5,
            },
            "billing_country": "IN",
            "timestamp": iso(now - timedelta(hours=random.randint(1, 100))),
            "label_fraud": 1,
        }
        orders.append(order)
        order_seq += 1

    # -------------------------------------------------------- listings ---
    listings = []
    for i in range(1, N_LISTINGS + 1):
        brand = random.choice(BRANDS)
        category = random.choice(list(CATEGORY_MSRP.keys()))
        msrp = CATEGORY_MSRP[category]
        seller = f"SEL_{i}"
        listings.append({
            "listing_id": rand_id("LST", i),
            "seller_id": seller,
            "title": f"{brand} {category} item {i}",
            "description": f"A {category} product from {brand}, model {rand_str(4)}.",
            "price": round(random.uniform(msrp * 0.7, msrp * 1.1), 2),
            "brand": brand,
            "category": category,
            "images": [f"img_{rand_str(6)}"],
            "gtin": f"890{random.randint(1000000,9999999)}",
            "label_counterfeit": 0,
        })

    # -------- embedded fraud pattern #3: counterfeit listing ------------
    listings.append({
        "listing_id": "LST_COUNTERFEIT_1",
        "seller_id": "SEL_UNAUTH_999",
        "title": "L'Oreal Revitalift Serum - Clearance Sale",
        "description": "L'Oreal branded serum, brand new, clearance price, limited stock.",
        "price": round(CATEGORY_MSRP["cosmetics"] * 0.18, 2),  # ~82% below MSRP
        "brand": "L'Oreal",
        "category": "cosmetics",
        "images": ["img_counterfeit_low_quality_scan"],
        "gtin": "8901030000000",
        "label_counterfeit": 1,
    })

    # ---------------------------------------------------------- reviews --
    reviews = []
    review_seq = 1
    valid_order_ids = [o["order_id"] for o in orders]

    for _ in range(N_REVIEWS - 30):
        acc = random.choice(accounts)
        has_order = random.random() < 0.8
        order_id = random.choice(valid_order_ids) if has_order else None
        # if we pick an order, make sure account matches for "verified" case sometimes
        if order_id and random.random() < 0.7:
            matching = [o for o in orders if o["order_id"] == order_id][0]
            acc = matching["account_id"]
        reviews.append({
            "review_id": rand_id("REV", review_seq),
            "account_id": acc,
            "product_id": rand_id("PRD", random.randint(1, N_LISTINGS)),
            "device_id": account_device.get(acc, rand_id("DEV", 9000)),
            "rating": random.choice([3, 4, 4, 5, 5, 5, 2, 1]),
            "text": random.choice(GENUINE_REVIEW_TEMPLATES),
            "order_id": order_id,
            "timestamp": iso(now - timedelta(hours=random.randint(0, 24 * 30))),
            "label_fake": 0,
        })
        review_seq += 1

    # -------- embedded fraud pattern #4: coordinated review ring --------
    ring_review_device = "DEV_REVRING_1"
    ring_time = now - timedelta(hours=1)
    ring_review_accounts = [f"ACC_REVRING_{i}" for i in range(1, 8)]
    shared_product = "PRD_REVRING_TARGET"
    for i, acc in enumerate(ring_review_accounts):
        reviews.append({
            "review_id": rand_id("REV", review_seq),
            "account_id": acc,
            "product_id": shared_product,
            "device_id": ring_review_device,
            "rating": 5,
            "text": random.choice(FAKE_REVIEW_TEMPLATES),
            "order_id": None,
            "timestamp": iso(ring_time + timedelta(minutes=i * 2)),
            "label_fake": 1,
        })
        review_seq += 1

    # product descriptions (for semantic relevance checks)
    product_descriptions = {
        shared_product: "A premium wireless charging pad with fast-charge support.",
    }
    for i in range(1, N_LISTINGS + 1):
        product_descriptions[rand_id("PRD", i)] = f"Product description for item {i}, category varies."

    # ------------------------------------------------------------ write --
    with open(os.path.join(DATA_DIR, "orders.json"), "w") as f:
        json.dump(orders, f, indent=2)
    with open(os.path.join(DATA_DIR, "listings.json"), "w") as f:
        json.dump(listings, f, indent=2)
    with open(os.path.join(DATA_DIR, "reviews.json"), "w") as f:
        json.dump(reviews, f, indent=2)
    with open(os.path.join(DATA_DIR, "product_descriptions.json"), "w") as f:
        json.dump(product_descriptions, f, indent=2)

    print(f"Wrote {len(orders)} orders, {len(listings)} listings, {len(reviews)} reviews to {DATA_DIR}")
    print("Embedded fraud patterns:")
    print(f"  - device '{ring_device}' shared across {len(ring_accounts)} accounts")
    print(f"  - IP '{burst_ip}' burst of {15} orders within 5 minutes")
    print(f"  - counterfeit listing 'LST_COUNTERFEIT_1' (82% below MSRP, unauthorized seller)")
    print(f"  - review ring on device '{ring_review_device}' / product '{shared_product}' ({len(ring_review_accounts)} reviews)")


if __name__ == "__main__":
    main()
