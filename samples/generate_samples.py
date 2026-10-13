#!/usr/bin/env python
"""ML Arena - bundled sample dataset generator.

    python samples/generate_samples.py

Writes seven reproducible CSV datasets into ``<project root>/samples``.  Every
dataset is generated from a *fixed* random seed and is byte-for-byte identical
on every run (idempotent), so the shipped CSVs are reproducible from this file.

The datasets are not random noise.  Each one contains a *planted signal* that
the matching ML Arena use case is supposed to discover, so a brand new user
sees the product find something real within the first thirty seconds.

The filename prefix encodes the recommended use-case key, e.g.
``orders_discount_leakage.csv`` -> use case ``discount_leakage``.

Nothing here touches the network, and nothing here imports any other ML Arena
module except ``mlarena.config`` (which must be imported before polars).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --- make the project root importable when run as a plain script --------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mlarena import config  # noqa: E402  -- MUST precede the polars import

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

SEED = 42

OUT_DIR = Path(config.SAMPLES_DIR) if hasattr(config, "SAMPLES_DIR") else Path(__file__).resolve().parent


# =============================================================================
# small helpers
# =============================================================================
def _rng(offset: int) -> np.random.Generator:
    """Deterministic generator; every dataset gets its own stream."""
    return np.random.default_rng(SEED + offset)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _calibrate_intercept(z: np.ndarray, target_rate: float) -> float:
    """Find c so that mean(sigmoid(z + c)) == target_rate (bisection)."""
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if float(_sigmoid(z + mid).mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _choice(rng: np.random.Generator, values, weights, size: int) -> np.ndarray:
    return rng.choice(np.asarray(values, dtype=object), size=size, p=np.asarray(weights, dtype=float))


def _cat(rng: np.random.Generator, mapping: dict, size: int) -> np.ndarray:
    keys = list(mapping.keys())
    weights = np.asarray([mapping[k] for k in keys], dtype=float)
    return _choice(rng, keys, weights / weights.sum(), size)


def _dates(rng: np.random.Generator, n: int, start: str, end: str) -> np.ndarray:
    """n datetime64[D] values uniformly spread across [start, end]."""
    lo = np.datetime64(start, "D")
    hi = np.datetime64(end, "D")
    span = int((hi - lo).astype(int))
    offs = rng.integers(0, span + 1, size=n)
    return (lo + offs.astype("timedelta64[D]")).astype("datetime64[D]")


def _date_series(name: str, dates: np.ndarray) -> pl.Series:
    return pl.Series(name, dates).cast(pl.Date)


def _str_series(name: str, values, null_mask: np.ndarray | None = None) -> pl.Series:
    """String column, optionally with a boolean mask of missing values."""
    out = np.asarray(values).astype(object)
    if null_mask is not None and bool(null_mask.any()):
        out = out.copy()
        out[null_mask] = None
    return pl.Series(name, out.tolist(), dtype=pl.String)


def _money(x: np.ndarray) -> np.ndarray:
    """Round to paise (2dp) the way an ERP would store it."""
    return np.round(np.asarray(x, dtype=float), 2)


def _rat(x: np.ndarray, digits: int = 6) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float), digits)


# =============================================================================
# shared business vocabulary  (kept consistent across every order dataset)
# =============================================================================
SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Government"]
SEG_WEIGHTS = {"Enterprise": 0.22, "Mid-Market": 0.31, "SMB": 0.38, "Government": 0.09}
SEG_DISCOUNT = {"Enterprise": 0.205, "Mid-Market": 0.145, "SMB": 0.085, "Government": 0.155}

REGIONS = ["North", "South", "East", "West", "Central", "North-East", "International"]
REGION_WEIGHTS = {
    "North": 0.19,
    "South": 0.17,
    "East": 0.14,
    "West": 0.18,
    "Central": 0.11,
    "North-East": 0.08,
    "International": 0.13,
}
REGION_COUNTRIES = {
    "North": ["India"],
    "South": ["India"],
    "East": ["India"],
    "West": ["India"],
    "Central": ["India"],
    "North-East": ["India"],
    "International": ["UAE", "Singapore", "United Kingdom", "Australia", "Germany"],
}
REGION_COUNTRY_WEIGHTS = {
    "North": [1.0],
    "South": [1.0],
    "East": [1.0],
    "West": [1.0],
    "Central": [1.0],
    "North-East": [1.0],
    "International": [0.30, 0.24, 0.20, 0.15, 0.11],
}

CHANNELS = ["Direct", "Partner", "Distributor", "Online"]
CHANNEL_WEIGHTS = {"Direct": 0.42, "Partner": 0.24, "Distributor": 0.16, "Online": 0.18}
CHANNEL_ADJ = {"Direct": 0.000, "Partner": 0.045, "Distributor": 0.030, "Online": -0.030}

PAYMENT_TERMS = ["Net 30", "Net 45", "Net 60", "Prepaid"]
PAYMENT_WEIGHTS = {"Net 30": 0.45, "Net 45": 0.20, "Net 60": 0.25, "Prepaid": 0.10}

ORDER_STATUS = ["Completed", "Shipped", "Cancelled", "Returned", "Pending"]
ORDER_STATUS_WEIGHTS = {"Completed": 0.86, "Shipped": 0.08, "Cancelled": 0.03, "Returned": 0.02, "Pending": 0.01}

# category -> [(product name, list price in INR, cost as a fraction of list price)]
PRODUCTS: dict[str, list[tuple[str, float, float]]] = {
    "Software": [
        ("Atlas CRM Suite", 68000, 0.25),
        ("Nimbus ERP Core", 125000, 0.24),
        ("Vertex Analytics Pro", 48000, 0.26),
        ("Helios HR Platform", 42000, 0.27),
        ("Orion Billing Engine", 32000, 0.28),
    ],
    "Hardware": [
        ("Titan Rack Server R740", 780000, 0.70),
        ("Vector Laptop V15", 92000, 0.66),
        ("Nova Storage Array 8TB", 410000, 0.71),
        ("Pulse Network Switch 48P", 135000, 0.64),
    ],
    "Cloud Services": [
        ("Cirrus Compute Credits", 78000, 0.42),
        ("Cirrus Object Storage", 38000, 0.40),
        ("Cirrus Managed Database", 62000, 0.44),
        ("Cirrus CDN Plan", 24000, 0.39),
    ],
    "Support": [
        ("Premium Support Gold", 72000, 0.35),
        ("Premium Support Platinum", 98000, 0.34),
        ("24x7 Support Standard", 30000, 0.37),
    ],
    "Professional Services": [
        ("Implementation Package", 145000, 0.56),
        ("Data Migration Service", 88000, 0.54),
        ("Custom Integration Sprint", 190000, 0.55),
        ("Training Workshop", 42000, 0.58),
    ],
    "Networking": [
        ("Falcon Router FX-9", 68000, 0.62),
        ("Falcon Firewall FG-500", 115000, 0.63),
        ("Falcon Access Point AX6", 19000, 0.61),
    ],
    "Security": [
        ("Sentinel Endpoint Security", 48000, 0.44),
        ("Sentinel SIEM Licence", 120000, 0.46),
        ("Sentinel Identity Guard", 72000, 0.45),
    ],
}
CATEGORIES = list(PRODUCTS.keys())
CATEGORY_WEIGHTS = {
    "Software": 0.19,
    "Hardware": 0.15,
    "Cloud Services": 0.17,
    "Support": 0.13,
    "Professional Services": 0.11,
    "Networking": 0.13,
    "Security": 0.12,
}
# category-level discount offset: low-margin kit simply cannot be discounted hard
CATEGORY_DISCOUNT_ADJ = {
    "Software": 0.030,
    "Hardware": -0.075,
    "Cloud Services": 0.010,
    "Support": -0.010,
    "Professional Services": -0.020,
    "Networking": -0.060,
    "Security": -0.015,
}
# how many units a "typical" order contains, relative to the segment norm: nobody
# buys six rack servers the way they buy six software seats
CATEGORY_QTY_MULT = {
    "Software": 1.00,
    "Hardware": 0.22,
    "Cloud Services": 1.45,
    "Support": 0.70,
    "Professional Services": 0.35,
    "Networking": 0.50,
    "Security": 0.95,
}
SEG_QTY_SCALE = {"Enterprise": 62.0, "Mid-Market": 30.0, "SMB": 9.0, "Government": 38.0}

REP_NAMES = [
    "Aarav Sharma", "Priya Nair", "Rohan Mehta", "Ananya Iyer", "Vikram Singh", "Neha Gupta",
    "Karthik Reddy", "Sneha Desai", "Arjun Rao", "Divya Menon", "Rahul Verma", "Pooja Joshi",
    "Sanjay Kulkarni", "Meera Pillai", "Aditya Bose", "Ritu Agarwal", "Nikhil Chauhan", "Tanvi Shah",
    "Manish Tiwari", "Kavya Krishnan", "Suresh Patil", "Ishita Banerjee", "Harsh Vardhan", "Deepa Shetty",
    "Gaurav Malhotra", "Lakshmi Narayan", "Farhan Khan", "Anjali Saxena", "Varun Kapoor", "Shreya Ghosh",
    "Rajesh Nambiar", "Pallavi Rane", "Siddharth Jain", "Nandini Rao", "Mohit Bansal", "Aisha Siddiqui",
    "Prakash Mishra", "Swati Deshpande", "Yash Thakur", "Rekha Varma",
]
# 40 reps spread across the seven sales regions (6+6+5+6+5+5+7 = 40)
REP_HOME: list[str] = (
    ["North"] * 6 + ["South"] * 6 + ["East"] * 5 + ["West"] * 6 + ["Central"] * 5 + ["North-East"] * 5 + ["International"] * 7
)
# Reps who habitually over-discount.  Deliberately clustered in North / West so the
# driver analysis has a genuine, concentrated story to tell.
LEAKY_REPS = {1, 3, 5, 8, 18, 21, 25}
LEAKY_REP_PROB = 0.48          # share of a leaky rep's orders that blow the budget
NORMAL_LEAK_PROB = 0.012       # background noise from everyone else
REGION_LEAK_MULT = {"North": 1.16, "West": 1.12, "South": 1.04, "Central": 1.02}


def _rep_index(rng: np.random.Generator, regions: np.ndarray) -> np.ndarray:
    """Pick a sales rep *within* each row's region (reps belong to a territory)."""
    reps_by_region: dict[str, np.ndarray] = {}
    for region in REGIONS:
        reps_by_region[region] = np.array([i for i, r in enumerate(REP_HOME) if r == region], dtype=np.int64)
    out = np.zeros(len(regions), dtype=np.int64)
    for region, pool in reps_by_region.items():
        mask = regions == region
        k = int(mask.sum())
        if k:
            out[mask] = rng.choice(pool, size=k)
    return out


# =============================================================================
# 1 + 7.  discount-leakage orders  (60k flagship, and the 500k benchmark)
# =============================================================================
def build_discount_orders(n: int, offset: int, start: str = "2024-01-01", end: str = "2025-12-31") -> pl.DataFrame:
    rng = _rng(offset)

    dates = _dates(rng, n, start, end)
    segments = _cat(rng, SEG_WEIGHTS, n)
    regions = _cat(rng, REGION_WEIGHTS, n)
    countries = np.array(
        [_choice(rng, REGION_COUNTRIES[r], REGION_COUNTRY_WEIGHTS[r], 1)[0] for r in regions], dtype=object
    )
    channels = _cat(rng, CHANNEL_WEIGHTS, n)
    categories = _cat(rng, CATEGORY_WEIGHTS, n)

    # product is drawn inside its category, so product name and price always agree
    product_name = np.empty(n, dtype=object)
    list_price = np.empty(n, dtype=float)
    cost_ratio = np.empty(n, dtype=float)
    for cat in CATEGORIES:
        mask = categories == cat
        k = int(mask.sum())
        if not k:
            continue
        picks = rng.integers(0, len(PRODUCTS[cat]), size=k)
        product_name[mask] = [PRODUCTS[cat][p][0] for p in picks]
        list_price[mask] = [PRODUCTS[cat][p][1] for p in picks]
        cost_ratio[mask] = [PRODUCTS[cat][p][2] for p in picks]

    rep_idx = _rep_index(rng, regions)
    sales_rep = np.array([REP_NAMES[i] for i in rep_idx], dtype=object)

    # quantity: enterprise buys in bulk, SMB buys a handful, and heavy kit is
    # bought in smaller counts than per-seat software
    seg_qty_scale = np.array([SEG_QTY_SCALE[s] for s in segments])
    cat_qty_mult = np.array([CATEGORY_QTY_MULT[c] for c in categories])
    quantity = np.maximum(
        1, np.round(rng.lognormal(mean=np.log(seg_qty_scale * cat_qty_mult) - 0.45, sigma=0.95))
    ).astype(np.int64)
    quantity = np.clip(quantity, 1, 900)

    # --- expected discount: a sensible function of context ---------------------
    qty_effect = 0.035 * np.clip(np.log10(quantity.astype(float)), 0.0, 2.4)
    ch_effect = np.array([CHANNEL_ADJ[c] for c in channels])
    seg_effect = np.array([SEG_DISCOUNT[s] for s in segments])
    cat_effect = np.array([CATEGORY_DISCOUNT_ADJ[c] for c in categories])
    base_discount = seg_effect + cat_effect + ch_effect + qty_effect + rng.normal(0.0, 0.018, n)

    # --- planted leakage: a few reps (and their regions) go far past the norm ---
    is_leaky_rep = np.isin(rep_idx, list(LEAKY_REPS))
    leak_prob = np.where(is_leaky_rep, LEAKY_REP_PROB, NORMAL_LEAK_PROB)
    leak_mask = rng.random(n) < leak_prob
    leak_mag = np.where(
        is_leaky_rep,
        rng.uniform(0.085, 0.215, n),
        rng.uniform(0.045, 0.090, n),
    )
    region_mult = np.array([REGION_LEAK_MULT.get(r, 1.0) for r in regions])
    leak = np.where(leak_mask, leak_mag * region_mult, 0.0)

    discount_pct = np.clip(base_discount + leak, 0.0, 0.72)

    unit_price = _money(list_price * (1.0 - discount_pct))
    # a volume rebate lands on top of the negotiated price for some orders, so
    # unit_price and net_price are related but not clones
    rebate = np.where(rng.random(n) < 0.20, rng.uniform(0.002, 0.022, n), 0.0)
    rebate = np.where(quantity >= 120, np.maximum(rebate, rng.uniform(0.006, 0.020, n)), rebate)
    net_price = _money(unit_price * (1.0 - rebate))
    discount_pct = _rat(np.where(list_price > 0, (list_price - net_price) / list_price, 0.0))

    revenue = _money(quantity * net_price)
    unit_cost = list_price * cost_ratio * rng.normal(1.0, 0.035, n)
    cost = _money(quantity * unit_cost)
    gross_margin_pct = _rat(np.where(revenue > 0, (revenue - cost) / revenue, 0.0))

    # --- dates / identifiers ---------------------------------------------------
    order = np.argsort(dates, kind="stable")
    dates, segments, regions, countries = dates[order], segments[order], regions[order], countries[order]
    channels, categories = channels[order], categories[order]
    product_name, list_price, cost_ratio = product_name[order], list_price[order], cost_ratio[order]
    sales_rep, quantity = sales_rep[order], quantity[order]
    unit_price, net_price, discount_pct = unit_price[order], net_price[order], discount_pct[order]
    revenue, cost, gross_margin_pct = revenue[order], cost[order], gross_margin_pct[order]

    year = dates.astype("datetime64[Y]").astype(int) + 1970
    seq = np.arange(1, n + 1)
    order_id = np.array([f"ORD-{y}-{s:06d}" for y, s in zip(year, seq)], dtype=object)
    customer_id = np.array([f"CUST-{i:05d}" for i in rng.integers(1, 4001, n)], dtype=object)

    payment_terms = _cat(rng, PAYMENT_WEIGHTS, n)
    order_status = _cat(rng, ORDER_STATUS_WEIGHTS, n)

    # --- realistic mess: missing values, duplicates ----------------------------
    null_terms = rng.random(n) < 0.014
    null_country = rng.random(n) < 0.009
    null_rep = rng.random(n) < 0.006

    df = pl.DataFrame(
        [
            _str_series("order_id", order_id),
            _date_series("order_date", dates),
            _str_series("customer_id", customer_id),
            _str_series("customer_segment", segments),
            _str_series("region", regions),
            _str_series("country", countries, null_country),
            _str_series("product_category", categories),
            _str_series("product_name", product_name),
            _str_series("channel", channels),
            _str_series("sales_rep", sales_rep, null_rep),
            pl.Series("quantity", quantity, dtype=pl.Int64),
            pl.Series("list_price", list_price, dtype=pl.Float64),
            pl.Series("unit_price", unit_price, dtype=pl.Float64),
            pl.Series("net_price", net_price, dtype=pl.Float64),
            pl.Series("discount_pct", discount_pct, dtype=pl.Float64),
            pl.Series("revenue", revenue, dtype=pl.Float64),
            pl.Series("cost", cost, dtype=pl.Float64),
            pl.Series("gross_margin_pct", gross_margin_pct, dtype=pl.Float64),
            _str_series("payment_terms", payment_terms, null_terms),
            _str_series("order_status", order_status),
        ]
    )

    # a handful of genuine duplicate rows for the data-quality report
    dup_idx = rng.choice(n, size=max(1, int(n * 0.0007)), replace=False)
    df = pl.concat([df, df[dup_idx.tolist()]]).sort("order_date", maintain_order=True)
    return df


# =============================================================================
# 2.  margin-overflow orders  (40k)
# =============================================================================
def build_margin_orders(n: int, offset: int, target_overflow: float = 0.20) -> pl.DataFrame:
    rng = _rng(offset)
    dates = _dates(rng, n, "2024-01-01", "2025-12-31")
    segments = _cat(rng, SEG_WEIGHTS, n)
    regions = _cat(rng, REGION_WEIGHTS, n)
    countries = np.array(
        [_choice(rng, REGION_COUNTRIES[r], REGION_COUNTRY_WEIGHTS[r], 1)[0] for r in regions], dtype=object
    )
    channels = _cat(rng, CHANNEL_WEIGHTS, n)
    categories = _cat(rng, CATEGORY_WEIGHTS, n)

    product_name = np.empty(n, dtype=object)
    list_price = np.empty(n, dtype=float)
    cost_ratio = np.empty(n, dtype=float)
    for cat in CATEGORIES:
        mask = categories == cat
        k = int(mask.sum())
        if not k:
            continue
        picks = rng.integers(0, len(PRODUCTS[cat]), size=k)
        product_name[mask] = [PRODUCTS[cat][p][0] for p in picks]
        list_price[mask] = [PRODUCTS[cat][p][1] for p in picks]
        cost_ratio[mask] = [PRODUCTS[cat][p][2] for p in picks]

    rep_idx = _rep_index(rng, regions)
    sales_rep = np.array([REP_NAMES[i] for i in rep_idx], dtype=object)

    seg_qty_scale = np.array([SEG_QTY_SCALE[s] for s in segments])
    cat_qty_mult = np.array([CATEGORY_QTY_MULT[c] for c in categories])
    quantity = np.clip(
        np.maximum(1, np.round(rng.lognormal(np.log(seg_qty_scale * cat_qty_mult) - 0.45, 0.95))), 1, 900
    ).astype(np.int64)

    seg_effect = np.array([SEG_DISCOUNT[s] for s in segments])
    cat_effect = np.array([CATEGORY_DISCOUNT_ADJ[c] for c in categories])
    ch_effect = np.array([CHANNEL_ADJ[c] for c in channels])
    qty_effect = 0.035 * np.clip(np.log10(quantity.astype(float)), 0.0, 2.4)
    discount_pct = np.clip(
        seg_effect + cat_effect + ch_effect + qty_effect + rng.normal(0.0, 0.017, n), 0.0, 0.60
    )

    # The pricing policy allows a discount (with a safety buffer).  Everything the
    # policy implies - including the floor margin - follows from that number.
    policy_discount = np.clip(seg_effect + cat_effect + 0.087, 0.03, 0.45)
    policy_margin_floor = _rat(1.0 - cost_ratio / (1.0 - policy_discount))

    unit_price = _money(list_price * (1.0 - discount_pct))
    revenue = _money(quantity * unit_price)
    unit_cost = list_price * cost_ratio * rng.normal(1.0, 0.030, n)
    cost = _money(quantity * unit_cost)
    margin = np.where(revenue > 0, (revenue - cost) / revenue, 0.0)

    # --- plant margin erosion until ~target_overflow of orders breach the floor --
    below = margin < policy_margin_floor
    need = int(round(target_overflow * n)) - int(below.sum())
    if need > 0:
        # context weights: two regions and three categories carry the problem,
        # plus the same handful of over-discounting reps as the flagship sample
        hot_region = np.isin(regions, ["North", "West"]).astype(float)
        hot_cat = np.isin(categories, ["Hardware", "Networking", "Security"]).astype(float)
        hot_rep = np.isin(rep_idx, list(LEAKY_REPS)).astype(float)
        w = 0.35 + 2.6 * hot_region * hot_cat + 1.9 * hot_rep + 0.6 * hot_region
        w = np.where(below, 0.0, w)
        p = w / w.sum()
        picked = rng.choice(n, size=min(need, int((~below).sum())), replace=False, p=p)
        overshoot = rng.uniform(0.004, 0.095, picked.size)
        target_margin = policy_margin_floor[picked] - overshoot
        target_margin = np.clip(target_margin, -0.35, 0.95)
        via_discount = rng.random(picked.size) < 0.62
        # 62%: the discount is too deep
        sel = picked[via_discount]
        tm = target_margin[via_discount]
        new_disc = np.clip(1.0 - cost_ratio[sel] / (1.0 - tm), 0.0, 0.85)
        unit_price[sel] = _money(list_price[sel] * (1.0 - new_disc))
        discount_pct[sel] = new_disc
        revenue[sel] = _money(quantity[sel] * unit_price[sel])
        # 38%: the cost came in too high (procurement / rework / freight)
        sel2 = picked[~via_discount]
        tm2 = target_margin[~via_discount]
        cost[sel2] = _money(revenue[sel2] * (1.0 - tm2))

    gross_margin_pct = _rat(np.where(revenue > 0, (revenue - cost) / revenue, 0.0))
    below_floor = (gross_margin_pct < policy_margin_floor).astype(np.int64)

    order = np.argsort(dates, kind="stable")
    (dates, segments, regions, countries, channels, categories, product_name, list_price,
     sales_rep, quantity, unit_price, discount_pct, revenue, cost, gross_margin_pct,
     policy_margin_floor, below_floor) = (
        a[order] for a in (dates, segments, regions, countries, channels, categories, product_name,
                           list_price, sales_rep, quantity, unit_price, discount_pct, revenue, cost,
                           gross_margin_pct, policy_margin_floor, below_floor)
    )

    year = dates.astype("datetime64[Y]").astype(int) + 1970
    order_id = np.array([f"MO-{y}-{s:06d}" for y, s in zip(year, np.arange(1, n + 1))], dtype=object)
    customer_id = np.array([f"CUST-{i:05d}" for i in rng.integers(1, 3201, n)], dtype=object)
    payment_terms = _cat(rng, PAYMENT_WEIGHTS, n)
    order_status = _cat(rng, ORDER_STATUS_WEIGHTS, n)

    null_terms = rng.random(n) < 0.011
    null_region = rng.random(n) < 0.007

    return pl.DataFrame(
        [
            _str_series("order_id", order_id),
            _date_series("order_date", dates),
            _str_series("customer_id", customer_id),
            _str_series("customer_segment", segments),
            _str_series("region", regions, null_region),
            _str_series("country", countries),
            _str_series("product_category", categories),
            _str_series("product_name", product_name),
            _str_series("channel", channels),
            _str_series("sales_rep", sales_rep),
            pl.Series("quantity", quantity, dtype=pl.Int64),
            pl.Series("list_price", list_price, dtype=pl.Float64),
            pl.Series("unit_price", unit_price, dtype=pl.Float64),
            pl.Series("discount_pct", _rat(discount_pct), dtype=pl.Float64),
            pl.Series("revenue", revenue, dtype=pl.Float64),
            pl.Series("cost", cost, dtype=pl.Float64),
            pl.Series("gross_margin_pct", gross_margin_pct, dtype=pl.Float64),
            pl.Series("policy_margin_floor", policy_margin_floor, dtype=pl.Float64),
            pl.Series("below_margin_floor", below_floor, dtype=pl.Int64),
            _str_series("payment_terms", payment_terms, null_terms),
            _str_series("order_status", order_status),
        ]
    )


# =============================================================================
# 3.  demand series  (30k)
# =============================================================================
DEMAND_REGIONS = ["North", "South", "East", "West", "Central", "North-East"]
DEMAND_CATEGORIES = CATEGORIES
DEMAND_BASE = {
    "Software": 118.0,
    "Hardware": 44.0,
    "Cloud Services": 262.0,
    "Support": 88.0,
    "Professional Services": 33.0,
    "Networking": 61.0,
    "Security": 76.0,
}
DEMAND_PRICE = {
    "Software": 11000.0,
    "Hardware": 52000.0,
    "Cloud Services": 4200.0,
    "Support": 9000.0,
    "Professional Services": 26000.0,
    "Networking": 21000.0,
    "Security": 14000.0,
}
DEMAND_REGION_MULT = {
    "North": 1.42,
    "South": 1.28,
    "East": 0.92,
    "West": 1.35,
    "Central": 0.78,
    "North-East": 0.61,
}
# Mon..Sun - a B2B buying week: quiet weekends
WEEKDAY_FACTOR = np.array([1.10, 1.16, 1.13, 1.08, 0.94, 0.44, 0.39])


def build_demand_series(offset: int, start: str = "2024-01-01", end: str = "2025-12-31") -> pl.DataFrame:
    rng = _rng(offset)
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + np.timedelta64(1, "D"))
    doy = (days - days.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    dow = ((days.astype(int) + 3) % 7)  # 1970-01-01 was a Thursday -> Monday == 0

    t = np.arange(days.size, dtype=float)
    trend = 1.0 + 0.00034 * t                                     # ~+25% over two years
    weekly = WEEKDAY_FACTOR[dow]
    annual = (
        1.0
        + 0.24 * np.exp(-(((doy - 78) / 24.0) ** 2))              # fiscal year-end push (Mar)
        + 0.19 * np.exp(-(((doy - 305) / 28.0) ** 2))             # festive season (Nov)
        - 0.16 * np.exp(-(((doy - 358) / 12.0) ** 2))             # December shutdown
    )

    frames: list[pl.DataFrame] = []
    for region in DEMAND_REGIONS:
        for cat in DEMAND_CATEGORIES:
            level = DEMAND_BASE[cat] * DEMAND_REGION_MULT[region] * rng.uniform(0.88, 1.12)
            promo_p = np.clip(0.075 + 0.16 * (annual - 0.85), 0.02, 0.30)
            promo = (rng.random(days.size) < promo_p).astype(np.int64)
            promo_lift = np.where(promo == 1, rng.uniform(1.22, 1.58, days.size), 1.0)
            demand_index = level * trend * weekly * annual * promo_lift

            # stock-outs bite hardest exactly when demand is highest
            stock_lam = np.clip(0.10 + 0.0022 * demand_index + 0.45 * promo, 0.02, 6.0)
            stockouts = np.minimum(rng.poisson(stock_lam), 9)
            drag = np.clip(1.0 - 0.045 * stockouts, 0.55, 1.0)

            units = np.maximum(1, np.round(demand_index * drag * rng.normal(1.0, 0.085, days.size))).astype(np.int64)
            price = (
                DEMAND_PRICE[cat]
                * (DEMAND_REGION_MULT[region] ** 0.28)
                * np.where(promo == 1, rng.uniform(0.86, 0.94, days.size), 1.0)
                * np.exp(rng.normal(0.0, 0.012, days.size))
            )
            price = _money(price)
            revenue = _money(units * price)

            frames.append(
                pl.DataFrame(
                    [
                        _date_series("date", days),
                        _str_series("region", np.full(days.size, region, dtype=object)),
                        _str_series("product_category", np.full(days.size, cat, dtype=object)),
                        pl.Series("units_sold", units, dtype=pl.Int64),
                        pl.Series("avg_price", price, dtype=pl.Float64),
                        pl.Series("promo_flag", promo, dtype=pl.Int64),
                        pl.Series("stockouts", stockouts.astype(np.int64), dtype=pl.Int64),
                        pl.Series("revenue", revenue, dtype=pl.Float64),
                    ]
                )
            )
    return pl.concat(frames).sort(["date", "region", "product_category"])


# =============================================================================
# 4.  customer churn  (25k)
# =============================================================================
def build_churn(offset: int, n: int = 25000, snapshot: str = "2026-06-30") -> pl.DataFrame:
    rng = _rng(offset)
    ref = np.datetime64(snapshot, "D")

    contract = _cat(rng, {"Monthly": 0.48, "Annual": 0.34, "Two-Year": 0.18}, n)
    tenure = np.empty(n, dtype=float)
    for ctype, lo, hi in (("Monthly", 30, 900), ("Annual", 180, 1150), ("Two-Year", 400, 1500)):
        m = contract == ctype
        tenure[m] = rng.uniform(lo, hi, int(m.sum()))

    ticket_lambda = np.array([{"Monthly": 2.30, "Annual": 1.30, "Two-Year": 0.90}[c] for c in contract]) * np.clip(
        tenure / 400.0, 0.5, 3.0
    )
    support_tickets = np.minimum(rng.poisson(ticket_lambda), 15).astype(np.int64)

    contract_bonus = np.array([{"Monthly": 0.0, "Annual": 0.55, "Two-Year": 0.95}[c] for c in contract])
    satisfaction = np.clip(
        7.35 - 0.34 * support_tickets + contract_bonus + rng.normal(0.0, 1.15, n), 1.0, 10.0
    )
    satisfaction = np.round(satisfaction, 1)

    # --- latent churn propensity (this is what makes the label learnable) ------
    contract_effect = np.array([{"Monthly": 0.90, "Annual": -0.25, "Two-Year": -0.98}[c] for c in contract])
    z = (
        0.52 * (support_tickets - 2.0)
        - 0.60 * (satisfaction - 7.0)
        + contract_effect
        - 0.00072 * (tenure - 420.0)
        + rng.normal(0.0, 0.92, n)
    )
    z = z + _calibrate_intercept(z, 0.30)
    is_churned = (rng.random(n) < _sigmoid(z)).astype(np.int64)

    # --- recency: churned customers went quiet, but with real overlap ----------
    days_since = np.where(
        is_churned == 1,
        rng.uniform(90, 540, n),
        rng.uniform(0, 200, n),
    )
    # make the two groups genuinely overlap so the model has to work for it
    flip = rng.random(n) < 0.16
    days_since = np.where(flip, rng.uniform(60, 230, n), days_since)
    days_since = np.minimum(np.round(days_since), tenure - 5.0)
    days_since = np.maximum(days_since, 0.0).astype(np.int64)

    last_purchase = ref - days_since.astype("timedelta64[D]")
    signup = ref - np.round(tenure).astype("timedelta64[D]")

    freq_lambda = np.clip(1.2 + tenure / 210.0 - 1.6 * is_churned, 0.4, 30.0)
    total_orders = np.maximum(1, rng.poisson(freq_lambda)).astype(np.int64)
    aov = np.round(rng.lognormal(np.log(42000.0), 0.85, n), 2)
    aov = np.where(contract == "Two-Year", aov * 1.22, np.where(contract == "Annual", aov * 1.09, aov))
    total_revenue = _money(total_orders * aov)
    avg_order_value = _money(total_revenue / np.maximum(total_orders, 1))

    customer_id = np.array([f"CUST-{i:05d}" for i in rng.permutation(np.arange(1, n + 1))], dtype=object)
    null_sat = rng.random(n) < 0.013

    return (
        pl.DataFrame(
            [
                _str_series("customer_id", customer_id),
                _date_series("signup_date", signup.astype("datetime64[D]")),
                _date_series("last_purchase_date", last_purchase.astype("datetime64[D]")),
                pl.Series("total_orders", total_orders, dtype=pl.Int64),
                pl.Series("total_revenue", total_revenue, dtype=pl.Float64),
                pl.Series("avg_order_value", avg_order_value, dtype=pl.Float64),
                pl.Series("days_since_last_order", days_since, dtype=pl.Int64),
                pl.Series("support_tickets", support_tickets, dtype=pl.Int64),
                pl.Series("satisfaction_score", satisfaction, dtype=pl.Float64).set(
                    pl.Series(null_sat), None
                ),
                _str_series("contract_type", contract),
                pl.Series("is_churned", is_churned, dtype=pl.Int64),
            ]
        )
        .sort("customer_id")
    )


# =============================================================================
# 5.  loan credit default  (30k)
# =============================================================================
def build_credit(offset: int, n: int = 30000) -> pl.DataFrame:
    rng = _rng(offset)
    app_date = _dates(rng, n, "2024-01-01", "2025-12-31")

    credit_score = np.clip(rng.normal(692.0, 79.0, n), 300.0, 850.0)
    annual_income = np.round(rng.lognormal(np.log(880000.0), 0.62, n), 0)
    annual_income = np.clip(annual_income, 180000, 12000000)
    employment_years = np.clip(rng.gamma(2.0, 3.4, n), 0.0, 38.0).round(1)

    dti_target = np.clip(
        0.14 + (760.0 - credit_score) / 420.0 + rng.normal(0.0, 0.11, n) + 0.004 * np.maximum(0.0, 6.0 - employment_years),
        0.02,
        0.86,
    )
    existing_debt = np.round(dti_target * annual_income, 0)
    debt_to_income = np.round(np.clip(existing_debt / annual_income, 0.02, 0.90), 4)

    num_late_payments = np.minimum(
        rng.poisson(np.clip(1.9 - (credit_score - 690.0) / 110.0, 0.05, 6.0)), 9
    ).astype(np.int64)
    num_prior_loans = np.minimum(rng.poisson(1.6, n), 7).astype(np.int64)

    term_months = _choice(rng, [12, 24, 36, 48, 60, 84], [0.10, 0.17, 0.30, 0.14, 0.21, 0.08], n).astype(np.int64)
    interest_rate = np.clip(
        10.4
        + (712.0 - credit_score) / 34.0
        + 2.6 * (debt_to_income - 0.35)
        + 0.09 * num_late_payments
        + rng.normal(0.0, 0.85, n),
        6.9,
        26.5,
    )
    interest_rate = np.round(interest_rate, 2)

    loan_amount = np.round(np.clip(rng.lognormal(np.log(620000.0), 0.85, n), 75000, 6000000), -3)

    loan_purpose = _cat(
        rng,
        {"Home": 0.22, "Auto": 0.19, "Personal": 0.26, "Education": 0.11, "Business": 0.15, "Gold": 0.07},
        n,
    )
    home_ownership = _cat(rng, {"Rent": 0.38, "Own": 0.31, "Mortgage": 0.31}, n)

    # --- default is a learnable function of score, DTI, late payments and rate --
    z = (
        1.00 * (700.0 - credit_score) / 70.0
        + 0.80 * (debt_to_income - 0.35) / 0.15
        + 0.44 * num_late_payments
        + 0.30 * (interest_rate - 13.0) / 3.0
        + 0.10 * num_prior_loans
        - 0.030 * employment_years
        + np.where(np.isin(loan_purpose, ["Personal", "Business"]), 0.24, -0.06)
        + rng.normal(0.0, 0.55, n)
    )
    z = z + _calibrate_intercept(z, 0.12)
    default_flag = (rng.random(n) < _sigmoid(z)).astype(np.int64)

    loan_id = np.array([f"LN-{i:07d}" for i in rng.permutation(np.arange(1, n + 1))], dtype=object)
    customer_id = np.array([f"CUST-{i:05d}" for i in rng.integers(1, 26001, n)], dtype=object)

    return pl.DataFrame(
        [
            _str_series("loan_id", loan_id),
            _str_series("customer_id", customer_id),
            _date_series("application_date", app_date),
            pl.Series("loan_amount", loan_amount, dtype=pl.Float64),
            pl.Series("term_months", term_months, dtype=pl.Int64),
            pl.Series("interest_rate", interest_rate, dtype=pl.Float64),
            pl.Series("annual_income", annual_income, dtype=pl.Float64),
            pl.Series("employment_years", np.round(employment_years, 1), dtype=pl.Float64),
            pl.Series("credit_score", np.round(credit_score, 0), dtype=pl.Int64),
            pl.Series("existing_debt", existing_debt, dtype=pl.Float64),
            pl.Series("debt_to_income", debt_to_income, dtype=pl.Float64),
            pl.Series("num_prior_loans", num_prior_loans, dtype=pl.Int64),
            pl.Series("num_late_payments", num_late_payments, dtype=pl.Int64),
            _str_series("loan_purpose", loan_purpose),
            _str_series("home_ownership", home_ownership),
            pl.Series("default_flag", default_flag, dtype=pl.Int64),
        ]
    ).sort("application_date")


# =============================================================================
# 6.  transaction anomaly  (50k)
# =============================================================================
MERCHANT_CATEGORIES = [
    "Grocery", "Restaurants", "Fuel", "Travel", "Electronics", "Apparel",
    "Utilities", "Healthcare", "Entertainment", "Online Retail", "ATM Withdrawal",
    "Professional Services",
]
MERCHANT_WEIGHTS = [0.17, 0.13, 0.10, 0.07, 0.06, 0.09, 0.08, 0.06, 0.07, 0.09, 0.05, 0.03]
TXN_CHANNELS = ["Card Present", "Card Not Present", "Online", "ATM", "UPI"]
TXN_CHANNEL_WEIGHTS = [0.34, 0.20, 0.18, 0.06, 0.22]
DEVICES = ["Mobile", "Desktop", "POS Terminal", "ATM"]
DEVICE_WEIGHTS = [0.44, 0.20, 0.30, 0.06]
HOUR_WEIGHTS = np.array(
    [0.6, 0.35, 0.25, 0.2, 0.2, 0.3, 0.8, 1.6, 2.6, 3.4, 3.8, 3.9,
     3.7, 3.6, 3.5, 3.3, 3.1, 2.9, 2.7, 2.4, 2.0, 1.6, 1.2, 0.8],
    dtype=float,
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
CURRENCIES = ["INR", "USD", "AED", "GBP", "SGD"]
CURRENCY_WEIGHTS = [0.955, 0.017, 0.012, 0.009, 0.007]
CURRENCY_COUNTRY = {"INR": "India", "USD": "United States", "AED": "UAE", "GBP": "United Kingdom", "SGD": "Singapore"}


def build_transactions(offset: int, n: int = 50000, anomaly_rate: float = 0.015) -> pl.DataFrame:
    rng = _rng(offset)

    # timestamps spread over twelve months, with a realistic time-of-day curve
    base_day = np.datetime64("2025-07-01", "D")
    day_off = rng.integers(0, 365, n)
    hour = rng.choice(np.arange(24), size=n, p=HOUR_WEIGHTS)
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)
    ts = base_day + day_off.astype("timedelta64[D]") + hour.astype("timedelta64[h]") + minute.astype("timedelta64[m]") + second.astype("timedelta64[s]")
    ts = np.sort(ts)

    n_accounts = 4000
    account_id = np.array([f"ACC-{i:06d}" for i in rng.integers(1, n_accounts + 1, n)], dtype=object)

    merchant = _choice(rng, MERCHANT_CATEGORIES, np.asarray(MERCHANT_WEIGHTS, dtype=float), n)
    channel = _choice(rng, TXN_CHANNELS, np.asarray(TXN_CHANNEL_WEIGHTS, dtype=float), n)
    device = _choice(rng, DEVICES, np.asarray(DEVICE_WEIGHTS, dtype=float), n)
    currency = _choice(rng, CURRENCIES, np.asarray(CURRENCY_WEIGHTS, dtype=float), n)
    country = np.array([CURRENCY_COUNTRY[c] for c in currency], dtype=object)

    amount = np.round(rng.lognormal(np.log(2450.0), 1.02, n), 2)
    amount = np.where(merchant == "Travel", amount * 2.6, amount)
    amount = np.where(merchant == "Electronics", amount * 1.9, amount)
    amount = np.where(merchant == "ATM Withdrawal", np.round(amount / 500.0) * 500.0, amount)
    amount = np.clip(amount, 20.0, 900000.0)

    txn_count_24h = np.minimum(rng.poisson(2.4, n) + 1, 40).astype(np.int64)
    is_international = (currency != "INR").astype(np.int64)

    # --- inject genuine, *detectable* anomalies -------------------------------
    n_anom = int(round(anomaly_rate * n))
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    kind = rng.integers(0, 3, n_anom)
    truth = np.zeros(n, dtype=np.int64)
    truth[anom_idx] = 1

    a_amt = anom_idx[kind == 0]                       # unusually large amount
    a_hour = anom_idx[kind == 1]                     # odd hour + elevated amount
    a_intl = anom_idx[kind == 2]                     # international + high velocity

    amount[a_amt] = np.round(amount[a_amt] * rng.uniform(8.0, 55.0, a_amt.size), 2)
    amount[a_hour] = np.round(amount[a_hour] * rng.uniform(3.5, 12.0, a_hour.size), 2)
    amount[a_intl] = np.round(amount[a_intl] * rng.uniform(4.0, 18.0, a_intl.size), 2)

    if a_hour.size:
        odd_hours = rng.choice(np.array([0, 1, 2, 3, 4, 5]), size=a_hour.size)
        cur_hour = ((ts[a_hour].astype("datetime64[h]").astype(int) % 24)).astype(int)
        delta_h = (odd_hours - cur_hour).astype("timedelta64[h]")
        ts[a_hour] = ts[a_hour] + delta_h

    if a_intl.size:
        is_international[a_intl] = 1
        currency[a_intl] = rng.choice(["USD", "AED", "GBP", "SGD"], size=a_intl.size)
        country[a_intl] = [CURRENCY_COUNTRY[c] for c in currency[a_intl]]
        txn_count_24h[a_intl] = rng.integers(14, 41, a_intl.size)
        channel[a_intl] = rng.choice(["Card Not Present", "Online"], size=a_intl.size)
        device[a_intl] = "Desktop"

    amount = np.clip(amount, 20.0, 4_000_000.0)

    ts_str = np.datetime_as_string(ts, unit="s").astype(object)
    ts_str = np.char.replace(ts_str.astype(str), "T", " ").astype(object)

    order = np.argsort(ts, kind="stable")
    txn_id = np.array([f"TXN-{i:08d}" for i in range(1, n + 1)], dtype=object)

    return pl.DataFrame(
        [
            _str_series("transaction_id", txn_id),
            _str_series("timestamp", ts_str[order]),
            _str_series("account_id", account_id[order]),
            _str_series("merchant_category", merchant[order]),
            _str_series("channel", channel[order]),
            pl.Series("amount", amount[order], dtype=pl.Float64),
            _str_series("currency", currency[order]),
            _str_series("country", country[order]),
            pl.Series("is_international", is_international[order], dtype=pl.Int64),
            _str_series("device_type", device[order]),
            pl.Series("txn_count_24h", txn_count_24h[order], dtype=pl.Int64),
            pl.Series("_truth_anomaly", truth[order], dtype=pl.Int64),
        ]
    )


# =============================================================================
# dataset registry + writer
# =============================================================================
DATASETS = [
    ("orders_discount_leakage.csv", "discount_leakage", "Flags orders where the discount far exceeds what the context justifies", lambda: build_discount_orders(60_000, offset=1)),
    ("orders_margin_overflow.csv", "margin_overflow", "Flags orders whose realised margin falls below the pricing policy floor", lambda: build_margin_orders(40_000, offset=2)),
    ("sales_demand_series.csv", "demand_forecast", "Forecasts units sold over time with trend + seasonality + promo lift", lambda: build_demand_series(offset=3)),
    ("customers_churn.csv", "churn_risk", "Scores customers likely to stop buying and sizes the revenue at stake", lambda: build_churn(offset=4)),
    ("loan_credit_default.csv", "credit_default", "Predicts probability of loan default from credit bureau signals", lambda: build_credit(offset=5)),
    ("transactions_anomaly.csv", "transaction_anomaly", "Unsupervised: finds card transactions that do not look like the others", lambda: build_transactions(offset=6)),
    ("ml_arena_large_benchmark.csv", "discount_leakage", "500k-row stress test: same leakage structure, proves the engine stays fast", lambda: build_discount_orders(500_000, offset=7)),
]


def _print_table(rows: list[list[str]]) -> None:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "-+-".join("-" * w for w in widths)
    for i, row in enumerate(rows):
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))
        if i == 0:
            print(sep)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ML Arena sample generator - seed {SEED}")
    print(f"Output directory: {OUT_DIR}")
    print()

    summary: list[list[str]] = [["file", "rows", "cols", "size (MB)", "use case", "gen (s)"]]
    failures: list[str] = []

    for filename, usecase, _blurb, builder in DATASETS:
        t0 = time.perf_counter()
        try:
            df = builder()
        except Exception as exc:  # pragma: no cover - surfaced in the summary
            failures.append(f"{filename}: {exc}")
            print(f"  !! {filename} FAILED: {exc}")
            continue
        path = OUT_DIR / filename
        df.write_csv(path)
        gen_s = time.perf_counter() - t0
        size_mb = path.stat().st_size / (1024 * 1024)
        summary.append([filename, f"{df.height:,}", f"{df.width}", f"{size_mb:.1f}", usecase, f"{gen_s:.1f}"])
        print(f"  wrote {filename:<34} {df.height:>9,} rows x {df.width:>2} cols  {size_mb:>7.1f} MB  ({gen_s:.1f}s)")

    print()
    _print_table(summary)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nAll datasets written. Re-running this script reproduces identical files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
