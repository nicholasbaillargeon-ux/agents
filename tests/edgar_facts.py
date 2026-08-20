"""Builders for EDGAR `companyfacts` payloads.

Real NVDA numbers, because the arithmetic is the point: a fiscal-Q4 hole is
only interesting if the two possible answers are far apart, and here they are
253.49B (right) versus 229.42B (what a gap-jumping scan returns).
"""

from __future__ import annotations

B = 1_000_000_000

# NVDA, as EDGAR actually holds it: quarterly facts for Q1-Q3, a full year, and
# year-to-date figures — but no standalone Q4, which only exists as FY - 9M.
NVDA_Q1_FY27 = 81.61 * B      # 2026-01-26 .. 2026-04-26
NVDA_FY26 = 215.94 * B        # 2025-01-27 .. 2026-01-25
NVDA_9M_FY26 = 147.81 * B     # 2025-01-27 .. 2025-10-26
NVDA_Q3_FY26 = 57.01 * B      # 2025-07-28 .. 2025-10-26
NVDA_6M_FY26 = 90.81 * B      # 2025-01-27 .. 2025-07-27
NVDA_Q2_FY26 = 46.74 * B      # 2025-04-28 .. 2025-07-27
NVDA_Q1_FY26 = 44.06 * B      # 2025-01-27 .. 2025-04-27

NVDA_DERIVED_Q4_FY26 = NVDA_FY26 - NVDA_9M_FY26
NVDA_CORRECT_TTM = NVDA_Q1_FY27 + NVDA_DERIVED_Q4_FY26 + NVDA_Q3_FY26 + NVDA_Q2_FY26
NVDA_GAP_JUMPING_TTM = NVDA_Q1_FY27 + NVDA_Q3_FY26 + NVDA_Q2_FY26 + NVDA_Q1_FY26

NVDA_PERIODS = [
    ("2026-01-26", "2026-04-26", NVDA_Q1_FY27),
    ("2025-01-27", "2026-01-25", NVDA_FY26),
    ("2025-01-27", "2025-10-26", NVDA_9M_FY26),
    ("2025-07-28", "2025-10-26", NVDA_Q3_FY26),
    ("2025-01-27", "2025-07-27", NVDA_6M_FY26),
    ("2025-04-28", "2025-07-27", NVDA_Q2_FY26),
    ("2025-01-27", "2025-04-27", NVDA_Q1_FY26),
]

# The abandoned tag: real facts, but none of them newer than FY2022.
STALE_PERIODS = [
    ("2021-11-01", "2022-01-30", 7.64 * B),
    ("2021-08-02", "2021-10-31", 7.10 * B),
    ("2021-05-03", "2021-08-01", 6.51 * B),
    ("2021-02-01", "2021-05-02", 5.66 * B),
]


def durations(periods, *, form="10-Q", filed="2026-05-20") -> list[dict]:
    return [{"start": s, "end": e, "val": v, "form": form, "filed": filed,
             "fy": 2026, "fp": "Q1"} for s, e, v in periods]


def instants(pairs, *, form="10-Q") -> list[dict]:
    return [{"end": e, "val": v, "form": form, "filed": e} for e, v in pairs]


def facts(**tags) -> dict:
    """facts(Revenues=[...], dei__EntityCommonStockSharesOutstanding=[...])"""
    out: dict = {"us-gaap": {}, "dei": {}}
    for tag, points in tags.items():
        ns, _, name = tag.partition("__")
        if not name:
            ns, name = "us-gaap", ns
        unit = "USD/shares" if "PerShare" in name else ("shares" if "Shares" in name else "USD")
        out[ns][name] = {"units": {unit: points}}
    return out


def nvda_facts(*, with_stale_tag: bool = True, shares: float = 24.2e9) -> dict:
    tags = {
        "Revenues": durations(NVDA_PERIODS),
        "NetIncomeLoss": durations([(s, e, v * 0.63) for s, e, v in NVDA_PERIODS]),
        "EarningsPerShareDiluted": durations([(s, e, round(v / shares, 2))
                                              for s, e, v in NVDA_PERIODS]),
        "StockholdersEquity": instants([("2026-04-26", 100.0 * B)]),
        "Assets": instants([("2026-04-26", 180.0 * B)]),
        "dei__EntityCommonStockSharesOutstanding": instants([("2026-05-20", shares)]),
    }
    if with_stale_tag:
        tags["RevenueFromContractWithCustomerExcludingAssessedTax"] = durations(STALE_PERIODS)
    return facts(**tags)
