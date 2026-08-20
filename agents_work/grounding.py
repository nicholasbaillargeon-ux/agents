"""Check model prose against the facts it was given.

The research agent hands the model a dossier and tells it to invent no numbers.
Models mostly comply, and "mostly" is the problem: one fabricated margin in an
otherwise accurate brief is worse than a brief that admits it knows nothing,
because nothing on the page marks which sentence to distrust.

So every figure in the generated prose is checked back against the dossier and
the unmatched ones are listed under the section that contains them. This does
not stop the model inventing a number — nothing does — it just refuses to let
the invention pass silently.
"""

from __future__ import annotations

import re

# $1,234.5  ·  12.4%  ·  -3.2  ·  240bp  ·  1.2B
#
# The lookbehind is what stops "Nasdaq-100" being read as negative one hundred
# and "$10,000-to-$66,000" as negative sixty-six thousand. A hyphen only means
# minus at the start of a token, never glued to the end of a word.
_NUMBER = re.compile(
    r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|bps?|[KMBT]\b)?", re.I)
_SCALES = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _parse(token: str) -> tuple[float, int, float] | None:
    """(value, decimals written, unit scale). None if it is not really a number.

    The scale is carried separately because precision is written on the
    mantissa, not the value: "$253.5B" is precise to 0.1 *billion*, so judging
    it against a tolerance of 0.1 would reject the very fact it came from.
    """
    raw = token.strip().lower().replace("$", "").replace(",", "").replace(" ", "")
    scale = 1.0
    if raw.endswith("bps"):
        raw, scale = raw[:-3], 0.01
    elif raw.endswith("bp"):
        raw, scale = raw[:-2], 0.01
    elif raw.endswith("%"):
        raw = raw[:-1]
    elif raw and raw[-1] in _SCALES:
        raw, scale = raw[:-1], _SCALES[raw[-1]]
    if not raw or raw in ("-", "."):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    places = len(raw.split(".")[1]) if "." in raw else 0
    return value * scale, places, scale


def numbers(text: str) -> list[tuple[str, float, int, float]]:
    out = []
    for m in _NUMBER.finditer(text):
        parsed = _parse(m.group())
        if parsed is not None:
            out.append((m.group().strip(), *parsed))
    return out


def _grounded(value: float, places: int, scale: float, facts: list[float]) -> bool:
    """True when some fact rounds to the written figure at its own precision.

    Precision matters: a dossier holding 253,490,000,000 grounds "$253.5B" and
    "253.49B" but not "$260B". Percentages are also checked against their
    fractional twin, since the dossier stores 0.63 for what prose calls 63%.

    Magnitude is compared, not sign. English carries direction in words —
    "sits 0.9% off its 52-week high" is the correct rendering of a -0.9 dossier
    figure, and flagging it fires the warning on almost every brief written.
    """
    # Two tolerances, whichever is wider. The first is the precision the figure
    # was written to: "$253.5B" is precise to a tenth of a billion. The second is
    # a relative band, because prose restates rather than transcribes -- "70%+
    # growth" of a 70.7% figure is a true statement, and "about $250B" of
    # $253.49B is the same number said out loud. The band stays narrow enough to
    # catch what matters: an invented 74.2% against a real 63.0% is ten bands
    # away, and a $500B backlog against $253B revenue is sixty.
    tolerance = max(scale * 10.0 ** (-places) / 2.0, abs(value) * RELATIVE_BAND)
    target = abs(value)
    for fact in facts:
        for multiplier in _UNIT_MULTIPLIERS:
            if abs(abs(fact * multiplier) - target) <= tolerance:
                return True
    return False


# Financial writing changes units mid-sentence: a dossier line reading
# "11,729" under a millions heading is the same fact the model writes as
# "$11,729M", and 0.63 is the same fact as "63%". Matching only the literal
# value flags a dozen correct figures per brief, which trains the reader to
# skip the warning — so scale differences are treated as the same number.
_UNIT_MULTIPLIERS = (1.0, 100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)


# Two figures agreeing this closely are the same figure restated, not two claims.
RELATIVE_BAND = 0.015

_HAS_UNIT = re.compile(r"[$%]|bps?\b|[KMBT]\b", re.I)

# Years, small counts and round percentages are not claims about the company.
_ALWAYS_OK = set(range(0, 13)) | {15, 20, 25, 30, 50, 52, 75, 100}


def ungrounded(prose: str, dossier: str, *, limit: int = 8) -> list[str]:
    """Figures in `prose` that no figure in `dossier` supports."""
    facts = [v for _, v, _, _ in numbers(dossier)]
    if not facts:
        return []
    flagged: list[str] = []
    for token, value, places, scale in numbers(prose):
        if value in _ALWAYS_OK and places == 0:
            continue
        if 1900 <= value <= 2100 and places == 0:  # a year
            continue
        # A bare integer with no unit is an index name, a count or an ordinal —
        # "the Nasdaq-100", "22 stocks at highs", "three segments". A fabricated
        # financial claim essentially always carries a unit: $, %, bp, or a
        # magnitude suffix. Checking bare integers costs far more false alarms
        # than it catches.
        if places == 0 and scale == 1.0 and not _HAS_UNIT.search(token):
            continue
        if _grounded(value, places, scale, facts):
            continue
        if token not in flagged:
            flagged.append(token)
        if len(flagged) >= limit:
            break
    return flagged
