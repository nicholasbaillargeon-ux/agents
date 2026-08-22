"""Job boards: Greenhouse, Lever, Ashby, SmartRecruiters, Workday, Oracle
Recruiting and Eightfold.

The board slugs below were each verified to return postings on 2026-08-20. They
rot — firms move ATS vendors — so `Boards.fetch_all` reports which sources
answered and which came back empty, and an empty board is a logged event rather
than a silently shorter result list.

That reporting earned its keep on day one: Optiver had moved from the
`optiver` board to `optiverus`, and Plaid had left Lever for Ashby entirely.
Both showed up as "no postings returned" in the brief rather than as twenty
boards quietly becoming eighteen. `tests/test_boards_live.py` re-checks every
slug against the real APIs (`AGENTS_LIVE=1 pytest -m live`) so the next
migration is a failing test rather than a silent gap.

**Why seven vendors and not three.** The original registry was Greenhouse/Lever/
Ashby, which is where prop shops and startups post — and almost nowhere else.
Banks, brokers, exchanges and enterprise IT firms run on Workday, Oracle
Recruiting, Eightfold or an iCIMS/Phenom front end, so a registry limited to the
first three cannot see Cantor Fitzgerald, Nasdaq or Capital One at all. Each
vendor here is a public JSON feed, no scraping and no key.

**Boards deliberately absent.** AHEAD, WWT, Insight, ePlus, Slalom, EPAM,
Goldman Sachs, Morgan Stanley, BlackRock, Citadel, Two Sigma, DE Shaw, SIG, DRW,
Bridgewater, Jefferies, CME, ICE, Cboe, Tradeweb and MarketAxess were each
probed against all seven vendors here and none exposes a public JSON board —
they run iCIMS, Phenom, JazzHR or bespoke career sites that render server-side.
Close peers cover the same ground instead: CDW and Trace3 for AHEAD, Nasdaq and
LSEG for the exchanges, Point72 and Tower Research for the funds. Reaching the
rest means an HTML adapter, which is scraping and a different maintenance
promise from "the firm publishes this feed for exactly this purpose".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..netcache import Fetcher

log = logging.getLogger(__name__)

GREENHOUSE = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER = "https://api.lever.co/v0/postings/{slug}?mode=json"
ASHBY = "https://api.ashbyhq.com/posting-api/job-board/{slug}"
SMARTRECRUITERS = "https://api.smartrecruiters.com/v1/companies/{slug}/postings"
SMARTRECRUITERS_JOB = "https://jobs.smartrecruiters.com/{slug}/{job_id}"
WORKDAY = "https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
WORKDAY_JOB = "https://{tenant}.{wd}.myworkdayjobs.com/en-US/{site}{path}"
ORACLE = ("https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
          "?onlyData=true&expand=requisitionList"
          "&finder=findReqs;siteNumber={site},limit={limit},sortBy=POSTING_DATES_DESC")
ORACLE_JOB = "https://{host}/hcmUI/CandidateExperience/en/sites/{site}/job/{job_id}"
EIGHTFOLD = ("https://{sub}.eightfold.ai/api/apply/v2/jobs"
             "?domain={domain}&start={start}&num={num}&sort_by=timestamp")

VENDORS = ("greenhouse", "lever", "ashby", "smartrecruiters", "workday", "oracle",
           "eightfold")

# How each vendor is named to a reader. The keys are the registry's, so a
# vendor added without a label still renders, just less prettily.
VENDOR_LABELS = {
    "greenhouse": "Greenhouse", "lever": "Lever", "ashby": "Ashby",
    "smartrecruiters": "SmartRecruiters", "workday": "Workday",
    "oracle": "Oracle Recruiting", "eightfold": "Eightfold",
}

# Sectors the registry is allowed to use. Not decoration: the coverage table
# groups by these, and a typo would quietly create a one-firm sector.
SECTORS = ("quant", "fintech", "ai", "bank", "broker", "exchange", "asset",
           "enterprise")

# (vendor, slug, display name, sector)
#
# The `slug` column is vendor-shaped, because these vendors do not all key a
# board by one opaque word:
#   greenhouse/lever/ashby/smartrecruiters   the board slug
#   workday                                  "tenant/wdN/SiteName"
#   oracle                                   "host/siteNumber"
#   eightfold                                "subdomain/domain"
# Keeping it one string keeps the registry a flat 4-tuple, which is what the
# coverage table and the live slug test both iterate.
REGISTRY: tuple[tuple[str, str, str, str], ...] = (
    # -- prop trading / quant funds ------------------------------------------
    ("greenhouse", "janestreet", "Jane Street", "quant"),
    ("greenhouse", "jumptrading", "Jump Trading", "quant"),
    ("greenhouse", "imc", "IMC Trading", "quant"),
    ("greenhouse", "optiverus", "Optiver", "quant"),   # moved from "optiver" 2026-08
    ("greenhouse", "akunacapital", "Akuna Capital", "quant"),
    ("greenhouse", "oldmissioncapital", "Old Mission Capital", "quant"),
    ("greenhouse", "virtu", "Virtu Financial", "quant"),
    ("greenhouse", "flowtraders", "Flow Traders", "quant"),
    ("greenhouse", "squarepointcapital", "Squarepoint Capital", "quant"),
    ("greenhouse", "point72", "Point72", "quant"),
    ("greenhouse", "towerresearchcapital", "Tower Research Capital", "quant"),
    ("greenhouse", "schonfeld", "Schonfeld", "quant"),
    ("greenhouse", "gsacapital", "GSA Capital", "quant"),
    ("greenhouse", "engineersgate", "Engineers Gate", "quant"),
    ("greenhouse", "exoduspoint", "ExodusPoint", "quant"),
    # A talent-community board rather than a full one, but it is the only
    # public JSON HRT publishes and it does carry the campus FPGA roles.
    ("greenhouse", "hrttalentcommunity", "Hudson River Trading", "quant"),
    ("ashby", "voleon", "Voleon", "quant"),
    ("lever", "belvederetrading", "Belvedere Trading", "quant"),
    ("eightfold", "mlp/mlp.com", "Millennium", "quant"),

    # -- banks, brokers and advisory -----------------------------------------
    # Cantor Fitzgerald and BGC share one Oracle site; Newmark posts there too.
    ("oracle", "hdow.fa.us6.oraclecloud.com/CX_1003",
     "Cantor Fitzgerald / BGC", "broker"),
    ("oracle", "icbpjb.fa.ocs.oraclecloud.com/LazardProfessionalCareer",
     "Lazard", "broker"),
    ("workday", "moelis/wd1/Experienced-Hires", "Moelis", "broker"),
    ("workday", "pjtpartners/wd1/Careers", "PJT Partners", "broker"),
    ("workday", "raymondjames/wd1/RaymondjamesCareers", "Raymond James", "broker"),
    ("workday", "guggenheim/wd1/Guggenheim_Careers", "Guggenheim Partners", "broker"),
    ("oracle", "egug.fa.us2.oraclecloud.com/CX_1", "American Express", "bank"),
    ("workday", "capitalone/wd12/Capital_One", "Capital One", "bank"),

    # -- exchanges and market infrastructure ---------------------------------
    ("workday", "nasdaq/wd1/Global_External_Site", "Nasdaq", "exchange"),
    ("workday", "lseg/wd3/Careers", "LSEG", "exchange"),

    # -- asset managers -------------------------------------------------------
    ("greenhouse", "aqr", "AQR Capital Management", "asset"),
    ("greenhouse", "mangroup", "Man Group", "asset"),

    # -- fintech --------------------------------------------------------------
    ("greenhouse", "stripe", "Stripe", "fintech"),
    ("greenhouse", "robinhood", "Robinhood", "fintech"),
    ("greenhouse", "coinbase", "Coinbase", "fintech"),
    ("greenhouse", "gemini", "Gemini", "fintech"),
    ("greenhouse", "mercury", "Mercury", "fintech"),
    ("greenhouse", "block", "Block", "fintech"),
    ("greenhouse", "affirm", "Affirm", "fintech"),
    ("greenhouse", "brex", "Brex", "fintech"),
    ("greenhouse", "chime", "Chime", "fintech"),
    ("greenhouse", "sofi", "SoFi", "fintech"),
    ("greenhouse", "betterment", "Betterment", "fintech"),
    ("greenhouse", "carta", "Carta", "fintech"),
    ("greenhouse", "ripple", "Ripple", "fintech"),
    ("greenhouse", "fireblocks", "Fireblocks", "fintech"),
    ("greenhouse", "bitgo", "BitGo", "fintech"),
    ("greenhouse", "alloy", "Alloy", "fintech"),
    ("lever", "wealthfront", "Wealthfront", "fintech"),
    ("ashby", "ramp", "Ramp", "fintech"),
    ("ashby", "plaid", "Plaid", "fintech"),           # left Lever for Ashby 2026-08
    ("ashby", "paxos", "Paxos", "fintech"),
    ("ashby", "moderntreasury", "Modern Treasury", "fintech"),
    ("smartrecruiters", "Wise", "Wise", "fintech"),

    # -- AI -------------------------------------------------------------------
    ("greenhouse", "databricks", "Databricks", "ai"),
    ("greenhouse", "scaleai", "Scale AI", "ai"),
    ("greenhouse", "figma", "Figma", "ai"),
    ("greenhouse", "anthropic", "Anthropic", "ai"),
    ("greenhouse", "togetherai", "Together AI", "ai"),
    ("ashby", "openai", "OpenAI", "ai"),
    ("ashby", "cohere", "Cohere", "ai"),
    ("ashby", "cerebras", "Cerebras", "ai"),
    ("ashby", "perplexity", "Perplexity", "ai"),
    ("workday", "nvidia/wd5/NVIDIAExternalCareerSite", "NVIDIA", "ai"),

    # -- enterprise IT / infrastructure (the AHEAD-adjacent segment) ----------
    # AHEAD itself has no public JSON board — see the module docstring. These
    # are its direct competitors and the platform vendors it builds on.
    ("workday", "cdw/wd5/Careers", "CDW", "enterprise"),
    ("greenhouse", "trace3", "Trace3", "enterprise"),
    ("greenhouse", "thoughtworks", "Thoughtworks", "enterprise"),
    ("workday", "kyndryl/wd5/KyndrylEarlyCareers", "Kyndryl (early careers)", "enterprise"),
    ("workday", "optiv/wd5/Optiv_Careers", "Optiv", "enterprise"),
    ("workday", "crowdstrike/wd5/CrowdstrikeCareers", "CrowdStrike", "enterprise"),
    ("workday", "paloaltonetworks/wd5/panwexternalcareers", "Palo Alto Networks", "enterprise"),
    ("greenhouse", "purestorage", "Pure Storage", "enterprise"),
    ("greenhouse", "rubrik", "Rubrik", "enterprise"),
    ("greenhouse", "zscaler", "Zscaler", "enterprise"),
    ("greenhouse", "okta", "Okta", "enterprise"),
    ("greenhouse", "datadog", "Datadog", "enterprise"),
    ("greenhouse", "cloudflare", "Cloudflare", "enterprise"),
    ("greenhouse", "mongodb", "MongoDB", "enterprise"),
    ("greenhouse", "elastic", "Elastic", "enterprise"),
    ("greenhouse", "fivetran", "Fivetran", "enterprise"),
    ("greenhouse", "grafanalabs", "Grafana Labs", "enterprise"),
    ("greenhouse", "cockroachlabs", "Cockroach Labs", "enterprise"),
    ("greenhouse", "vercel", "Vercel", "enterprise"),
    ("ashby", "snowflake", "Snowflake", "enterprise"),
    ("ashby", "confluent", "Confluent", "enterprise"),
    ("ashby", "redis", "Redis", "enterprise"),
)

# Banks and brokers do not say "intern". They say "2027 Summer Analyst",
# "Global Markets Analyst Program", "off-cycle placement" — which is why the
# original quant/startup-shaped pattern saw nothing at Cantor Fitzgerald or
# Lazard. `analyst` alone stays out: every firm here has hundreds of senior
# analysts, and matching it would turn the internship scout into a job board.
INTERN_PATTERNS = re.compile(
    r"\b(intern|internship|co-?op|new\s?grad(?:uate)?|campus|university|student|"
    r"summer\s?20\d\d|placement|apprentice|trainee|early\s?career|"
    r"(?:summer|spring|winter|off-?cycle|sophomore|freshman|penultimate)"
    r"\s+(?:analyst|program(?:me)?|insight|associate)|"
    r"(?:analyst|associate|developer|engineer|technology)\s+(?:program(?:me)?|academy)|"
    r"graduate\s+(?:analyst|program(?:me)?|scheme|role|position)|"
    r"rotational\s+program(?:me)?)\b", re.I)

# Words that mean "this is the kind of work the user is aiming at".
RELEVANT = {
    "quantitative": 3, "quant": 3, "trading": 3, "trader": 3, "research": 2,
    "software": 2, "engineer": 2, "engineering": 2, "developer": 2,
    "machine learning": 3, "ml": 2, "ai": 2, "data": 1, "python": 2,
    "c++": 2, "infrastructure": 1, "platform": 1, "backend": 1, "systems": 1,
    # Added with the banks, brokers and enterprise-IT boards: the same work
    # goes by different names on those job families.
    "technology": 2, "technologist": 2, "cloud": 2, "devops": 2, "sre": 2,
    "site reliability": 2, "analytics": 2, "algorithm": 2, "algorithmic": 2,
    "low latency": 3, "fpga": 2, "distributed": 2, "database": 1, "sql": 1,
    "java": 2, "golang": 2, "rust": 2, "risk": 1, "electronic trading": 3,
    "market data": 2, "automation": 1, "security engineer": 2, "networking": 1,
    # The AI half of "AI and finance", spelled the way postings spell it. "ai"
    # and "ml" above catch the acronyms; these catch the roles that never use
    # them, and an LLM or inference-infrastructure internship is the single
    # closest match on these boards to what the candidate is aiming at.
    "deep learning": 3, "llm": 3, "generative": 2, "neural": 2, "inference": 2,
    "computer vision": 2, "nlp": 2, "pytorch": 2, "cuda": 2, "gpu": 2,
    # Computer engineering proper -- the hardware/software line the candidate
    # is on, and where the quant boards keep their most technical internships.
    "computer engineering": 3, "embedded": 2, "firmware": 2, "hardware": 1,
    "verilog": 2, "rtl": 2, "asic": 2, "signal processing": 2, "robotics": 1,
    "compiler": 2, "kernel": 1,
}
IRRELEVANT = {
    "sales": -4, "recruiting": -4, "recruiter": -4, "marketing": -3, "legal": -4,
    "compliance": -2, "hr": -3, "people": -2, "design": -2, "accounting": -3,
    "office": -2, "executive assistant": -4, "customer support": -4, "brand": -3,
    # These arrive with the bank and enterprise-IT boards in volume.
    "audit": -3, "tax": -3, "payroll": -3, "facilities": -3, "procurement": -3,
    "communications": -2, "wealth advisor": -3, "financial advisor": -3,
    "account executive": -4, "account manager": -3, "customer success": -3,
    "collections": -3, "teller": -4, "branch": -3, "underwriting": -2,
}


def _word_regex(word: str) -> re.Pattern:
    """`word` as a whole-word pattern, tolerating punctuation-tailed tokens.

    A plain substring test scored "Retail Sales Intern" as an AI role ("ai" is
    inside "Retail") and "HTML Developer" as machine learning ("ml"). Two-letter
    tokens are exactly the ones worth the most points, so the false positives
    landed on the highest weights. `\b` cannot follow "c++", hence the
    conditional edges rather than a blanket `\b...\b`.
    """
    left = r"\b" if word[:1].isalnum() else ""
    right = r"\b" if word[-1:].isalnum() else ""
    return re.compile(left + re.escape(word) + right, re.I)


_RELEVANT_RE = {w: _word_regex(w) for w in RELEVANT}
_IRRELEVANT_RE = {w: _word_regex(w) for w in IRRELEVANT}
_METRO_RE = re.compile(
    r"\b(new york|nyc|manhattan|chicago|remote|boston|jersey city|stamford)\b", re.I)


# --- Geography gate -------------------------------------------------------
# Range is the United States, Canada and London. A hard filter, not a score
# penalty: a Singapore desk is not a weaker match than a New York one, it is not
# a match at all, and the last sweep carried 19 Singapore rows the model still
# paid to triage.
#
# Allow is tested before deny, and both are whole-word. That ordering is the
# whole design. It is what makes multi-site postings work -- "London; Amsterdam"
# and "New York, London, or Paris" are in range because one site is -- and it is
# what resolves the city names that exist on both sides of the line:
# "Manchester, NH" and "Birmingham, AL" match their US state and never reach the
# UK entries below.
IN_RANGE = (
    "united states", "united states of america", "usa", "america",
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
    # Cities, because plenty of boards give a city and stop there.
    "nyc", "manhattan", "brooklyn", "jersey city", "hoboken", "stamford",
    "greenwich", "princeton", "chicago", "boston", "cambridge, ma", "somerville",
    "san francisco", "bay area", "silicon valley", "palo alto", "menlo park",
    "mountain view", "sunnyvale", "santa clara", "san mateo",
    "redwood city", "cupertino", "irvine", "los angeles", "san diego",
    "seattle", "bellevue", "redmond", "portland", "austin", "dallas", "frisco",
    "houston", "atlanta", "charlotte", "raleigh", "durham", "miami", "sunrise",
    "tampa", "orlando", "phoenix", "tempe", "denver", "boulder", "salt lake city",
    "las vegas", "minneapolis", "detroit", "ann arbor", "columbus", "cleveland",
    "pittsburgh", "philadelphia", "baltimore", "arlington", "reston", "mclean",
    "nashville", "st. louis", "saint louis", "kansas city", "madison",
    # Canada.
    "canada", "canadian", "toronto", "montreal", "montréal", "vancouver",
    "ottawa", "calgary", "edmonton", "winnipeg", "waterloo", "kitchener",
    "mississauga", "ontario", "quebec", "québec", "british columbia", "alberta",
    "nova scotia", "manitoba", "saskatchewan",
    # The one European city in range, by request.
    "london",
)

# Everything the sweep actually returns from outside the range, plus the obvious
# neighbours. Only reached when nothing in IN_RANGE matched, so the UK cities
# here cost nothing to the US cities that share their names.
OUT_OF_RANGE = (
    "singapore", "hong kong", "china", "shanghai", "beijing", "shenzhen",
    "taiwan", "taipei", "japan", "tokyo", "osaka", "korea", "seoul",
    "india", "mumbai", "bengaluru", "bangalore", "hyderabad", "pune", "chennai",
    "gurgaon", "gurugram", "noida", "new delhi", "sri lanka", "colombo",
    "philippines", "manila", "vietnam", "hanoi", "ho chi minh", "thailand",
    "bangkok", "malaysia", "kuala lumpur", "indonesia", "jakarta",
    "australia", "sydney", "melbourne", "new zealand", "auckland",
    "netherlands", "amsterdam", "hoofddorp", "rotterdam", "the hague",
    "france", "paris", "germany", "berlin", "munich", "frankfurt", "hamburg",
    "ireland", "dublin", "switzerland", "zurich", "zürich", "geneva",
    "spain", "madrid", "barcelona", "italy", "milan", "rome", "portugal",
    "lisbon", "porto", "poland", "warsaw", "krakow", "kraków", "wroclaw",
    "czech", "prague", "hungary", "budapest", "romania", "bucharest",
    "serbia", "belgrade", "bulgaria", "sofia", "greece", "athens",
    "sweden", "stockholm", "norway", "oslo", "denmark", "copenhagen",
    "finland", "helsinki", "austria", "vienna", "belgium", "brussels",
    "luxembourg", "iceland", "reykjavik", "estonia", "tallinn", "lithuania",
    "latvia", "riga", "cyprus", "malta",
    "israel", "tel aviv", "jerusalem", "haifa", "uae", "dubai", "abu dhabi",
    "saudi", "riyadh", "qatar", "doha", "bahrain", "kuwait", "turkey",
    "istanbul", "egypt", "cairo", "morocco", "casablanca",
    "south africa", "johannesburg", "cape town", "nigeria", "lagos",
    "kenya", "nairobi", "ghana", "accra",
    "brazil", "são paulo", "sao paulo", "rio de janeiro", "argentina",
    "buenos aires", "chile", "santiago", "colombia", "bogota", "bogotá",
    "peru", "lima", "mexico", "mexico city", "guadalajara", "monterrey",
    "costa rica", "panama", "uruguay", "montevideo",
    "russia", "moscow", "ukraine", "kyiv", "kiev", "belarus", "kazakhstan",
    # The UK and Ireland beyond London, which the range deliberately excludes.
    "uk", "u.k", "united kingdom", "gb", "scotland", "wales", "northern ireland",
    "edinburgh", "glasgow", "belfast", "cardiff", "bristol",
    # Manchester and Birmingham are deliberately absent: New Hampshire and
    # Alabama have one each, the boards write them the same way, and a bare
    # "Manchester" is genuinely ambiguous. "uk" above catches the English ones.
    "leeds", "liverpool", "sheffield", "oxford", "isle of man",
    "emea", "apac", "latam",
)

_IN_RANGE_RE = re.compile("|".join(_word_regex(w).pattern for w in IN_RANGE), re.I)
_OUT_OF_RANGE_RE = re.compile(
    "|".join(_word_regex(w).pattern for w in OUT_OF_RANGE), re.I)


def location_in_range(title: str, location: str) -> bool:
    """Is this posting inside the US / Canada / London range?

    Reads the title as well as the location field, because on several boards the
    location field does not hold the location. Cloudflare files every posting
    under "In-Office" and puts the city in the title ("Software Engineer Intern
    (Fall 2026) - Austin, TX"); matching the location alone dropped all eleven of
    its internships as unlocatable.

    An unrecognised string is kept, not dropped. Capital One and NVIDIA file
    multi-site postings as "2 Locations" and "8 Locations", which names no place
    at all -- that is a board's formatting, not evidence of an out-of-range
    office, and dropping it would silently lose real US roles. Those rows reach
    the model, whose triage prompt carries the same range rule and can skip them.

    There is no two-letter-code pass. It would have to be case-sensitive to keep
    \bOR\b out of "New York, London, or Paris", and even then half the US state
    codes are the ISO code of an excluded country -- NL, DE, IL, IN, CO, MA, PA,
    PE -- so "Amsterdam, NL" would read as Newfoundland. Codes only ever decide
    rows that no name matched, and those are kept anyway.
    """
    text = f"{title} {location}"
    if _IN_RANGE_RE.search(text):
        return True
    return not _OUT_OF_RANGE_RE.search(text)
# Query parameters that carry no identity. Everything else is kept, because on
# several of these boards the query string IS the identity: Databricks, Stripe,
# Jump Trading, Squarepoint, Gemini and Old Mission all point every posting at
# one generic careers page and distinguish them only by `gh_jid`/`id`/`token`.
# Dropping the whole query collapsed 107 Jump Trading roles into a single key,
# so 106 of them could never appear in a diff. Measured on the live registry:
# gh_jid has 233 distinct values for one firm, `t` has exactly one (`gh_src=`).
TRACKING_PARAMS = {
    "t", "gh_src", "src", "source", "ref", "referrer", "utm_source", "utm_medium",
    "utm_campaign", "utm_term", "utm_content",
}


def posting_key(url: str) -> str:
    """Canonical identity for a posting URL.

    Shared rather than inlined because three things must agree on it — the
    nightly diff, the verdict lookup, and the backlog accounting. Drops the
    fragment and known tracking parameters, keeps identifying ones, and sorts
    what remains so a board reordering its query string is not a new posting.
    """
    if not url:
        return ""
    parts = urlsplit(url)
    kept = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                  if k.lower() not in TRACKING_PARAMS)
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"),
                       urlencode(kept), ""))


# --- how long has it been open ----------------------------------------------
#
# Every vendor dates a posting differently, so each adapter normalises to one
# ISO date in `posted` and the age is computed at render time. Storing the age
# instead would freeze it: a posting first surfaced eight days ago is eight days
# older today, and the digest would still say "1 day".

def _iso_date(value) -> str:
    """First ten characters of an ISO timestamp, if they look like a date."""
    text = str(value or "")[:10]
    return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else ""


def _ms_date(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _epoch_date(seconds) -> str:
    try:
        return datetime.fromtimestamp(int(seconds), timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


_WORKDAY_TODAY = re.compile(r"posted\s+today", re.I)
_WORKDAY_YESTERDAY = re.compile(r"posted\s+yesterday", re.I)
_WORKDAY_AGE = re.compile(r"posted\s+(\d+)(\+)?\s*days?\s+ago", re.I)


def workday_posted(text: str, today: date | None = None) -> tuple[str, bool]:
    """Workday's relative age string -> (ISO posting date, is_a_floor).

    Workday's board API gives "Posted Yesterday" / "Posted 30+ Days Ago" and no
    real date, so the date is reconstructed from the day it was fetched. That is
    worth doing rather than storing the phrase: "30+ days ago" recorded a month
    from now still means the same posting date, whereas the phrase would still
    read "30+" forever. The bucket's floor is kept honest by the flag, which the
    brief renders as "60+" rather than a number it cannot support.
    """
    today = today or datetime.now(timezone.utc).date()
    if _WORKDAY_TODAY.search(text or ""):
        return today.isoformat(), False
    if _WORKDAY_YESTERDAY.search(text or ""):
        return date.fromordinal(today.toordinal() - 1).isoformat(), False
    m = _WORKDAY_AGE.search(text or "")
    if m:
        days = min(int(m.group(1)), 3650)
        return date.fromordinal(max(1, today.toordinal() - days)).isoformat(), bool(m.group(2))
    return "", False


def days_open(posted: str, today: date | None = None) -> int | None:
    """Whole days between the posting date and today, or None if undated."""
    if not posted:
        return None
    try:
        then = date.fromisoformat(posted[:10])
    except ValueError:
        return None
    delta = ((today or datetime.now(timezone.utc).date()) - then).days
    return delta if delta >= 0 else 0


def format_days_open(posted: str, is_floor: bool = False,
                     today: date | None = None) -> str:
    """The 'Days open' cell. An em dash where the board publishes no date —
    a blank column reads as zero, and zero is a claim."""
    n = days_open(posted, today)
    if n is None:
        return "—"
    return f"{n}+" if is_floor else str(n)


@dataclass
class Posting:
    company: str
    title: str
    location: str
    url: str
    sector: str = ""
    posted: str = ""
    posted_is_floor: bool = False
    source: str = ""
    score: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identity for the nightly diff. The URL carries the ATS id;
        titles and locations get edited in place and would false-positive."""
        return posting_key(self.url) or f"{self.company}:{self.title}"

    @property
    def is_internship(self) -> bool:
        return bool(INTERN_PATTERNS.search(self.title))

    @property
    def days_open(self) -> int | None:
        return days_open(self.posted)

    def as_dict(self) -> dict:
        return {"company": self.company, "title": self.title, "location": self.location,
                "url": self.url, "sector": self.sector, "posted": self.posted,
                "posted_is_floor": self.posted_is_floor,
                "source": self.source, "score": self.score}


def score(posting: Posting) -> int:
    """Deterministic relevance score. The LLM re-ranks later; this decides what
    is even worth spending a token on, and works when the LLM is down."""
    text = f"{posting.title} {posting.location}"
    total, reasons = 0, []
    if posting.is_internship:
        total += 6
        reasons.append("internship/new-grad title")
    for word, pts in RELEVANT.items():
        if _RELEVANT_RE[word].search(text):
            total += pts
            reasons.append(f"+{pts} {word}")
    for word, pts in IRRELEVANT.items():
        if _IRRELEVANT_RE[word].search(text):
            total += pts
            reasons.append(f"{pts} {word}")
    if _METRO_RE.search(text):
        total += 1
        reasons.append("+1 target metro")
    posting.score = total
    posting.reasons = reasons
    return total


# Page sizes and stopping rules, per vendor. Workday hands out twenty postings
# a request and NVIDIA alone has two thousand of them, so an exhaustive sweep of
# every board would be a thousand requests a night for postings the title filter
# throws away. Boards under `WORKDAY_FULL_SWEEP` are swept whole; larger ones are
# queried with the early-career terms below, and the coverage table says which
# of the two happened so a truncated board never reads as a complete one.
WORKDAY_PAGE = 20
WORKDAY_FULL_SWEEP = 200
WORKDAY_MAX_PAGES = 10
WORKDAY_QUERIES = ("intern", "new grad", "graduate", "campus")
ORACLE_LIMIT = 200
EIGHTFOLD_PAGE = 100
EIGHTFOLD_MAX_PAGES = 5
SMARTRECRUITERS_PAGE = 100
SMARTRECRUITERS_MAX_PAGES = 5


# One retry per request, and a status that says what went wrong. A sweep of
# eighty boards is a couple of hundred requests over three minutes, so a single
# timeout is now an ordinary event rather than a rare one -- and "no postings
# returned" is the wrong thing to say about it, because that is the status that
# means "this firm has moved ATS vendor, go probe" and costs an afternoon.
BOARD_RETRIES = 1


class Boards:
    def __init__(self, fetcher: Fetcher) -> None:
        self.f = fetcher
        self.source_status: dict[str, str] = {}
        self._failure = ""

    def fetch_all(self, registry=REGISTRY, *, ttl: int = 3600) -> list[Posting]:
        out: list[Posting] = []
        for vendor, slug, company, sector in registry:
            self._failure = ""
            try:
                got, note = self._fetch_one(vendor, slug, company, sector, ttl)
            except Exception as e:  # noqa: BLE001 - one bad board never kills the sweep
                log.warning("board %s/%s blew up: %s", vendor, slug, e)
                self.source_status[company] = f"error: {type(e).__name__}"
                continue
            if got:
                self.source_status[company] = f"{len(got)} postings" + (f" ({note})" if note else "")
            else:
                # "no postings returned" reads as "the slug is dead" and sends
                # you off probing ATS vendors. When the request itself failed,
                # say so: that is a retry tomorrow, not a registry edit.
                self.source_status[company] = self._failure or "no postings returned"
            out.extend(got)
        return out

    def _fetch_one(self, vendor: str, slug: str, company: str, sector: str,
                   ttl: int) -> tuple[list[Posting], str]:
        handler = {
            "greenhouse": self._greenhouse, "lever": self._lever, "ashby": self._ashby,
            "smartrecruiters": self._smartrecruiters, "workday": self._workday,
            "oracle": self._oracle, "eightfold": self._eightfold,
        }[vendor]
        postings, note = handler(slug, company, sector, ttl)
        deduped, seen = [], set()
        for p in postings:
            if not (p.title and p.url) or p.key in seen:
                continue
            seen.add(p.key)
            p.source = vendor
            deduped.append(p)
        return deduped, note

    # -- one-request boards ---------------------------------------------------

    def _json(self, url: str, ttl: int, *, body: dict | None = None):
        for attempt in range(BOARD_RETRIES + 1):
            resp = self.f.fetch(url, ttl=ttl, body=body)
            if resp is None:
                self._failure = "unreachable (network or timeout)"
            elif not resp.ok:
                self._failure = f"HTTP {resp.status}"
            else:
                data = resp.json()
                if data is not None:
                    return data
                self._failure = "non-JSON body"
            if attempt < BOARD_RETRIES:
                log.info("retrying %s after %s", url, self._failure)
        return None

    def _greenhouse(self, slug, company, sector, ttl):
        data = self._json(GREENHOUSE.format(slug=slug), ttl)
        out = []
        for j in (data or {}).get("jobs", []) or []:
            loc = (j.get("location") or {}).get("name", "") or ""
            # `first_published` is when the posting went live; `updated_at`
            # moves every time a recruiter touches the requisition, so using it
            # for "days open" would reset the clock on an eight-month-old role.
            posted = _iso_date(j.get("first_published")) or _iso_date(j.get("updated_at"))
            out.append(Posting(company=company, title=(j.get("title") or "").strip(),
                               location=loc, url=j.get("absolute_url", ""),
                               sector=sector, posted=posted))
        return out, ""

    def _lever(self, slug, company, sector, ttl):
        data = self._json(LEVER.format(slug=slug), ttl)
        out = []
        for j in data or []:
            cats = j.get("categories") or {}
            out.append(Posting(company=company, title=(j.get("text") or "").strip(),
                               location=cats.get("location", "") or "",
                               url=j.get("hostedUrl", ""), sector=sector,
                               posted=_ms_date(j.get("createdAt"))))
        return out, ""

    def _ashby(self, slug, company, sector, ttl):
        data = self._json(ASHBY.format(slug=slug), ttl)
        out = []
        for j in (data or {}).get("jobs", []) or []:
            url = j.get("jobUrl") or j.get("applyUrl") or j.get("externalLink") or ""
            out.append(Posting(company=company, title=(j.get("title") or "").strip(),
                               location=j.get("location", "") or "", url=url,
                               sector=sector, posted=_iso_date(j.get("publishedAt"))))
        return out, ""

    # -- paginated boards -----------------------------------------------------

    def _smartrecruiters(self, slug, company, sector, ttl):
        out, offset = [], 0
        for _ in range(SMARTRECRUITERS_MAX_PAGES):
            url = (f"{SMARTRECRUITERS.format(slug=slug)}"
                   f"?limit={SMARTRECRUITERS_PAGE}&offset={offset}")
            data = self._json(url, ttl)
            page = (data or {}).get("content") or []
            for j in page:
                loc = (j.get("location") or {})
                where = ", ".join(x for x in (loc.get("city"), loc.get("region"),
                                              loc.get("country")) if x)
                out.append(Posting(
                    company=company, title=(j.get("name") or "").strip(),
                    location=where, sector=sector,
                    url=SMARTRECRUITERS_JOB.format(slug=slug, job_id=j.get("id", "")),
                    posted=_iso_date(j.get("releasedDate"))))
            offset += len(page)
            if len(page) < SMARTRECRUITERS_PAGE or offset >= (data or {}).get("totalFound", 0):
                break
        return out, ""

    def _workday(self, slug, company, sector, ttl):
        try:
            tenant, wd, site = slug.split("/")
        except ValueError:
            raise ValueError(f"workday slug must be 'tenant/wdN/Site', got {slug!r}") from None
        url = WORKDAY.format(tenant=tenant, wd=wd, site=site)
        today = datetime.now(timezone.utc).date()

        def page(offset: int, search: str):
            return self._json(url, ttl, body={"appliedFacets": {}, "limit": WORKDAY_PAGE,
                                              "offset": offset, "searchText": search})

        first = page(0, "")
        if first is None:
            return [], ""
        total = int(first.get("total") or 0)
        searches = [""] if total <= WORKDAY_FULL_SWEEP else list(WORKDAY_QUERIES)
        note = "" if total <= WORKDAY_FULL_SWEEP else f"early-career search of {total}"

        out = []
        for search in searches:
            for n in range(WORKDAY_MAX_PAGES):
                data = first if (search == "" and n == 0) else page(n * WORKDAY_PAGE, search)
                jobs = (data or {}).get("jobPostings") or []
                for j in jobs:
                    posted, is_floor = workday_posted(j.get("postedOn") or "", today)
                    out.append(Posting(
                        company=company, title=(j.get("title") or "").strip(),
                        location=j.get("locationsText", "") or "", sector=sector,
                        url=WORKDAY_JOB.format(tenant=tenant, wd=wd, site=site,
                                               path=j.get("externalPath", "")),
                        posted=posted, posted_is_floor=is_floor))
                if len(jobs) < WORKDAY_PAGE:
                    break
        return out, note

    def _oracle(self, slug, company, sector, ttl):
        try:
            host, site = slug.split("/")
        except ValueError:
            raise ValueError(f"oracle slug must be 'host/siteNumber', got {slug!r}") from None
        data = self._json(ORACLE.format(host=host, site=site, limit=ORACLE_LIMIT), ttl)
        items = (data or {}).get("items") or []
        if not items:
            return [], ""
        reqs = items[0].get("requisitionList") or []
        total = int(items[0].get("TotalJobsCount") or len(reqs))
        out = [Posting(company=company, title=(r.get("Title") or "").strip(),
                       location=r.get("PrimaryLocation", "") or "", sector=sector,
                       url=ORACLE_JOB.format(host=host, site=site, job_id=r.get("Id", "")),
                       posted=_iso_date(r.get("PostedDate")))
               for r in reqs]
        return out, (f"newest {len(out)} of {total}" if total > len(out) else "")

    def _eightfold(self, slug, company, sector, ttl):
        try:
            sub, domain = slug.split("/")
        except ValueError:
            raise ValueError(f"eightfold slug must be 'sub/domain', got {slug!r}") from None
        out, start = [], 0
        for _ in range(EIGHTFOLD_MAX_PAGES):
            data = self._json(EIGHTFOLD.format(sub=sub, domain=domain, start=start,
                                               num=EIGHTFOLD_PAGE), ttl)
            page = (data or {}).get("positions") or []
            for j in page:
                out.append(Posting(
                    company=company, title=(j.get("name") or "").strip(),
                    location=j.get("location", "") or "", sector=sector,
                    url=j.get("canonicalPositionUrl", ""),
                    posted=_epoch_date(j.get("t_create"))))
            start += len(page)
            if len(page) < EIGHTFOLD_PAGE or start >= int((data or {}).get("count") or 0):
                break
        return out, ""
