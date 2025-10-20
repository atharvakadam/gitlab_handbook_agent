# filters.py
# Helper functions for: allowlists, safe $vectorSearch.filter building, and simple intent->section suggestions.
from datetime import datetime
import re

ALLOW = None  # populated by load_allowlists()

def load_allowlists(coll):
    """Call this once at service startup and cache globally."""
    global ALLOW
    def distinct_nonempty(field):
        vals = [v for v in coll.distinct(field) if isinstance(v, str) and v.strip()]
        return set(vals)
    ALLOW = {
        "sections": distinct_nonempty("section"),
        "sp1":      distinct_nonempty("section_prefix1"),
        "sp2":      distinct_nonempty("section_prefix2"),
        "access":   set(sum([g if isinstance(g, list) else [g] for g in coll.distinct("access_groups")], [])),
        "sources":  distinct_nonempty("source"),
        "tags":     set(sum([t if isinstance(t, list) else [t] for t in coll.distinct("tags")], [])),
    }
    return ALLOW

def _valid_list(vals, allowed):
    if not vals: return []
    return [v for v in vals if v in allowed]

def parse_iso_dt(s):
    if not s: return None
    s = s.replace("Z","")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

# Simple deterministic suggestion based on query text -> section_prefix1
SECTION_PREFIX1_RULES = [
    (r"\b(leave|parental|maternity|paternity|pto|vacation|sick|absence)\b", ["people-group", "total-rewards"]),
    (r"\b(benefit|insurance|medical|dental|vision|perk)\b", ["total-rewards"]),
    (r"\b(policy|legal|contract|nda|compliance|ethic)\b", ["legal"]),
    (r"\b(okr|goal|roadmap|planning)\b", ["product-development"]),
    (r"\b(sev|incident|on[- ]?call|pagerduty)\b", ["security", "engineering"]),
    (r"\b(hiring|recruit|referral|interview)\b", ["hiring", "people-group"]),
    (r"\b(expense|reimburse|procure|invoice|vendor)\b", ["finance", "business-technology"]),
]

def suggest_sp1(query, max_pick=2):
    q = query.lower()
    hits = []
    for pat, groups in SECTION_PREFIX1_RULES:
        if re.search(pat, q):
            hits.extend(groups)
    # keep only allowed + unique
    if ALLOW and "sp1" in ALLOW:
        hits = [h for h in hits if h in ALLOW["sp1"]]
    seen = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return seen[:max_pick]

def build_vector_filter(
    *,
    user_groups=None,
    sources=None,
    sections_any=None,   # full section strings
    sp1_any=None,        # section_prefix1
    sp2_any=None,        # section_prefix2
    tags_all=None,
    tags_any=None,
    updated_after=None,
    updated_before=None,
):
    """Return a dict suitable for $vectorSearch.filter, or None if no valid constraints."""
    if ALLOW is None:
        raise RuntimeError("filters.load_allowlists(coll) must be called once at startup")
    clauses = []

    # access groups (default to 'all')
    groups = _valid_list(user_groups, ALLOW["access"]) or ["all"]
    clauses.append({"access_groups": {"$in": groups}})

    # sources
    srcs = _valid_list(sources, ALLOW["sources"])
    if srcs:
        clauses.append({"source": {"$in": srcs}})

    # sections
    secs = _valid_list(sections_any, ALLOW["sections"])
    if secs:
        clauses.append({"section": {"$in": secs}})
    sp1  = _valid_list(sp1_any, ALLOW["sp1"])
    if sp1:
        clauses.append({"section_prefix1": {"$in": sp1}})
    sp2  = _valid_list(sp2_any, ALLOW["sp2"])
    if sp2:
        clauses.append({"section_prefix2": {"$in": sp2}})

    # tags
    if tags_all:
        clauses.append({"tags": {"$all": tags_all}})
    if tags_any:
        clauses.append({"tags": {"$in": tags_any}})

    # dates (expect updated_at_dt as Date)
    gte = parse_iso_dt(updated_after)
    lte = parse_iso_dt(updated_before)
    if gte or lte:
        rng = {}
        if gte: rng["$gte"] = gte
        if lte: rng["$lte"] = lte
        clauses.append({"updated_at_dt": rng})

    clauses = [c for c in clauses if c]
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]
