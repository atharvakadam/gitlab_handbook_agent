from langchain_core.tools import tool
from agent.openai_wrapper import OpenAIWrapper
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import os, json, asyncio
from datetime import datetime
from agent.filters import load_allowlists

openai_client = OpenAIWrapper()

load_dotenv()

# --- Optional deps (install: pip install pymongo sentence-transformers) ---
from pymongo import MongoClient
HAS_ST = None  # will be determined lazily

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MONGO_URI not set")

client = MongoClient(mongo_uri)
db = os.getenv("ATLAS_DB", "gitlab_internal_documentation")
collection = os.getenv("ATLAS_COLLECTION", "handbook")
index = os.getenv("ATLAS_INDEX", "handbook_knn")

# ----------------- Helpers -----------------
def _stitch_adjacent(results: List[Dict[str, Any]], max_per_doc: int = 2) -> List[Dict[str, Any]]:
    by_doc = {}
    for r in results:
        by_doc.setdefault(r["doc_id"], []).append(r)
    stitched = []
    for doc_id, items in by_doc.items():
        items.sort(key=lambda x: x["chunk_index"])
        groups = []
        cur = [items[0]]
        for a, b in zip(items, items[1:]):
            if b["chunk_index"] == a["chunk_index"] + 1:
                cur.append(b)
            else:
                groups.append(cur); cur = [b]
        groups.append(cur)
        for g in groups[:max_per_doc]:
            text = "\n".join(x["chunk_text"] for x in g)
            stitched.append({
                "doc_id": doc_id,
                "start_index": g[0]["chunk_index"],
                "end_index": g[-1]["chunk_index"],
                "web_url": g[0]["web_url"],
                "title": g[0]["title"],
                "score": float(sum(x["score"] for x in g)/len(g)),
                "context": text
            })
    return sorted(stitched, key=lambda x: x["score"], reverse=True)

def _parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    # Accept "YYYY-MM-DD" or full ISO
    try:
        if len(dt) == 10:
            return datetime.fromisoformat(dt)  # naive date
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None

def _build_vector_filter(
    *,
    user_groups: Optional[List[str]],
    sources: Optional[List[str]],
    tags_all: Optional[List[str]],
    tags_any: Optional[List[str]],
    updated_after: Optional[str],
    updated_before: Optional[str],
) -> Optional[dict]:
    """Return a Mongo query usable as $vectorSearch.filter, or None."""
    clauses = []

    # Only docs readable by the requester (groups)
    if user_groups:
        clauses.append({"access_groups": {"$in": user_groups}})

    # Restrict to specific sources
    if sources:
        clauses.append({"source": {"$in": sources}})

    # Tags constraints
    if tags_all:   # doc must contain ALL
        clauses.append({"tags": {"$all": tags_all}})
    if tags_any:   # doc must contain ANY
        clauses.append({"tags": {"$in": tags_any}})

    # Updated_at bounds (requires ISODate in Mongo)
    gte = _parse_iso(updated_after)
    lte = _parse_iso(updated_before)
    if gte or lte:
        rng = {}
        if gte:
            rng["$gte"] = gte
        if lte:
            rng["$lte"] = lte
        clauses.append({"updated_at": rng})

    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]

# # filters.py
# # Helper functions for: allowlists, safe $vectorSearch.filter building, and simple intent->section suggestions.
# from datetime import datetime
# import re

# ALLOW = None  # populated by load_allowlists()

# def load_allowlists():
#     """Call this once at service startup and cache globally."""
#     global ALLOW
#     coll = client.db.collection
#     def distinct_nonempty(field):
#         vals = [v for v in coll.distinct(field) if isinstance(v, str) and v.strip()]
#         return set(vals)
#     ALLOW = {
#         "sections": distinct_nonempty("section"),
#         "sp1":      distinct_nonempty("section_prefix1"),
#         "sp2":      distinct_nonempty("section_prefix2"),
#         "access":   set(sum([g if isinstance(g, list) else [g] for g in coll.distinct("access_groups")], [])),
#         "sources":  distinct_nonempty("source"),
#         "tags":     set(sum([t if isinstance(t, list) else [t] for t in coll.distinct("tags")], [])),
#     }
#     return ALLOW

# def _valid_list(vals, allowed):
#     if not vals: return []
#     return [v for v in vals if v in allowed]

# def parse_iso_dt(s):
#     if not s: return None
#     s = s.replace("Z","")
#     try:
#         return datetime.fromisoformat(s)
#     except Exception:
#         return None

# # Simple deterministic suggestion based on query text -> section_prefix1
# SECTION_PREFIX1_RULES = [
#     (r"\b(leave|parental|maternity|paternity|pto|vacation|sick|absence)\b", ["people-group", "total-rewards"]),
#     (r"\b(benefit|insurance|medical|dental|vision|perk)\b", ["total-rewards"]),
#     (r"\b(policy|legal|contract|nda|compliance|ethic)\b", ["legal"]),
#     (r"\b(okr|goal|roadmap|planning)\b", ["product-development"]),
#     (r"\b(sev|incident|on[- ]?call|pagerduty)\b", ["security", "engineering"]),
#     (r"\b(hiring|recruit|referral|interview)\b", ["hiring", "people-group"]),
#     (r"\b(expense|reimburse|procure|invoice|vendor)\b", ["finance", "business-technology"]),
# ]

# def suggest_sp1(query, max_pick=2):
#     q = query.lower()
#     hits = []
#     for pat, groups in SECTION_PREFIX1_RULES:
#         if re.search(pat, q):
#             hits.extend(groups)
#     # keep only allowed + unique
#     if ALLOW and "sp1" in ALLOW:
#         hits = [h for h in hits if h in ALLOW["sp1"]]
#     seen = []
#     for h in hits:
#         if h not in seen:
#             seen.append(h)
#     return seen[:max_pick]

# def build_vector_filter(
#     *,
#     user_groups=None,
#     sources=None,
#     sections_any=None,   # full section strings
#     sp1_any=None,        # section_prefix1
#     sp2_any=None,        # section_prefix2
#     tags_all=None,
#     tags_any=None,
#     updated_after=None,
#     updated_before=None,
# ):
#     """Return a dict suitable for $vectorSearch.filter, or None if no valid constraints."""
#     if ALLOW is None:
#         raise RuntimeError("filters.load_allowlists(coll) must be called once at startup")
#     clauses = []

#     # access groups (default to 'all')
#     groups = _valid_list(user_groups, ALLOW["access"]) or ["all"]
#     clauses.append({"access_groups": {"$in": groups}})

#     # sources
#     srcs = _valid_list(sources, ALLOW["sources"])
#     if srcs:
#         clauses.append({"source": {"$in": srcs}})

#     # sections
#     secs = _valid_list(sections_any, ALLOW["sections"])
#     if secs:
#         clauses.append({"section": {"$in": secs}})
#     sp1  = _valid_list(sp1_any, ALLOW["sp1"])
#     if sp1:
#         clauses.append({"section_prefix1": {"$in": sp1}})
#     sp2  = _valid_list(sp2_any, ALLOW["sp2"])
#     if sp2:
#         clauses.append({"section_prefix2": {"$in": sp2}})

#     # tags
#     if tags_all:
#         clauses.append({"tags": {"$all": tags_all}})
#     if tags_any:
#         clauses.append({"tags": {"$in": tags_any}})

#     # dates (expect updated_at_dt as Date)
#     gte = parse_iso_dt(updated_after)
#     lte = parse_iso_dt(updated_before)
#     if gte or lte:
#         rng = {}
#         if gte: rng["$gte"] = gte
#         if lte: rng["$lte"] = lte
#         clauses.append({"updated_at_dt": rng})

#     clauses = [c for c in clauses if c]
#     if not clauses:
#         return None
#     return {"$and": clauses} if len(clauses) > 1 else clauses[0]


# ALLOW = load_allowlists()
# print("[filters] loaded allowlists:",
#       {k: (len(v) if hasattr(v, "__len__") else "ok") for k,v in ALLOW.items()})

# ----------------- Embedding -----------------
# --- SentenceTransformers model cache + warmup flags ---
_ST_MODEL = None
_ST_MODEL_NAME = os.getenv("VECTOR_EMBED_MODEL", "intfloat/e5-small-v2")  # smaller default helps latency
_WARMED = False
_WARM_TASK = None

async def _get_st_model(model_name: str = None):
    """
    Lazily load and cache the SentenceTransformer model in a worker thread
    to avoid blocking and to ensure import happens post-fork.
    """
    global _ST_MODEL, HAS_ST
    name = model_name or _ST_MODEL_NAME

    if HAS_ST is False:
        raise ImportError("sentence_transformers not available")

    if _ST_MODEL is None:
        def _load():
            try:
                from sentence_transformers import SentenceTransformer  # import here, post-fork
            except Exception as e:
                # Mark unavailable so we fall back to OpenAI embeddings
                import traceback; traceback.print_exc()
                raise
            return SentenceTransformer(name, device="cpu")
        _ST_MODEL = await asyncio.to_thread(_load)
        HAS_ST = True
    return _ST_MODEL

async def _embed_query(query: str) -> List[float]:
    """
    VECTOR_EMBED_PROVIDER=openai|sentence-transformers
    VECTOR_EMBED_MODEL=intfloat/e5-small-v2 OR text-embedding-3-small
    """
    provider = os.getenv("VECTOR_EMBED_PROVIDER", "").lower()
    model = os.getenv("VECTOR_EMBED_MODEL", _ST_MODEL_NAME)
    print(f"[embed] provider={provider or 'sentence-transformers'} model={model}")

    # 1) OpenAI path (fastest cloud option)
    if provider == "openai":
        oai_model = model if "text-embedding" in model else "text-embedding-3-small"
        return await openai_client.generate_embedding(query, model=oai_model)

    # 2) SentenceTransformers path (local)
    try:
        bi = await _get_st_model(model)
        def _encode():
            return bi.encode(
                f"query: {query}",
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16
            ).tolist()
        return await asyncio.to_thread(_encode)
    except Exception as e:
        # Fallback to OpenAI if ST unavailable
        print(f"[embed] ST failed ({e}); falling back to OpenAI embeddings")
        return await openai_client.generate_embedding(query, model="text-embedding-3-small")

# ---------- Warmup (runs once, non-blocking) ----------
async def _warm_embeddings():
    """
    Preload the ST model and run one dummy encode so the first request
    doesn't pay load/JIT. Skips if provider=openai.
    """
    global _WARMED, HAS_ST
    if _WARMED:
        return
    if os.getenv("VECTOR_EMBED_PROVIDER", "").lower() == "openai":
        return
    try:
        bi = await _get_st_model(_ST_MODEL_NAME)   # imports ST post-fork
        def _encode():
            return bi.encode(
                "query: warmup",
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16
            )
        await asyncio.to_thread(_encode)
        HAS_ST = True
        _WARMED = True
        print("[warmup] sentence-transformers ready")
    except Exception as e:
        HAS_ST = False
        print(f"[warmup] skipped: {e}")

def _schedule_warm():
    """
    Schedule warmup after the event loop exists (post-fork).
    """
    if os.getenv("VECTOR_EMBED_PROVIDER", "").lower() == "openai":
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_warm_embeddings())
    except RuntimeError:
        import threading
        threading.Thread(target=lambda: asyncio.run(_warm_embeddings()), daemon=True).start()

_schedule_warm()


import time

# ----------------- Tool: vector retrieval -----------------
@tool("generate_retrieval_response")
async def generate_retrieval_response(user_query: str,
                                      k: int = 15,
                                      top: int = 6,
                                      max_per_doc: int = 2,
                                      # NEW: optional prefilters
                                      user_groups: Optional[List[str]] = None,     # e.g., ["all", "eng", "sales"]
                                      sources: Optional[List[str]] = None,         # e.g., ["gitlab-handbook"]
                                      tags_all: Optional[List[str]] = None,        # require ALL tags
                                      tags_any: Optional[List[str]] = None,        # require ANY tag
                                      updated_after: Optional[str] = None,         # ISO8601 "2025-01-01T00:00:00Z"
                                      updated_before: Optional[str] = None,        # ISO8601
                                    ):
    """
    Retrieve relevant passages from MongoDB Atlas Vector Search for a user's info-seeking query.
    Returns a JSON string with:
    {
      "stitched": [
        {"title": str, "web_url": str, "doc_id": str, "start_index": int, "end_index": int, "score": float, "context": str},
        ...
      ],
      "total_hits": int
    }
    """
    t0 = time.perf_counter()
    print(f"[vector] query => {user_query}")
    print("[vector] args:", json.dumps({
        "user_groups": user_groups,
        "sources": sources,
        "tags_all": tags_all,
        "tags_any": tags_any,
        "updated_after": updated_after,
        "updated_before": updated_before,
        }, default=str))

    # 1) Embed
    try:
        t_embed0 = time.perf_counter()
        qvec = await _embed_query(user_query)
        t_embed1 = time.perf_counter()
        print(f"[vector] embedding time: {t_embed1 - t_embed0:.2f}s")
    except Exception as e:
        return json.dumps({"error": f"embedding_failed: {e!s}"})

    # 2) Query Atlas (offload blocking PyMongo to a worker thread)
    try:
        coll = client[db][collection]

        # Build prefilter for Atlas Vector Search
        avs_filter = _build_vector_filter(
            user_groups=user_groups,
            sources=sources,
            tags_all=tags_all,
            tags_any=tags_any,
            updated_after=updated_after,
            updated_before=updated_before,
        )
        if avs_filter:
            print(f"[vector] filter: {json.dumps(avs_filter, default=str)}")

        vs = {
            "index": index,
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": max(int(k) * 3, 120),
            "limit": int(k),
        }
        if avs_filter:
            vs["filter"] = avs_filter

        # >>> add this:
        print("[vector] $vectorSearch:", json.dumps({k: (v if k != "queryVector" else "[...vector...]") for k, v in vs.items()}, default=str))

        pipeline = [
            {"$vectorSearch": vs},
            {"$project": {
                "_id": 0,
                "doc_id": 1,
                "chunk_index": 1,
                "title": 1,
                "web_url": 1,
                "chunk_text": 1,
                "score": {"$meta": "vectorSearchScore"},
                "source": 1,
                "tags": 1,
                "updated_at": 1,
                "access_groups": 1,
            }},
        ]

        def _run_agg():
            return list(coll.aggregate(pipeline))

        t_db0 = time.perf_counter()
        hits = await asyncio.to_thread(_run_agg)
        t_db1 = time.perf_counter()
        print(f"[vector] db query time: {t_db1 - t_db0:.2f}s")

    except Exception as e:
        return json.dumps({"error": f"mongo_query_failed: {e!s}"})

    if not hits:
        return json.dumps({"stitched": [], "total_hits": 0})

    t_stitch0 = time.perf_counter()
    stitched = _stitch_adjacent(hits, max_per_doc=int(max_per_doc))
    t_stitch1 = time.perf_counter()
    print(f"[vector] stitching time: {t_stitch1 - t_stitch0:.2f}s")
    stitched = stitched[: int(top)]

    # Build a flat sources list that the LLM can easily cite
    sources = [
        {
            "i": i + 1,
            "title": s.get("title"),
            "url": s.get("web_url"),
            "doc_id": s.get("doc_id"),
            "span": f"{s.get('start_index')}–{s.get('end_index')}",
            "score": s.get("score"),
        }
        for i, s in enumerate(stitched)
    ]

    payload = {
        "stitched": stitched,   # full text blocks (for synthesis)
        "sources": sources,     # lightweight linkable list
        "total_hits": len(hits),
        # NEW: timing breakdown (milliseconds)
        "timings_ms": {
            "embed": round((t_embed1 - t_embed0)*1000),
            "db":     round((t_db1 - t_db0)*1000),
            "stitch": round((t_stitch1 - t_stitch0)*1000),
            "total_tool": round((t_stitch1 - t0)*1000),
        },
        "debug": {
            "index": index,
            "k": int(k),
            "numCandidates": max(int(k) * 3, 120),
            "filter": avs_filter,          # will be null if none
            "provider": os.getenv("VECTOR_EMBED_PROVIDER",""),
            "model": os.getenv("VECTOR_EMBED_MODEL", _ST_MODEL_NAME),
        }
    }
    return json.dumps(payload)

@tool("flag_harmful_content")
async def moderate_text(text: str) -> str:
    """Flags harmful text to avoid interacting and responding to harmful inputs and questions"""
    print("Moderating text...")
    flagged = await openai_client.moderate_text(text)
    if flagged:
        print("Flagged !!!")
        return "Message flagged as potentially harmful"

tools = [
    generate_retrieval_response,
    moderate_text,
]
