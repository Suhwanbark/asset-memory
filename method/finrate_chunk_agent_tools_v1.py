"""Read-only FIN-RATE tools; no query routing, gold access, or authored memory."""
from __future__ import annotations

import time
import regex
import tiktoken

import finrate_bm25_top10_v1 as bm25

ENC = tiktoken.get_encoding("o200k_base")
READ_TOKENS = 8000
RESULT_TOKENS = 10000


def tokens(text):
    return len(ENC.encode(text, disallowed_special=()))


def schema(properties, required):
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


TOOLS = [
    {"type": "function", "name": "search", "description":
     "BM25 keyword search over all 15,311 original FIN-RATE chunk bodies. "
     "Returns ranked chunk IDs, original titles, scores and exact excerpts. "
     "No automatic company/year filtering. Change the query or paginate as desired.",
     "inputSchema": schema({
         "query": {"type": "string"},
         "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
         "offset": {"type": "integer", "minimum": 0, "default": 0}}, ["query"])},
    {"type": "function", "name": "grep", "description":
     "Find a literal phrase or regular expression in chunk titles and full bodies. "
     "Search all chunks or specified doc_ids. Returns exact matched spans with "
     "surrounding text, field (title/text), character offsets and pagination. "
     "Results follow chunk ID, field and match position, not relevance rank.",
     "inputSchema": schema({
         "pattern": {"type": "string"},
         "doc_ids": {"type": ["array", "null"], "items": {"type": "string"}},
         "regex": {"type": "boolean", "default": False},
         "case_sensitive": {"type": "boolean", "default": False},
         "offset": {"type": "integer", "minimum": 0, "default": 0},
         "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20}}, ["pattern"])},
    {"type": "function", "name": "read", "description":
     "Read original chunk body text at a zero-based character offset. length is "
     "a requested character count (default 24000); each response contains at most "
     "8000 text tokens. next_start allows further reading without source truncation. "
     "A doc ID may come from either search or grep.",
     "inputSchema": schema({
         "doc_id": {"type": "string"},
         "start": {"type": "integer", "minimum": 0, "default": 0},
         "length": {"type": "integer", "minimum": 1, "default": 24000}}, ["doc_id"])},
]


def integer(value, name, low, high=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer in [{low}, {high or 'unbounded'}]")
    return value


def exact_excerpt(doc, field, start, end):
    text = doc.get(field, "")
    start, end = max(0, start), min(len(text), end)
    return {"doc_id": doc["_id"], "field": field, "start": start, "end": end,
            "text": text[start:end]}


class ChunkTools:
    def __init__(self, docs, retriever):
        self.docs = sorted(docs, key=lambda d: d["_id"])
        self.by_id = {d["_id"]: d for d in self.docs}
        self.retriever = retriever

    @classmethod
    def load(cls):
        docs, digest = bm25.load_corpus()
        retriever, manifest = bm25.load_or_build(docs, digest)
        return cls(docs, retriever), digest, manifest

    def search(self, query, top_k=10, offset=0):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        top_k = integer(top_k, "top_k", 1, 50)
        offset = integer(offset, "offset", 0)
        ranking = [r for r in bm25.search(query, self.retriever, self.docs, k=len(self.docs)) if r["score"] > 0]
        terms = list(dict.fromkeys(bm25.tokenize(query)))
        pattern = regex.compile(r"\b(?:" + "|".join(regex.escape(t) for t in terms) + r")\b", regex.I) if terms else None
        results = []
        for row in ranking[offset:offset + top_k]:
            doc = self.by_id[row["doc_id"]]
            excerpts = []
            if pattern:
                for match in pattern.finditer(doc["text"], timeout=1):
                    if excerpts and match.start() < excerpts[-1]["end"]:
                        continue
                    excerpts.append(exact_excerpt(doc, "text", match.start() - 160, match.end() + 240))
                    if len(excerpts) == 2:
                        break
            results.append({**row, "body_chars": len(doc["text"]), "excerpts": excerpts})
        return {"query": query, "scope": "all_original_chunk_bodies", "positive_score_count": len(ranking),
                "offset": offset, "results": results,
                "next_offset": offset + len(results) if offset + len(results) < len(ranking) else None}

    def grep(self, pattern, doc_ids=None, regex=False, case_sensitive=False, offset=0, limit=20):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be nonempty")
        if len(pattern) > 2000:
            raise ValueError("pattern exceeds 2000 characters")
        if not isinstance(regex, bool) or not isinstance(case_sensitive, bool):
            raise ValueError("regex and case_sensitive must be booleans")
        offset = integer(offset, "offset", 0)
        limit = integer(limit, "limit", 1, 50)
        if doc_ids is None:
            docs = self.docs
        else:
            if not isinstance(doc_ids, list) or not all(isinstance(x, str) for x in doc_ids):
                raise ValueError("doc_ids must be a list of original chunk IDs")
            unknown = [x for x in doc_ids if x not in self.by_id]
            if unknown:
                raise ValueError(f"Unknown doc_ids: {unknown[:5]}")
            docs = [self.by_id[x] for x in sorted(set(doc_ids))]
        import regex as regex_module
        compiled = regex_module.compile(pattern if regex else regex_module.escape(pattern),
                                        0 if case_sensitive else regex_module.I)
        matches, seen, started = [], 0, time.monotonic()
        for doc in docs:
            for field in ("title", "text"):
                if time.monotonic() - started > 20:
                    raise TimeoutError("grep exceeded 20 seconds; narrow pattern or doc_ids")
                for match in compiled.finditer(doc.get(field, ""), timeout=1):
                    if seen >= offset:
                        if len(matches) == limit:
                            return {"pattern": pattern, "offset": offset, "matches": matches,
                                    "next_offset": offset + len(matches)}
                        matches.append({"title": doc.get("title", ""),
                                        "match_start": match.start(), "match_end": match.end(),
                                        **exact_excerpt(doc, field, match.start() - 180, match.end() + 260)})
                    seen += 1
        return {"pattern": pattern, "offset": offset, "matches": matches, "next_offset": None}

    def read(self, doc_id, start=0, length=24000):
        if doc_id not in self.by_id:
            raise ValueError(f"Unknown chunk: {doc_id}")
        start = integer(start, "start", 0)
        length = integer(length, "length", 1)
        doc = self.by_id[doc_id]
        body = doc["text"]
        if start > len(body):
            raise ValueError(f"start exceeds body_chars={len(body)}")
        end = min(len(body), start + length)
        requested_end = end
        if tokens(body[start:end]) > READ_TOKENS:
            # Work in source characters so Unicode boundaries and offsets remain exact.
            low, high = start, end
            while low < high:
                mid = (low + high + 1) // 2
                if tokens(body[start:mid]) <= READ_TOKENS:
                    low = mid
                else:
                    high = mid - 1
            end = low
        return {"title": doc.get("title", ""), "body_chars": len(body),
                **exact_excerpt(doc, "text", start, end),
                "page_limited": end < requested_end,
                "next_start": end if end < len(body) else None}

    def call(self, name, arguments):
        if name not in {"search", "grep", "read"}:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        result = getattr(self, name)(**arguments)
        # Keep the entire JSON response valid; preserve pagination when limiting output.
        import json
        key = "results" if name == "search" else "matches" if name == "grep" else None
        while tokens(json.dumps(result, ensure_ascii=False)) > RESULT_TOKENS and key and result[key]:
            result[key].pop()
            result["next_offset"] = result["offset"] + len(result[key])
            result["page_limited"] = True
        if tokens(json.dumps(result, ensure_ascii=False)) > RESULT_TOKENS:
            raise ValueError("Tool response exceeds output ceiling")
        return result
