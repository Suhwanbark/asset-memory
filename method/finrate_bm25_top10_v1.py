"""Full-original-chunk BM25 retrieval for FIN-RATE; no learned models or reranker.

The paper does not publish its BM25 analyzer/parameters. These choices are an
explicit reconstruction: bm25s 0.3.13, Lucene scoring, k1=1.2, b=0.75, lowercase
alphanumeric tokens, no stemming/stopword removal, body-only indexing.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

import bm25s
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ZIP = ROOT / "data/finrate_public_v1/corpus/corpus.zip"
INDEX = ROOT / "data/finrate_bm25_original_chunks_v1"
TOKEN_PATTERN = r"[a-z0-9]+"
TOKEN_RE = re.compile(TOKEN_PATTERN)
POLICY = {
    "library": "bm25s", "version": "0.3.13", "method": "lucene",
    "k1": 1.2, "b": 0.75, "token_pattern": TOKEN_PATTERN,
    "lowercase": True, "stopwords": None, "stemming": False,
    "indexed_field": "text", "include_title": False,
    "segmentation": "original distributed chunks, no subdivision or truncation",
    "scope": "all 15311 chunks, no company/year/gold filtering",
    "query": "original question, no rewrite or expansion",
    "tie_break": "ascending original chunk ID",
    "paper_status": "BM25 family follows paper; analyzer/parameters are declared reconstruction choices",
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def check_disk():
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError("Disk below 10 GiB; stop")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_corpus():
    check_disk()
    source = json.loads((ROOT / "data/finrate_public_v1/source_manifest.json").read_text())
    expected = next(x["sha256"] for x in source["files"] if x["path"] == "corpus/corpus.zip")
    actual = sha_file(CORPUS_ZIP)
    if actual != expected:
        raise ValueError("Corpus source SHA256 mismatch")
    with zipfile.ZipFile(CORPUS_ZIP) as archive:
        docs = [json.loads(line) for line in archive.open("corpus.jsonl")]
    docs.sort(key=lambda d: d["_id"])
    if len(docs) != 15311 or len({d["_id"] for d in docs}) != 15311:
        raise ValueError("Unexpected corpus IDs/count")
    return docs, actual


def load_or_build(docs, corpus_sha):
    check_disk()
    if importlib.metadata.version("bm25s") != POLICY["version"]:
        raise ValueError("BM25 library version changed")
    manifest_file = INDEX / "manifest.json"
    identity = {"policy": POLICY, "corpus_zip_sha256": corpus_sha,
                "retrieval_source_sha256": sha_file(__file__),
                "document_ids": [d["_id"] for d in docs]}
    if manifest_file.exists():
        saved = json.loads(manifest_file.read_text())
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError("Frozen index identity changed")
        for name, expected in saved["index_files_sha256"].items():
            if sha_file(INDEX / name) != expected:
                raise ValueError(f"Index hash mismatch: {name}")
        return bm25s.BM25.load(str(INDEX), mmap=True, show_progress=False), saved
    if INDEX.exists() and any(INDEX.iterdir()):
        raise RuntimeError("Partial index exists; inspect before rebuilding")
    print(json.dumps({"stage": "tokenizing", "chunks": len(docs)}), flush=True)
    # Integer token IDs avoid retaining a Python string for every occurrence.
    tokenized = bm25s.tokenize([d["text"] for d in docs], lower=True,
                              token_pattern=TOKEN_PATTERN, stopwords=[],
                              return_ids=True, show_progress=False)
    check_disk()
    retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75,
                           backend="numpy", csc_backend="numpy")
    retriever.index(tokenized, show_progress=False)
    del tokenized
    if len(retriever.get_scores(["revenue"])) != len(docs):
        raise ValueError("Index document count mismatch")
    check_disk()
    INDEX.mkdir(parents=True, exist_ok=True)
    retriever.save(str(INDEX), show_progress=False)
    files = {p.name: sha_file(p) for p in sorted(INDEX.iterdir()) if p.is_file()}
    saved = {**identity, "index_files_sha256": files}
    manifest_file.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "indexed", "chunks": len(docs),
                      "index_bytes": sum(p.stat().st_size for p in INDEX.iterdir())}), flush=True)
    return retriever, saved


def search(question, retriever, docs, k=100):
    """Only question and corpus index enter retrieval; gold IDs are never used."""
    tokens = tokenize(question)
    scores = retriever.get_scores(tokens) if tokens else np.zeros(len(docs))
    if len(scores) != len(docs) or not np.isfinite(scores).all():
        raise ValueError("Invalid retrieval scores")
    # docs are ordered by ID, so stable sorting gives deterministic tied ranks.
    order = np.argsort(-scores, kind="stable")[:k]
    return [{"rank": n + 1, "doc_id": docs[int(i)]["_id"],
             "score": float(scores[i]), "title": docs[int(i)].get("title", "")}
            for n, i in enumerate(order)]


def evidence_metrics(ranking, gold_ids):
    gold = set(gold_ids)
    out = {"gold_count": len(gold)}
    for k in (5, 10):
        found = [r["doc_id"] for r in ranking[:k] if r["doc_id"] in gold]
        out[str(k)] = {"matched_gold_ids": found, "hit_any": bool(found),
                      "matched_count": len(found), "coverage": len(found) / len(gold),
                      "all_gold_found": len(found) == len(gold)}
    return out


def validate_implementation():
    """Independently verify BM25 math and retrieval beyond an embedding-sized prefix."""
    texts = ["alpha alpha beta", "beta gamma", "filler " * 600 + "tailanchor"]
    tokens = [tokenize(t) for t in texts]
    model = bm25s.BM25(method="lucene", k1=1.2, b=0.75, backend="numpy")
    model.index(tokens, show_progress=False)
    q = ["alpha", "beta"]
    lengths = [len(t) for t in tokens]
    avg = sum(lengths) / len(lengths)
    expected = []
    for words, length in zip(tokens, lengths):
        freq = Counter(words)
        value = 0.0
        for term in q:
            df = sum(term in ws for ws in tokens)
            idf = math.log(1 + (len(tokens) - df + 0.5) / (df + 0.5))
            tf = freq[term]
            value += idf * tf / (tf + 1.2 * (1 - 0.75 + 0.75 * length / avg))
        expected.append(value)
    np.testing.assert_allclose(model.get_scores(q), expected, rtol=1e-5, atol=1e-6)
    if int(np.argmax(model.get_scores(["tailanchor"]))) != 2:
        raise AssertionError("Full text tail not searchable")
    if not np.all(model.get_scores(["absentterm"]) == 0):
        raise AssertionError("Unknown term should not match")
    return {"bm25_formula": True, "tail_after_600_tokens_searchable": True,
            "unknown_term_scores_zero": True}
