"""Build exactly ten previously acquired accession-linked public HTML copies.

Selection uses only company/form/filing metadata, not questions or outcomes.
No network, LLM, GPU, original-data mutations, or corpus-wide expansion.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from finrate_raw_store_v1 import ROOT, DEFAULT_STORE, SCHEMA, RawStore, original_paths, project_html, sha256

SOURCE = ROOT / "data/finrate_source_audit_20261007"
RESULT = ROOT / "result/finrate_raw_smoke10_v1"
MIN_FREE = 10 * 1024**3


def guard():
    if shutil.disk_usage(ROOT).free < MIN_FREE:
        raise RuntimeError("STOP: free disk below 10 GiB")


def dump(path, value):
    guard()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select():
    candidates = []
    for path in sorted(SOURCE.glob("*.meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("status") == "compared" and (SOURCE / (meta["filing_key"] + ".mirror.html")).exists():
            meta["metadata_source_path"] = str(path.relative_to(ROOT))
            candidates.append(meta)
    chosen = {}
    for ticker in sorted({m["ticker"] for m in candidates}):
        meta = next(m for m in candidates if m["ticker"] == ticker)
        chosen[meta["filing_key"]] = meta
    for form in sorted({m["actual_sec_form"] for m in candidates}):
        if not any(m["actual_sec_form"] == form for m in chosen.values()):
            meta = next(m for m in candidates if m["actual_sec_form"] == form)
            chosen[meta["filing_key"]] = meta
    for meta in candidates:
        if len(chosen) >= 10:
            break
        chosen.setdefault(meta["filing_key"], meta)
    if len(chosen) != 10:
        raise RuntimeError("Selection does not contain exactly ten filings")
    return candidates, sorted(chosen.values(), key=lambda m: m["filing_key"])


def metadata_row(meta, cik):
    """Independently recheck accession identity against cached SEC submission rows."""
    company = json.loads((SOURCE / "submissions" / f"CIK{cik:010d}.json").read_text())
    tables = [company["filings"]["recent"]]
    for path in sorted((SOURCE / "submissions").glob(f"CIK{cik:010d}-submissions-*.json")):
        tables.append(json.loads(path.read_text()))
    found = []
    for table in tables:
        for i, accession in enumerate(table["accessionNumber"]):
            if accession == meta["accession"]:
                found.append({k: table[k][i] for k in ["accessionNumber", "filingDate", "reportDate",
                                                      "acceptanceDateTime", "form", "primaryDocument"]})
    if not found or any(row != found[0] for row in found):
        raise ValueError("Missing/conflicting SEC accession metadata")
    row = found[0]
    if row != meta["metadata_candidates"][0]:
        raise ValueError("Prior audit and SEC metadata disagree")
    return company["name"], row


def validate_projection(data, projected):
    # Resolve every text span against the unmodified source DOM. This catches
    # offsets accidentally pointing into the pruned rather than original DOM.
    soup = BeautifulSoup(data, "html.parser")
    paths = original_paths(soup)
    nodes = {paths[id(node)]: node for node in soup.descendants if type(node) is NavigableString}
    for span in projected["spans"]:
        expected = re.sub(r"\s+", " ", str(nodes[span["dom_path"]]))
        actual = projected["text"][span["start"]:span["end"]]
        if actual != expected:
            raise AssertionError("Text span does not resolve to original HTML")
    return len(projected["spans"])


def main():
    guard()
    if DEFAULT_STORE.exists() or RESULT.exists():
        raise SystemExit("Refusing to overwrite existing smoke artifacts")
    candidates, selected = select()
    DEFAULT_STORE.mkdir(parents=True)
    RESULT.mkdir(parents=True)
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "storage/tool smoke, not QA evaluation",
            "source_pool": "12 locally cached accession-linked public HTML mirrors",
            "selection_rule": "Sort filing_key; first per ticker, fill uncovered SEC forms, then sorted remaining to ten",
            "not_random_or_representative": True, "questions_or_gold_used": False,
            "candidate_keys": [m["filing_key"] for m in candidates],
            "selected": selected, "primary_document_only": True,
            "external_images_and_exhibits": "not downloaded; original HTML references preserved",
            "sec_bytes_verified": False, "direct_sec_download": False}
    dump(DEFAULT_STORE / "selection.json", plan)
    db = sqlite3.connect(DEFAULT_STORE / "manifest.sqlite")
    db.executescript(SCHEMA)
    for key, value in {"schema_version": 2, "offsets": "Unicode characters, end-exclusive",
                       "structure_sidecar_policy": "omitted; reconstruct from retained HTML when needed",
                       "scope": "primary HTML only; no external images/exhibits",
                       "provenance": "cached Yahoo Finance mirrors, SEC accession metadata verified",
                       "original_dom_paths": "BeautifulSoup html.parser before projection pruning",
                       "build_status": "building"}.items():
        db.execute("INSERT INTO store_metadata VALUES (?,?)", (key, json.dumps(value)))
    with (ROOT / "result/finrate_corpus_inventory_20261007/filings.csv").open(encoding="utf-8-sig") as f:
        inventory = {f'{r["ticker"]}_{r["title_date"]}_{r["form_label"]}': r for r in csv.DictReader(f)}
    details = []
    verified_spans = 0
    for meta in selected:
        guard()
        key = meta["filing_key"]
        raw_path = SOURCE / (key + ".mirror.html")
        data = raw_path.read_bytes()
        if sha256(data) != meta["html_sha256"]:
            raise AssertionError("Source HTML hash changed")
        cik = int(re.search(r"/data/(\d+)/", meta["original_sec_url"])[1])
        company, sec = metadata_row(meta, cik)
        filing_id = meta["accession"]
        document_id = filing_id + ":primary"
        folder = DEFAULT_STORE / "filings" / f"{cik:010d}" / filing_id.replace("-", "")
        folder.mkdir(parents=True)
        (folder / "primary.html").write_bytes(data)
        projected = project_html(data)
        verified_spans += validate_projection(data, projected)
        text = projected["text"]
        (folder / "primary.txt").write_text(text, encoding="utf-8")
        dump(folder / "provenance.json", {"source_metadata": meta, "sec_metadata": sec,
                                           "projection": projected["projection"],
                                           "historical_company_name": "not inferred from current submission name"})
        rel = lambda name: str((folder / name).relative_to(DEFAULT_STORE))
        db.execute("INSERT INTO filings VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (filing_id, filing_id, f"{cik:010d}", meta["ticker"], company, sec["form"],
                    sec["filingDate"], sec["acceptanceDateTime"], sec["reportDate"] or None,
                    int(sec["form"].endswith("/A")), None))
        db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (document_id, filing_id, "primary", sec["primaryDocument"], meta["original_sec_url"],
                    meta["html_download_url"], meta["copy_provenance"], 0, rel("primary.html"),
                    rel("primary.txt"), sha256(data), sha256(text.encode()),
                    len(text), meta["metadata_source_path"]))
        root_id = document_id + ":section:00000"
        for i, section in enumerate(projected["sections"]):
            db.execute("INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (f"{document_id}:section:{i:05d}", document_id, None if i == 0 else root_id,
                        section["kind"], section["title"], section["start"], section["end"],
                        section["dom_path"], section["source_anchor"], section["boundary_basis"]))
        db.executemany("INSERT INTO text_spans VALUES (?,?,?,?)",
                       [(document_id, s["start"], s["end"], s["dom_path"]) for s in projected["spans"]])
        # Only document-level mapping is currently justified; do not invent
        # character offsets by stitching or fuzzy-matching benchmark chunks.
        chunk_ids = inventory[key]["chunk_ids_in_title_sequence"].split(";")
        if len(chunk_ids) != meta["chunk_count"]:
            raise AssertionError("Inventory chunk count changed")
        db.executemany("INSERT INTO finrate_mapping VALUES (?,?,?,?,?,?,?,?,?)",
                       [(chunk_id, key, document_id, meta["title_date"], meta["title_form"],
                         "document_candidate_from_prior_audit;chunk_span_unmapped",
                         meta["identity_matching_basis"], None, None) for chunk_id in chunk_ids])
        db.commit()
        details.append({"ticker": meta["ticker"], "title_filing_key": key, "filing_id": filing_id,
                        "document_id": document_id, "form": sec["form"], "filed_at": sec["filingDate"],
                        "report_date": sec["reportDate"], "html_bytes": len(data), "mapped_chunk_ids": len(chunk_ids),
                        "projection": projected["projection"]})
        print(json.dumps({"saved": key, "characters": len(text)}, ensure_ascii=False), flush=True)
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or db.execute("PRAGMA foreign_key_check").fetchall():
        raise AssertionError("SQLite integrity failure")
    db.execute("UPDATE store_metadata SET value=? WHERE key='build_status'", (json.dumps("ready"),))
    db.commit()
    counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
              for table in ["filings", "documents", "sections", "text_spans", "finrate_mapping"]}
    db.close()
    store = RawStore()
    calls = 0
    with (RESULT / "tool_trace.jsonl").open("w", encoding="utf-8") as trace:
        def call(tool, arguments):
            nonlocal calls
            result = getattr(store, tool)(**arguments)
            trace.write(json.dumps({"tool": tool, "arguments": arguments, "result": result}, ensure_ascii=False) + "\n")
            calls += 1
            return result
        listed, offset = [], 0
        while True:
            result = call("list_filings", {"offset": offset, "limit": 3, "include_sections": True})
            listed.extend(result["items"])
            if result["next_offset"] is None:
                break
            offset = result["next_offset"]
        assert len(listed) == 10 and len({r["document_id"] for r in listed}) == 10
        for item in listed:
            document_id = item["document_id"]
            full = store._text(store._document(document_id))
            pages, start = [], 0
            while True:
                result = call("read", {"document_id": document_id, "start": start, "length": 24000})
                pages.append(result["text"])
                if result["next_start"] is None:
                    break
                start = result["next_start"]
            assert "".join(pages) == full
            hits = call("grep", {"pattern": "Item", "document_ids": [document_id], "limit": 2})
            assert hits["matches"], document_id
            for hit in hits["matches"]:
                assert full[hit["start"]:hit["end"]] == hit["text"]
            for section in item["sections"]:
                if section["kind"] == "item":
                    read = call("read", {"document_id": document_id, "section_id": section["section_id"], "length": 500})
                    assert read["start"] == section["start_offset"]
                    break
        hits1 = call("grep", {"pattern": "Item", "limit": 3})
        hits2 = call("grep", {"pattern": "Item", "offset": hits1["next_offset"], "limit": 3})
        assert {(r["document_id"],r["match_start"]) for r in hits1["matches"]}.isdisjoint(
            {(r["document_id"],r["match_start"]) for r in hits2["matches"]})
        filtered = call("list_filings", {"ticker": "VLO", "form": "10-Q"})
        assert filtered["total"] == 1
    store.close()
    source_hashes = {}
    (RESULT / "source_snapshot").mkdir()
    for name in ["finrate_raw_store_v1.py", "build_finrate_raw_smoke10_v1.py"]:
        code = (ROOT / "method" / name).read_bytes()
        (RESULT / "source_snapshot" / name).write_bytes(code)
        source_hashes[name] = sha256(code)
    report = {"status": "passed", "counts": counts, "companies": sorted({m["ticker"] for m in selected}),
              "forms": dict(Counter(m["actual_sec_form"] for m in selected)), "filings": details,
              "checks": {"original_copy_hashes": 10, "sec_metadata_rows_verified": 10,
                         "text_spans_resolved_to_original_dom": verified_spans,
                         "full_document_read_roundtrips": 10, "sqlite_integrity": "ok",
                         "sqlite_foreign_keys": "ok", "tool_trace_calls": calls,
                         "listing_and_grep_pagination": "passed", "section_navigation": "passed"},
              "source_sha256": source_hashes, "model_calls": 0, "network_requests": 0,
              "disk_free_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 3),
              "data_bytes": sum(p.stat().st_size for p in DEFAULT_STORE.rglob("*") if p.is_file()),
              "limitations": ["Storage smoke only; no QA/model scores", "Public mirrors, not byte-verified against SEC",
                              "Primary HTML only; referenced images and exhibits not downloaded",
                              "Text projection is not browser rendering; colspan/rowspan remain in retained HTML",
                              "Company names in filings table are current SEC metadata snapshot names, not historical names",
                              "FIN-RATE group-to-document candidates retained; chunk-level source offsets remain NULL",
                              "Section index is navigation-only, not an exhaustive semantic section classifier",
                              "Smoke grep supports literal case-insensitive/sensitive strings only"]}
    dump(RESULT / "report.json", report)
    print(json.dumps({"status": report["status"], "counts": counts, "data_bytes": report["data_bytes"],
                      "disk_free_gib": report["disk_free_gib"]}), flush=True)


if __name__ == "__main__":
    main()
