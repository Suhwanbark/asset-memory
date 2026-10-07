"""Download pinned public data and audit EVERY question before any sampling."""
from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import zipfile

import requests

from .runtime import check_disk, file_digest, write_json

HF_ID = "GGLabYale/Fin-RATE"
HF_REVISION = "284b1b4f6cd1e58574f1f1d17c29b35c6aa72cce"
TASKS = ("DR-QA", "EC-QA", "LT-QA")
MODEL = "gpt-5.6-luna"
MODEL_CUTOFF = "2026-02-16"
MODEL_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"


def download(url, path, *, max_bytes=100_000_000):
    check_disk()
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    count = 0
    try:
        with requests.get(url, timeout=(10, 60), stream=True) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                for block in response.iter_content(1024 * 1024):
                    check_disk()
                    count += len(block)
                    if count > max_bytes:
                        raise ValueError("Download exceeded predeclared size limit")
                    f.write(block)
        partial.replace(path)
    except BaseException:
        # Keep partial bytes for audit; never interpret them as a complete source.
        raise


def acquire(folder):
    folder = Path(folder)
    base = f"https://huggingface.co/datasets/{HF_ID}/resolve/{HF_REVISION}/"
    files = ["corpus/corpus.zip"] + [f"qa/{task}.json" for task in TASKS]
    manifest = {"dataset": HF_ID, "revision": HF_REVISION, "files": []}
    for relative in files:
        path = folder / relative
        print("ACQUIRE", relative, flush=True)
        download(base + relative, path)
        manifest["files"].append({"path": relative, "url": base + relative,
                                  "sha256": file_digest(path), "bytes": path.stat().st_size})
    write_json(folder / "source_manifest.json", manifest, immutable=True)
    return manifest


def documents(folder):
    # Stream the archive; no duplicate 250+ MB extracted corpus is needed.
    with zipfile.ZipFile(Path(folder) / "corpus/corpus.zip") as archive:
        names = [n for n in archive.namelist() if n.endswith("corpus.jsonl")]
        if len(names) != 1:
            raise ValueError(f"Unexpected corpus members: {names}")
        with archive.open(names[0]) as f:
            for line in f:
                yield json.loads(line)


def metadata(doc):
    title = doc["title"]
    matches = re.findall(r"(?<!\d)(20\d{2})[-_/](\d{2})[-_/](\d{2})(?!\d)", title)
    dates = []
    for parts in matches:
        try:
            dates.append(date(*map(int, parts)).isoformat())
        except ValueError:
            pass
    ticker = re.split(r"[ _/|]", title.strip())[0]
    return {"doc_id": doc["_id"], "title": title, "ticker": ticker,
            "filing_date": dates[0] if len(set(dates)) == 1 else None}


def audit(folder, out, *, cutoff=MODEL_CUTOFF):
    cutoff = date.fromisoformat(cutoff).isoformat()
    out = Path(out)
    lookup = {}
    for doc in documents(folder):
        if doc["_id"] in lookup:
            raise ValueError("Duplicate corpus ID")
        lookup[doc["_id"]] = metadata(doc)
    dates = [m["filing_date"] for m in lookup.values() if m["filing_date"]]
    rows = []
    counts = {}
    for task in TASKS:
        questions = json.loads((Path(folder) / f"qa/{task}.json").read_text())
        counts[task] = len(questions)
        for q in questions:
            ids = q.get("doc_ids") or [q.get("doc_id")]
            missing = [i for i in ids if i not in lookup]
            evidence_dates = [lookup[i]["filing_date"] for i in ids if i in lookup]
            if missing or not evidence_dates or None in evidence_dates:
                reason = "unresolved_source_date_or_id"
            elif max(evidence_dates) <= cutoff:
                reason = "all_gold_filings_on_or_before_model_cutoff"
            else:
                # Publication after cutoff is NECESSARY but NOT sufficient.
                # A later filing can merely repeat an old fact or forecast.
                reason = "requires_new_fact_dependency_audit"
            rows.append({"q_id": q["q_id"], "task": task, "status": reason,
                         "gold_doc_ids": ids, "gold_filing_dates": evidence_dates,
                         "missing_ids": missing})
    if len({r["q_id"] for r in rows}) != len(rows):
        raise ValueError("Question IDs are not globally unique")
    candidates = [r for r in rows if r["status"] == "requires_new_fact_dependency_audit"]
    report = {
        "model": MODEL, "knowledge_cutoff": cutoff, "cutoff_source": MODEL_SOURCE,
        "dataset_revision": HF_REVISION, "questions": counts, "total_questions": len(rows),
        "corpus_chunks": len(lookup), "known_filing_date_min": min(dates) if dates else None,
        "known_filing_date_max": max(dates) if dates else None,
        "unknown_filing_dates": len(lookup) - len(dates),
        "post_cutoff_corpus_chunks": sum(d > cutoff for d in dates),
        "question_status_counts": dict(Counter(r["status"] for r in rows)),
        "date_candidates": len(candidates), "approved_eligible_questions": 0,
        "eligibility_rule": "Requires a gold-supported fact first disclosed after model cutoff; dates alone never approve a question.",
        "state": "requires_dependency_audit" if candidates else "stopped_no_post_cutoff_candidates",
        "limitation": "Filing dates parsed from released titles; date parsing does not certify first disclosure of a fact.",
    }
    write_json(out / "cutoff_audit.json", report)
    write_json(out / "question_audit.json", rows)
    write_json(out / "document_metadata.json", list(lookup.values()))
    return report
