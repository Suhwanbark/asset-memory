"""Inventory the published corpus without model calls or source-document downloads.

Filing keys and item labels are derived from titles, not SEC accession numbers.
An uninterrupted chunk sequence does not establish source-content completeness.
"""
import csv
import hashlib
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/finrate_public_v1"
OUT = ROOT / "result/finrate_corpus_inventory_20261007"
TITLE = re.compile(
    r"^(?P<ticker>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<form>.+?)_(?P<sequence>\d+)_(?P<section>.*)$"
)


def write_csv(name, records):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    manifest = json.loads((DATA / "source_manifest.json").read_text())
    archive = DATA / "corpus/corpus.zip"
    actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected_sha = next(x["sha256"] for x in manifest["files"] if x["path"] == "corpus/corpus.zip")
    if actual_sha != expected_sha:
        raise ValueError("Corpus archive differs from the pinned source manifest")
    with zipfile.ZipFile(archive) as z:
        with z.open("corpus.jsonl") as f:
            corpus = [json.loads(line) for line in f]
    groups = defaultdict(list)
    field_sets = Counter()
    word_lengths = []
    for row in corpus:
        match = TITLE.fullmatch(row["title"])
        if not match:
            raise ValueError(f"Unrecognized corpus title: {row['title']}")
        meta = match.groupdict()
        field_sets[tuple(sorted(row))] += 1
        word_lengths.append(len(row["text"].split()))
        groups[(meta["ticker"], meta["date"], meta["form"])].append((meta, row))
    filing_rows = []
    duplicate_sequences = []
    for (ticker, date, form), rows in sorted(groups.items()):
        seq_counts = Counter(int(m["sequence"]) for m, _ in rows)
        gaps = sorted(set(range(max(seq_counts) + 1)) - set(seq_counts))
        duplicates = {str(k): v for k, v in seq_counts.items() if v > 1}
        if duplicates:
            duplicate_sequences.append({"ticker": ticker, "date": date, "form": form, "duplicate_sequences": duplicates})
        labels = set()
        for meta, _ in rows:
            item = re.match(r"item_(\d+[a-z]?)(?:_|$)", meta["section"])
            if item:
                labels.add(item.group(1).upper())
        filing_rows.append({
            "ticker": ticker,
            "title_date": date,
            "form_label": form,
            "filing_key_basis": "ticker+title_date+form_label; not accession",
            "chunk_count": len(rows),
            "whitespace_units_including_markdown": sum(len(r["text"].split()) for _, r in rows),
            "title_item_labels_only": ";".join(sorted(labels, key=lambda x: (int(re.match(r"\d+", x).group()), x))),
            "item_unknown_chunks": sum("item_unknown" in m["section"] for m, _ in rows),
            "prologue_chunks": sum(m["section"] == "prologue" for m, _ in rows),
            "sequence_min": min(seq_counts),
            "sequence_max": max(seq_counts),
            "sequence_gap_count": len(gaps),
            "duplicate_sequence_positions": len(duplicates),
            "source_completeness": "unverified_no_accession_or_original_comparison",
            "chunk_ids_in_title_sequence": ";".join(r["_id"] for m, r in sorted(rows, key=lambda x: int(x[0]["sequence"]))),
        })
    years = sorted({r["title_date"][:4] for r in filing_rows})
    company_rows = []
    for ticker in sorted({r["ticker"] for r in filing_rows}):
        selected = [r for r in filing_rows if r["ticker"] == ticker]
        counts = Counter(r["title_date"][:4] for r in selected)
        company_rows.append({
            "ticker": ticker,
            "first_title_date": min(r["title_date"] for r in selected),
            "last_title_date": max(r["title_date"] for r in selected),
            "observed_years": ";".join(sorted(counts)),
            "years_without_records_in_corpus_span": ";".join(y for y in years if not counts[y]),
            "filing_groups": len(selected),
            "chunks": sum(r["chunk_count"] for r in selected),
            "10K_filing_groups": sum(r["form_label"] == "10-K" for r in selected),
            "10Q_filing_groups": sum(r["form_label"] == "10-Q" for r in selected),
            **{f"filings_{y}": counts[y] for y in years},
        })
    year_rows = []
    for year in years:
        selected = [r for r in filing_rows if r["title_date"].startswith(year)]
        year_rows.append({"title_year": year, "companies": len({r["ticker"] for r in selected}), "filing_groups": len(selected), "chunks": sum(r["chunk_count"] for r in selected)})
    form_rows = []
    for form in sorted({r["form_label"] for r in filing_rows}):
        selected = [r for r in filing_rows if r["form_label"] == form]
        form_rows.append({"form_label_from_title": form, "filing_groups": len(selected), "chunks": sum(r["chunk_count"] for r in selected), "companies": len({r["ticker"] for r in selected})})
    summary = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        "corpus_sha256_verified": actual_sha,
        "source_provenance": {"paper": "https://arxiv.org/html/2602.07294v4#S3.SS1", "original_source_reported_by_paper": "SEC EDGAR", "distribution": "https://huggingface.co/datasets/GGLabYale/Fin-RATE"},
        "companies_by_title_ticker": len(company_rows),
        "filing_groups_by_ticker_date_form": len(filing_rows),
        "chunks": len(corpus),
        "unique_chunk_ids": len({r["_id"] for r in corpus}),
        "min_title_date": min(r["title_date"] for r in filing_rows),
        "max_title_date": max(r["title_date"] for r in filing_rows),
        "companies_present_in_all_six_calendar_years": sum(not r["years_without_records_in_corpus_span"] for r in company_rows),
        "field_sets": [{"fields": list(k), "count": v} for k, v in field_sets.items()],
        "whitespace_length_including_markdown": {"mean": statistics.mean(word_lengths), "median": statistics.median(word_lengths), "min": min(word_lengths), "max": max(word_lengths)},
        "groups_with_sequence_gaps": sum(r["sequence_gap_count"] > 0 for r in filing_rows),
        "groups_with_duplicate_sequence_positions": duplicate_sequences,
        "year_counts": year_rows,
        "form_counts": form_rows,
        "limitations": [
            "Filing identity uses title-derived ticker/date/form, not SEC accession; same-day filings can collide.",
            "Title dates are not fiscal years and have not been verified against original SEC metadata.",
            "Form labels DEF, SC, NT are abbreviated in the distributed titles; do not infer full subtypes per record.",
            "Item labels may include table-of-contents entries and parsing artifacts; their presence does not prove complete Item bodies.",
            "No original-file comparison or exhaustive SEC filing-history comparison was performed.",
            "No sequence gaps only establishes continuity of distributed numbering, not preservation of original content.",
            "2020-2025 coverage does not establish complete annual/quarterly coverage for any company.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("companies.csv", company_rows)
    write_csv("filings.csv", filing_rows)
    write_csv("years.csv", year_rows)
    write_csv("forms.csv", form_rows)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"groups_with_duplicate_sequence_positions", "limitations", "field_sets"}}, ensure_ascii=False, indent=2))
    print("groups_with_duplicate_sequence_positions", len(duplicate_sequences))
    print("output_directory", OUT)


if __name__ == "__main__":
    main()
