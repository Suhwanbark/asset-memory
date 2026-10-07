"""Flag development-label issues without changing the frozen cohort or scores."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re

from finrate_poc.data import documents
from finrate_poc.runtime import ROOT, write_json


def main():
    out = ROOT / "result/finrate_luna_v1"
    questions = json.loads((out / "cohort/gold/dev.json").read_text())
    selected = {i for q in questions for i in q["source_ids"]}
    sources = {}
    report_dates = defaultdict(list)
    prefix_re = re.compile(r"^([A-Za-z0-9.-]+_\d{4}-\d{2}-\d{2}_10-K)_")
    period_re = re.compile(r"(?i)fiscal year ended (May|June) (\d{1,2}),? (20\d{2})")
    for d in documents(ROOT / "data/finrate_public_v1"):
        if d["_id"] in selected:
            sources[d["_id"]] = {"id": d["_id"], "title": d["title"]}
        m = prefix_re.match(d["title"])
        if not m or not d["title"].startswith("LW_"):
            continue
        # Whitespace normalization is recorded explicitly, not an exact-span claim.
        text = " ".join(d["text"].split())
        for found in period_re.finditer(text):
            month, day, year = found.groups()
            try:
                end = datetime.strptime(f"{month} {day} {year}", "%B %d %Y").date().isoformat()
            except ValueError:
                continue
            filing_date = d["title"].split("_")[1]
            if end <= filing_date:
                report_dates[m[1]].append({"source_id": d["_id"], "period_end": end,
                    "quote_whitespace_normalized": text[max(0,found.start()-60):found.end()+100]})
    latest = {p: max(items, key=lambda x:x["period_end"]) for p, items in report_dates.items()}
    rows = []
    for q in questions:
        question_years = sorted(set(re.findall(r"\b20\d{2}\b", q["question"])))
        linked = []
        for source_id in q["source_ids"]:
            m = prefix_re.match(sources[source_id]["title"])
            if m and m[1] in latest:
                linked.append({"gold_source_id": source_id, "filing_prefix": m[1], **latest[m[1]]})
        source_years = sorted({r["period_end"][:4] for r in linked})
        flag = bool(question_years and source_years and not set(question_years) <= set(source_years))
        rows.append({"q_id": q["q_id"], "question_years": question_years,
                     "linked_report_fiscal_years": source_years,
                     "possible_year_misalignment": flag, "fiscal_year_evidence": linked})
    report = {"split": "development_only", "questions_audited": len(rows),
              "scope": "Lamb Weston report-period evidence; no claim to exhaustive gold verification",
              "year_flags": sum(r["possible_year_misalignment"] for r in rows),
              "interpretation": "A review flag, not automatic invalidation: later reports can legitimately discuss earlier years.",
              "cohort_changed": False, "gold_changed": False, "heldout_opened": False,
              "manual_findings": [
                  {"q_id": "qa_2_001837", "finding": "Question labels the Item 3 comparison 2022/2023; linked filings are 2023/2024, and the 2024 filing explicitly covers fiscal year ended May 26, 2024. Its Item 3 merely cross-references Note 14. Check year assignment independently of model scores."},
                  {"q_id": "qa_2_001865", "finding": "Gold infers a stable operational/legal environment from no pending legal proceedings. Treat that implication as potentially broader than the supplied disclosure; do not automatically reward unsupported operational conclusions."}],
              "rows": rows}
    write_json(out / "development_quality_audit.json", report, immutable=True)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
