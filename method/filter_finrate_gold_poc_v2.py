"""Apply the user's long-context exclusion to the already frozen PoC cohort."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/finrate_gold100_gpt6_luna_v1"
DEST = ROOT / "data/finrate_gold_poc_gpt6_luna_v2"
CHUNK_LIMIT = 100_000
CONTEXT_LIMIT = 240_000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = DEST / name
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    encoded = text.encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"Frozen artifact differs: {path}")
    if not path.exists():
        path.write_bytes(encoded)


def main():
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError("Free disk space below 10 GiB")
    parent = json.loads((BASE / "manifest.json").read_text())
    for name, expected in parent["artifacts_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError(f"Parent artifact hash mismatch: {name}")
    corpus_path = ROOT / "data/finrate_public_v1/corpus/corpus.zip"
    expected = next(f["sha256"] for f in parent["source_files"] if f["path"] == "corpus/corpus.zip")
    if digest(corpus_path) != expected:
        raise RuntimeError("Corpus hash mismatch")
    inputs = json.loads((BASE / "inputs.json").read_text())
    gold = json.loads((BASE / "gold_annotations.json").read_text())
    old_rows = {r["q_id"]: r for r in csv.DictReader((BASE / "questions.csv").open())}
    wanted = {i for q in inputs for i in q["gold_doc_ids"]}
    with zipfile.ZipFile(corpus_path) as archive:
        docs = {d["_id"]: d for line in archive.open("corpus.jsonl")
                for d in [json.loads(line)] if d["_id"] in wanted}
    enc = tiktoken.get_encoding("o200k_base")
    sizes = {i: len(enc.encode(d["text"], disallowed_special=())) for i, d in docs.items()}
    kept, excluded, rows, measurements = [], [], [], []
    for q in inputs:
        lengths = {i: sizes[i] for i in q["gold_doc_ids"]}
        total = int(old_rows[q["q_id"]]["gold_text_tokens_estimate"])
        record = {"q_id": q["q_id"], "task": q["task"], "chunk_tokens": lengths,
                  "max_chunk_tokens": max(lengths.values()), "gold_context_tokens": total}
        measurements.append(record)
        reasons = []
        if record["max_chunk_tokens"] > CHUNK_LIMIT:
            reasons.append("single_gold_chunk_over_100000_tokens")
        if total > CONTEXT_LIMIT:
            reasons.append("gold_context_over_240000_tokens")
        if reasons:
            excluded.append({**record, "reasons": reasons})
        else:
            kept.append(q)
            rows.append({**old_rows[q["q_id"]], "max_gold_chunk_tokens_estimate": record["max_chunk_tokens"]})
    kept_ids = {q["q_id"] for q in kept}
    selected_gold = [q for q in gold if q["q_id"] in kept_ids]
    assert len(kept) + len(excluded) == len(inputs)
    assert len(kept) == len(selected_gold) == len(kept_ids)
    assert not kept_ids.intersection(q["q_id"] for q in excluded)
    DEST.mkdir(parents=True, exist_ok=True)
    save("inputs.json", kept)
    save("gold_annotations.json", selected_gold)
    save("exclusions.json", excluded)
    save("length_audit.json", measurements)
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    save("questions.csv", table.getvalue())
    counts = {task: sum(q["task"] == task for q in kept) for task in ("LT-QA", "EC-QA")}
    manifest = {
        "status": "prepared_only_no_model_calls", "dataset": parent["dataset"],
        "revision": parent["revision"], "parent_cohort": str(BASE),
        "parent_manifest_sha256": digest(BASE / "manifest.json"),
        "filter_reason": "User requested excluding excessively long gold chunks from current validation set",
        "limits": {"single_gold_chunk_tokens_max": CHUNK_LIMIT, "gold_context_tokens_max": CONTEXT_LIMIT},
        "token_estimator": "tiktoken/o200k_base", "comparison": "exclude if strictly greater than either limit",
        "unit_of_exclusion": "whole_question", "corpus_changed": False,
        "replacement_sampling": False, "context_truncation": False, "outcomes_used": False,
        "requested_answer_model": "gpt-6-luna", "proposed_reasoning_effort": "medium",
        "not_an_untouched_final_test_split": True, "counts": counts,
        "excluded_ids": [q["q_id"] for q in excluded],
        "artifacts_sha256": {name: digest(DEST / name) for name in (
            "inputs.json", "gold_annotations.json", "questions.csv", "exclusions.json", "length_audit.json")},
    }
    save("manifest.json", manifest)
    print(json.dumps({"counts": counts, "excluded": excluded, "path": str(DEST)}, indent=2))


if __name__ == "__main__":
    main()
