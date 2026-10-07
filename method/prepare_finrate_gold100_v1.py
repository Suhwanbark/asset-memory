"""Freeze 50 LT-QA + 50 EC-QA questions and inspect full gold-context size.

Local preparation only: no inference, no judge calls, no outcome-based filtering.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import statistics
import zipfile

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/finrate_public_v1"
DEST = ROOT / "data/finrate_gold100_gpt6_luna_v1"
SEED = 20261007


def save(name, text):
    path = DEST / name
    if path.exists() and path.read_bytes().decode("utf-8") != text:
        raise RuntimeError(f"Frozen artifact differs: {path}")
    if not path.exists():
        path.write_text(text)


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main():
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError("Free disk space below 10 GiB")
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    used_files = [f for f in manifest["files"] if f["path"] in {
        "corpus/corpus.zip", "qa/LT-QA.json", "qa/EC-QA.json"}]
    for entry in used_files:
        actual = hashlib.sha256((SOURCE / entry["path"]).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {entry['path']}")
    with zipfile.ZipFile(SOURCE / "corpus/corpus.zip") as archive:
        corpus = {row["_id"]: row for line in archive.open("corpus.jsonl")
                  for row in [json.loads(line)]}
    enc = tiktoken.get_encoding("o200k_base")
    inputs, gold, rows, summaries = [], [], [], {}
    for task in ("LT-QA", "EC-QA"):
        population = sorted(json.loads((SOURCE / "qa" / f"{task}.json").read_text()),
                            key=lambda q: q["q_id"])
        selected = random.Random(f"{SEED}:{task}").sample(population, 50)
        companies, unique_docs, lengths, counts = set(), set(), [], []
        for q in selected:
            docs = [corpus[doc_id] for doc_id in q["doc_ids"]]
            text = "\n\n".join(d["text"] for d in docs)
            # Text-only estimate; final request overhead/usage is recorded at execution.
            ntokens = len(enc.encode(text, disallowed_special=()))
            assets = sorted({d["title"].split("_")[0] for d in docs})
            inputs.append({"q_id": q["q_id"], "task": task,
                           "question": q["question"], "gold_doc_ids": q["doc_ids"]})
            gold.append({"task": task, **q})
            rows.append({"q_id": q["q_id"], "task": task,
                         "question": q["question"], "companies": ";".join(assets),
                         "gold_chunk_count": len(docs), "gold_text_tokens_estimate": ntokens,
                         "gold_doc_ids": ";".join(q["doc_ids"])})
            companies.update(assets)
            unique_docs.update(q["doc_ids"])
            lengths.append(ntokens)
            counts.append(len(docs))
        summaries[task] = {
            "population": len(population), "sample_size": len(selected),
            "companies": len(companies), "unique_gold_chunks": len(unique_docs),
            "gold_chunks_per_question": {
                "min": min(counts), "max": max(counts), "mean": statistics.mean(counts)},
            "gold_text_tokens_estimate": {
                "min": min(lengths), "max": max(lengths), "mean": statistics.mean(lengths),
                "median": statistics.median(lengths), "sum": sum(lengths)},
            "contexts_over_240000_estimated_tokens": sum(n > 240000 for n in lengths),
            "selected_ids": [q["q_id"] for q in selected],
        }
    assert len(inputs) == len({q["q_id"] for q in inputs}) == 100
    assert all("answer" not in q and "key_points" not in q for q in inputs)
    DEST.mkdir(parents=True, exist_ok=True)
    save("inputs.json", dump(inputs))
    save("gold_annotations.json", dump(gold))
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    save("questions.csv", table.getvalue())
    output = {
        "status": "prepared_only_no_model_calls", "dataset": manifest["dataset"],
        "revision": manifest["revision"], "source_files": used_files,
        "seed": SEED, "sample_algorithm": "random.Random(f'{seed}:{task}').sample(sorted_by_q_id, 50)",
        "selection_filters": [], "outcomes_used_for_selection": False,
        "context_truncation": False, "question_replacement_for_context_length": False,
        "requested_answer_model": "gpt-6-luna", "proposed_reasoning_effort": "medium",
        "gold_text_token_estimator": "tiktoken/o200k_base; excludes request overhead",
        "not_an_untouched_final_test_split": True, "tasks": summaries,
        "artifacts_sha256": {name: hashlib.sha256((DEST / name).read_bytes()).hexdigest()
                             for name in ("inputs.json", "gold_annotations.json", "questions.csv")},
    }
    save("manifest.json", dump(output))
    print(dump({task: {k: v for k, v in summary.items() if k != "selected_ids"}
                for task, summary in summaries.items()}))


if __name__ == "__main__":
    main()
