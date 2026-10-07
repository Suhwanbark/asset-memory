"""Validate fixed inputs, complete top10 evidence, actual inference traces and scores."""
from collections import Counter
import csv
import io
import json
from pathlib import Path

import run_finrate_bm25_top10_gpt6_luna_v3 as run


def main():
    manifest = json.loads((run.RUN / "manifest.json").read_text())
    report = json.loads((run.RUN / "report.json").read_text())
    qs = json.loads((run.COHORT / "inputs.json").read_text())
    original = json.loads((run.ORIGINAL / "inputs.json").read_text())
    refs = {q["q_id"]: q for q in json.loads((run.COHORT / "gold_annotations.json").read_text())}
    old_report = json.loads((run.PREVIOUS / "report.json").read_text())
    old_rows = {r["q_id"]: r for r in old_report["rows"]}
    rows = {r["q_id"]: r for r in report["rows"]}
    ranks = json.loads((run.COHORT / "rankings.json").read_text())
    metrics = json.loads((run.COHORT / "retrieval_metrics.json").read_text())
    docs, corpus_sha = run.retrieval.load_corpus()
    corpus = {d["_id"]: d for d in docs}
    assert report["state"] == "completed"
    assert corpus_sha == manifest["corpus_zip_sha256"]
    assert run.retrieval.sha_file(run.PREVIOUS / "manifest.json") == manifest["previous_manifest_sha256"]
    assert run.retrieval.sha_file(run.PREVIOUS / "report.json") == manifest["previous_report_sha256"]
    assert run.retrieval.sha_file(run.ORIGINAL / "inputs.json") == manifest["original_cohort_sha256"]
    assert run.retrieval.sha_file(run.ORIGINAL / "gold_annotations.json") == manifest["original_annotations_sha256"]
    assert [q["q_id"] for q in qs] == [q["q_id"] for q in original] == manifest["selected_ids"]
    for name, expected in manifest["artifacts_sha256"].items():
        assert run.retrieval.sha_file(run.COHORT / name) == expected
    for key, path in [("runner_source_sha256", run.__file__),
                      ("transport_source_sha256", run.transport.__file__),
                      ("retrieval_source_sha256", run.retrieval.__file__),
                      ("long_transport_source_sha256", run.long_transport.__file__)]:
        assert run.retrieval.sha_file(path) == manifest[key]
        assert Path(path).read_bytes() == (run.RUN / "source_snapshot" / Path(path).name).read_bytes()
    usage = Counter()
    stage_usage = {"answer": Counter(), "judge": Counter()}
    comparisons = []
    turn_ids = set()
    for q, prev in zip(qs, original):
        qid = q["q_id"]
        assert q["question"] == prev["question"] == refs[qid]["question"]
        ranked = ranks[qid]["ranked_candidates"]
        ids = q["retrieved_doc_ids"]
        assert ids == [r["doc_id"] for r in ranked[:10]] and len(set(ids)) == 10
        assert all(a["score"] >= b["score"] for a, b in zip(ranked, ranked[1:]))
        expected_prompt = run.transport.answer_prompt(q["question"], [corpus[i] for i in ids])
        folder = run.RUN / qid
        assert (folder / "answer/prompt.txt").read_text() == expected_prompt
        assert run.sha(expected_prompt) == q["prompt_sha256"]
        answer = json.loads((folder / "answer/response.json").read_text())
        assert answer["model"] == run.MODEL and answer["reasoning_effort"] == "medium"
        a_events = [json.loads(x) for x in (folder / "answer/attempt_01/events.jsonl").read_text().splitlines()]
        valid, messages, a_usage, forbidden, window = run.long_transport.inspect_events(a_events)
        assert valid and not forbidden and answer["text"] == messages[0]["text"]
        assert a_usage == answer["usage"]
        assert a_usage[0]["input_tokens"] >= 0.9 * q["prompt_tokens_estimate"]
        requests = [json.loads(x) for x in (folder / "answer/attempt_01/request.jsonl").read_text().splitlines()]
        injected = next(r for r in requests if r.get("method") == "thread/inject_items")
        assert injected["params"]["items"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": expected_prompt}]}]
        turn_req = next(r for r in requests if r.get("method") == "turn/start")
        assert turn_req["params"]["input"] == [{"type": "text", "text": run.long_transport.TRIGGER}]
        completed = next(e for e in a_events if e.get("method") == "turn/completed")
        turn_id = completed["params"]["turn"]["id"]
        assert turn_id not in turn_ids
        turn_ids.add(turn_id)
        judge = json.loads((folder / "judge/response.json").read_text())
        ref = refs[qid]
        expected_judge = run.transport.load_judge()(q["question"], ref["answer"], answer["text"], ref["key_points"])
        assert (folder / "judge/prompt.txt").read_text() == expected_judge
        assert judge["model"] == run.MODEL and judge["reasoning_effort"] == "medium"
        j_events = [json.loads(x) for x in (folder / "judge/attempt_01/events.jsonl").read_text().splitlines()]
        valid, messages, j_usage, warnings, forbidden = run.transport.inspect_events(j_events)
        assert valid and not forbidden and messages[0] == judge["text"]
        assert j_usage == judge["usage"]
        grade = run.transport.parse_judge(judge["text"], len(ref["key_points"]))
        assert grade["verdict"] == rows[qid]["verdict"]
        assert rows[qid]["answer"] == answer["text"] and rows[qid]["status"] == "graded"
        for stage, values in [("answer", a_usage), ("judge", j_usage)]:
            for u in values:
                stage_usage[stage].update(u)
                usage.update(u)
        comparisons.append({"q_id": qid, "task": q["task"], "gold_context_verdict": old_rows[qid]["verdict"],
                            "bm25_top10_verdict": grade["verdict"], "gold_count": metrics[qid]["gold_count"],
                            "gold_found_top10": metrics[qid]["10"]["matched_count"],
                            "bm25_prompt_tokens_estimate": q["prompt_tokens_estimate"]})
    assert dict(usage) == report["provider_usage"]
    assert report["attempts_finished"] == 40
    assert len(list(run.RUN.glob("*/answer/response.json"))) == 20
    assert len(list(run.RUN.glob("*/judge/response.json"))) == 20
    canary = run.ROOT / "result/finrate_bm25_appserver_probe_v2"
    assert (canary / "events.jsonl").read_bytes() == (run.RUN / "qa_3_001691/answer/attempt_01/events.jsonl").read_bytes()
    by_task = {}
    for task in ["LT-QA", "EC-QA", "ALL"]:
        rr = [r for r in comparisons if task == "ALL" or r["task"] == task]
        by_task[task] = {"n": len(rr), "gold_correct": sum(r["gold_context_verdict"] == "CORRECT" for r in rr),
                         "bm25_correct": sum(r["bm25_top10_verdict"] == "CORRECT" for r in rr)}
    validation = {"completed": True, "same_fixed_questions": 20, "all_original_chunks_indexed": len(docs),
                  "full_top10_prompt_verified": 20, "actual_model_and_effort_verified": 40,
                  "unchanged_official_judge_prompt_verified": 20, "native_tool_calls": 0, "compactions": 0,
                  "answers": 20, "judges": 20, "successful_calls": 40, "successful_answers_regenerated": 0,
                  "first_canary_reused_byte_identically": True, "unresolved_failures": 0,
                  "input_usage_sum_verified": True, "original_cohort_gold_and_report_unchanged": True,
                  "provider_usage": dict(usage), "usage_by_stage": {k: dict(v) for k, v in stage_usage.items()},
                  "comparison": by_task, "paired_rows": comparisons,
                  "comparison_limitations": ["same questions/model/medium/rubric; evidence and input token totals differ",
                                              "answer transport has extra fixed continuation trigger for every BM25 query",
                                              "single Luna judge; no regrading or score overrides"]}
    run.write(run.RUN / "validation.json", validation, frozen=True)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(comparisons[0]))
    writer.writeheader(); writer.writerows(comparisons)
    run.freeze_text(run.RUN / "paired_comparison.csv", out.getvalue())
    lines = ["BM25 top10 vs previous full gold-context; same 20 questions and GPT-6 Luna answer/judge"]
    for task, r in by_task.items():
        lines.append(f"{task}: gold {r['gold_correct']}/{r['n']}, BM25 {r['bm25_correct']}/{r['n']}")
    lines.extend(validation["comparison_limitations"])
    run.freeze_text(run.RUN / "comparison.txt", "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "paired_rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
