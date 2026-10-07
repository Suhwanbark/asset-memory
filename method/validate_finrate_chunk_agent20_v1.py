"""Validate completed full20 and produce paired results plus trace diagnostics."""
from collections import Counter
import csv
import io
import json
from pathlib import Path

import finrate_bm25_top10_v1 as bm25
import run_finrate_gold20_gpt56_luna_judge6_v1 as common

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "result/finrate_chunk_agent20_gpt6_luna_v1"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    report = json.loads((RUN / "report.json").read_text())
    assert report["state"].startswith("completed")
    original = json.loads((ROOT / "data/finrate_gold20_gpt6_luna_v1/inputs.json").read_text())
    questions = json.loads((ROOT / "data/finrate_chunk_agent_v1/inputs.json").read_text())
    assert questions == [{k: q[k] for k in ("q_id", "task", "question")} for q in original]
    assert [q["q_id"] for q in questions] == manifest["question_ids"] == [r["q_id"] for r in report["rows"]]
    for name, digest in manifest["source_sha256"].items():
        assert bm25.sha_file(RUN / "source_snapshot" / name) == bm25.sha_file(ROOT / "method" / name) == digest
    for qid, imported in manifest["reused_smoke"].items():
        for name, digest in imported["files_sha256"].items():
            assert bm25.sha_file(Path(imported["source"]) / name) == digest
            assert bm25.sha_file(RUN / qid / name) == digest
    docs, corpus_sha = bm25.load_corpus()
    assert corpus_sha == manifest["corpus_zip_sha256"]
    source = {d["_id"]: d for d in docs}
    gold = {q["q_id"]: q for q in json.loads((ROOT / "data/finrate_gold20_gpt6_luna_v1/gold_annotations.json").read_text())}
    diagnostics = []
    span_count, total_calls, tool_errors = 0, 0, 0
    all_tools = Counter()
    for row in report["rows"]:
        q = next(q for q in questions if q["q_id"] == row["q_id"])
        folder = Path(row["artifact_dir"])
        response = json.loads((folder / "agent/response.json").read_text())
        calls = jsonl(folder / "agent/tools.jsonl")
        requests = jsonl(folder / "agent/requests.jsonl")
        events = jsonl(folder / "agent/events.jsonl")
        start = next(x for x in requests if x.get("method") == "turn/start")
        assert start["params"]["input"] == [{"type": "text", "text": q["question"]}]
        thread = next(x for x in requests if x.get("method") == "thread/start")
        assert thread["params"]["baseInstructions"] == manifest["instructions"]
        assert thread["params"]["dynamicTools"] == manifest["tools"]
        resolved = next(e["result"] for e in events if e.get("id") == 2 and "result" in e)
        assert resolved["model"] == "gpt-6-luna" and resolved["reasoningEffort"] == "medium"
        assert response["tool_calls"] == len(calls)
        server_calls = [e for e in events if e.get("method") == "item/tool/call"]
        delivered = [e["params"]["item"] for e in events if e.get("method") == "item/completed"
                     and e["params"]["item"].get("type") == "dynamicToolCall"]
        regex_omissions, empty_searches, empty_greps = [], 0, 0
        for call in calls:
            total_calls += 1
            all_tools[call["tool"]] += 1
            request = next(e["params"] for e in server_calls if e["params"]["callId"] == call["call_id"])
            assert request["tool"] == call["tool"] and request["arguments"] == call["arguments"]
            completed = [item for item in delivered if item["id"] == call["call_id"]]
            if completed:
                assert json.loads(completed[0]["contentItems"][0]["text"]) == call["result"]
            elif response["state"] == "completed":
                raise AssertionError("Missing completed tool output in successful generation")
            if not call["success"]:
                tool_errors += 1
                continue
            result = call["result"]
            if call["tool"] == "search":
                spans = [span for r in result["results"] for span in r["excerpts"]]
                empty_searches += not result["results"]
            elif call["tool"] == "grep":
                spans = result["matches"]
                empty_greps += not spans
                if "|" in call["arguments"]["pattern"] and not call["arguments"].get("regex", False):
                    regex_omissions.append({"call_index": call["index"], "pattern": call["arguments"]["pattern"], "matches": len(spans)})
            else:
                spans = [result]
            for span in spans:
                assert span["text"] == source[span["doc_id"]][span["field"]][span["start"]:span["end"]]
                span_count += 1
        usage = response["usage"]
        for key in ("inputTokens", "outputTokens", "cachedInputTokens", "reasoningOutputTokens"):
            assert sum(u["last"][key] for u in usage) == response["provider_total"].get(key, 0)
        if row["state"] == "graded":
            assert response["state"] == "completed" and not response["error"] and not response["forbidden_items"]
            assert len(calls) == len(server_calls) == len(delivered)
            ref = gold[row["q_id"]]
            points = ref["key_points"] if isinstance(ref["key_points"], list) else [ref["key_points"]]
            expected_prompt = common.load_judge()(q["question"], ref["answer"], response["answer"], points)
            assert (folder / "judge/prompt.txt").read_text() == expected_prompt
            judge = json.loads((folder / "judge/response.json").read_text())
            assert judge["model"] == "gpt-6-luna" and judge["reasoning_effort"] == "medium"
            assert row["verdict"] == common.parse_judge(judge["text"], len(points))["verdict"]
        diagnostics.append({"q_id": row["q_id"], "task": row["task"], "state": row["state"],
            "verdict": row.get("verdict"), "tool_counts": dict(Counter(c["tool"] for c in calls)),
            "model_responses": response["model_responses"], "regex_flag_omissions": regex_omissions,
            "empty_searches": empty_searches, "empty_greps": empty_greps,
            "cited_without_body_seen": row.get("cited_without_body_seen", []),
            "error": row.get("error")})
    for task, stats in report["tasks"].items():
        rows = [r for r in report["rows"] if task == "ALL" or r["task"] == task]
        assert stats["selected"] == len(rows)
        assert stats["correct"] == sum(r.get("verdict") == "CORRECT" for r in rows)
        assert stats["accuracy"] == stats["correct"] / len(rows)
    result = {"passed": True, "updated": common.now(), "questions": 20, "reused_smoke": 2,
              "source_spans_verified": span_count, "tool_calls": total_calls, "tool_counts": dict(all_tools),
              "tool_errors": tool_errors, "regex_omission_calls": sum(len(d["regex_flag_omissions"]) for d in diagnostics),
              "regex_omission_questions": sum(bool(d["regex_flag_omissions"]) for d in diagnostics),
              "checks": ["original20 unchanged", "smoke2 byte-identical", "prompt/tools/budget/source hashes unchanged",
                         "provider-resolved model and effort", "all source spans exact", "tool dispatch/output consistency",
                         "provider usage reconciles", "official judge unchanged", "scores recounted"]}
    common.write(RUN / "validation.json", result, frozen=True)
    common.write(RUN / "trace_diagnostics.json", diagnostics, frozen=True)
    baselines = {}
    for name, directory in (("bm25", "finrate_bm25_top10_gpt6_luna_v3"), ("gold", "finrate_gold20_gpt6_luna_v1")):
        baseline = json.loads((ROOT / "result" / directory / "report.json").read_text())
        baselines[name] = {r["q_id"]: r for r in baseline["rows"]}
        assert set(baselines[name]) == {q["q_id"] for q in questions}
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["q_id", "task", "agent", "bm25", "gold", "model_responses", "tool_calls"])
    writer.writeheader()
    for row in report["rows"]:
        writer.writerow({"q_id": row["q_id"], "task": row["task"], "agent": row.get("verdict", row["state"]),
                         "bm25": baselines["bm25"][row["q_id"]].get("verdict"), "gold": baselines["gold"][row["q_id"]].get("verdict"),
                         "model_responses": row.get("model_responses"), "tool_calls": row.get("tool_calls")})
    common.freeze_text(RUN / "paired_comparison.csv", stream.getvalue())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
