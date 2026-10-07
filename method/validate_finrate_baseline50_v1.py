"""Audit paired cohort, full source inputs, exposed traces, judge and provenance."""
from collections import Counter
import csv
import io
from pathlib import Path
import argparse

import run_finrate_baseline50_v1 as r


def jsonl(path):
    return [r.json.loads(line) for line in path.read_text().splitlines()]


def cli_response(folder, prompt):
    response = r.read(folder / "response.json")
    assert response["model"] == "gpt-6-luna" and response["reasoning_effort"] == "medium"
    assert (folder / "prompt.txt").read_text() == prompt
    assert response["prompt_sha256"] == r.common.sha(prompt)
    event_path = folder / f"attempt_{response['attempt']:02d}" / "events.jsonl"
    valid, messages, usage, warnings, forbidden = r.common.inspect_events(jsonl(event_path))
    assert valid and not forbidden and messages[0] == response["text"]
    assert response["usage"] == usage
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-agent", action="store_true")
    parser.add_argument("--answer-model", choices=["gpt-6-astra", "gpt-6.1-sol"])
    args = parser.parse_args()
    if args.answer_model:
        import run_finrate_keyword_agent_model50_v1 as variant
        variant.configure(args.answer_model)
    elif args.keyword_agent:
        import run_finrate_keyword_agent50_v1 as narrowed
        narrowed.configure()
    manifest = r.read(r.RUN / "manifest.json")
    report = r.read(r.RUN / "report.json")
    assert report["state"] in {"completed", "completed_with_errors"}
    for name, digest in manifest["source_sha256"].items():
        assert r.bm25.sha_file(r.RUN / "source_snapshot" / name) == digest
        assert r.bm25.sha_file(r.ROOT / "method" / name) == digest
    assert r.bm25.sha_file(r.DATA / "inputs.json") == manifest["inputs_sha256"]
    assert r.bm25.sha_file(r.DATA / "gold_annotations.json") == manifest["annotations_sha256"]
    if not args.answer_model:
        assert r.bm25.sha_file(r.RUN / "imports.json") == manifest["imports_sha256"]
    imports = r.read(r.RUN / "imports.json") if (r.RUN / "imports.json").exists() else {}
    imported = 0
    if args.answer_model:
        parent = Path(manifest["parent_run"])
        assert r.bm25.sha_file(parent / "manifest.json") == manifest["parent_manifest_sha256"]
        assert r.bm25.sha_file(parent / "report.json") == manifest["parent_report_sha256"]
        old = r.read(parent / "manifest.json")
        assert old["selected_ids"] == manifest["selected_ids"]
        assert old["bm25"] == manifest["bm25"]
        assert old["judge_source_sha256"] == manifest["judge_source_sha256"]
        assert old["agent"]["tools"] == manifest["agent"]["tools"]
        assert old["agent"]["instructions"] == manifest["agent"]["instructions"]
        assert {**manifest["agent"]["policy"], "model": old["agent"]["policy"]["model"]} == old["agent"]["policy"]
    elif args.keyword_agent:
        for pair, meta in imports.items():
            folder = r.RUN / pair
            source = Path(meta["source"])
            if "assessment_sha256" in meta:
                assert folder.is_symlink() and folder.resolve() == source.resolve()
                assert r.bm25.sha_file(folder / "assessment.json") == meta["assessment_sha256"]
                imported += 1
            for stage, info in meta["stages"].items():
                stage_source = Path(info["source"])
                if "response_sha256" in info:
                    assert r.bm25.sha_file(folder / stage / "response.json") == info["response_sha256"]
                else:
                    trace = stage_source / info["selected_attempt"]
                    assert r.bm25.sha_file(trace / "events.jsonl") == info["events_sha256"]
                    assert r.bm25.sha_file(trace / "request.json") == info["request_sha256"]
        parent = Path(manifest["parent_run"])
        assert r.bm25.sha_file(parent / "manifest.json") == manifest["parent_manifest_sha256"]
        # Verify original20 artifacts also through the complete import ancestry.
        for method, meta in r.read(parent / "imports.json").items():
            if method not in r.OLD:
                continue
            for qid, files in meta["files"].items():
                for name, digest in files.items():
                    assert r.bm25.sha_file(Path(meta["source"]) / qid / name) == digest
    else:
        for method, meta in imports.items():
            assert r.bm25.sha_file(Path(meta["source"]) / "manifest.json") == meta["manifest_sha256"]
            for qid, files in meta["files"].items():
                folder = r.RUN / method / qid
                assert folder.is_symlink() and folder.resolve() == (Path(meta["source"]) / qid).resolve()
                for name, digest in files.items():
                    assert r.bm25.sha_file(folder / name) == digest
                imported += 1
    qs = {q["q_id"]: q for q in r.read(r.DATA / "inputs.json")}
    gold = {q["q_id"]: q for q in r.read(r.DATA / "gold_annotations.json")}
    assert set(qs) == set(gold) == set(manifest["selected_ids"])
    expected_pairs = {(m, qid) for m in r.OLD for qid in qs}
    assert expected_pairs == {(x["method"], x["q_id"]) for x in report["rows"]}
    assert len(report["rows"]) == len(qs) * len(r.OLD)
    env, digest, _ = r.tools.ChunkTools.load()
    assert digest == manifest["corpus_zip_sha256"]
    totals = {m: {"answer": Counter(), "judge": Counter()} for m in r.OLD}
    spans_verified = 0
    tool_counts = Counter()
    diagnostics = []
    errors = []
    judged = 0
    for row in report["rows"]:
        method, qid = row["method"], row["q_id"]
        q, ref = qs[qid], gold[qid]
        folder = Path(row["artifact_dir"])
        if row["state"] != "graded":
            errors.append({k: row.get(k) for k in ("method", "q_id", "state", "error", "artifact_dir")})
            continue
        if method == "naive_agent":
            response = r.read(folder / "agent/response.json")
            requests = jsonl(folder / "agent/requests.jsonl")
            events = jsonl(folder / "agent/events.jsonl")
            calls = jsonl(folder / "agent/tools.jsonl")
            thread = next(x["params"] for x in requests if x.get("method") == "thread/start")
            turn = next(x["params"] for x in requests if x.get("method") == "turn/start")
            assert turn["input"] == [{"type": "text", "text": q["question"]}]
            assert thread["baseInstructions"] == manifest["agent"]["instructions"]
            assert thread["dynamicTools"] == manifest["agent"]["tools"]
            resolved = next(x["result"] for x in events if x.get("id") == 2 and "result" in x)
            assert resolved["model"] == manifest["answer_model"] and resolved["reasoningEffort"] == "medium"
            assert response["state"] == "completed" and not response["error"] and not response["forbidden_items"]
            assert response["answer"] == row["answer"]
            dispatched = {e["params"]["callId"]: e["params"] for e in events if e.get("method") == "item/tool/call"}
            delivered = {e["params"]["item"]["id"]: e["params"]["item"] for e in events
                         if e.get("method") == "item/completed" and e["params"]["item"].get("type") == "dynamicToolCall"}
            assert response["tool_calls"] == len(calls) == len(dispatched) == len(delivered)
            regex_omissions = 0
            for c in calls:
                tool_counts[c["tool"]] += 1
                assert dispatched[c["call_id"]]["arguments"] == c["arguments"]
                assert dispatched[c["call_id"]]["tool"] == c["tool"]
                assert r.json.loads(delivered[c["call_id"]]["contentItems"][0]["text"]) == c["result"]
                if not c["success"]:
                    continue
                result = c["result"]
                if c["tool"] == "search":
                    spans = [s for item in result["results"] for s in item["excerpts"]]
                elif c["tool"] == "grep":
                    spans = result["matches"]
                    regex_omissions += "|" in c["arguments"]["pattern"] and not c["arguments"].get("regex", False)
                else:
                    spans = [result]
                for s in spans:
                    assert s["text"] == env.by_id[s["doc_id"]][s["field"]][s["start"]:s["end"]]
                    spans_verified += 1
            for source_key, target_key in (("inputTokens", "input_tokens"), ("outputTokens", "output_tokens"),
                                           ("cachedInputTokens", "cached_input_tokens"), ("reasoningOutputTokens", "reasoning_output_tokens")):
                assert sum(u["last"][source_key] for u in response["usage"]) == response["provider_total"].get(source_key, 0)
                totals[method]["answer"][target_key] += response["provider_total"].get(source_key, 0)
            diagnostics.append({"q_id": qid, "tool_calls": len(calls), "model_responses": response["model_responses"],
                                "regex_flag_omissions": regex_omissions})
        else:
            if method == "bm25":
                retrieval = r.read(folder / "retrieval.json")
                ids = [x["doc_id"] for x in r.bm25.search(q["question"], env.retriever, env.docs, k=100)[:10]]
                assert ids == retrieval["context_order"] and len(set(ids)) == 10
            else:
                ids = ref["doc_ids"]
            prompt = r.common.answer_prompt(q["question"], [env.by_id[i] for i in ids])
            if args.answer_model and method == "bm25":
                assert prompt == (Path(manifest["parent_run"]) / "bm25" / qid / "answer/prompt.txt").read_text()
            if method == "gold":
                response = cli_response(folder / "answer", prompt)
            else:
                response = r.read(folder / "answer/response.json")
                assert all(response[k] == v for k, v in r.long_transport.identity(prompt).items())
                assert (folder / "answer/prompt.txt").read_text() == prompt
                attempt = folder / "answer" / f"attempt_{response['attempt']:02d}"
                valid, messages, usage, forbidden, window = r.long_transport.inspect_events(jsonl(attempt / "events.jsonl"))
                assert valid and not forbidden and messages[0]["text"] == response["text"] and usage == response["usage"]
                requests = jsonl(attempt / "request.jsonl")
                injected = next(x["params"] for x in requests if x.get("method") == "thread/inject_items")
                assert injected["items"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]
            assert response["text"] == row["answer"]
            for u in response["usage"]:
                totals[method]["answer"].update(u)
        points = ref["key_points"] if isinstance(ref["key_points"], list) else [ref["key_points"]]
        prompt = r.common.load_judge()(q["question"], ref["answer"], row["answer"], points)
        judge = cli_response(folder / "judge", prompt)
        assert r.common.parse_judge(judge["text"], len(points))["verdict"] == row["verdict"]
        for u in judge["usage"]:
            totals[method]["judge"].update(u)
        judged += 1
    for method, tasks in report["tasks"].items():
        for task, stats in tasks.items():
            selected = [x for x in report["rows"] if x["method"] == method and (task == "ALL" or x["task"] == task)]
            assert stats["selected"] == len(selected)
            assert stats["correct"] == sum(x.get("verdict") == "CORRECT" for x in selected)
            assert stats["accuracy"] == stats["correct"] / len(selected)
    result = {"passed": True, "checked_at": r.common.now(), "questions": 50, "conditions": list(r.OLD),
              "answer_model": manifest["answer_model"], "judge_model": manifest["judge_model"],
              "imported_pairs_unchanged": imported, "graded_pairs_verified": judged,
              "source_spans_verified": spans_verified, "tool_counts": dict(tool_counts), "execution_errors": errors,
              "usage_successful_pairs_including_reused": totals,
              "checks": ["same frozen cohort", "original imported artifacts unchanged", "source snapshots unchanged",
                         "exact full original static context", "agent question only and unchanged tools/policy",
                         "actual provider model/effort", "tool trace exact source spans", "official judge and score recount"]}
    r.common.write(r.RUN / "validation.json", result, frozen=True)
    r.common.write(r.RUN / "trace_diagnostics.json", diagnostics, frozen=True)
    out = io.StringIO()
    fields = ["q_id", "task", *r.OLD]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for qid, q in qs.items():
        writer.writerow({"q_id": qid, "task": q["task"], **{
            m: next(x for x in report["rows"] if x["method"] == m and x["q_id"] == qid).get("verdict", "EXECUTION_ERROR")
            for m in r.OLD}})
    r.common.freeze_text(r.RUN / "paired_comparison.csv", out.getvalue())
    print(r.json.dumps({k: v for k, v in result.items() if k != "usage_successful_pairs_including_reused"}))


if __name__ == "__main__":
    main()
