"""First LT and EC questions from frozen gold20; native tools, then original judge."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import re
import signal

import run_finrate_gold20_gpt56_luna_judge6_v1 as common
import finrate_bm25_top10_v1 as bm25
import finrate_chunk_agent_tools_v1 as tool_module
import finrate_chunk_agent_transport_v1 as agent

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/finrate_chunk_agent_v1"
RUN = ROOT / "result/finrate_chunk_agent_smoke_v1"
ORIGINAL = ROOT / "data/finrate_gold20_gpt6_luna_v1"


def prepare(environment, corpus_sha, index_manifest):
    original = json.loads((ORIGINAL / "inputs.json").read_text())
    questions = [{k: q[k] for k in ("q_id", "task", "question")} for q in original]
    assert Counter(q["task"] for q in questions) == {"LT-QA": 10, "EC-QA": 10}
    smoke = [next(q for q in questions if q["task"] == task) for task in ("LT-QA", "EC-QA")]
    common.write(COHORT / "inputs.json", questions, frozen=True)
    common.write(COHORT / "manifest.json", {
        "parent": str(ORIGINAL), "parent_inputs_sha256": bm25.sha_file(ORIGINAL / "inputs.json"),
        "question_ids": [q["q_id"] for q in questions], "count": 20,
        "selection": "Exact existing gold20 cohort; remove gold_doc_ids from agent-facing inputs",
        "smoke_ids": [q["q_id"] for q in smoke],
        "smoke_selection": "First existing question per task, before running the agent; no outcome selection",
    }, frozen=True)
    sources = [Path(__file__), Path(agent.__file__), Path(tool_module.__file__), Path(bm25.__file__), Path(common.__file__)]
    manifest = {
        "policy": agent.POLICY, "instructions": agent.INSTRUCTIONS, "isolation": agent.ISOLATION,
        "tools": tool_module.TOOLS, "question_ids": [q["q_id"] for q in questions],
        "smoke_ids": [q["q_id"] for q in smoke], "corpus_chunks": len(environment.docs),
        "corpus_zip_sha256": corpus_sha, "index_policy": index_manifest["policy"],
        "cohort_inputs_sha256": bm25.sha_file(COHORT / "inputs.json"),
        "source_sha256": {p.name: bm25.sha_file(p) for p in sources},
        "judge_model": "gpt-6-luna", "judge_reasoning_effort": "medium",
        "judge_source_sha256": bm25.sha_file(common.REFERENCE / "evaluation/qa_llm_judge.py"),
        "judge_has_raw_context": False, "judge_prompt": "unchanged official create_kp_judge_prompt",
        "smoke_workers": 2, "full20_started": False,
        "protocol_reference": "https://learn.chatgpt.com/docs/app-server#dynamic-tool-calls-experimental",
        "trace_scope": "All sent RPC requests/responses, received provider events, tool arguments/exact results, messages and provider usage; hidden internal reasoning is not exposed",
    }
    common.write(RUN / "manifest.json", manifest, frozen=True)
    for path in sources:
        common.freeze_text(RUN / "source_snapshot" / path.name, path.read_text())
    return questions, smoke


def evidence_audit(answer, calls, gold_ids):
    body_spans = []
    returned_ids = set()
    for call in calls:
        if not call["success"]:
            continue
        result = call["result"]
        if call["tool"] == "search":
            returned_ids.update(row["doc_id"] for row in result["results"])
            body_spans.extend(span for row in result["results"] for span in row["excerpts"])
        elif call["tool"] == "grep":
            returned_ids.update(span["doc_id"] for span in result["matches"])
            body_spans.extend(span for span in result["matches"] if span["field"] == "text")
        else:
            returned_ids.add(result["doc_id"])
            body_spans.append({k: result[k] for k in ("doc_id", "field", "start", "end", "text")})
    body_ids = {span["doc_id"] for span in body_spans if span["end"] > span["start"]}
    cited = set(re.findall(r"\bdoc_\d{6}\b", answer or ""))
    return {"returned_ids": sorted(returned_ids), "body_seen_ids": sorted(body_ids),
            "read_ids": sorted({c["arguments"]["doc_id"] for c in calls if c["tool"] == "read" and c["success"]}),
            "cited_ids": sorted(cited), "cited_without_body_seen": sorted(cited - body_ids),
            "gold_ids": gold_ids, "gold_returned": sorted(set(gold_ids) & returned_ids),
            "gold_body_seen": sorted(set(gold_ids) & body_ids),
            "body_spans": body_spans,
            "interpretation": "Partial body exposure is not full gold evidence coverage; citation existence is not entailment"}


def worker(q, environment):
    folder = RUN / q["q_id"]
    row = {"q_id": q["q_id"], "task": q["task"], "question": q["question"]}
    try:
        generated = agent.invoke(q["question"], environment, folder / "agent")
        row.update({k: generated[k] for k in ("state", "error", "answer", "tool_calls", "tool_errors", "model_responses", "provider_total")})
        common.write(folder / "prediction.json", row, frozen=True)
        if generated["state"] != "completed":
            return row
        # Gold is loaded only after inference, never put into the agent tools/context.
        gold = {g["q_id"]: g for g in json.loads((ORIGINAL / "gold_annotations.json").read_text())}
        ref = gold[q["q_id"]]
        calls = [json.loads(line) for line in (folder / "agent/tools.jsonl").read_text().splitlines()]
        audit = evidence_audit(generated["answer"], calls, ref["doc_ids"])
        common.write(folder / "evidence_audit.json", audit, frozen=True)
        points = ref["key_points"]
        if isinstance(points, str):
            points = [points]
        prompt = common.load_judge()(q["question"], ref["answer"], generated["answer"], points)
        assessed = common.invoke(prompt, folder / "judge")
        row.update(common.parse_judge(assessed["text"], len(points)))
        row.update(state="graded", judge_usage=assessed["usage"], judge_response=assessed["text"],
                   gold_returned=audit["gold_returned"], gold_body_seen=audit["gold_body_seen"],
                   gold_count=len(ref["doc_ids"]), cited_without_body_seen=audit["cited_without_body_seen"])
        common.write(folder / "assessment.json", row, frozen=True)
    except Exception as exc:
        row.update(state="execution_error", error=f"{type(exc).__name__}: {exc}")
        common.write(folder / "error.json", row)
    print(json.dumps({k: row.get(k) for k in ("q_id", "state", "verdict", "tool_calls", "error")}), flush=True)
    return row


def report(smoke, rows, final=False):
    payload = {"state": "completed" if final else "running", "updated": common.now(),
               "selected": len(smoke), "processed": len(rows), "graded": sum(r["state"] == "graded" for r in rows),
               "labels": dict(Counter(r["verdict"] for r in rows if r.get("verdict"))),
               "rows": rows, "scope": "two-question smoke only, not the full20 accuracy"}
    if final and any(r["state"] != "graded" for r in rows):
        payload["state"] = "completed_with_errors"
    common.write(RUN / "report.json", payload)
    common.write(RUN / "status.json", {k: v for k, v in payload.items() if k != "rows"})
    if final:
        lines = ["FIN-RATE naive chunk agent smoke", "Model / judge: gpt-6-luna / medium", "Fixed cohort20; smoke: first LT + first EC."]
        for r in rows:
            lines.append(f"{r['task']} {r['q_id']}: {r.get('verdict', r['state'])}; tool calls {r.get('tool_calls')}; model responses {r.get('model_responses')}")
        common.freeze_text(RUN / "report.txt", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common.RUN = RUN
        common.JUDGE_MODEL = "gpt-6-luna"
        def stop(signum, frame):
            common.STOP.set()
            with common.LOCK:
                for proc in list(common.ACTIVE.values()):
                    common.stop_process(proc)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        environment, corpus_sha, index_manifest = tool_module.ChunkTools.load()
        _, smoke = prepare(environment, corpus_sha, index_manifest)
        print(json.dumps({"prepared": True, "smoke_ids": [q["q_id"] for q in smoke]}), flush=True)
        if args.prepare_only:
            return
        common.write(RUN / "started.json", {"pid": os.getpid(), "started": common.now()}, frozen=True)
        rows = []
        report(smoke, rows)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, q, environment) for q in smoke]
            for future in as_completed(futures):
                rows.append(future.result())
                report(smoke, rows)
        rows.sort(key=lambda r: [q["q_id"] for q in smoke].index(r["q_id"]))
        report(smoke, rows, final=True)


if __name__ == "__main__":
    main()
