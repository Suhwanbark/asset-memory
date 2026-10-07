"""Extend validated smoke2 to the unchanged full20; preserve every original attempt."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import signal

import run_finrate_chunk_agent_smoke_v2 as smoke
import finrate_chunk_agent_transport_v2 as agent
import finrate_chunk_agent_tools_v1 as tool_module
import finrate_bm25_top10_v1 as bm25
import run_finrate_gold20_gpt56_luna_judge6_v1 as common

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "result/finrate_chunk_agent20_gpt6_luna_v1"
SMOKE = ROOT / "result/finrate_chunk_agent_smoke_v2"
COHORT = ROOT / "data/finrate_chunk_agent_v1"
WORKERS = 20


def prepare(environment, corpus_sha):
    original = json.loads((smoke.ORIGINAL / "inputs.json").read_text())
    questions = json.loads((COHORT / "inputs.json").read_text())
    assert questions == [{k: q[k] for k in ("q_id", "task", "question")} for q in original]
    source = json.loads((SMOKE / "manifest.json").read_text())
    assert json.loads((SMOKE / "validation.json").read_text())["passed"]
    assert source["policy"] == agent.POLICY
    assert source["instructions"] == agent.INSTRUCTIONS and source["isolation"] == agent.ISOLATION
    assert source["tools"] == tool_module.TOOLS and source["corpus_zip_sha256"] == corpus_sha
    for name, digest in source["source_sha256"].items():
        assert bm25.sha_file(ROOT / "method" / name) == digest
    assert source["question_ids"] == [q["q_id"] for q in questions]
    imports = {}
    for qid in source["smoke_ids"]:
        src, dst = SMOKE / qid, RUN / qid
        files = {str(p.relative_to(src)): bm25.sha_file(p) for p in src.rglob("*") if p.is_file()}
        if not dst.exists():
            shutil.copytree(src, dst)
        assert all(bm25.sha_file(dst / name) == digest for name, digest in files.items())
        imports[qid] = {"source": str(src), "files_sha256": files}
    sources = [Path(__file__), Path(smoke.__file__), Path(agent.__file__),
               Path(tool_module.__file__), Path(bm25.__file__), Path(common.__file__)]
    manifest = {**source, "scope": "full original20, LT10 and EC10", "full20_started": True,
                "workers": WORKERS, "smoke_manifest_sha256": bm25.sha_file(SMOKE / "manifest.json"),
                "reused_smoke": imports, "source_sha256": {p.name: bm25.sha_file(p) for p in sources},
                "retry_policy": "After all questions, one retry only for transient transport errors without a completed answer; no budget/content retries",
                "no_method_changes_after_smoke": True, "corpus_chunks": len(environment.docs)}
    common.write(RUN / "manifest.json", manifest, frozen=True)
    for p in sources:
        common.freeze_text(RUN / "source_snapshot" / p.name, p.read_text())
    return questions, list(imports)


def save_report(questions, rows, phase):
    tasks = {}
    for task in ("LT-QA", "EC-QA", "ALL"):
        qs = [q for q in questions if task == "ALL" or q["task"] == task]
        rr = [rows[q["q_id"]] for q in qs if q["q_id"] in rows]
        labels = Counter(r["verdict"] for r in rr if r.get("state") == "graded")
        tasks[task] = {"selected": len(qs), "processed": len(rr), "graded": sum(labels.values()),
                       "correct": labels["CORRECT"], "labels": dict(labels),
                       "accuracy": labels["CORRECT"] / len(qs),
                       "execution_errors": sum(r.get("state") != "graded" for r in rr),
                       "pending": len(qs) - len(rr)}
    totals = {"agent_input_tokens": 0, "agent_output_tokens": 0,
              "agent_cached_input_tokens": 0, "judge_input_tokens": 0, "judge_output_tokens": 0}
    for response_path in RUN.glob("**/agent/response.json"):
        response = json.loads(response_path.read_text())
        usage = response.get("provider_total", {})
        totals["agent_input_tokens"] += usage.get("inputTokens", 0)
        totals["agent_output_tokens"] += usage.get("outputTokens", 0)
        totals["agent_cached_input_tokens"] += usage.get("cachedInputTokens", 0)
    # Count all completed judge attempts, including any failed transport attempt costs.
    for path in RUN.glob("**/judge/attempt_*/outcome.json"):
        for usage in json.loads(path.read_text()).get("usage", []):
            totals["judge_input_tokens"] += usage.get("input_tokens", 0)
            totals["judge_output_tokens"] += usage.get("output_tokens", 0)
    payload = {"state": phase, "updated": common.now(), "model": "gpt-6-luna", "judge_model": "gpt-6-luna",
               "reasoning_effort": "medium", "workers": WORKERS, "tasks": tasks,
               "provider_usage": totals, "rows": [rows[q["q_id"]] for q in questions if q["q_id"] in rows],
               "reused_smoke_count": 2, "prompt_tools_budget_unchanged": True}
    common.write(RUN / "report.json", payload)
    common.write(RUN / "status.json", {"state": phase, "updated": common.now(), "pid": os.getpid(),
        **tasks["ALL"], "active_model_processes": len(common.ACTIVE),
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 2)})
    if phase.startswith("completed"):
        lines = ["FIN-RATE naive chunk agent20", "Answer / judge: gpt-6-luna / medium", "Same original20; smoke2 reused byte-identically; remaining18 newly evaluated."]
        for task, data in tasks.items():
            lines.append(f"{task}: CORRECT {data['correct']}/{data['selected']} ({data['accuracy']:.0%}); labels {data['labels']}; execution_errors {data['execution_errors']}")
        lines.append(f"Provider usage including reused smoke and all recorded attempts: {totals}")
        common.freeze_text(RUN / "report.txt", "\n".join(lines) + "\n")
        stream = io.StringIO()
        fields = ["q_id", "task", "state", "verdict", "tool_calls", "tool_errors", "model_responses", "error", "artifact_dir"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in payload["rows"])
        common.freeze_text(RUN / "results.csv", stream.getvalue())
    return payload


def transient_retry(row):
    if row.get("state") == "graded" or row.get("answer") or common.STOP.is_set():
        return False
    error = (row.get("error") or "").lower()
    if any(s in error for s in ("budget", "unexpected", "compaction", "router", "mismatch")):
        return False
    return any(s in error for s in ("timeout", "capacity", "429", "rate limit", "connection", "temporarily unavailable"))


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / "started.json").exists():
            raise RuntimeError("Existing run retained; inspect before any resume")
        common.RUN = RUN
        common.JUDGE_MODEL = "gpt-6-luna"
        smoke.RUN = RUN
        def stop(signum, frame):
            common.STOP.set()
            with common.LOCK:
                for proc in list(common.ACTIVE.values()):
                    common.stop_process(proc)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        environment, corpus_sha, _ = tool_module.ChunkTools.load()
        questions, reused = prepare(environment, corpus_sha)
        rows = {qid: {**json.loads((RUN / qid / "assessment.json").read_text()),
                      "artifact_dir": str(RUN / qid), "reused_smoke": True} for qid in reused}
        common.write(RUN / "started.json", {"started": common.now(), "pid": os.getpid(), "new_questions": 18}, frozen=True)
        save_report(questions, rows, "running")
        print(json.dumps({"state": "started", "new": 18, "reused": 2, "workers": WORKERS}), flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(smoke.worker, q, environment): q for q in questions if q["q_id"] not in reused}
            for future in as_completed(futures):
                q = futures[future]
                row = future.result()
                row.update(artifact_dir=str(RUN / q["q_id"]), reused_smoke=False)
                rows[q["q_id"]] = row
                save_report(questions, rows, "running")
        retry_questions = [q for q in questions if transient_retry(rows[q["q_id"]])]
        if retry_questions:
            retry_root = RUN / "retry_01"
            smoke.RUN = retry_root
            common.write(RUN / "retry_plan.json", {"question_ids": [q["q_id"] for q in retry_questions],
                         "reason": "transient transport failure, no completed answer", "started": common.now()}, frozen=True)
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(smoke.worker, q, environment): q for q in retry_questions}
                for future in as_completed(futures):
                    q = futures[future]
                    row = future.result()
                    row.update(artifact_dir=str(retry_root / q["q_id"]), reused_smoke=False, retry_of=str(RUN / q["q_id"]))
                    rows[q["q_id"]] = row
                    save_report(questions, rows, "retrying_transport_failures")
        final = "completed" if all(r["state"] == "graded" for r in rows.values()) else "completed_with_errors"
        save_report(questions, rows, final)
        print(json.dumps({"state": final}), flush=True)


if __name__ == "__main__":
    main()
