"""Expand the three completed original-chunk conditions to one frozen LT25/EC25 set.

No old answer is regenerated. Other planned, unimplemented conditions are not
silently counted as evaluated. Each worker generates then immediately judges.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal

import finrate_bm25_top10_v1 as bm25
import finrate_bm25_appserver_transport_v1 as long_transport
import finrate_chunk_agent_tools_v1 as tools
import finrate_chunk_agent_transport_v2 as agent
import run_finrate_gold20_gpt56_luna_judge6_v1 as common
import run_finrate_chunk_agent_smoke_v2 as smoke

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "data/finrate_gold_poc_gpt6_luna_v2"
DATA = ROOT / "data/finrate_baseline50_v1"
RUN = ROOT / "result/finrate_baseline50_gpt6_luna_v1"
OLD = {
    "bm25": ROOT / "result/finrate_bm25_top10_gpt6_luna_v3",
    "gold": ROOT / "result/finrate_gold20_gpt6_luna_v1",
    "naive_agent": ROOT / "result/finrate_chunk_agent20_gpt6_luna_v1",
}
WORKERS = 20


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    common.RUN = RUN
    common.MODEL = common.JUDGE_MODEL = "gpt-6-luna"
    common.WORKERS = WORKERS


def prepare(env, corpus_sha, index_manifest):
    common.check_disk()
    parent = read(PARENT / "inputs.json")
    questions = [{k: q[k] for k in ("q_id", "task", "question")}
                 for task in ("LT-QA", "EC-QA")
                 for q in [q for q in parent if q["task"] == task][:25]]
    assert Counter(q["task"] for q in questions) == {"LT-QA": 25, "EC-QA": 25}
    gold = {q["q_id"]: q for q in read(PARENT / "gold_annotations.json")}
    old_questions = read(ROOT / "data/finrate_gold20_gpt6_luna_v1/inputs.json")
    assert {q["q_id"] for q in old_questions} <= {q["q_id"] for q in questions}
    for q in questions:
        assert q["question"] == gold[q["q_id"]]["question"]
    common.write(DATA / "inputs.json", questions, frozen=True)
    common.write(DATA / "gold_annotations.json", [gold[q["q_id"]] for q in questions], frozen=True)
    common.write(DATA / "manifest.json", {
        "selection": "First 25 per task in frozen random v2 pool order, without outcome selection",
        "parent": str(PARENT), "parent_manifest_sha256": bm25.sha_file(PARENT / "manifest.json"),
        "parent_inputs_sha256": bm25.sha_file(PARENT / "inputs.json"),
        "counts": dict(Counter(q["task"] for q in questions)),
        "question_ids": [q["q_id"] for q in questions],
        "retained_original20": [q["q_id"] for q in old_questions],
    }, frozen=True)
    imports = {}
    for method, source in OLD.items():
        imports[method] = {"source": str(source), "manifest_sha256": bm25.sha_file(source / "manifest.json"), "files": {}}
        for q in old_questions:
            qid = q["q_id"]
            assessment = read(source / qid / "assessment.json")
            assert assessment.get("state", assessment.get("status")) == "graded"
            assert assessment["task"] == q["task"]
            target = RUN / method / qid
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(source / qid, target_is_directory=True)
            assert target.is_symlink() and target.resolve() == (source / qid).resolve()
            imports[method]["files"][qid] = {
                str(p.relative_to(source / qid)): bm25.sha_file(p)
                for p in sorted((source / qid).rglob("*")) if p.is_file()
            }
    common.write(RUN / "imports.json", imports, frozen=True)
    sources = [Path(__file__), Path(bm25.__file__), Path(common.__file__),
               Path(long_transport.__file__), Path(agent.__file__), Path(tools.__file__), Path(smoke.__file__)]
    manifest = {
        "scope": "Three implemented conditions expanded to 50 total; additional planned baselines are separate work",
        "methods": list(OLD), "selected_ids": [q["q_id"] for q in questions],
        "counts": {"LT-QA": 25, "EC-QA": 25}, "workers_total": WORKERS,
        "answer_model": "gpt-6-luna", "judge_model": "gpt-6-luna", "reasoning_effort": "medium",
        "schedule": "One shared worker pool across all methods; judge immediately after each answer",
        "corpus_chunks": len(env.docs), "corpus_zip_sha256": corpus_sha,
        "inputs_sha256": bm25.sha_file(DATA / "inputs.json"),
        "annotations_sha256": bm25.sha_file(DATA / "gold_annotations.json"),
        "imports_sha256": bm25.sha_file(RUN / "imports.json"),
        "bm25": {"policy": index_manifest["policy"], "top_k": 10, "reranker": None,
                 "context": "Full original bodies in rank order", "prompt_token_limit": 780000,
                 "transport": "unchanged app-server history input", "trigger": long_transport.TRIGGER},
        "gold": {"context": "All annotated gold bodies in annotated order, no reference answer",
                 "prompt_token_limit": 240000, "transport": "unchanged isolated CLI"},
        "agent": {"policy": agent.POLICY, "instructions": agent.INSTRUCTIONS,
                  "isolation": agent.ISOLATION, "tools": tools.TOOLS},
        "judge_source_sha256": bm25.sha_file(common.REFERENCE / "evaluation/qa_llm_judge.py"),
        "source_sha256": {p.name: bm25.sha_file(p) for p in sources},
        "completed_answer_retry": False,
        "retry_policy": "Static transport attempts at most two; transient agent failure without answer once after initial grid",
        "context_overflow": "Record execution error, never truncate or replace question",
        "local_gpu": False, "disk_stop_bytes": common.MIN_FREE,
        "trace": "All provider-visible requests/events, tool arguments/results, final answers, judgments, actual usage",
    }
    common.write(RUN / "manifest.json", manifest, frozen=True)
    for source in sources:
        common.freeze_text(RUN / "source_snapshot" / source.name, source.read_text())
    return questions, gold


def worker(method, q, gold, env, retry=False):
    qid = q["q_id"]
    folder = RUN / method / ("retry_01" if retry else "") / qid
    row = {"method": method, "q_id": qid, "task": q["task"], "artifact_dir": str(folder)}
    try:
        common.check_disk()
        if common.STOP.is_set():
            raise RuntimeError("Stopped")
        if (folder / "assessment.json").exists():
            old = read(folder / "assessment.json")
            return {**old, **row, "state": "graded", "reused": folder.is_symlink()}
        ref = gold[qid]
        if method == "naive_agent":
            cached = folder / "agent/response.json"
            generated = read(cached) if cached.exists() else agent.invoke(q["question"], env, folder / "agent")
            row.update({k: generated[k] for k in ("state", "error", "answer", "tool_calls", "tool_errors", "model_responses", "provider_total")})
            common.write(folder / "prediction.json", row, frozen=True)
            if generated["state"] != "completed":
                return row
            calls = [json.loads(line) for line in (folder / "agent/tools.jsonl").read_text().splitlines()]
            audit = smoke.evidence_audit(row["answer"], calls, ref["doc_ids"])
            common.write(folder / "evidence_audit.json", audit, frozen=True)
        else:
            if method == "bm25":
                ranking = bm25.search(q["question"], env.retriever, env.docs, k=100)
                ids = [r["doc_id"] for r in ranking[:10]]
                common.write(folder / "retrieval.json", {"query": q["question"], "ranked_candidates": ranking,
                             "context_order": ids, "context_truncated": False}, frozen=True)
                invoke, limit = long_transport.invoke, 780000
            else:
                ids = ref["doc_ids"]
                invoke, limit = common.invoke, 240000
            prompt = common.answer_prompt(q["question"], [env.by_id[i] for i in ids])
            count = len(common.ENC.encode(prompt, disallowed_special=()))
            row.update(context_doc_ids=ids, prompt_tokens_estimate=count, prompt_sha256=common.sha(prompt))
            common.write(folder / "input_manifest.json", row, frozen=True)
            if count > limit:
                raise ValueError(f"input_length_error: {count} > {limit}; full source preserved")
            generated = invoke(prompt, folder / "answer")
            row.update(answer=generated["text"], answer_usage=generated["usage"])
            common.write(folder / "prediction.json", row, frozen=True)
        points = ref["key_points"]
        if isinstance(points, str):
            points = [points]
        prompt = common.load_judge()(q["question"], ref["answer"], row["answer"], points)
        judged = common.invoke(prompt, folder / "judge")
        row.update(common.parse_judge(judged["text"], len(points)))
        row.update(state="graded", judge_usage=judged["usage"], judge_response=judged["text"])
        common.write(folder / "assessment.json", row, frozen=True)
    except Exception as exc:
        row.update(state="execution_error", error=f"{type(exc).__name__}: {exc}")
        if not folder.is_symlink():
            common.write(folder / "error.json", row)
    print(json.dumps({k: row.get(k) for k in ("method", "q_id", "state", "verdict", "error")}), flush=True)
    return row


def save(questions, rows, phase):
    summary = {}
    for method in OLD:
        summary[method] = {}
        for task in ("LT-QA", "EC-QA", "ALL"):
            selected = [q for q in questions if task == "ALL" or q["task"] == task]
            subset = [rows[(method, q["q_id"])] for q in selected if (method, q["q_id"]) in rows]
            labels = Counter(r.get("verdict") for r in subset if r.get("state") == "graded")
            summary[method][task] = {"selected": len(selected), "processed": len(subset),
                "graded": sum(labels.values()), "labels": dict(labels), "correct": labels["CORRECT"],
                "accuracy": labels["CORRECT"] / len(selected) if len(subset) == len(selected) else None,
                "execution_errors": sum(r.get("state") != "graded" for r in subset)}
    status = {"state": phase, "updated": common.now(), "pid": os.getpid(), "workers": WORKERS,
              "selected_questions": len(questions), "selected_pairs": len(questions) * len(OLD),
              "processed": len(rows), "graded": sum(r.get("state") == "graded" for r in rows.values()),
              "tasks": summary}
    common.write(RUN / "status.json", status)
    common.write(RUN / "report.json", {**status, "rows": list(rows.values())})
    if phase != "running":
        out = io.StringIO()
        fields = ["method", "q_id", "task", "state", "verdict", "error", "artifact_dir"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in rows.values())
        common.freeze_text(RUN / "results.csv", out.getvalue())
        lines = ["FIN-RATE LT25 + EC25; GPT-6 Luna answer and judge", f"State: {phase}"]
        for method in OLD:
            lines.append(method + ": " + json.dumps(summary[method], ensure_ascii=False))
        common.freeze_text(RUN / "report.txt", "\n".join(lines) + "\n")


def retryable(row):
    if row.get("answer") or row.get("state") == "graded" or common.STOP.is_set():
        return False
    error = (row.get("error") or "").lower()
    if any(s in error for s in ("budget", "unexpected", "compaction", "router", "mismatch")):
        return False
    return any(s in error for s in ("timeout", "capacity", "429", "rate limit", "connection", "temporarily unavailable"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    configure()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame):
            common.STOP.set()
            with common.LOCK:
                for proc in list(common.ACTIVE.values()):
                    common.stop_process(proc)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        env, digest, index_manifest = tools.ChunkTools.load()
        qs, gold = prepare(env, digest, index_manifest)
        print(json.dumps({"prepared": True, "questions": len(qs), "methods": list(OLD)}), flush=True)
        if args.prepare_only:
            return
        common.write(RUN / "started.json", {"pid": os.getpid(), "started": common.now()}, frozen=True)
        rows = {}
        jobs = []
        for q in qs:
            for method in OLD:
                if (RUN / method / q["q_id"] / "assessment.json").exists():
                    rows[(method, q["q_id"])] = worker(method, q, gold, env)
                else:
                    jobs.append((method, q))
        save(qs, rows, "running")
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(worker, m, q, gold, env): (m, q["q_id"]) for m, q in jobs}
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
                save(qs, rows, "running")
        retry = [q for q in qs if retryable(rows[("naive_agent", q["q_id"])])]
        if retry:
            common.write(RUN / "retry_plan.json", {"ids": [q["q_id"] for q in retry],
                         "reason": "Transient transport failure with no completed answer"}, frozen=True)
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(worker, "naive_agent", q, gold, env, True): q for q in retry}
                for future in as_completed(futures):
                    rows[("naive_agent", futures[future]["q_id"])] = future.result()
                    save(qs, rows, "running")
        phase = "stopped" if common.STOP.is_set() else ("completed" if all(r.get("state") == "graded" for r in rows.values()) else "completed_with_errors")
        save(qs, rows, phase)
        print(json.dumps({"state": phase}), flush=True)


if __name__ == "__main__":
    main()
