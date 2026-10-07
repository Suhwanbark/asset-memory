"""Controlled answer-model comparison; frozen LT25/EC25 and Luna judge unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import io
import os
from pathlib import Path
import signal

import run_finrate_baseline50_v1 as r
import finrate_transport_review_v1 as review

PARENT = r.ROOT / "result/finrate_keyword_agent50_gpt6_luna_v1"
METHODS = ("bm25", "naive_agent")
MODEL = None
RUN = None


def configure(model):
    global MODEL, RUN
    MODEL = model
    RUN = r.ROOT / f"result/finrate_keyword_agent50_{model.replace('-', '_')}_judge6_v1"
    r.RUN = RUN
    r.OLD = {m: PARENT / m for m in METHODS}
    r.configure()
    r.common.MODEL = model
    r.common.JUDGE_MODEL = "gpt-6-luna"
    r.long_transport.MODEL = model
    r.agent.MODEL = model
    r.agent.POLICY = {**r.agent.POLICY, "model": model}
    review.install()


def prepare(env, digest):
    r.common.check_disk()
    qs = r.read(r.DATA / "inputs.json")
    gold = {x["q_id"]: x for x in r.read(r.DATA / "gold_annotations.json")}
    parent = r.read(PARENT / "manifest.json")
    assert r.read(PARENT / "validation.json")["passed"]
    assert digest == parent["corpus_zip_sha256"]
    assert [q["q_id"] for q in qs] == parent["selected_ids"]
    assert {**r.agent.POLICY, "model": parent["agent"]["policy"]["model"]} == parent["agent"]["policy"]
    assert r.agent.INSTRUCTIONS == parent["agent"]["instructions"]
    assert r.tools.TOOLS == parent["agent"]["tools"]
    prompts = {}
    for q in qs:
        r.common.check_disk()
        ranking = r.bm25.search(q["question"], env.retriever, env.docs, k=100)
        ids = [x["doc_id"] for x in ranking[:10]]
        prompt = r.common.answer_prompt(q["question"], [env.by_id[i] for i in ids])
        old = PARENT / "bm25" / q["q_id"] / "answer/prompt.txt"
        assert prompt == old.read_text()
        prompts[q["q_id"]] = {"doc_ids": ids, "prompt_sha256": r.common.sha(prompt)}
    smoke = [next(q for q in qs if q["task"] == task)["q_id"] for task in ("LT-QA", "EC-QA")]
    sources = [Path(__file__), Path(r.__file__), Path(review.__file__), Path(r.bm25.__file__),
               Path(r.common.__file__), Path(r.long_transport.__file__), Path(r.agent.__file__),
               Path(r.tools.__file__), Path(r.smoke.__file__)]
    manifest = {
        "scope": "Same 50 questions and two methods; change only answer model, retain GPT-6 Luna judge",
        "answer_model": MODEL, "judge_model": "gpt-6-luna", "reasoning_effort": "medium",
        "parent_run": str(PARENT), "parent_manifest_sha256": r.bm25.sha_file(PARENT / "manifest.json"),
        "parent_report_sha256": r.bm25.sha_file(PARENT / "report.json"),
        "selected_ids": [q["q_id"] for q in qs], "counts": {"LT-QA": 25, "EC-QA": 25},
        "methods": list(METHODS), "workers_total": 20, "smoke_ids": smoke,
        "smoke_policy": "First frozen question per task and method; functional checks only, no quality gate; reuse these results in full50",
        "inputs_sha256": r.bm25.sha_file(r.DATA / "inputs.json"),
        "annotations_sha256": r.bm25.sha_file(r.DATA / "gold_annotations.json"),
        "corpus_zip_sha256": digest, "corpus_chunks": len(env.docs),
        "bm25": parent["bm25"], "identical_bm25_prompts": prompts,
        "agent": {"policy": r.agent.POLICY, "instructions": r.agent.INSTRUCTIONS,
                  "isolation": r.agent.ISOLATION, "tools": r.tools.TOOLS},
        "judge_source_sha256": parent["judge_source_sha256"],
        "schedule": "Generation then immediate judge in one shared pool of20",
        "source_sha256": {p.name: r.bm25.sha_file(p) for p in sources},
        "trace": parent["trace"], "local_gpu": False, "disk_stop_bytes": r.common.MIN_FREE,
        "quality_retry": False, "gold_context_new_calls": False,
        "retry_policy": "Only transient no-answer agent errors once after all initial jobs; original attempts retained",
        "model_guidance": "https://learn.chatgpt.com/docs/models",
        "migration_guidance": "https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md#migration-quickstart",
    }
    r.common.write(RUN / "manifest.json", manifest, frozen=True)
    r.common.write(RUN / "preflight.json", {"passed": True, "same50_questions": True,
                   "identical_bm25_full_prompts": len(prompts), "agent_only_model_changed": True,
                   "judge_unchanged": True}, frozen=True)
    for p in sources:
        r.common.freeze_text(RUN / "source_snapshot" / p.name, p.read_text())
    return qs, gold, smoke


def save(qs, rows, phase):
    tasks = {}
    for method in METHODS:
        tasks[method] = {}
        for task in ("LT-QA", "EC-QA", "ALL"):
            selected = [q for q in qs if task == "ALL" or q["task"] == task]
            subset = [rows[(method, q["q_id"])] for q in selected if (method, q["q_id"]) in rows]
            labels = Counter(x.get("verdict") for x in subset if x.get("state") == "graded")
            tasks[method][task] = {"selected": len(selected), "processed": len(subset),
                "graded": sum(labels.values()), "labels": dict(labels), "correct": labels["CORRECT"],
                "accuracy": labels["CORRECT"] / len(selected) if len(subset) == len(selected) else None,
                "execution_errors": sum(x.get("state") != "graded" for x in subset)}
    status = {"state": phase, "updated": r.common.now(), "pid": os.getpid(),
              "answer_model": MODEL, "judge_model": "gpt-6-luna", "workers": 20,
              "selected_questions": len(qs), "selected_pairs": len(qs) * 2,
              "processed": len(rows), "graded": sum(x.get("state") == "graded" for x in rows.values()),
              "tasks": tasks}
    r.common.write(RUN / "status.json", status)
    r.common.write(RUN / "report.json", {**status, "rows": list(rows.values())})
    if phase.startswith("completed") or phase in {"stopped", "smoke_transport_error"}:
        out = io.StringIO()
        fields = ["method", "q_id", "task", "state", "verdict", "error", "artifact_dir"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: x.get(k, "") for k in fields} for x in rows.values())
        r.common.freeze_text(RUN / "results.csv", out.getvalue())
        lines = [f"FIN-RATE LT25 + EC25; answer {MODEL}; judge gpt-6-luna; medium", f"State: {phase}"]
        for method in METHODS:
            lines.append(method + ": " + r.json.dumps(tasks[method]))
        r.common.freeze_text(RUN / "report.txt", "\n".join(lines) + "\n")


def execute(jobs, gold, env, qs, rows, phase, retry=False):
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(r.worker, m, q, gold, env, retry): (m, q["q_id"]) for m, q in jobs}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
            save(qs, rows, phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-6-astra", choices=["gpt-6-astra", "gpt-6.1-sol"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    configure(args.model)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame):
            r.common.STOP.set()
            with r.common.LOCK:
                for proc in list(r.common.ACTIVE.values()):
                    r.common.stop_process(proc)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        env, digest, _ = r.tools.ChunkTools.load()
        qs, gold, smoke = prepare(env, digest)
        print(r.json.dumps({"prepared": True, "answer_model": MODEL, "same_questions": 50}), flush=True)
        if args.prepare_only:
            return
        r.common.write(RUN / "started.json", {"started": r.common.now(), "pid": os.getpid()}, frozen=True)
        rows = {}
        save(qs, rows, "smoke")
        execute([(m, q) for q in qs if q["q_id"] in smoke for m in METHODS], gold, env, qs, rows, "smoke")
        # A normal answer-quality label never controls whether the full set runs.
        broken = [x for x in rows.values() if x["state"] != "graded" and "budget exhausted" not in (x.get("error") or "").lower()]
        if broken or r.common.STOP.is_set():
            save(qs, rows, "stopped" if r.common.STOP.is_set() else "smoke_transport_error")
            return
        r.common.write(RUN / "smoke_passed.json", {"ids": smoke, "methods": list(METHODS),
                       "functional_check": "Completed judge or normal declared budget boundary; quality not used to gate full run"}, frozen=True)
        execute([(m, q) for q in qs if q["q_id"] not in smoke for m in METHODS], gold, env, qs, rows, "running")
        retry = [("naive_agent", q) for q in qs if r.retryable(rows[("naive_agent", q["q_id"])])]
        if retry:
            r.common.write(RUN / "retry_plan.json", {"ids": [q["q_id"] for m, q in retry]}, frozen=True)
            execute(retry, gold, env, qs, rows, "retrying", retry=True)
        phase = "stopped" if r.common.STOP.is_set() else ("completed" if all(x["state"] == "graded" for x in rows.values()) else "completed_with_errors")
        save(qs, rows, phase)
        print(r.json.dumps({"state": phase}), flush=True)


if __name__ == "__main__":
    main()
