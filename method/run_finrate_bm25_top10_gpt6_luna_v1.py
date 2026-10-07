"""BM25 top-10 original chunks -> GPT-6 Luna -> immediate GPT-6 Luna judge.

Uses the exact frozen gold20 questions and the already validated inference
transport/prompt templates. This is BM25-only, not the paper's BM25+R/top-5.
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
import shutil
import signal
import threading

import run_finrate_gold20_gpt56_luna_judge6_v1 as transport
import finrate_bm25_top10_v1 as retrieval

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/finrate_gold20_gpt6_luna_v1"
PREVIOUS = ROOT / "result/finrate_gold20_gpt6_luna_v1"
COHORT = ROOT / "data/finrate_bm25_top10_gpt6_luna_v1"
RUN = ROOT / "result/finrate_bm25_top10_gpt6_luna_v1"
WORKERS = 20
MAX_PROMPT_TOKENS = 240000
MODEL = "gpt-6-luna"
TOP_K = 10

# Changing module variables affects only this process, not old code/artifacts.
transport.RUN = RUN
transport.MODEL = MODEL
transport.JUDGE_MODEL = MODEL
transport.WORKERS = WORKERS
write = transport.write
freeze_text = transport.freeze_text
sha = transport.sha
now = transport.now


def prepare():
    transport.check_disk()
    questions = json.loads((ORIGINAL / "inputs.json").read_text())
    annotations = json.loads((ORIGINAL / "gold_annotations.json").read_text())
    gold = {q["q_id"]: q for q in annotations}
    previous = json.loads((PREVIOUS / "manifest.json").read_text())
    previous_report = json.loads((PREVIOUS / "report.json").read_text())
    assert len(questions) == len(gold) == 20
    assert [q["q_id"] for q in questions] == previous["selected_ids"]
    assert Counter(q["task"] for q in questions) == {"LT-QA": 10, "EC-QA": 10}
    assert previous["answer_model"] == previous["judge_model"] == MODEL
    assert previous["isolation_instructions"] == transport.ISOLATION
    assert previous["judge_source_sha256"] == retrieval.sha_file(transport.REFERENCE / "evaluation/qa_llm_judge.py")
    assert previous["generation_source_sha256"] == retrieval.sha_file(transport.REFERENCE / "generation/qa_generation.py")
    impl_checks = retrieval.validate_implementation()
    docs, corpus_sha = retrieval.load_corpus()
    model, index_manifest = retrieval.load_or_build(docs, corpus_sha)
    corpus = {d["_id"]: d for d in docs}
    planned = []
    rankings = {}
    lengths = {}
    metrics = {}
    for q in questions:
        transport.check_disk()
        qid = q["q_id"]
        assert q["question"] == gold[qid]["question"]
        assert q["gold_doc_ids"] == gold[qid]["doc_ids"]
        ranks = retrieval.search(q["question"], model, docs, k=100)
        retrieved = [r["doc_id"] for r in ranks[:TOP_K]]
        assert len(retrieved) == len(set(retrieved)) == TOP_K
        prompt = transport.answer_prompt(q["question"], [corpus[i] for i in retrieved])
        count = len(transport.ENC.encode(prompt, disallowed_special=()))
        lengths[qid] = count
        record = {"q_id": qid, "task": q["task"], "question": q["question"],
                  "retrieved_doc_ids": retrieved, "prompt_tokens_estimate": count,
                  "prompt_sha256": sha(prompt)}
        planned.append(record)
        rankings[qid] = {"query": q["question"], "query_tokens": retrieval.tokenize(q["question"]),
                         "top_k": TOP_K, "ranked_candidates": ranks,
                         "context_order": retrieved, "context_truncated": False,
                         "context_sha256": sha("\n\n".join(corpus[i]["text"] for i in retrieved))}
        # Gold labels enter only this post-retrieval measurement, never scoring.
        metrics[qid] = retrieval.evidence_metrics(ranks, q["gold_doc_ids"])
        freeze_text(RUN / qid / "answer/prompt.txt", prompt)
        write(RUN / qid / "retrieval.json", rankings[qid], frozen=True)
        print(json.dumps({"stage": "retrieved", "q_id": qid, "input_tokens": count}), flush=True)
    write(COHORT / "inputs.json", planned, frozen=True)
    write(COHORT / "gold_annotations.json", annotations, frozen=True)
    write(COHORT / "rankings.json", rankings, frozen=True)
    write(COHORT / "retrieval_metrics.json", metrics, frozen=True)
    policy = {
        "answer_model": MODEL, "judge_model": MODEL, "reasoning_effort": "medium",
        "workers": WORKERS, "judging_schedule": "Immediately after each answer in the same worker pool",
        "selection": "Exact same 20 questions as frozen gold20, unchanged order, no replacement/filtering",
        "selected_ids": [q["q_id"] for q in questions], "corpus_count": len(docs),
        "dataset_revision": transport.REV, "top_k": TOP_K, "reranker": None,
        "retrieval_policy": retrieval.POLICY, "index_manifest_sha256": retrieval.sha_file(retrieval.INDEX / "manifest.json"),
        "corpus_zip_sha256": corpus_sha, "original_cohort_sha256": retrieval.sha_file(ORIGINAL / "inputs.json"),
        "original_annotations_sha256": retrieval.sha_file(ORIGINAL / "gold_annotations.json"),
        "previous_manifest_sha256": retrieval.sha_file(PREVIOUS / "manifest.json"),
        "previous_report_sha256": retrieval.sha_file(PREVIOUS / "report.json"),
        "previous_gold_accuracy": {task: row["accuracy"] for task, row in previous_report["tasks"].items()},
        "judge_source_sha256": previous["judge_source_sha256"],
        "generation_source_sha256": previous["generation_source_sha256"],
        "runner_source_sha256": retrieval.sha_file(__file__),
        "transport_source_sha256": retrieval.sha_file(transport.__file__),
        "retrieval_source_sha256": retrieval.sha_file(retrieval.__file__),
        "artifacts_sha256": {n: retrieval.sha_file(COHORT / n) for n in
                             ["inputs.json", "gold_annotations.json", "rankings.json", "retrieval_metrics.json"]},
        "context": "Full text of 10 original chunks in BM25 rank order, joined with blank lines; no titles, truncation or summary",
        "answer_reference_leakage": False, "judge_has_raw_context": False,
        "judge_blinding": "No retrieval method label, previous answer/grade or answer model identity in judge prompt",
        "prompt_tokens_estimate": lengths, "estimator": "o200k_base",
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "over_limit_qids": [qid for qid, n in lengths.items() if n > MAX_PROMPT_TOKENS],
        "over_limit_policy": "Record input_length_error; never silently truncate, drop a chunk or replace a question",
        "model_context_window": 272000, "auto_compact_token_limit": 250000,
        "output_token_limit": "CLI/provider default; same as previous gold run",
        "isolation_instructions": transport.ISOLATION, "native_tools": "disabled",
        "local_gpu": False, "disk_stop_bytes": transport.MIN_FREE,
        "judge": "same official English rubric, single GPT-6 Luna judge; not paper ensemble",
        "paper_differences": ["top10 instead of top5", "BM25 without reranker", "20 paired questions rather than 500 per task",
                              "No new gold-count<=5 filter", "BM25 analyzer and parameters are declared reconstruction choices"],
        "implementation_checks": impl_checks,
    }
    write(RUN / "manifest.json", policy, frozen=True)
    for source in [Path(__file__), Path(transport.__file__), Path(retrieval.__file__)]:
        freeze_text(RUN / "source_snapshot" / source.name, source.read_text())
    return planned, gold, metrics


def worker(q, gold):
    qid = q["q_id"]
    folder = RUN / qid
    row = {"q_id": qid, "task": q["task"], "retrieved_doc_ids": q["retrieved_doc_ids"]}
    try:
        if q["prompt_tokens_estimate"] > MAX_PROMPT_TOKENS:
            raise ValueError("input_length_error: complete top10 exceeds declared input limit")
        prompt = (folder / "answer/prompt.txt").read_text()
        assert sha(prompt) == q["prompt_sha256"]
        transport.set_state(qid, "generating")
        generated = transport.invoke(prompt, folder / "answer")
        row.update(answer=generated["text"], answer_usage=generated["usage"])
        write(folder / "prediction.json", row, frozen=True)
        transport.set_state(qid, "judging")
        ref = gold[qid]
        points = ref["key_points"]
        if isinstance(points, str):
            points = [points]
        assert isinstance(ref["answer"], str) and all(isinstance(p, str) for p in points)
        judge_prompt = transport.load_judge()(q["question"], ref["answer"], generated["text"], points)
        assessed = transport.invoke(judge_prompt, folder / "judge")
        row.update(transport.parse_judge(assessed["text"], len(points)))
        row.update(status="graded", judge_usage=assessed["usage"], judge_response=assessed["text"])
        write(folder / "assessment.json", row, frozen=True)
        transport.set_state(qid, "graded", verdict=row["verdict"])
        print(json.dumps({"q_id": qid, "task": q["task"], "verdict": row["verdict"]}), flush=True)
    except Exception as exc:
        row.update(status="execution_error", error=f"{type(exc).__name__}: {exc}")
        write(folder / "error.json", row)
        transport.set_state(qid, "execution_error", error=row["error"])
        print(json.dumps({"q_id": qid, "error": row["error"]}), flush=True)
    return row


def report(questions, rows, metrics, final=False):
    tasks = {}
    for task in ["LT-QA", "EC-QA", "ALL"]:
        selected = [q for q in questions if task == "ALL" or q["task"] == task]
        rr = [r for r in rows if task == "ALL" or r["task"] == task]
        labels = Counter(r["verdict"] for r in rr if r.get("status") == "graded")
        n = len(selected)
        rm = [metrics[q["q_id"]] for q in selected]
        tasks[task] = {"selected": n, "graded": sum(labels.values()), "labels": dict(labels),
                       "correct": labels["CORRECT"],
                       "accuracy": labels["CORRECT"] / n if sum(labels.values()) == n else None,
                       "accuracy_among_graded": labels["CORRECT"] / sum(labels.values()) if labels else None,
                       "ungraded": n - sum(labels.values()),
                       "retrieval": {str(k): {"any_gold_hit_queries": sum(m[str(k)]["hit_any"] for m in rm),
                                               "any_gold_hit_rate": sum(m[str(k)]["hit_any"] for m in rm) / n,
                                               "all_gold_found_queries": sum(m[str(k)]["all_gold_found"] for m in rm),
                                               "mean_gold_coverage": sum(m[str(k)]["coverage"] for m in rm) / n,
                                               "gold_chunks_retrieved": sum(m[str(k)]["matched_count"] for m in rm),
                                               "total_gold_chunks": sum(m["gold_count"] for m in rm)}
                                     for k in (5, 10)}}
    usage = Counter()
    attempts = 0
    for p in RUN.glob("*/[aj]*/attempt_*/outcome.json"):
        out = json.loads(p.read_text())
        attempts += 1
        for u in out.get("usage", []):
            for key, value in u.items():
                if isinstance(value, (int, float)):
                    usage[key] += value
    done = tasks["ALL"]["graded"] == len(questions)
    state = "completed" if final and done else "finished_with_errors" if final else "running"
    result = {"state": state, "updated": now(), "tasks": tasks,
              "provider_usage": dict(usage), "attempts_finished": attempts,
              "workers": WORKERS, "model": MODEL, "judge_model": MODEL, "reasoning_effort": "medium",
              "baseline": "BM25 original chunk top10; no reranker", "rows": sorted(rows, key=lambda r: r["q_id"])}
    write(RUN / "report.json", result)
    with transport.LOCK:
        write(RUN / "status.json", {"state": state, "pid": os.getpid(), "updated": now(),
              "active_model_processes": len(transport.ACTIVE),
              "phases": dict(Counter(v["phase"] for v in transport.STATES.values())),
              "questions": dict(transport.STATES), "free_disk_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 2)})
    if final:
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["q_id", "task", "status", "verdict", "key_point_coverage", "error"])
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in writer.fieldnames} for r in result["rows"])
        (RUN / "results.csv").write_text(out.getvalue())
        lines = ["FIN-RATE BM25 top10 — GPT-6 Luna answer / GPT-6 Luna judge / medium", f"완료 시각: {now()}",
                 "이전 gold20과 동일 LT10/EC10. 전체15311 원래chunk의 본문 BM25. 재분할·reranker·본문절단 없음.",
                 "worker20, 답변 완료 직후 채점. 단일 Luna judge이며 논문 ensemble과 다름."]
        for task, t in tasks.items():
            lines.append(f"{task}: 정답 {t['correct']}/{t['selected']}; 채점 {t['graded']}; 분포 {t['labels']}")
            lines.append(f"  top10 gold 하나 이상 검색: {t['retrieval']['10']['any_gold_hit_queries']}/{t['selected']}; "
                         f"전체 gold 검색: {t['retrieval']['10']['all_gold_found_queries']}/{t['selected']}; "
                         f"평균 gold coverage: {t['retrieval']['10']['mean_gold_coverage']:.4f}")
        lines += [f"Provider usage: {dict(usage)}", "이전 gold context와 근거 개수/입력길이는 다르므로 동일 token 예산 비교가 아님."]
        (RUN / "report.txt").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        qs, gold, metrics = prepare()
        if args.prepare_only:
            print(json.dumps({"prepared": len(qs), "max_prompt_tokens": max(q["prompt_tokens_estimate"] for q in qs),
                              "over_limit_qids": [q["q_id"] for q in qs if q["prompt_tokens_estimate"] > MAX_PROMPT_TOKENS]}), flush=True)
            return
        for q in qs:
            transport.set_state(q["q_id"], "queued")
        write(RUN / "started.json", {"pid": os.getpid(), "started": now(), "model": MODEL})
        def interrupted(signum, frame):
            transport.STOP.set()
            with transport.LOCK:
                procs = list(transport.ACTIVE.values())
            for proc in procs:
                transport.stop_process(proc)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        rows = []
        monitor_done = threading.Event()
        def monitor():
            while not monitor_done.wait(3):
                try:
                    transport.check_disk()
                    with transport.LOCK:
                        report(qs, list(rows), metrics)
                except Exception as exc:
                    transport.STOP.set()
                    write(RUN / "monitor_error.json", {"error": str(exc), "time": now()})
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        report(qs, rows, metrics)
        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(worker, q, gold) for q in qs]
                for future in as_completed(futures):
                    row = future.result()
                    with transport.LOCK:
                        rows.append(row)
                    report(qs, list(rows), metrics)
        finally:
            monitor_done.set()
            thread.join(timeout=5)
        result = report(qs, rows, metrics, final=True)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
