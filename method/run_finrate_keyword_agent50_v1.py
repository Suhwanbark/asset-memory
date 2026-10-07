"""User-corrected scope: only keyword BM25 and naive chunk agent, same 50."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fcntl
import os
import signal

import run_finrate_baseline50_v1 as r
import finrate_transport_review_v1 as review

PARENT = r.RUN
RUN = r.ROOT / "result/finrate_keyword_agent50_gpt6_luna_v1"
METHODS = ("bm25", "naive_agent")


def configure():
    r.RUN = RUN
    r.OLD = {m: PARENT / m for m in METHODS}
    r.configure()
    review.install()


def prepare():
    qs = r.read(r.DATA / "inputs.json")
    gold = {q["q_id"]: q for q in r.read(r.DATA / "gold_annotations.json")}
    imported = {}
    for method in METHODS:
        for q in qs:
            qid = q["q_id"]
            source, folder = PARENT / method / qid, RUN / method / qid
            imported[f"{method}/{qid}"] = {"source": str(source), "stages": {}}
            if (source / "assessment.json").exists():
                folder.parent.mkdir(parents=True, exist_ok=True)
                if not folder.exists():
                    folder.symlink_to(source, target_is_directory=True)
                imported[f"{method}/{qid}"]["assessment_sha256"] = r.bm25.sha_file(source / "assessment.json")
                continue
            folder.mkdir(parents=True, exist_ok=True)
            stage = "agent" if method == "naive_agent" else "answer"
            kind = "agent" if method == "naive_agent" else "app"
            for name, parser in ((stage, kind), ("judge", "cli")):
                recovery = review.import_stage(source / name, folder / name, parser)
                if recovery:
                    imported[f"{method}/{qid}"]["stages"][name] = recovery
    r.common.write(RUN / "imports.json", imported, frozen=True)
    parent_manifest = r.read(PARENT / "manifest.json")
    sources = [Path(__file__), Path(review.__file__), Path(r.__file__), Path(r.common.__file__),
               Path(r.long_transport.__file__), Path(r.agent.__file__), Path(r.tools.__file__)]
    manifest = {**parent_manifest, "scope": "User clarified keyword BM25 and naive agent only",
                "methods": list(METHODS), "parent_run": str(PARENT),
                "parent_manifest_sha256": r.bm25.sha_file(PARENT / "manifest.json"),
                "imports_sha256": r.bm25.sha_file(RUN / "imports.json"),
                "source_sha256": {p.name: r.bm25.sha_file(p) for p in sources},
                "transport_change": "Recognize verified recoverable 403 reconnect/HTTPS fallback; require one successful turn, no tools or compaction in static inference",
                "gold_additional_calls": False, "completed_answer_regeneration": False}
    r.common.write(RUN / "manifest.json", manifest, frozen=True)
    for p in sources:
        r.common.freeze_text(RUN / "source_snapshot" / p.name, p.read_text())
    return qs, gold


def main():
    configure()
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
        r.common.check_disk()
        qs, gold = prepare()
        env, _, _ = r.tools.ChunkTools.load()
        r.common.write(RUN / "started.json", {"pid": os.getpid(), "started": r.common.now()}, frozen=True)
        rows, jobs = {}, []
        for q in qs:
            for method in METHODS:
                if (RUN / method / q["q_id"] / "assessment.json").exists():
                    rows[(method, q["q_id"])] = r.worker(method, q, gold, env)
                else:
                    jobs.append((method, q))
        r.save(qs, rows, "running")
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(r.worker, m, q, gold, env): (m, q["q_id"]) for m, q in jobs}
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
                r.save(qs, rows, "running")
        retry = [q for q in qs if r.retryable(rows[("naive_agent", q["q_id"])])]
        if retry:
            r.common.write(RUN / "retry_plan.json", {"ids": [q["q_id"] for q in retry]}, frozen=True)
            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = {pool.submit(r.worker, "naive_agent", q, gold, env, True): q for q in retry}
                for future in as_completed(futures):
                    rows[("naive_agent", futures[future]["q_id"])] = future.result()
                    r.save(qs, rows, "running")
        phase = "stopped" if r.common.STOP.is_set() else ("completed" if all(x.get("state") == "graded" for x in rows.values()) else "completed_with_errors")
        r.save(qs, rows, phase)
        print(r.json.dumps({"state": phase}), flush=True)


if __name__ == "__main__":
    main()
