"""Answer FIN-RATE questions, one isolated sandbox per question, and keep the full trace.

Only the question text enters the sandbox. q_id, task and every gold field stay on the host.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from .contamination import BLOCKED_HOSTS, POLICY_VERSION, detect, opened_pages
from .environment import DockerEnvironment, SandboxError
from .session import CodexSession, save_json

RESULT_SCHEMA = "sandbox-result/v1"
METHOD = "sandbox_web"
PROMPT_PATH = Path(__file__).resolve().parent / "resources" / "answer_prompt.md"
INPUT_KEYS = ("q_id", "task", "question")
GOLD_KEYS = frozenset({"answer", "key_points", "doc_ids"})
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def load_questions(path: Path) -> list[dict]:
    """Keep q_id, task and question; refuse a gold annotation file passed by mistake."""
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError("inputs must be a JSON list of question records")
    questions, seen = [], set()
    for row in rows:
        leaked = sorted(GOLD_KEYS & set(row))
        if leaked:
            raise ValueError(f"{row.get('q_id')}: input carries gold fields {leaked}; pass inputs.json")
        question = {key: row[key] for key in INPUT_KEYS}
        if not _SAFE_ID.fullmatch(str(question["q_id"])) or question["q_id"] in seen:
            raise ValueError(f"unsafe or duplicate q_id: {question['q_id']!r}")
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError(f"{question['q_id']}: empty question")
        seen.add(question["q_id"])
        questions.append(question)
    return questions


def render_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _answer_document(raw):
    """Return (answer, sources) from /work/answer.json, or None if it is unusable."""
    try:
        doc = json.loads(raw)
    except (TypeError, UnicodeError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("answer"), str) or not doc["answer"].strip():
        return None
    sources = doc.get("sources", [])
    sources = [item for item in sources if isinstance(item, str)] if isinstance(sources, list) else []
    return doc["answer"], sources


def trace_activity(directory: Path) -> dict:
    """Count agent actions in the event trace and list the pages it opened."""
    searches = page_actions = commands = 0
    opened = []
    for trace in sorted(directory.glob("turn_*.jsonl")):
        for line in trace.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") == "command_execution":
                commands += 1
            elif item.get("type") == "web_search":
                action = item.get("action") if isinstance(item.get("action"), dict) else {}
                if action.get("type") == "search":
                    searches += 1
                else:
                    page_actions += 1
                    opened.extend(url for url, _ in opened_pages(item))
    return {"web_searches": searches, "web_page_actions": page_actions, "commands": commands,
            "opened_urls": list(dict.fromkeys(opened))}


def _usage(turns) -> dict:
    total = {key: 0 for key in USAGE_KEYS}
    for turn in turns:
        usage = turn.get("usage") if isinstance(turn.get("usage"), dict) else {}
        for key in USAGE_KEYS:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                total[key] += value
    return total


async def run_query(question: dict, directory: Path, *, docker, auth: bytes, model: str, effort: str,
                    timeout: float, template: str, environment_factory=DockerEnvironment) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "input.json", question)
    started = time.monotonic()
    environment = environment_factory(docker)
    session = CodexSession(environment, directory, model=model, effort=effort)
    result = {"schema_version": RESULT_SCHEMA, "q_id": question["q_id"], "task": question["task"],
              "status": "failed", "answer": None, "sources": [], "answer_source": None,
              "model": model, "reasoning_effort": effort, "timeout_seconds": timeout,
              "image": docker.image, "errors": []}
    turn_ok = False
    try:
        await environment.start()
        await session.initialize(auth)
        try:
            await session.turn(render_prompt(template, question["question"]), timeout=timeout)
            turn_ok = True
        except (SandboxError, TimeoutError) as exc:
            result["errors"].append(f"turn: {type(exc).__name__}: {exc}")
        try:
            parsed = _answer_document(await environment.read_file("/work/answer.json"))
        except SandboxError as exc:
            parsed = None
            result["errors"].append(f"answer.json: {exc}")
        if parsed is not None:
            result["answer"], result["sources"] = parsed
            result["answer_source"] = "answer.json"
        elif session.last_message.strip():
            result["answer"], result["answer_source"] = session.last_message, "final_message"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if environment.container_id and not environment.stopped:
            try:
                await environment.freeze()
                result["errors"].extend(await environment.collect(directory))
            except Exception as exc:
                result["errors"].append(f"artifact recovery: {exc}")
        try:
            await environment.close()
        except Exception as exc:
            result["errors"].append(f"cleanup: {exc}")
    if result["answer"]:
        result["status"] = "completed" if turn_ok else "incomplete"
    contamination = detect(directory, question["question"])
    save_json(directory / "contamination.json", contamination)
    result.update(
        contamination_status=contamination["status"],
        contamination_summary=contamination["summary"],
        score_eligible=result["status"] == "completed" and contamination["status"] != "blocked",
        cli_version=session.cli_version, thread_id=session.thread_id,
        usage=_usage(session.turns), activity=trace_activity(directory),
        elapsed_seconds=round(time.monotonic() - started, 3))
    save_json(directory / "result.json", result)
    # Same shape the existing FIN-RATE judge runners read (state/answer per question).
    save_json(directory / "prediction.json", {
        "method": METHOD, "q_id": result["q_id"], "task": result["task"], "artifact_dir": str(directory),
        "state": result["status"], "error": "; ".join(result["errors"]) or None, "answer": result["answer"],
        "contamination_status": result["contamination_status"], "score_eligible": result["score_eligible"]})
    return result


def summarize(questions, results: dict) -> dict:
    rows = [results[q["q_id"]] for q in questions if q["q_id"] in results]
    by_task = {}
    for task in sorted({q["task"] for q in questions}):
        subset = [row for row in rows if row["task"] == task]
        by_task[task] = {"selected": sum(q["task"] == task for q in questions), "processed": len(subset),
                         "status": dict(Counter(row["status"] for row in subset)),
                         "contamination": dict(Counter(row["contamination_status"] for row in subset)),
                         "score_eligible": sum(bool(row["score_eligible"]) for row in subset)}
    usage = {key: sum(row["usage"].get(key, 0) for row in rows) for key in USAGE_KEYS}
    return {"selected": len(questions), "processed": len(rows),
            "status": dict(Counter(row["status"] for row in rows)),
            "contamination": dict(Counter(row["contamination_status"] for row in rows)),
            "blocked": sorted(row["q_id"] for row in rows if row["contamination_status"] == "blocked"),
            "score_eligible": sum(bool(row["score_eligible"]) for row in rows),
            "by_task": by_task, "usage": usage,
            "cli_versions": sorted({row["cli_version"] for row in rows if row.get("cli_version")}),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows)}


async def run_batch(questions: list[dict], output_dir: Path, *, docker, auth: bytes, model: str, effort: str,
                    timeout: float, concurrency: int, template: str, provenance: dict | None = None,
                    environment_factory=DockerEnvironment) -> dict:
    """Run every question not already finished; a resumed directory must keep its settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"method": METHOD, "model": model, "reasoning_effort": effort, "timeout_seconds": timeout,
                "image_id": docker.image, "cpus": docker.cpus, "memory_mb": docker.memory_mb,
                "prompt_sha256": sha256_text(template), "contamination_policy": POLICY_VERSION,
                "blocked_hosts": list(BLOCKED_HOSTS)}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())["settings"]
        changed = sorted(key for key in settings if previous.get(key) != settings[key])
        if changed:
            raise ValueError(f"{output_dir} was started with different settings: {changed}")
    else:
        save_json(manifest_path, {"settings": settings, "provenance": provenance or {},
                                  "created_at_unix": time.time()})
        (output_dir / "answer_prompt.md").write_text(template)
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    async def one(question):
        directory = output_dir / question["q_id"]
        done = directory / "result.json"
        if done.exists():
            results[question["q_id"]] = json.loads(done.read_text())
            return
        async with semaphore:
            if directory.exists():
                # An interrupted attempt: keep its trace, then start over.
                aborted = output_dir / "_aborted" / f"{question['q_id']}-{int(time.time())}"
                aborted.parent.mkdir(exist_ok=True)
                shutil.move(str(directory), str(aborted))
            result = await run_query(question, directory, docker=docker, auth=auth, model=model, effort=effort,
                                     timeout=timeout, template=template, environment_factory=environment_factory)
        results[question["q_id"]] = result
        print(json.dumps({key: result.get(key) for key in
                          ("q_id", "status", "contamination_status", "elapsed_seconds", "errors")}), flush=True)

    await asyncio.gather(*(one(question) for question in questions))
    with (output_dir / "predictions.jsonl").open("w") as handle:
        for question in questions:
            row = results[question["q_id"]]
            handle.write(json.dumps({"q_id": row["q_id"], "task": row["task"], "state": row["status"],
                                     "answer": row["answer"], "contamination_status": row["contamination_status"],
                                     "score_eligible": row["score_eligible"],
                                     "artifact_dir": str(output_dir / row["q_id"])}, ensure_ascii=False) + "\n")
    summary = summarize(questions, results)
    save_json(output_dir / "summary.json", summary)
    return summary
