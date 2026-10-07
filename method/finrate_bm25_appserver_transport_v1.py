"""Tool-disabled Luna inference through documented Codex app-server history input.

turn/start has a 1 Mi-character UI limit. The supported thread/inject_items RPC
supplies the complete official prompt as user history; a short fixed user turn
starts inference. This is explicit input packaging, not summarization. Use the
same packaging for all 20 answers, and retain the exact judge prompt separately.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

import run_finrate_gold20_gpt56_luna_judge6_v1 as shared

MODEL = "gpt-6-luna"
TRIGGER = "Answer the preceding question using its provided context."
CONTEXT_WINDOW = 1050000
COMPACTION_LIMIT = 950000


def identity(prompt):
    return {"model": MODEL, "reasoning_effort": "medium", "prompt_sha256": shared.sha(prompt),
            "transport": "codex_app_server_0.160.1_inject_user_history",
            "context_window": CONTEXT_WINDOW, "auto_compact_token_limit": COMPACTION_LIMIT,
            "trigger": TRIGGER}


def inspect_events(events):
    turns = [e["params"]["turn"] for e in events if e.get("method") == "turn/completed"]
    starts = [e for e in events if e.get("method") == "turn/started"]
    usage_events = [e["params"]["tokenUsage"] for e in events if e.get("method") == "thread/tokenUsage/updated"]
    items = [e["params"]["item"] for e in events if e.get("method") == "item/completed"]
    messages = [i for i in items if i.get("type") == "agentMessage"]
    forbidden = [i.get("type") for i in items if i.get("type") not in {"agentMessage", "reasoning", "userMessage"}]
    for e in events:
        method = e.get("method", "")
        if "compact" in method.lower() or "compact" in e.get("params", {}).get("item", {}).get("type", "").lower():
            forbidden.append(method or "compaction")
        if method == "error" or e.get("error") is not None or (e.get("id") is not None and method):
            forbidden.append(method or "RPC error")
    model_results = [e["result"] for e in events if e.get("id") == 2 and "result" in e]
    valid_model = len(model_results) == 1 and model_results[0].get("model") == MODEL and model_results[0].get("reasoningEffort") == "medium"
    valid = valid_model and len(turns) == len(starts) == len(messages) == 1 and bool(usage_events)
    valid = valid and turns[0]["status"] == "completed" and bool(messages[0].get("text", "").strip()) and not forbidden
    usage = []
    if usage_events:
        u = usage_events[-1]["last"]
        usage = [{"input_tokens": u["inputTokens"], "cached_input_tokens": u.get("cachedInputTokens", 0),
                  "output_tokens": u["outputTokens"], "reasoning_output_tokens": u.get("reasoningOutputTokens", 0)}]
        valid = valid and u["inputTokens"] > 0
    return valid, messages, usage, forbidden, usage_events[-1].get("modelContextWindow") if usage_events else None


def accept_existing(prompt, folder, probe):
    """Import the first successful canary without making another model request."""
    folder = Path(folder)
    probe = Path(probe)
    if (folder / "response.json").exists():
        return
    if (probe / "prompt.txt").read_text() != prompt:
        raise ValueError("Canary prompt differs")
    events = [json.loads(line) for line in (probe / "events.jsonl").read_text().splitlines()]
    valid, messages, usage, forbidden, effective_window = inspect_events(events)
    if not valid:
        raise ValueError(f"Canary trace invalid: {forbidden}")
    requests = [json.loads(line) for line in (probe / "request.jsonl").read_text().splitlines()]
    injected = next(r for r in requests if r.get("method") == "thread/inject_items")
    assert injected["params"]["items"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    turn = next(r for r in requests if r.get("method") == "turn/start")
    assert turn["params"]["input"] == [{"type": "text", "text": TRIGGER}]
    attempt = folder / "attempt_01"
    shared.freeze_text(folder / "prompt.txt", prompt)
    for name in ["events.jsonl", "request.jsonl", "stderr.txt"]:
        shared.freeze_text(attempt / name, (probe / name).read_text())
    outcome = {"usage": usage, "valid_completed_turn": True, "forbidden_item_types": [],
               "compaction_events": [], "error": None, "effective_model_context_window": effective_window,
               "imported_from": str(probe), "finished": shared.now()}
    shared.write(attempt / "request.json", {**identity(prompt), "imported_from": str(probe)}, frozen=True)
    shared.write(attempt / "outcome.json", outcome, frozen=True)
    shared.write(folder / "response.json", {**identity(prompt), **outcome, "text": messages[0]["text"], "attempt": 1,
                  "usage_source": "provider_reported_thread/tokenUsage/updated.last"}, frozen=True)
    shared.freeze_text(folder / "response.txt", messages[0]["text"])


def invoke(prompt, folder):
    shared.check_disk()
    folder = Path(folder)
    cached = folder / "response.json"
    ident = identity(prompt)
    if cached.exists():
        saved = json.loads(cached.read_text())
        if any(saved.get(k) != v for k, v in ident.items()):
            raise ValueError("Cached response identity mismatch")
        return saved
    shared.freeze_text(folder / "prompt.txt", prompt)
    last_error = None
    for attempt_n in (1, 2):
        if shared.STOP.is_set():
            raise RuntimeError("Stopped")
        attempt = folder / f"attempt_{attempt_n:02d}"
        if (attempt / "outcome.json").exists():
            last_error = json.loads((attempt / "outcome.json").read_text()).get("error")
            continue
        if (attempt / "request.json").exists():
            raise RuntimeError("Unfinished prior attempt; inspect before resuming")
        attempt.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="finrate-bm25-appserver-") as tmp:
            cfg = {"model": MODEL, "model_reasoning_effort": "medium", "model_context_window": CONTEXT_WINDOW,
                   "model_auto_compact_token_limit": COMPACTION_LIMIT, "web_search": "disabled",
                   "project_doc_max_bytes": 0, "mcp_servers": {}, "apps._default.enabled": False,
                   "developer_instructions": shared.ISOLATION, "features.skip_host_skill_discovery": True}
            for feature in shared.DISABLED:
                cfg[f"features.{feature}"] = False
            argv = ["/home/hob/.local/bin/codex", "app-server", "--listen", "stdio://"]
            for key, val in cfg.items():
                argv += ["-c", f"{key}={json.dumps(val)}"]
            env = os.environ.copy()
            for key in ("CODEX_REMOTE_PAYLOAD", "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            shared.write(attempt / "request.json", {**ident, "argv": argv, "started": shared.now(),
                         "attempt": attempt_n, "prompt_file": str(folder / "prompt.txt")}, frozen=True)
            started = time.monotonic()
            events = []
            error = None
            with (attempt / "stderr.txt").open("wb") as err, (attempt / "events.jsonl").open("w") as event_log, (attempt / "request.jsonl").open("w") as request_log:
                proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                        text=True, bufsize=1, cwd=tmp, env=env, start_new_session=True)
                with shared.LOCK:
                    shared.ACTIVE[proc.pid] = proc
                incoming = queue.Queue()
                def reader():
                    try:
                        for line in proc.stdout:
                            event_log.write(line); event_log.flush()
                            incoming.put(json.loads(line))
                    except Exception as exc:
                        incoming.put(exc)
                    finally:
                        incoming.put(None)
                reader_thread = threading.Thread(target=reader, daemon=True)
                reader_thread.start()
                def send(message):
                    data = json.dumps(message, ensure_ascii=False)
                    request_log.write(data + "\n"); request_log.flush()
                    proc.stdin.write(data + "\n"); proc.stdin.flush()
                def get():
                    while True:
                        shared.check_disk()
                        if shared.STOP.is_set():
                            raise RuntimeError("Stopped")
                        if time.monotonic() - started > shared.TIMEOUT:
                            raise TimeoutError("Model call exceeded timeout")
                        try:
                            event = incoming.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        if event is None:
                            raise RuntimeError("App-server exited before completion")
                        if isinstance(event, Exception):
                            raise event
                        events.append(event)
                        if event.get("id") is not None and event.get("method"):
                            raise RuntimeError("Unexpected server tool/request")
                        return event
                def rpc(number, method, params):
                    send({"id": number, "method": method, "params": params})
                    while True:
                        event = get()
                        if event.get("id") == number:
                            if "error" in event:
                                raise RuntimeError(f"{method}: {event['error']}")
                            return event["result"]
                try:
                    rpc(1, "initialize", {"clientInfo": {"name": "finrate_bm25", "version": "1.0"},
                                          "capabilities": {"experimentalApi": True}})
                    send({"method": "initialized"})
                    thread = rpc(2, "thread/start", {"model": MODEL, "cwd": tmp, "approvalPolicy": "never",
                                 "sandbox": "read-only", "ephemeral": True,
                                 "developerInstructions": shared.ISOLATION, "dynamicTools": [], "config": cfg})
                    if thread.get("model") != MODEL or thread.get("reasoningEffort") != "medium":
                        raise RuntimeError("Resolved model/effort mismatch")
                    tid = thread["thread"]["id"]
                    rpc(3, "thread/inject_items", {"threadId": tid, "items": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]})
                    rpc(4, "turn/start", {"threadId": tid, "input": [{"type": "text", "text": TRIGGER}],
                                         "model": MODEL, "effort": "medium"})
                    while True:
                        event = get()
                        if event.get("method") == "turn/completed":
                            break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    shared.stop_process(proc)
                    reader_thread.join(timeout=5)
                    with shared.LOCK:
                        shared.ACTIVE.pop(proc.pid, None)
            valid, messages, usage, forbidden, effective_window = inspect_events(events)
            outcome = {"usage": usage, "valid_completed_turn": valid, "forbidden_item_types": forbidden,
                       "compaction_events": [x for x in forbidden if "compact" in x.lower()],
                       "error": error, "elapsed_s": time.monotonic() - started, "finished": shared.now(),
                       "effective_model_context_window": effective_window}
            if not error and valid:
                result = {**ident, **outcome, "text": messages[0]["text"], "attempt": attempt_n,
                          "usage_source": "provider_reported_thread/tokenUsage/updated.last"}
                shared.write(attempt / "outcome.json", outcome, frozen=True)
                shared.write(cached, result, frozen=True)
                shared.freeze_text(folder / "response.txt", result["text"])
                return result
            outcome["error"] = error or f"Invalid completed turn; forbidden={forbidden}"
            shared.write(attempt / "outcome.json", outcome, frozen=True)
            last_error = outcome["error"]
            # Never regenerate a completed/partially delivered answer or bypass a tool/compaction failure.
            if messages or forbidden or any(e.get("method") == "turn/completed" for e in events):
                break
        if attempt_n < 2 and not shared.STOP.is_set():
            shared.STOP.wait(5)
    raise RuntimeError(f"No accepted response: {last_error}")
