"""Accept completed turns after specifically identified recoverable 403 fallback.

Raw traces stay unchanged. A terminal failure, unknown error, tool or compaction
still fails validation. This changes transport interpretation, never the model
prompt, benchmark answer or rubric.
"""
import re
from pathlib import Path

import run_finrate_baseline50_v1 as r

CLI_INSPECT = r.common.inspect_events
APP_INSPECT = r.long_transport.inspect_events


def cli_filter(events):
    kept, warnings = [], []
    for e in events:
        message = e.get("message", "") if e.get("type") == "error" else e.get("item", {}).get("message", "")
        is_403 = "unexpected status 403 Forbidden:" in message and "wss://chatgpt.com/backend-api/codex/responses" in message
        reconnect = e.get("type") == "error" and re.match(r"^Reconnecting\.\.\. [2-5]/5 \(", message)
        fallback = e.get("type") == "item.completed" and e.get("item", {}).get("type") == "error" and message.startswith("Falling back from WebSockets to HTTPS transport. ")
        if is_403 and (reconnect or fallback):
            warnings.append(e)
        else:
            kept.append(e)
    return kept, warnings


def app_filter(events):
    kept, warnings = [], []
    started = [e["params"]["turn"]["id"] for e in events if e.get("method") == "turn/started"]
    for e in events:
        p = e.get("params", {})
        error = p.get("error") or {}
        message = error.get("message", "") if isinstance(error, dict) else ""
        recoverable = (
            e.get("method") == "error" and p.get("willRetry") is True
            and len(started) == 1 and p.get("turnId") == started[0]
            and re.fullmatch(r"Reconnecting\.\.\. [2-5]/5", message)
            and error.get("codexErrorInfo", {}).get("responseStreamDisconnected", {}).get("httpStatusCode") == 403
            and "wss://chatgpt.com/backend-api/codex/responses" in error.get("additionalDetails", "")
        )
        if recoverable:
            warnings.append(e)
        else:
            kept.append(e)
    return kept, warnings


def cli_inspect(events):
    kept, warnings = cli_filter(events)
    valid, messages, usage, startup, forbidden = CLI_INSPECT(kept)
    return valid, messages, usage, startup + [str(e) for e in warnings], forbidden


def app_inspect(events):
    return APP_INSPECT(app_filter(events)[0])


def install():
    r.common.inspect_events = cli_inspect
    r.long_transport.inspect_events = app_inspect


def import_stage(source, target, kind):
    """Reuse an accepted response or earliest verified stored complete turn."""
    source, target = Path(source), Path(target)
    if not source.exists():
        return None
    if (source / "response.json").exists():
        saved = r.read(source / "response.json")
        if kind == "agent" and saved.get("state") != "completed":
            return None
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
        return {"source": str(source), "response_sha256": r.bm25.sha_file(source / "response.json"), "mode": "unchanged_completed_response"}
    if kind == "agent" or not (source / "prompt.txt").exists():
        return None
    prompt = (source / "prompt.txt").read_text()
    identity = r.long_transport.identity(prompt) if kind == "app" else {
        "model": "gpt-6-luna", "reasoning_effort": "medium", "prompt_sha256": r.common.sha(prompt)}
    for attempt in sorted(source.glob("attempt_*")):
        if not (attempt / "events.jsonl").exists() or not (attempt / "request.json").exists():
            continue
        request = r.read(attempt / "request.json")
        assert all(request.get(k) == v for k, v in identity.items())
        events = [r.json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
        checked = app_inspect(events) if kind == "app" else cli_inspect(events)
        if not checked[0]:
            continue
        messages, usage = checked[1:3]
        answer = messages[0]["text"] if kind == "app" else messages[0]
        review = {"source": str(source), "selected_attempt": attempt.name,
                  "mode": "earliest_stored_completed_turn_after_verified_403_fallback",
                  "events_sha256": r.bm25.sha_file(attempt / "events.jsonl"),
                  "request_sha256": r.bm25.sha_file(attempt / "request.json"),
                  "review_source_sha256": r.bm25.sha_file(__file__), "new_model_calls": 0}
        target.mkdir(parents=True, exist_ok=True)
        for old in sorted(source.glob("attempt_*")):
            link = target / old.name
            if not link.exists():
                link.symlink_to(old, target_is_directory=True)
        r.common.freeze_text(target / "prompt.txt", prompt)
        r.common.freeze_text(target / "response.txt", answer)
        result = {**identity, "text": answer, "usage": usage, "attempt": int(attempt.name.split("_")[-1]),
                  "error": None, "transport_review": review, "valid_completed_turn": True,
                  "usage_source": "unchanged original provider completed-turn usage"}
        r.common.write(target / "response.json", result, frozen=True)
        r.common.write(target / "transport_review.json", review, frozen=True)
        return review
    return None
