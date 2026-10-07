"""Recover verified completed responses after harmless pre-turn CLI warnings.

Preserves the frozen runner, all original attempt records and initial reports.
No accepted answer is regenerated. Judge calls still use the frozen prompts and
20-worker settings. Unknown warnings, tools, failed turns or compaction fail closed.
"""
import json
import shutil
from pathlib import Path

import run_finrate_gold20_gpt6_luna_v1 as base

ORIGINAL_INVOKE = base.invoke
WARNINGS = {
    'Under-development features enabled: skip_host_skill_discovery. Under-development features are incomplete and may behave unpredictably. To suppress this warning, set `suppress_unstable_features_warning = true` in /home/hob/.codex/config.toml.',
    'Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.',
}


def recover(prompt, folder):
    identity = {"model": base.MODEL, "reasoning_effort": "medium", "prompt_sha256": base.sha(prompt)}
    if (folder / "response.json").exists():
        return None
    for attempt in sorted(folder.glob("attempt_*")):
        if not (attempt / "outcome.json").exists():
            continue
        outcome = json.loads((attempt / "outcome.json").read_text())
        request = json.loads((attempt / "request.json").read_text())
        if outcome["returncode"] != 0:
            continue
        assert all(request[k] == v for k, v in identity.items())
        assert base.sha((folder / "prompt.txt").read_bytes()) == identity["prompt_sha256"]
        events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
        starts = [i for i, e in enumerate(events) if e.get("type") == "turn.started"]
        ends = [i for i, e in enumerate(events) if e.get("type") == "turn.completed"]
        if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
            continue
        messages, warnings = [], []
        accepted = True
        for i, event in enumerate(events):
            kind = event.get("type")
            if kind not in {"thread.started", "turn.started", "turn.completed", "item.completed"}:
                accepted = False
                break
            if kind != "item.completed":
                continue
            item = event["item"]
            if item.get("type") == "error" and i < starts[0] and item.get("message") in WARNINGS:
                warnings.append(item["message"])
            elif starts[0] < i < ends[0] and item.get("type") in {"agent_message", "reasoning"}:
                if item["type"] == "agent_message":
                    messages.append(item["text"])
            else:
                accepted = False
                break
        if not accepted or len(messages) != 1 or not messages[0].strip():
            continue
        usage = events[ends[0]].get("usage")
        if not isinstance(usage, dict) or usage.get("input_tokens", 0) <= 0:
            continue
        review = {
            "reviewed": base.now(), "decision": "accept_exact_completed_response",
            "reason": "Only allowlisted informational CLI warnings before turn.started; no native tools or compaction; one successful completed turn",
            "attempt": attempt.name, "warnings": warnings,
            "original_outcome_sha256": base.sha((attempt / "outcome.json").read_bytes()),
            "events_sha256": base.sha((attempt / "events.jsonl").read_bytes()),
            "request_sha256": base.sha((attempt / "request.json").read_bytes()),
            "review_code_sha256": base.sha(Path(__file__).read_bytes()),
        }
        result = {**identity, **outcome, "error": None, "forbidden_item_types": [],
                  "text": messages[0], "usage": [usage], "attempt": request["attempt"],
                  "usage_source": "provider_reported_turn.completed; prompt estimate is separate",
                  "transport_review": review}
        base.write(attempt / "transport_review.json", review, frozen=True)
        base.write(folder / "response.json", result, frozen=True)
        base.freeze_text(folder / "response.txt", result["text"])
        return result
    return None


def reviewed_invoke(prompt, folder):
    base.check_disk()
    response = recover(prompt, folder)
    if response is not None:
        return response
    try:
        return ORIGINAL_INVOKE(prompt, folder)
    except RuntimeError:
        response = recover(prompt, folder)
        if response is not None:
            return response
        raise


if __name__ == "__main__":
    archive = base.RUN / "initial_transport_report"
    archive.mkdir(exist_ok=True)
    for name in ("report.json", "status.json", "results.csv", "report.txt", "started.json"):
        source, target = base.RUN / name, archive / name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)
    base.write(base.RUN / "transport_revision.json", {
        "frozen_runner_sha256": base.sha(Path(base.__file__).read_bytes()),
        "review_runner_sha256": base.sha(Path(__file__).read_bytes()),
        "policy": "Only allowlisted pre-turn CLI warnings reclassified; prompts/model/cohort unchanged; exact saved responses reused",
        "initial_reports": str(archive),
        "original_errors": "Retained per-question error.json records refer to the initial parser error and are superseded by successful assessment.json where present",
    }, frozen=True)
    base.invoke = reviewed_invoke
    base.main()
