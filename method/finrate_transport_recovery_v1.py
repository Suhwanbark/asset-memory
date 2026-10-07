"""Accept a completed original inference rejected only for a fallback notice.

The frozen transport and its request/event files remain unchanged. Recovery
creates a previously absent response.json plus a provenance sidecar, without
another model call. Terminal failures and native tools still fail closed.
"""
import json
from pathlib import Path

import jsonschema

from finrate_autonomous_v10.transport import ANSWER, IsolatedLuna
from finrate_poc.data import MODEL
from finrate_poc.model import token_count
from finrate_poc.runtime import check_disk, digest, file_digest, write_json

NOTICE = "Falling back from WebSockets to HTTPS transport."


def recover(folder, prompt, schema, reasoning="low"):
    check_disk()
    folder = Path(folder)
    if (folder / "response.json").exists():
        raise ValueError("Do not replace an accepted response")
    request = json.loads((folder / "request.json").read_text())
    identity = {
        "model": MODEL, "reasoning": reasoning,
        "prompt_sha256": digest(prompt),
        "response_schema_sha256": digest(json.dumps(schema, sort_keys=True)),
        "transport": "codex_exec_ephemeral_v9_isolated_policy",
    }
    if any(request.get(k) != v for k, v in identity.items()):
        raise ValueError("Recovery input identity differs from original request")
    events = []
    for line in (folder / "events.jsonl").read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    if any(e.get("type") == "turn.failed" for e in events):
        raise ValueError("Terminal model failure cannot be recovered")
    completed = [e for e in events if e.get("type") == "turn.completed"]
    if len(completed) != 1:
        raise ValueError("Require exactly one completed turn")
    notices = 0
    answers = []
    for event in events:
        if not event.get("type", "").startswith("item."):
            continue
        item = event.get("item", {})
        kind = item.get("type")
        if kind in {"agent_message", "reasoning"}:
            if kind == "agent_message" and event["type"] == "item.completed":
                answers.append(item["text"])
        elif kind == "error" and item.get("message", "").startswith(NOTICE):
            notices += event["type"] == "item.completed"
        else:
            raise ValueError("Native tool or unrecognized item cannot be recovered")
    if notices < 1 or len(answers) != 1:
        raise ValueError("Require a fallback notice and one original final answer")
    jsonschema.validate(json.loads(answers[0]), schema)
    provenance = {
        "request_sha256": file_digest(folder / "request.json"),
        "events_sha256": file_digest(folder / "events.jsonl"),
        "recovery_code_sha256": file_digest(__file__),
        "same_original_prompt_and_schema": True,
        "new_model_calls": 0,
        "reason": "Informational WebSocket-to-HTTPS notice misclassified as native tool use",
    }
    result = {
        **identity, "text": answers[0], "elapsed_s": None,
        "elapsed_unavailable_for_recovered_call": True,
        "prompt_tokens": token_count(prompt),
        "answer_tokens": token_count(answers[0]),
        "usage": [e.get("usage", {}) for e in completed],
        "checkpoint_limitation": "Requested model alias; immutable weights not independently verified.",
        "recovery": provenance,
    }
    write_json(folder / "response_recovery.json", provenance, immutable=True)
    write_json(folder / "response.json", result, immutable=True)
    return answers[0]


class RecoveredLuna(IsolatedLuna):
    def invoke(self, prompt, folder, *, schema=None):
        schema = ANSWER if schema is None else schema
        try:
            return super().invoke(prompt, folder, schema=schema)
        except RuntimeError as exc:
            if str(exc) not in {
                "Unexpected tool use: response rejected and preserved",
                "Prior request without accepted response; no automatic regeneration",
            }:
                raise
            return recover(folder, prompt, schema, self.reasoning)
