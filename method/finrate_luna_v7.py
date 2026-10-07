"""Versioned exact-Luna transport: native shell/sleep disabled, final JSON schema.

Flags verified against local CLI features/help. The original event rejection
remains enforced. All v7 conditions use this transport, including static arms.
Failed requests are retained rather than automatically resampled.
"""
import json
from pathlib import Path
import tempfile
import time
from finrate_poc.data import MODEL
from finrate_poc.model import token_count
from finrate_poc.runtime import check_disk, command, digest, write_json
from finrate_poc.schemas import STRING, obj

ANSWER = obj({"answer": STRING, "claims": {"type": "array", "items": obj({
    "text": STRING, "evidence_ids": {"type": "array", "items": STRING}})}})


class LunaV7:
    def __init__(self, *, reasoning="low", max_calls=10000):
        self.reasoning = reasoning
        self.max_calls = max_calls
        self.calls = 0

    def invoke(self, prompt, folder, *, schema=None):
        check_disk()
        folder = Path(folder)
        if schema is None:
            schema = ANSWER
        if (folder / "request.json").exists() and not (folder / "response.json").exists():
            raise RuntimeError("Prior request without accepted response; no automatic regeneration")
        identity = {"model": MODEL, "reasoning": self.reasoning,
                    "prompt_sha256": digest(prompt), "transport": "codex_exec_ephemeral_v7_shell_sleep_disabled"}
        if schema is not None:
            identity["response_schema_sha256"] = digest(json.dumps(schema, sort_keys=True))
        cached = folder / "response.json"
        if cached.exists():
            result = json.loads(cached.read_text())
            if any(result[k] != v for k, v in identity.items()):
                raise ValueError("Cached call inputs changed; use a new versioned directory")
            return result["text"]
        if self.calls >= self.max_calls:
            raise RuntimeError("Configured model-call budget exhausted")
        folder.mkdir(parents=True, exist_ok=True)
        request = {**identity, "prompt": prompt}
        if schema is not None:
            request["response_schema"] = schema
        write_json(folder / "request.json", request, immutable=True)
        self.calls += 1
        with tempfile.TemporaryDirectory(prefix="finrate-luna-") as tmp:
            argv = ["/home/hob/.local/bin/codex", "exec", "--json", "--ephemeral",
                    "--ignore-user-config", "--skip-git-repo-check",
                    "--disable", "shell_tool", "--disable", "unified_exec",
                    "--disable", "sleep_tool", "-s", "read-only",
                    "-m", MODEL, "-c", f"model_reasoning_effort={self.reasoning}",
                    "-c", "web_search=disabled", "-c", "project_doc_max_bytes=0",
                    "-c", "mcp_servers={}", "-C", tmp, "-"]
            if schema is not None:
                schema_path = Path(tmp) / "response_schema.json"
                schema_path.write_text(json.dumps(schema))
                argv[-1:-1] = ["--output-schema", str(schema_path)]
            started = time.monotonic()
            returncode, stdout, stderr = command(argv, prompt=prompt, cwd=tmp)
        (folder / "events.jsonl").write_text(stdout)
        (folder / "stderr.txt").write_text(stderr)
        events = []
        for line in stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        items = [e.get("item", {}) for e in events if e.get("type") == "item.completed"]
        forbidden = [e for e in items if e.get("type") in {
            "command_execution", "mcp_tool_call", "web_search", "file_change"}]
        if forbidden:
            raise RuntimeError("Unexpected tool use: response rejected and preserved")
        answers = [e["text"] for e in items if e.get("type") == "agent_message"]
        if returncode or not answers:
            raise RuntimeError(f"Inference failed ({returncode}); raw trace preserved")
        result = {**identity, "text": answers[-1], "elapsed_s": time.monotonic() - started,
                  "prompt_tokens": token_count(prompt), "answer_tokens": token_count(answers[-1]),
                  "usage": [e.get("usage", {}) for e in events if e.get("type") == "turn.completed"],
                  "checkpoint_limitation": "Provider exposes the requested model alias; immutable weights are not independently verified."}
        write_json(cached, result, immutable=True)
        return result["text"]

