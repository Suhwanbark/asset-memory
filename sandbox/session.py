"""One Codex CLI thread inside one sandbox container.

Ported from finautorubric ``final_pipeline/session.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .environment import SandboxError


ENABLED_FEATURES = ("shell_tool", "unified_exec", "goals", "workspace_dependencies")

DISABLED_FEATURES = (
    "apps", "plugins", "browser_use", "browser_use_external", "computer_use",
    "in_app_browser", "hooks", "multi_agent", "multi_agent_v2", "memories",
    "skill_mcp_dependency_install", "skill_search", "tool_suggest",
    "image_generation",
)


def save_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class CodexSession:
    def __init__(self, environment, directory: Path, *, model, effort="medium"):
        self.environment, self.directory = environment, directory
        self.model, self.effort = model, effort
        self.thread_id = None
        self.turns = []
        self.features = set()
        self.failed = False
        self.last_message = ""
        self.cli_version = None

    async def initialize(self, auth: bytes):
        self.directory.mkdir(parents=True, exist_ok=True)
        await self.environment.put("/home/agent/.codex/auth.json", auth)
        code, out, _ = await self.environment.exec(["codex", "--version"])
        if code:
            raise SandboxError("Codex CLI is unavailable in sandbox image")
        self.cli_version = out.decode().strip()
        code, out, _ = await self.environment.exec(["codex", "features", "list"])
        if code:
            raise SandboxError("cannot inspect sandbox Codex feature configuration")
        self.features = {line.split()[0] for line in out.decode().splitlines() if line.split()}
        missing = set(ENABLED_FEATURES) - self.features
        if missing:
            raise SandboxError("sandbox Codex lacks required features: " + ", ".join(sorted(missing)))
        self.save()

    def argv(self):
        # --search enables the native Responses web_search tool (live web).
        args = ["codex", "--search", "-c", 'mcp_servers={}',
                "-c", "model_reasoning_effort=" + json.dumps(self.effort)]
        for feature in DISABLED_FEATURES:
            if feature in self.features:
                args.extend(["--disable", feature])
        for feature in ENABLED_FEATURES:
            args.extend(["--enable", feature])
        args.append("exec")
        if self.thread_id:
            args.extend(["resume", self.thread_id])
        # This command can only be dispatched through DockerEnvironment.exec.
        # No --ephemeral: the session rollout is the only record of opened pages.
        args.extend(["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
                     "--ignore-user-config", "--ignore-rules", "--json", "--model", self.model,
                     "--output-last-message", "/work/last_message.txt", "-"])
        return args

    def save(self):
        save_json(self.directory / "session.json", {
            "container_name": self.environment.name, "container_id": self.environment.container_id,
            "thread_id": self.thread_id, "cli_version": self.cli_version,
            "model": self.model, "reasoning_effort": self.effort, "turns": self.turns,
            "failed": self.failed})

    async def turn(self, prompt: str, *, timeout: float):
        if self.failed:
            raise SandboxError("failed session cannot start another turn")
        index = len(self.turns) + 1
        trace = self.directory / f"turn_{index:02d}.jsonl"
        (self.directory / f"prompt_{index:02d}.txt").write_text(prompt)
        record = {"turn": index, "resumed_thread_id": self.thread_id, "status": "running"}
        self.turns.append(record)
        self.save()
        try:
            code, _, err = await self.environment.exec(
                self.argv(), data=prompt.encode(), timeout=timeout, output=trace, limit=64 * 1024**2)
            (self.directory / f"turn_{index:02d}.stderr").write_bytes(err)
            record.update(status="complete" if code == 0 else "failed", exit_code=code)
            if code:
                self.failed = True
        except BaseException as exc:
            self.failed = True
            record.update(status="interrupted", error_type=type(exc).__name__)
            raise
        finally:
            if trace.exists():
                for line in trace.read_text(errors="replace").splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "thread.started":
                        try:
                            candidate = str(uuid.UUID(str(event.get("thread_id"))))
                        except ValueError:
                            self.failed = True
                            record["status"] = "invalid_thread_id"
                            continue
                        if self.thread_id and candidate != self.thread_id:
                            self.failed = True
                            record["status"] = "session_mismatch"
                        else:
                            self.thread_id = candidate
                    if event.get("type") == "turn.completed":
                        record["usage"] = event.get("usage")
                    if event.get("type") == "turn.failed":
                        self.failed = True
                        record["status"] = "failed"
            self.last_message = final_message(trace)
            record["final_message_bytes"] = len(self.last_message.encode("utf-8"))
            self.save()
        if not self.thread_id:
            self.failed = True
            self.save()
            raise SandboxError("Codex returned no thread ID")
        if self.failed:
            raise SandboxError("Codex turn failed")


def final_message(trace: Path):
    """Last agent message in a Codex JSONL trace."""
    text = ""
    if trace.exists():
        for line in trace.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
                item = event.get("item", {})
                if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                    if isinstance(item.get("text"), str):
                        text = item["text"]
            except (ValueError, AttributeError):
                continue
    return text
