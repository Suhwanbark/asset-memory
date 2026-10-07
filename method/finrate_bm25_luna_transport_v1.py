"""Long-context variant of the validated Luna transport; no changes to frozen runs.

The only inference changes are explicit context/compaction limits. GPT-6 Luna's
published context window is 1,050,000 tokens. Actual acceptance, compaction and
provider usage are validated from CLI events; no text truncation is allowed.
"""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
import time

import run_finrate_gold20_gpt56_luna_judge6_v1 as shared
from run_finrate_gold20_gpt56_luna_judge6_v1 import (
    sha, freeze_text, write, now, STOP, LOCK, ACTIVE, TIMEOUT, DISABLED, ISOLATION,
    inspect_events, stop_process, check_disk,
)

def invoke(prompt, folder, context_window=1050000, auto_compact_token_limit=950000):
    shared.check_disk()
    assert folder.name in {"answer", "judge"}
    call_model = "gpt-6-luna"
    identity = {"model": call_model, "reasoning_effort": "medium", "prompt_sha256": sha(prompt),
                "context_window": context_window, "auto_compact_token_limit": auto_compact_token_limit}
    cached = folder / "response.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        assert all(result[k] == v for k, v in identity.items())
        return result
    freeze_text(folder / "prompt.txt", prompt)
    last_error = None
    for attempt in range(1, 3):
        if STOP.is_set():
            raise RuntimeError("Stopped")
        attempt_dir = folder / f"attempt_{attempt:02d}"
        if (attempt_dir / "outcome.json").exists():
            last_error = json.loads((attempt_dir / "outcome.json").read_text()).get("error")
            continue
        if (attempt_dir / "request.json").exists():
            raise RuntimeError("Prior unfinished attempt retained; inspect before resume")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="finrate-bm25-top10-") as tmp:
            argv = ["/home/hob/.local/bin/codex", "exec", "--json", "--ephemeral",
                    "--ignore-user-config", "--skip-git-repo-check", "-s", "read-only",
                    "-m", call_model, "-c", 'model_reasoning_effort="medium"',
                    "-c", f"model_context_window={context_window}", "-c", f"model_auto_compact_token_limit={auto_compact_token_limit}",
                    "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0",
                    "-c", "mcp_servers={}", "-c", "apps._default.enabled=false",
                    "-c", "developer_instructions=" + json.dumps(ISOLATION),
                    "--enable", "skip_host_skill_discovery"]
            for feature in DISABLED:
                argv += ["--disable", feature]
            argv += ["-C", tmp, "-"]
            env = os.environ.copy()
            for key in ("CODEX_REMOTE_PAYLOAD", "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       TOKENIZERS_PARALLELISM="false")
            write(attempt_dir / "request.json", {**identity, "started": now(), "argv": argv,
                  "prompt_file": str(folder / "prompt.txt"), "attempt": attempt}, frozen=True)
            started = time.monotonic()
            error = None
            with (folder / "prompt.txt").open("rb") as inp, (attempt_dir / "events.jsonl").open("wb") as out, (attempt_dir / "stderr.txt").open("wb") as err:
                proc = subprocess.Popen(argv, stdin=inp, stdout=out, stderr=err,
                                        cwd=tmp, env=env, start_new_session=True)
                with LOCK:
                    ACTIVE[proc.pid] = proc
                try:
                    while proc.poll() is None:
                        check_disk()
                        if STOP.is_set():
                            raise RuntimeError("Run stopped")
                        if time.monotonic() - started > TIMEOUT:
                            raise TimeoutError("Model transport timeout")
                        time.sleep(0.5)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    stop_process(proc)
                finally:
                    with LOCK:
                        ACTIVE.pop(proc.pid, None)
            events = []
            for line in (attempt_dir / "events.jsonl").read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            valid, messages, usage, warnings, forbidden = inspect_events(events)
            compaction = [e for e in events if "compact" in e.get("type", "").lower()]
            outcome = {"returncode": proc.returncode, "elapsed_s": time.monotonic() - started,
                       "usage": usage, "forbidden_item_types": forbidden, "compaction_events": compaction,
                       "informational_warnings": warnings, "valid_completed_turn": valid,
                       "error": error, "finished": now()}
            if not error and proc.returncode == 0 and valid and not compaction:
                result = {**identity, **outcome, "text": messages[-1], "attempt": attempt,
                          "usage_source": "provider_reported_turn.completed; prompt estimate is separate"}
                write(attempt_dir / "outcome.json", outcome, frozen=True)
                write(cached, result, frozen=True)
                freeze_text(folder / "response.txt", result["text"])
                return result
            event_errors = [e.get("message", e.get("error")) for e in events
                            if e.get("type") in {"error", "turn.failed"}]
            outcome["error"] = error or f"exit={proc.returncode}; tools={forbidden}; errors={event_errors}; final_message={bool(messages)}"
            write(attempt_dir / "outcome.json", outcome, frozen=True)
            last_error = outcome["error"]
            if forbidden or compaction or messages:
                break  # No content-dependent resampling.
        if attempt < 2 and not STOP.is_set():
            STOP.wait(5)
    raise RuntimeError(f"No accepted response: {last_error}")
