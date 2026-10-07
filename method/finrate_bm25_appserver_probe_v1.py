"""Check the documented structured app-server input path with an actual fixed query.

Preserve all request/response events. A successful answer is never regenerated.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

import run_finrate_gold20_gpt56_luna_judge6_v1 as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/finrate_bm25_appserver_probe_v1"
QID = "qa_3_001691"


def main():
    shared.check_disk()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "request.jsonl").exists():
        raise RuntimeError("Existing probe preserved; do not regenerate")
    prompt = (ROOT / "result/finrate_bm25_top10_gpt6_luna_v2" / QID / "answer/prompt.txt").read_text()
    # Each text part is contiguous; concatenating all parts is exactly the prompt.
    parts = [prompt[i:i + 500000] for i in range(0, len(prompt), 500000)]
    assert "".join(parts) == prompt
    shared.freeze_text(OUT / "prompt.txt", prompt)
    with tempfile.TemporaryDirectory(prefix="finrate-bm25-appserver-") as tmp:
        cfg = {"model": "gpt-6-luna", "model_reasoning_effort": "medium",
               "model_context_window": 1050000, "model_auto_compact_token_limit": 950000,
               "web_search": "disabled", "project_doc_max_bytes": 0,
               "mcp_servers": {}, "apps._default.enabled": False,
               "developer_instructions": shared.ISOLATION,
               "features.skip_host_skill_discovery": True}
        for feature in shared.DISABLED:
            cfg[f"features.{feature}"] = False
        argv = ["/home/hob/.local/bin/codex", "app-server", "--listen", "stdio://"]
        for key, val in cfg.items():
            argv += ["-c", f"{key}={json.dumps(val)}"]
        env = os.environ.copy()
        for key in ("CODEX_REMOTE_PAYLOAD", "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (OUT / "stderr.txt").open("wb") as err, (OUT / "events.jsonl").open("w") as events_out, (OUT / "request.jsonl").open("w") as requests:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=err, text=True, bufsize=1, cwd=tmp, env=env, start_new_session=True)
            incoming = queue.Queue()
            def reader():
                for line in proc.stdout:
                    events_out.write(line); events_out.flush()
                    incoming.put(json.loads(line))
                incoming.put(None)
            threading.Thread(target=reader, daemon=True).start()
            started = time.monotonic()
            def send(message):
                data = json.dumps(message, ensure_ascii=False)
                requests.write(data + "\n"); requests.flush()
                proc.stdin.write(data + "\n"); proc.stdin.flush()
            def get():
                while True:
                    shared.check_disk()
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Probe exceeded 600s")
                    try:
                        event = incoming.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if event is None:
                        raise RuntimeError("App-server exited")
                    return event
            def rpc(ident, method, params):
                send({"id": ident, "method": method, "params": params})
                while True:
                    event = get()
                    if event.get("id") == ident:
                        if "error" in event:
                            raise RuntimeError(f"{method}: {event['error']}")
                        return event["result"]
            try:
                rpc(1, "initialize", {"clientInfo": {"name": "finrate_bm25", "version": "1.0"},
                                      "capabilities": {"experimentalApi": True}})
                send({"method": "initialized"})
                thread = rpc(2, "thread/start", {"model": "gpt-6-luna", "cwd": tmp,
                             "approvalPolicy": "never", "sandbox": "read-only", "ephemeral": True,
                             "developerInstructions": shared.ISOLATION, "dynamicTools": [],
                             "config": cfg})
                tid = thread["thread"]["id"]
                result = rpc(3, "turn/start", {"threadId": tid,
                             "input": [{"type": "text", "text": part} for part in parts],
                             "model": "gpt-6-luna", "effort": "medium"})
                print(json.dumps({"accepted": True, "parts": len(parts), "chars": len(prompt)}), flush=True)
                while True:
                    event = get()
                    if event.get("method") == "turn/completed":
                        shared.write(OUT / "completion.json", event, frozen=True)
                        print(json.dumps({"completed": event["params"]["turn"]["status"]}), flush=True)
                        break
                    if event.get("id") is not None and event.get("method"):
                        raise RuntimeError("Unexpected server tool/request")
            except Exception as exc:
                shared.write(OUT / "error.json", {"error": str(exc)}, frozen=True)
                print(json.dumps({"error": str(exc)}), flush=True)
            finally:
                shared.stop_process(proc)


if __name__ == "__main__":
    main()
