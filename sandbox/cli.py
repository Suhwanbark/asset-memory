"""python -m sandbox {build-image,preflight,run,scan}"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from .auth import load_codex_auth
from .contamination import detect
from .environment import DockerConfig, SandboxError, command
from .preflight import preflight
from .run import PROMPT_PATH, load_questions, run_batch
from .session import save_json

RESOURCES = Path(__file__).resolve().parent / "resources"
DEFAULT_IMAGE = "asset-memory-sandbox:v1"


def _docker_config(args) -> DockerConfig:
    return DockerConfig(image=args.image, executable=args.docker, cpus=args.cpus, memory_mb=args.memory_mb)


def _git_revision() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


async def _build_image(args) -> dict:
    dockerfile = RESOURCES / "Dockerfile"
    code, _, err = await command([args.docker, "build", "-f", str(dockerfile), "-t", args.image, str(RESOURCES)],
                                 timeout=1800, limit=64 * 1024**2)
    if code:
        raise SandboxError("Docker build failed: " + err.decode(errors="replace")[-2000:])
    return await preflight(_docker_config(args))


async def _run(args) -> dict:
    questions = load_questions(args.inputs)
    if args.q_id:
        wanted = set(args.q_id)
        questions = [q for q in questions if q["q_id"] in wanted]
        missing = wanted - {q["q_id"] for q in questions}
        if missing:
            raise ValueError("q_id not in inputs: " + ", ".join(sorted(missing)))
    if args.limit is not None:
        questions = questions[:args.limit]
    auth, auth_mode = load_codex_auth(args.auth_json, args.api_key_env)
    report = await preflight(_docker_config(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "preflight.json", report)
    if not report["ready"]:
        raise SandboxError("preflight failed; see " + str(args.output_dir / "preflight.json"))
    # Every container runs the exact image ID that passed preflight.
    docker = replace(_docker_config(args), image=report["image_id"])
    provenance = {"git": _git_revision(), "image_tag": args.image, "docker_server": report["docker_server"],
                  "codex_version": report["checks"]["codex"]["version"], "auth_mode": auth_mode,
                  "host_python": platform.python_version(), "inputs": str(args.inputs),
                  "inputs_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                  "question_ids": [q["q_id"] for q in questions]}
    return await run_batch(questions, args.output_dir, docker=docker, auth=auth, model=args.model,
                           effort=args.effort, timeout=args.timeout, concurrency=args.concurrency,
                           template=PROMPT_PATH.read_text(), provenance=provenance)


def _scan(args) -> dict:
    """Re-run contamination detection on a finished run, e.g. after the markers change."""
    counts = {}
    for result_path in sorted(args.run_dir.glob("*/result.json")):
        directory = result_path.parent
        question = json.loads((directory / "input.json").read_text())["question"]
        report = detect(directory, question)
        save_json(directory / "contamination.json", report)
        counts[directory.name] = report["status"]
    return {"scanned": len(counts), "status": counts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sandbox")
    sub = parser.add_subparsers(dest="command", required=True)

    def docker_flags(p):
        p.add_argument("--image", default=DEFAULT_IMAGE)
        p.add_argument("--docker", default="docker", help="Docker executable")
        p.add_argument("--cpus", type=float, default=1.0, help="CPU limit per question container")
        p.add_argument("--memory-mb", type=int, default=2048, help="memory limit per question container")

    docker_flags(sub.add_parser("build-image", help="build the image, then run preflight"))
    docker_flags(sub.add_parser("preflight", help="check image, isolation and Codex without model calls"))

    run = sub.add_parser("run", help="answer questions, one sandbox each")
    docker_flags(run)
    run.add_argument("--inputs", type=Path, required=True, help="inputs.json with q_id, task, question")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model", default="gpt-6-luna")
    run.add_argument("--effort", default="medium", help="model_reasoning_effort")
    run.add_argument("--timeout", type=float, default=900, help="wall seconds per question")
    run.add_argument("--concurrency", type=int, default=20)
    run.add_argument("--limit", type=int, help="only the first N questions")
    run.add_argument("--q-id", action="append", help="only these q_ids (repeatable)")
    run.add_argument("--auth-json", type=Path, help="Codex ChatGPT login file (default $CODEX_HOME/auth.json)")
    run.add_argument("--api-key-env", help="name of an environment variable holding an OpenAI API key")

    scan = sub.add_parser("scan", help="re-run contamination detection on a run directory")
    scan.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-image":
            report = asyncio.run(_build_image(args))
        elif args.command == "preflight":
            report = asyncio.run(preflight(_docker_config(args)))
        elif args.command == "run":
            if args.concurrency < 1:
                raise ValueError("--concurrency must be at least 1")
            report = asyncio.run(_run(args))
        else:
            report = _scan(args)
    except (OSError, SandboxError, ValueError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ready", True) else 1
