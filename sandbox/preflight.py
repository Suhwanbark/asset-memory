"""Model-free checks of the Docker image and sandbox envelope before a run.

Condensed from finautorubric ``final_pipeline/environment_checks.py``.
"""
from __future__ import annotations

from dataclasses import replace
import json
import time

from .environment import DockerConfig, DockerEnvironment, SandboxError, command
from .session import ENABLED_FEATURES

ENVIRONMENT_REVISION = "asset-memory.sandbox.env/v1"
REVISION_LABEL = "io.linq.sandbox.environment-revision"
CHECK_SCRIPT = "/opt/sandbox-environment/environment_check.py"
RESOLVE_SCRIPT = ("import json,socket,sys; "
                  "print(json.dumps({h: socket.gethostbyname(h) for h in sys.argv[1:]}))")


async def inspect_image(config: DockerConfig) -> dict:
    code, out, err = await command([config.executable, "image", "inspect", config.image])
    if code:
        raise SandboxError("build the sandbox image first: " + err.decode(errors="replace").strip())
    image = json.loads(out)[0]
    if image["Config"].get("Volumes"):
        raise SandboxError("sandbox image must not declare implicit volumes")
    labels = image["Config"].get("Labels") or {}
    if labels.get(REVISION_LABEL) != ENVIRONMENT_REVISION:
        raise SandboxError("sandbox image environment revision mismatch")
    return {"image_id": image["Id"], "repo_tags": image.get("RepoTags") or [],
            "architecture": image.get("Architecture"), "labels": labels}


async def preflight(config: DockerConfig) -> dict:
    """Start one real container and check tools, isolation and Codex; no model calls."""
    report = {"ready": False, "image": config.image, "checked_at_unix": time.time(), "model_calls": 0,
              "checks": {}}
    environment = None
    try:
        code, out, err = await command([config.executable, "version", "--format", "{{.Server.Version}}"])
        if code:
            raise SandboxError(err.decode(errors="replace").strip())
        report["docker_server"] = out.decode().strip()
        report.update(await inspect_image(config))
        environment = DockerEnvironment(replace(config, image=report["image_id"]))
        await environment.start()

        code, out, err = await environment.exec(
            ["python", CHECK_SCRIPT, "--expected-tmp", environment.tmp_root], timeout=180, limit=4 * 1024**2)
        functional = json.loads(out) if out.strip() else {"ready": False, "stderr": err.decode(errors="replace")}
        report["checks"]["functional_environment"] = {"passed": bool(functional.get("ready")) and code == 0,
                                                      "detail": functional}

        code, out, err = await environment.exec(["python", "-c", RESOLVE_SCRIPT, *config.blocked_hosts])
        resolved = json.loads(out) if code == 0 else {}
        report["checks"]["blocked_hosts"] = {
            "passed": code == 0 and all(resolved.get(host) == "0.0.0.0" for host in config.blocked_hosts),
            "detail": resolved or err.decode(errors="replace")[-500:]}

        code, version, _ = await environment.exec(["codex", "--version"])
        _, features, _ = await environment.exec(["codex", "features", "list"])
        available = {line.split()[0] for line in features.decode().splitlines() if line.split()}
        report["checks"]["codex"] = {"passed": code == 0 and set(ENABLED_FEATURES) <= available,
                                     "version": version.decode().strip(),
                                     "missing_features": sorted(set(ENABLED_FEATURES) - available)}
        report["ready"] = all(check["passed"] for check in report["checks"].values())
    except (OSError, SandboxError, TimeoutError, ValueError, KeyError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if environment is not None:
            try:
                await environment.close()
            except Exception as exc:
                report["ready"] = False
                report["cleanup_error"] = str(exc)
    return report
