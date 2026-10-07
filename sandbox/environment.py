"""One isolated Docker container per question. Agent code never runs in the host process.

Ported from finautorubric ``final_pipeline/environment.py`` (per-query container only).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import io
from pathlib import Path
import re
import tarfile
import uuid

from .contamination import BLOCKED_HOSTS


class SandboxError(RuntimeError):
    pass


MAX_SANDBOX_FILE_BYTES = 256 * 1024**2


async def command(argv, *, data=None, timeout=30, output=None, limit=8 * 1024**2):
    """Bound output and kill the local Docker client on timeout/cancellation.

    The caller must also stop the container to terminate an in-container exec.
    Never log stdin: setup uses it to transfer credentials.
    """
    process = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    async def read(stream, path=None):
        chunks, size = [], 0
        handle = path.open("wb") if path else None
        try:
            while chunk := await stream.read(65536):
                size += len(chunk)
                if size > limit:
                    raise SandboxError("subprocess output exceeded byte limit")
                if handle:
                    handle.write(chunk)
                    handle.flush()
                else:
                    chunks.append(chunk)
            return b"".join(chunks)
        finally:
            if handle:
                handle.close()

    async def write():
        try:
            if data:
                process.stdin.write(data)
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    tasks = [asyncio.create_task(read(process.stdout, output)),
             asyncio.create_task(read(process.stderr)), asyncio.create_task(write())]
    try:
        async with asyncio.timeout(timeout):
            stdout, stderr, _ = await asyncio.gather(*tasks)
            code = await process.wait()
            return code, stdout, stderr
    finally:
        forced = process.returncode is None
        if forced:
            process.kill()
        # Stop the pipe tasks before waiting; otherwise wait() can deadlock on a
        # full PIPE after a bounded reader has gone away.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        transport = getattr(process, "_transport", None)
        if forced and transport is not None:
            # On macOS a killed process can stay unreported while an inherited
            # Docker stdout pipe is open; closing the transport lets the loop reap it.
            transport.close()
        try:
            async with asyncio.timeout(2):
                await process.wait()
        except TimeoutError:
            if transport is not None:
                transport.close()


@dataclass(frozen=True)
class DockerConfig:
    image: str
    cpus: float = 1.0
    memory_mb: int = 2048
    pids_limit: int = 256
    executable: str = "docker"
    blocked_hosts: tuple = BLOCKED_HOSTS
    max_file_bytes: int = MAX_SANDBOX_FILE_BYTES

    def __post_init__(self):
        if not self.image or self.image.startswith("-"):
            raise ValueError("a prebuilt Docker image is required")
        if self.cpus <= 0 or self.memory_mb < 128 or self.pids_limit < 64 or self.max_file_bytes <= 0:
            raise ValueError("invalid sandbox resource limits")
        if any(not re.fullmatch(r"[A-Za-z0-9.-]+", host) for host in self.blocked_hosts):
            raise ValueError("blocked hosts must be plain hostnames")


class DockerEnvironment:
    def __init__(self, config: DockerConfig):
        self.config = config
        self.name = "sandbox-" + uuid.uuid4().hex
        self.tmp_root = "/tmp/.sandbox/" + self.name
        self.container_id = None
        self.stopped = False
        self.paused = False

    def create_command(self):
        c = self.config
        # No bind mounts, host network, Docker socket, or inherited home directory.
        # Private anonymous volumes remain accessible to Docker's archive API.
        return [c.executable, "create", "--name", self.name, "--init", "--read-only",
                "--user", "1000:1000", "--workdir", "/work", "--network", "bridge",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", str(c.pids_limit), "--cpus", str(c.cpus), "--memory", f"{c.memory_mb}m",
                "--ulimit", f"fsize={c.max_file_bytes}:{c.max_file_bytes}",
                "--mount", "type=volume,destination=/work",
                "--mount", "type=volume,destination=/home/agent",
                "--mount", "type=volume,destination=/tmp",
                # Benchmark/code-hosting hosts resolve to 0.0.0.0; the agent user
                # cannot edit /etc/hosts. Server-side web search is not covered.
                *[arg for host in c.blocked_hosts for arg in ("--add-host", host + ":0.0.0.0")],
                "--env", "HOME=/home/agent", "--env", "CODEX_HOME=/home/agent/.codex",
                "--env", "TMPDIR=" + self.tmp_root,
                "--entrypoint", "sleep", c.image, "infinity"]

    async def start(self):
        try:
            code, out, err = await command(self.create_command())
            if code:
                raise SandboxError("Docker create failed: " + err.decode(errors="replace"))
            self.container_id = out.decode().strip()
            code, _, err = await command([self.config.executable, "start", self.name])
            if code:
                raise SandboxError("Docker start failed: " + err.decode(errors="replace"))
            # Codex >= 0.159 refuses to start when CODEX_HOME does not exist.
            script = "import pathlib,sys; [pathlib.Path(p).mkdir(parents=True,exist_ok=True) for p in sys.argv[1:]]"
            code, _, err = await self.exec(["python3", "-c", script, self.tmp_root, "/home/agent/.codex"])
            if code:
                raise SandboxError("could not initialize sandbox scratch: " + err.decode(errors="replace")[:500])
        except BaseException:
            await self.close()
            raise

    async def exec(self, argv, *, data=None, timeout=30, output=None, limit=8 * 1024**2):
        if not self.container_id or self.stopped:
            raise SandboxError("sandbox is not running")
        return await command([self.config.executable, "exec", "-i", "--workdir", "/work", self.name, *argv],
                             data=data, timeout=timeout, output=output, limit=limit)

    async def put(self, path, data):
        # Only controller-chosen files are written; no source repository copy.
        script = ("import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                  "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read()); "
                  "os.chmod(p,0o600)")
        code, _, _ = await self.exec(["python3", "-c", script, path], data=data)
        if code:
            raise SandboxError("could not initialize sandbox file")

    async def read_file(self, path):
        # Read via the Docker API, not agent-modifiable Python/tar commands.
        code, out, _ = await command([self.config.executable, "cp", f"{self.name}:{path}", "-"])
        if code:
            return None
        try:
            with tarfile.open(fileobj=io.BytesIO(out), mode="r:") as archive:
                members = archive.getmembers()
                if len(members) != 1 or not members[0].isfile():
                    raise SandboxError("result must be a regular file, not a link or directory")
                if members[0].size > 4 * 1024**2:
                    raise SandboxError("result file exceeds byte limit")
                return archive.extractfile(members[0]).read()
        except tarfile.TarError as exc:
            raise SandboxError("invalid result archive from Docker") from exc

    async def collect(self, destination: Path, *, sessions="/home/agent/.codex/sessions"):
        """Store opaque archives; never extract agent-authored paths on the host.

        Auth and the rest of CODEX_HOME are deliberately excluded.
        """
        warnings = []
        destination.mkdir(parents=True, exist_ok=True)
        for source, filename in (("/work/.", "work.tar"), (sessions, "sessions.tar")):
            try:
                code, _, err = await command(
                    [self.config.executable, "cp", f"{self.name}:{source}", "-"],
                    output=destination / filename, limit=128 * 1024**2, timeout=30)
                if code:
                    raise SandboxError(err.decode(errors="replace")[:500])
            except (SandboxError, TimeoutError) as exc:
                warnings.append(f"{filename}: {exc}")
                (destination / filename).unlink(missing_ok=True)
        return warnings

    async def freeze(self):
        # Pause before copying so a timed-out child cannot keep changing files.
        if self.container_id and not self.stopped and not self.paused:
            code, _, _ = await command([self.config.executable, "pause", self.name])
            if code:
                raise SandboxError("could not pause sandbox for artifact recovery")
            self.paused = True

    async def close(self):
        try:
            code, _, err = await command([self.config.executable, "rm", "-f", "--volumes", self.name])
            if code and b"No such container" not in err:
                raise SandboxError("Docker cleanup failed: " + err.decode(errors="replace")[:500])
        finally:
            self.stopped = True
