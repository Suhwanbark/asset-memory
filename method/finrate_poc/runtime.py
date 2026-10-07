"""CPU-only execution, disk guard, immutable artifacts and explicit status."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
MIN_FREE_BYTES = 10_000_000_000


class DiskLow(RuntimeError):
    pass


def cpu_only():
    # Set before importing any tensor library. Never inspect or allocate a GPU.
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                      MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")


def check_disk(paths=(ROOT, Path(tempfile.gettempdir())), *, minimum=MIN_FREE_BYTES):
    observed = {}
    for path in paths:
        free = shutil.disk_usage(path).free
        observed[str(path)] = free
        if free < minimum:
            raise DiskLow(f"Free disk at {path}: {free} < {minimum} bytes")
    return observed


def digest(value):
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj, *, immutable=False):
    check_disk()
    path = Path(path)
    content = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if immutable and path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(content)
    os.replace(temp, path)


def append_json(path, obj):
    check_disk()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def watchdog(status_path, *, interval=2):
    """Only kill this worker's dedicated process group, never other server jobs."""
    done = threading.Event()

    def watch():
        while not done.wait(interval):
            try:
                check_disk()
            except DiskLow as exc:
                # A tiny final status is best effort; do not prune anyone's files.
                try:
                    Path(status_path).write_text(json.dumps({
                        "state": "stopped_low_disk", "reason": str(exc),
                        "pid": os.getpid(), "time": time.time()}) + "\n")
                finally:
                    if os.getpgrp() == os.getpid():
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                    os._exit(75)

    threading.Thread(target=watch, daemon=True).start()
    return done


def command(argv, *, prompt, timeout=420, cwd=None):
    """Bounded child, with disk monitoring throughout the call."""
    check_disk()
    with tempfile.TemporaryFile(mode="w+t") as out, tempfile.TemporaryFile(mode="w+t") as err:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                text=True, cwd=cwd)
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
            started = time.monotonic()
            while proc.poll() is None:
                check_disk()
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"Model call exceeded {timeout}s")
                time.sleep(1)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        out.seek(0)
        err.seek(0)
        return proc.returncode, out.read(), err.read()
