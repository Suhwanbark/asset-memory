"""Execution-only handoff: preserve in-flight workers and run seven conditions.

The original frozen experiment module, inputs, answers and scoring stay intact.
This coordinator replaces only its scheduler; it never regenerates an answer.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from finrate_poc.runtime import (
    ROOT, DiskLow, check_disk, cpu_only, file_digest, write_json,
)
from finrate_final_evaluation import single_executor
import finrate_completed100_v1 as experiment

BASE = experiment.BASE
RUN = BASE / 'parallel7'
ORDER = ['naive_llm', 'raw_agent', 'bank_agent', 'embedding', 'hybrid',
         'chunk_agent', 'flat_bank_agent']


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(pid):
    """Include kernel start time so a recycled PID cannot be adopted/signalled."""
    try:
        root = Path('/proc') / str(pid)
        tail = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'state': tail[0], 'pgid': int(tail[2]),
                'session': int(tail[3]), 'start_ticks': int(tail[19]),
                'uid': root.stat().st_uid,
                'argv': (root / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_process(saved, current):
    return bool(current and current['state'] not in {'Z', 'X'} and all(
        saved[k] == current[k] for k in ['pid', 'start_ticks', 'pgid', 'session', 'uid', 'argv']))


def alive(saved):
    return same_process(saved, identity(saved['pid']))


def expected_argv(condition):
    return [experiment.PYTHON, '-u', '-m', experiment.MODULE,
            'condition', '--condition', condition]


def owned_group_exists(pgid, latest_start):
    # A group recreated after handoff has no member this old. This also covers
    # still-running model children after an adopted worker's disk watchdog exits.
    for root in Path('/proc').iterdir():
        if not root.name.isdigit():
            continue
        p = identity(int(root.name))
        if (p and p['uid'] == os.getuid() and p['pgid'] == p['session'] == pgid
                and p['state'] not in {'Z', 'X'} and p['start_ticks'] <= latest_start):
            return True
    return False


def disk_watchdog(old_group, handoff_ticks):
    stop = threading.Event()

    def watch():
        while not stop.wait(2):
            try:
                check_disk()
            except DiskLow as exc:
                try:
                    (BASE / 'status.json').write_text(json.dumps({
                        'state': 'stopped_low_disk', 'reason': str(exc),
                        'pid': os.getpid(), 'time': time.time()}) + '\n')
                finally:
                    if owned_group_exists(old_group, handoff_ticks):
                        try:
                            os.killpg(old_group, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                    os._exit(75)

    threading.Thread(target=watch, daemon=True).start()
    return stop


def handoff():
    experiment.verify()
    if not (BASE / 'retrieval_manifest.json').exists():
        raise RuntimeError('Both retrieval conditions must already be prepared')
    original = read_json(BASE / 'launch.json')
    parent = identity(original['pid'])
    if not (parent and parent['argv'] == original['argv']
            and parent['pgid'] == parent['session'] == original['pgid']
            and parent['uid'] == os.getuid()):
        raise RuntimeError('Original coordinator identity differs; inspect before handoff')
    if (RUN / 'override.json').exists():
        raise RuntimeError('Handoff already recorded; do not repeat it')
    paused = False
    terminated = False
    try:
        os.kill(parent['pid'], signal.SIGSTOP)
        paused = True
        deadline = time.monotonic() + 5
        while identity(parent['pid'])['state'] not in {'T', 't'}:
            if time.monotonic() >= deadline:
                raise RuntimeError('Coordinator did not pause')
            time.sleep(.05)
        snapshots = {}
        adopted = {}
        completed = []
        for c in ORDER:
            folder = BASE / 'workers' / c
            launch = folder / 'launch.json'
            if not launch.exists():
                continue
            record = read_json(launch)
            status = read_json(folder / 'status.json') if (folder / 'status.json').exists() else {}
            snapshots[c] = {'launch': record, 'status': status}
            current = identity(record['pid'])
            if current and current['state'] not in {'Z', 'X'}:
                if not (current['argv'] == expected_argv(c)
                        and current['pgid'] == current['session'] == original['pgid']
                        and current['uid'] == os.getuid()):
                    raise RuntimeError('Unexpected existing worker: ' + c)
                adopted[c] = current
            elif status.get('state') == 'completed':
                completed.append(c)
            else:
                raise RuntimeError('Existing worker ended without completion: ' + c)
        write_json(RUN / 'before_switch.json', {
            'original_launch': original, 'original_status': read_json(BASE / 'status.json'),
            'original_identity': parent, 'workers': snapshots, 'time': time.time(),
        }, immutable=True)
        write_json(RUN / 'override.json', {
            'reason': 'User explicitly requested worker 7 and all methods in parallel.',
            'changes': {'worker_concurrency': {'before': 2, 'after': 7}},
            'original_manifest_sha256': file_digest(BASE / 'manifest.json'),
            'coordinator_sha256': file_digest(__file__), 'adopted': adopted,
            'already_completed': completed, 'new_conditions': [
                c for c in ORDER if c not in adopted and c not in completed],
            'frozen_experiment_unchanged': True, 'failed_answers_regenerated': 0,
            'cpu_only': True, 'min_free_disk_bytes': 10_000_000_000,
            'pid': os.getpid(), 'pgid': os.getpgrp(), 'time': time.time(),
        }, immutable=True)
        # SIGTERM targets only the paused scheduler. SIGCONT lets it exit;
        # neither signal targets its existing workers/process group.
        os.kill(parent['pid'], signal.SIGTERM)
        terminated = True
        os.kill(parent['pid'], signal.SIGCONT)
        deadline = time.monotonic() + 10
        while alive(parent):
            if time.monotonic() >= deadline:
                raise RuntimeError('Old coordinator did not exit')
            time.sleep(.1)
        return adopted, completed
    finally:
        if paused and not terminated and alive(parent):
            os.kill(parent['pid'], signal.SIGCONT)


def supervise(adopted, completed):
    active = dict(adopted)
    children = {}
    failed = []
    with single_executor(BASE / 'locks/pipeline.lock'):
        for c in ORDER:
            if c in active or c in completed:
                continue
            check_disk()
            folder = BASE / 'workers' / c
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'launch.json').exists():
                raise RuntimeError('Refusing duplicate worker: ' + c)
            argv = expected_argv(c)
            with (folder / 'stdout.log').open('a') as log:
                child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'PYTHONPATH': 'method'})
            children[c] = child
            current = identity(child.pid)
            if not current:
                raise RuntimeError('New worker exited before registration: ' + c)
            active[c] = current
            write_json(folder / 'launch.json', {
                'pid': child.pid, 'pgid': os.getpgrp(), 'argv': argv,
                'started': time.time(), 'coordinator': str(RUN / 'override.json'),
            }, immutable=True)
        write_json(RUN / 'workers_started.json', {'workers': active,
                   'adopted_conditions': list(adopted), 'time': time.time()}, immutable=True)
        while active:
            check_disk()
            for c, saved in list(active.items()):
                rc = children[c].poll() if c in children else None
                if alive(saved):
                    continue
                path = BASE / 'workers' / c / 'status.json'
                status = read_json(path) if path.exists() else {}
                ready = all((BASE / 'reports' / name).exists()
                            for name in [c + '.json', c + '_grounding.json'])
                if status.get('state') == 'completed' and ready and rc in {0, None}:
                    completed.append(c)
                else:
                    failed.append({'condition': c, 'returncode': rc, 'status': status})
                del active[c]
            write_json(BASE / 'status.json', {
                'state': 'needs_review' if failed else 'running', 'active': list(active),
                'completed_conditions': completed, 'failed': failed, 'queued': [],
                'pid': os.getpid(), 'questions': 100, 'conditions': 7,
                'worker_concurrency': 7, 'runtime_override': str(RUN / 'override.json'),
                'time': time.time(),
            })
            if active:
                time.sleep(2)
        if failed:
            return
        experiment.collect()
        write_json(BASE / 'status.json', {
            'state': 'completed', 'questions': 100, 'predictions': 700,
            'finished': time.time(), 'pid': os.getpid(), 'heldout_used': False,
            'worker_concurrency': 7, 'runtime_override': str(RUN / 'override.json'),
        })


def main():
    cpu_only()
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    check_disk()
    os.nice(10)
    if os.getpgrp() != os.getpid():
        raise RuntimeError('Coordinator requires a dedicated process group')
    original = read_json(BASE / 'launch.json')
    ticks = int(float(Path('/proc/uptime').read_text().split()[0]) * os.sysconf('SC_CLK_TCK'))
    stop = disk_watchdog(original['pgid'], ticks)
    try:
        with single_executor(RUN / 'handoff.lock'):
            adopted, completed = handoff()
            supervise(adopted, completed)
    except Exception as exc:
        write_json(RUN / 'error.json', {'state': 'failed', 'error': repr(exc),
                                      'pid': os.getpid(), 'time': time.time()})
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
