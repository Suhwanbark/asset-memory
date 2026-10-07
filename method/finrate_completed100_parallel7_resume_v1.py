"""Finish the recorded handoff after the old scheduler fully released its lock.

The first coordinator exited during the old process's brief lock-release race.
Its adopted inference workers never stopped. Preserve that attempt and adopt
only the exact recorded live processes, without repeating any model call.
"""
import os
from pathlib import Path
import time

import finrate_completed100_parallel7_v1 as scheduler
from finrate_poc.runtime import check_disk, cpu_only, file_digest, write_json
from finrate_final_evaluation import single_executor


def main():
    cpu_only()
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    check_disk()
    os.nice(10)
    if os.getpgrp() != os.getpid():
        raise RuntimeError('Dedicated process group required')
    prior = scheduler.RUN
    before = scheduler.read_json(prior / 'before_switch.json')
    override = scheduler.read_json(prior / 'override.json')
    launch = scheduler.read_json(prior / 'launch.json')
    for pid in [before['original_identity']['pid'], launch['pid']]:
        if scheduler.identity(pid) is not None:
            raise RuntimeError('Prior coordinator still present; wait for full exit')
    for name, sha in launch['code_sha256'].items():
        if file_digest(scheduler.ROOT / name) != sha:
            raise RuntimeError('Prior handoff code changed')
    scheduler.experiment.verify()
    adopted = override['adopted']
    if not all(scheduler.alive(p) for p in adopted.values()):
        raise RuntimeError('An adopted worker changed; inspect before resuming')
    for c in override['new_conditions']:
        if (scheduler.BASE / 'workers' / c / 'launch.json').exists():
            raise RuntimeError('A new condition was already launched: ' + c)
    scheduler.RUN = prior / 'continuation_v1'
    write_json(scheduler.RUN / 'override.json', {
        **override, 'pid': os.getpid(), 'pgid': os.getpgrp(), 'time': time.time(),
        'continuation_of': str(prior / 'override.json'),
        'coordinator_sha256': file_digest(__file__),
        'reason': 'Complete authorized worker7 handoff after old scheduler lock release; exact live workers retained.',
    }, immutable=True)
    old_group = before['original_identity']['pgid']
    ticks = int(float(Path('/proc/uptime').read_text().split()[0]) * os.sysconf('SC_CLK_TCK'))
    stop = scheduler.disk_watchdog(old_group, ticks)
    try:
        with single_executor(prior / 'handoff.lock'):
            scheduler.supervise(adopted, override['already_completed'])
    except Exception as exc:
        write_json(scheduler.RUN / 'error.json', {'error': repr(exc), 'pid': os.getpid(), 'time': time.time()})
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
