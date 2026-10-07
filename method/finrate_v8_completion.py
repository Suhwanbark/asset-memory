"""Wait for the four frozen DEV workers, then assemble the full read-only report."""
import json
import os
from pathlib import Path
import time

from finrate_final_evaluation import single_executor
from finrate_v5_development import verify_frozen
from finrate_v8_report import report
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json


def main():
    cpu_only(); check_disk(); os.nice(10)
    base = ROOT/'result/finrate_luna_v1'; out = base/'v8'; folder = out/'completion_audits'
    own = json.loads((folder/'manifest.json').read_text())
    for name, sha in own['code_sha256'].items():
        if file_digest(ROOT/name) != sha: raise ValueError('Auxiliary code changed')
    if file_digest(out/'method_manifest.json') != own['method_manifest_sha256']:
        raise ValueError('Method changed')
    status = folder/'status.json'; stop = watchdog(status)
    common = {'pid': os.getpid(), 'started': time.time(), 'cpu_only': True,
              'disk_stop_bytes': 10_000_000_000, 'new_model_calls': 0, 'evaluation_touched': False}
    try:
        with single_executor(folder/'execution.lock'):
            while True:
                check_disk(); states = {}; pending = []
                registry = json.loads((base/'active_run.json').read_text())
                if registry['version'] != 'v8': raise ValueError('Active experiment changed')
                workers = {w['label']: w for w in registry['workers']}
                for label in ['shared_planning', 'raw_agent', 'agentic', 'history_disabled']:
                    w = workers[label]; state = json.loads(Path(w['status_path']).read_text())['state']
                    states[label] = state
                    expected = 'shared_planning_ready' if label == 'shared_planning' else 'development_ready_for_review'
                    if state == expected: continue
                    if 'failed' in state or 'low_disk' in state: raise RuntimeError(f'Upstream worker terminal: {label} {state}')
                    try: argv = Path(f"/proc/{w['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0')
                    except FileNotFoundError: argv = None
                    if argv != w['argv']: raise RuntimeError(f'Active upstream worker absent: {label}')
                    pending.append(label)
                if not pending: break
                write_json(status, {**common, 'state': 'waiting_for_development', 'pending': pending, 'upstream_states': states})
                if time.time()-common['started'] > 8*3600: raise TimeoutError('Development incomplete after8hours; no restart attempted')
                time.sleep(15)
            verify_frozen(ROOT, json.loads((out/'method_manifest.json').read_text()))
            write_json(status, {**common, 'state': 'assembling_full_development_report'})
            result = report()
            write_json(status, {**common, 'state': 'audits_ready_for_review', 'finished': time.time(),
                'n_questions': result['n_questions'], 'conditions': result['conditions'],
                'report_sha256': file_digest(out/'paired_development_analysis.json')})
    except Exception as exc:
        write_json(status, {**common, 'state': 'audit_worker_failed', 'error': type(exc).__name__+': '+str(exc)})
        raise
    finally: stop.set()


if __name__ == '__main__': main()
