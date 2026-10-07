"""Observe registered experiment workers without changing or restarting them.

Only process identity, status, progress logs and call-file metadata are read.
Question, gold and model-response contents are never opened by this observer.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'result/finrate_luna_v1'
NORMAL_TERMINAL = {'asset_ready', 'development_ready_for_review', 'audits_ready_for_review', 'shared_planning_ready'}


def snapshot():
    registry = json.loads((BASE / 'active_run.json').read_text())
    rows = []
    issues = []
    for worker in registry['workers']:
        proc = Path('/proc') / str(worker['pid']) / 'cmdline'
        try:
            argv = proc.read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            argv = None
        state = json.loads(Path(worker['status_path']).read_text())
        live = argv == worker['argv']
        rows.append({'label': worker['label'], 'pid': worker['pid'],
                     'live_expected_argv': live, 'observed_argv': argv,
                     'status': state})
        if 'failed' in state['state'] or 'low_disk' in state['state']:
            issues.append({'label': worker['label'], 'reason': state})
        elif not live and state['state'] not in NORMAL_TERMINAL:
            issues.append({'label': worker['label'], 'reason': 'Expected process is absent',
                           'recorded_state': state['state']})
    progress = {}; assessment_progress = {}
    version = registry['version']
    if version in {'v7', 'v8'}:
        for condition in ['naive_llm', 'embedding', 'hybrid', 'raw_agent', 'agentic', 'history_disabled']:
            folder = BASE / version
            if condition == 'history_disabled': folder = folder / 'history_ablation'
            progress[condition] = len(list((folder / 'dev').glob(f'*/{condition}/prediction.json')))
            assessment_progress[condition] = {
                'original_gold': len(list((folder / 'dev').glob(f'*/{condition}/assessment.json'))),
                'source_grounding': len(list((folder / 'source_grounding').glob(f'*/{condition}/assessment.json')))}
    else:
        for asset in ['FIX', 'LW', 'NRT']:
            log = BASE / 'v5/workers' / asset / 'worker.log'
            lines = log.read_text().splitlines() if log.exists() else []
            progress[asset] = next((s for s in reversed(lines) if s.startswith('BATCH_MEMORY')), None)
    free = shutil.disk_usage(ROOT).free
    if free < 10_000_000_000:
        issues.append({'reason': 'Below disk threshold; experiment watchdogs are responsible for stopping their own process groups'})
    return {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'version': version, 'free_disk_bytes': free, 'workers': rows, 'progress': progress,
            'assessment_progress': assessment_progress, 'issues': issues}


def compact(state):
    return {'time': state['time'], 'free_disk_bytes': state['free_disk_bytes'],
            'workers': {w['label']: {'pid': w['pid'], 'live': w['live_expected_argv'],
                                   'state': w['status']['state']} for w in state['workers']},
            'progress': state['progress'], 'assessment_progress': state['assessment_progress'], 'issues': state['issues']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--until', choices=['online-start', 'development-ready', 'deadline'],
                        default='online-start')
    args = parser.parse_args()
    if not 0 <= args.seconds <= 3600 or not 1 <= args.interval <= 60:
        parser.error('Use 0..3600 seconds and a 1..60 second interval')
    start = time.monotonic()
    samples = []
    previous = None
    while True:
        state = snapshot()
        state['elapsed_seconds'] = round(time.monotonic() - start, 1)
        samples.append(state)
        signature = (state['progress'], state['assessment_progress'],
                     [(w['label'], w['status']['state']) for w in state['workers']])
        if signature != previous or state['issues']:
            print(json.dumps(compact(state)), flush=True)
            previous = signature
        online_started = any(w['label'] == 'agentic' and w['status']['state'] != 'waiting_for_asset_memories'
                             for w in state['workers'])
        states = {w['label']: w['status']['state'] for w in state['workers']}
        development_ready = (states.get('agentic') == 'development_ready_for_review'
                             and states.get('history_disabled') == 'development_ready_for_review'
                             and states.get('completion_audits') == 'audits_ready_for_review')
        if state['version'] == 'v7':
            development_ready = all(states.get(label) == 'development_ready_for_review'
                                    for label in ['static', 'raw_agent', 'agentic', 'history_disabled'])
        if state['version'] == 'v8':
            development_ready = (states.get('shared_planning') == 'shared_planning_ready'
                and all(states.get(label) == 'development_ready_for_review'
                        for label in ['raw_agent', 'agentic', 'history_disabled'])
                and states.get('completion_audits_v8') == 'audits_ready_for_review')
        reached_target = ((args.until == 'online-start' and online_started)
                          or (args.until == 'development-ready' and development_ready))
        if state['issues'] or reached_target or time.monotonic() - start >= args.seconds:
            break
        time.sleep(min(args.interval, max(0, args.seconds - (time.monotonic() - start))))
    now = datetime.datetime.now(datetime.timezone.utc)
    path = BASE / state['version'] / 'monitoring' / f'{now:%Y%m%dT%H%M%S%fZ}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'classification': 'terminal_state_detected' if state['issues'] else 'verified_wait',
              'wait_target': args.until, 'target_reached': reached_target,
              'samples': samples, 'new_model_calls_by_observer': 0,
              'questions_or_gold_read': False, 'upstream_restarts': 0,
              'elapsed_seconds': round(time.monotonic() - start, 1)}
    # A small status artifact is retained even when the disk guard fires.
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'saved': str(path), 'final': compact(state)}), flush=True)
    if state['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
