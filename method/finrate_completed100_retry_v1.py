"""One user-authorized retry round after all seven initial conditions finish.

Reuse accepted predictions and judgments, regardless of score. Retry only
execution failures in a separate output tree. Never overwrite attempt one.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

import finrate_completed100_v1 as experiment
from finrate_completed100_parallel7_v1 import identity
from finrate_final_evaluation import single_executor
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json

ORIGINAL = ROOT / 'result/finrate_completed100_v1'
BASE = ROOT / 'result/finrate_completed100_retry_v1'
MODULE = 'finrate_completed100_retry_v1'
CONDITIONS = experiment.CONDITIONS


def read(path):
    return json.loads(Path(path).read_text())


def retry_flags(prediction, assessment, grounding):
    """Operational status only; an incorrect/partial/abstaining answer stays."""
    generation = prediction['status'] == 'failed'
    return {
        'generation': generation,
        'reference': generation or assessment['label'] == 'UNASSESSED',
        'grounding': generation or grounding['state'] == 'assessment_failed',
    }


def primary_ready(status, worker_states, live_workers):
    return (status.get('state') == 'completed'
            and set(worker_states) == set(CONDITIONS)
            and all(s.get('state') == 'completed' for s in worker_states.values())
            and not live_workers)


def link_existing(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError('Refusing to replace retry artifact: ' + str(target))
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def seed_case(source, target, source_grounding, target_grounding, flags):
    target.mkdir(parents=True, exist_ok=True)
    if not flags['generation']:
        link_existing(source / 'prediction.json', target / 'prediction.json')
    if not flags['reference']:
        link_existing(source / 'assessment.json', target / 'assessment.json')
    if not flags['grounding']:
        link_existing(source_grounding, target_grounding)


def verify_registration():
    record = read(BASE / 'registration.json')
    if file_digest(ORIGINAL / 'manifest.json') != record['original_manifest_sha256']:
        raise RuntimeError('Original experiment identity changed')
    for name, sha in record['code_sha256'].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError('Registered retry code changed: ' + name)
    experiment.verify()
    return record


def wait_for_primary():
    while True:
        check_disk()
        status = read(ORIGINAL / 'status.json')
        workers = {}
        live = []
        for c in CONDITIONS:
            folder = ORIGINAL / 'workers' / c
            if (folder / 'status.json').exists():
                workers[c] = read(folder / 'status.json')
            if (folder / 'launch.json').exists():
                launch = read(folder / 'launch.json')
                p = identity(launch['pid'])
                if p and p['state'] not in {'Z', 'X'} and p['argv'] == launch['argv']:
                    live.append(c)
        if primary_ready(status, workers, live):
            if not (ORIGINAL / 'reports/reference.json').exists():
                raise RuntimeError('Primary completed without its comparison report')
            return
        write_json(BASE / 'status.json', {
            'state': 'waiting_for_initial_run', 'pid': os.getpid(), 'time': time.time(),
            'initial_state': status.get('state'), 'initial_active': live,
            'initial_worker_stages': {c: s.get('state') for c, s in workers.items()},
            'retry_started': False, 'additional_attempts_per_failed_case': 1,
        })
        if not live and status.get('state') in {'failed', 'needs_review', 'stopped_low_disk'}:
            raise RuntimeError('Initial run stopped before completion; no retry launched')
        p = identity(status.get('pid', -1))
        if not live and (not p or p['state'] in {'Z', 'X'}):
            raise RuntimeError('Initial coordinator and workers ended without completion')
        time.sleep(15)


def prepare():
    verify_registration()
    if (BASE / 'selection.json').exists():
        raise RuntimeError('Retry preparation already recorded; inspect before resuming')
    rows = []
    hashes = {}
    for q in experiment.inputs():
        for c in CONDITIONS:
            folder = ORIGINAL / 'answers' / q['q_id'] / c
            ground = ORIGINAL / 'source_grounding' / q['q_id'] / c
            pred = read(folder / 'prediction.json')
            grade = read(folder / 'assessment.json')
            source = read(ground / 'assessment.json')
            pred_sha = file_digest(folder / 'prediction.json')
            if grade['prediction_sha256'] != pred_sha or source['identity']['prediction_sha256'] != pred_sha:
                raise RuntimeError('Prior assessment does not match original prediction')
            if pred['status'] not in {'completed', 'failed'}:
                raise RuntimeError('Unknown generation status')
            if source['state'] == 'generation_failed' and pred['status'] != 'failed':
                raise RuntimeError('Inconsistent original grounding state')
            flags = retry_flags(pred, grade, source)
            rows.append({'q_id': q['q_id'], 'condition': c, **flags})
            for p in [folder / 'prediction.json', folder / 'assessment.json', ground / 'assessment.json']:
                hashes[str(p.relative_to(ORIGINAL))] = file_digest(p)
    if len(rows) != 700:
        raise RuntimeError('Expected the same fixed 700-case grid')
    counts = {c: {s: sum(r[s] for r in rows if r['condition'] == c)
                  for s in ['generation', 'reference', 'grounding']} for c in CONDITIONS}
    selection = {'rows': rows, 'counts': counts, 'original_artifact_sha256': hashes,
                 'original_report_sha256': file_digest(ORIGINAL / 'reports/reference.json'),
                 'one_additional_attempt': True, 'selection_uses_correctness_score': False,
                 'time': time.time()}
    write_json(BASE / 'selection.json', selection, immutable=True)
    # These immutable identities and retrievals remain owned by the first run.
    for name in ['manifest.json', 'retrieval_manifest.json', 'retrieval']:
        link_existing(ORIGINAL / name, BASE / name)
    for row in rows:
        qid, c = row['q_id'], row['condition']
        seed_case(ORIGINAL / 'answers' / qid / c, BASE / 'answers' / qid / c,
                  ORIGINAL / 'source_grounding' / qid / c, BASE / 'source_grounding' / qid / c, row)
    write_json(BASE / 'prepared.json', {
        'selection_sha256': file_digest(BASE / 'selection.json'), 'time': time.time(),
        'original_predictions_unchanged': True, 'counts': counts,
    }, immutable=True)


def verify_prepared():
    verify_registration()
    prepared = read(BASE / 'prepared.json')
    if file_digest(BASE / 'selection.json') != prepared['selection_sha256']:
        raise RuntimeError('Retry selection changed')
    selected = read(BASE / 'selection.json')
    for name, sha in selected['original_artifact_sha256'].items():
        if file_digest(ORIGINAL / name) != sha:
            raise RuntimeError('Original attempt changed: ' + name)
    if file_digest(ORIGINAL / 'reports/reference.json') != selected['original_report_sha256']:
        raise RuntimeError('Original report changed')
    return selected


def condition(c):
    verify_prepared()
    experiment.BASE = BASE
    status = BASE / 'workers' / c / 'status.json'
    # Frozen functions skip linked accepted results. Failed cases get a new
    # directory with the identical prompt/model and identical per-attempt budget.
    experiment.infer(c, status)
    experiment.grade(c, status)
    experiment.ground(c, status)
    write_json(status, {'state': 'completed', 'condition': c, 'pid': os.getpid(), 'time': time.time()})


def collect():
    selection = verify_prepared()
    experiment.BASE = BASE
    experiment.collect()
    first = read(ORIGINAL / 'reports/reference.json')
    second = read(BASE / 'reports/reference.json')
    # Traversal does not follow the reused prediction-file symlinks into original
    # request logs. Thus second's inference costs contain only new retry calls.
    costs = {}
    for c in CONDITIONS:
        old, extra = first['inference_costs'][c], second['inference_costs'][c]
        usage = Counter(old['provider_usage']); usage.update(extra['provider_usage'])
        costs[c] = {'first_attempt': old, 'retry_increment': extra,
                    'cumulative_provider_usage': dict(usage)}
    recovery = {}
    for c in CONDITIONS:
        eligible = [r for r in selection['rows'] if r['condition'] == c and r['generation']]
        recovered = sum(read(BASE / 'answers' / r['q_id'] / c / 'prediction.json')['status'] == 'completed'
                        for r in eligible)
        recovery[c] = {'generation_retried': len(eligible), 'generation_recovered': recovered,
                       'generation_still_failed': len(eligible) - recovered}
    report = {'first_attempt_summary': first['summary'], 'after_one_retry_summary': second['summary'],
              'recovery': recovery, 'inference_costs': costs,
              'source_grounding_states': second['source_grounding_states'],
              'policy': 'One extra attempt only for execution failures; accepted answers and judgments are reused regardless of score.',
              'original_artifacts_unchanged': True, 'successful_answers_resampled': 0,
              'same_100_questions': True, 'same_per_attempt_budget': True,
              'total_budget_includes_first_attempt_and_retry': True}
    write_json(BASE / 'reports/retry_comparison.json', report, immutable=True)
    lines = ['After one user-authorized retry of failed cases; same six companies / fixed100 questions.',
             'The first-attempt scores and failures are preserved separately. Total cost includes both attempts.']
    for c in CONDITIONS:
        a, b = first['summary']['all'][c], second['summary']['all'][c]
        lines.append(f"{c}: first correct {a['correct']}/100; after retry correct {b['correct']}/100; "
                     f"keypoints {b['key_point_lower']:.2%}..{b['key_point_upper']:.2%}; "
                     f"remaining generation failures {b['generation_failures']}; unassessed {b['unassessed']}")
    (BASE / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')


def pipeline():
    verify_registration()
    wait_for_primary()
    # Uniform cooldown after the first run, without changing the requested seven
    # condition workers or the experiment's model/budget settings.
    write_json(BASE / 'status.json', {'state': 'cooldown_before_retry', 'seconds': 60,
                                    'pid': os.getpid(), 'time': time.time()})
    time.sleep(60)
    prepare()
    active = {}; done = []; failed = []
    for c in CONDITIONS:
        folder = BASE / 'workers' / c; folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'launch.json').exists():
            raise RuntimeError('Retry worker already recorded: ' + c)
        argv = [experiment.PYTHON, '-u', '-m', MODULE, 'condition', '--condition', c]
        with (folder / 'stdout.log').open('a') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env={**os.environ, 'PYTHONPATH': 'method'},
                                     stdout=log, stderr=subprocess.STDOUT)
        active[c] = child
        write_json(folder / 'launch.json', {'pid': child.pid, 'pgid': os.getpgrp(),
                   'argv': argv, 'started': time.time()}, immutable=True)
    while active:
        check_disk()
        for c, child in list(active.items()):
            rc = child.poll()
            if rc is None:
                continue
            if rc == 0:
                done.append(c)
            else:
                failed.append({'condition': c, 'returncode': rc})
            del active[c]
        write_json(BASE / 'status.json', {'state': 'needs_review' if failed else 'retry_running',
                   'active': list(active), 'completed_conditions': done, 'failed': failed,
                   'worker_concurrency': 7, 'pid': os.getpid(), 'time': time.time()})
        if active:
            time.sleep(2)
    if failed:
        return
    collect()
    write_json(BASE / 'status.json', {'state': 'completed', 'pid': os.getpid(),
               'time': time.time(), 'questions': 100, 'conditions': 7, 'retry_rounds': 1})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pipeline', 'condition'])
    parser.add_argument('--condition', choices=CONDITIONS)
    args = parser.parse_args()
    cpu_only(); os.environ['OPENBLAS_NUM_THREADS'] = '2'; check_disk(); os.nice(10)
    if args.stage == 'pipeline' and os.getpgrp() != os.getpid():
        raise RuntimeError('Dedicated retry process group required')
    status = BASE / ('status.json' if args.stage == 'pipeline' else 'workers/' + args.condition + '/status.json')
    stop = watchdog(status)
    try:
        with single_executor(BASE / 'locks' / (args.condition or 'pipeline')):
            if args.stage == 'pipeline':
                pipeline()
            else:
                condition(args.condition)
    except Exception as exc:
        write_json(status, {'state': 'failed', 'error': repr(exc), 'pid': os.getpid(), 'time': time.time()})
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
