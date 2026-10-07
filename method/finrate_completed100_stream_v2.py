"""Six-condition evaluation with concurrent generation and seven grading slots.

Existing calls finish before handoff. Completed answers/grades are reused,
including low scores. Naive is archived and excluded from reporting and retry.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import finrate_completed100_v1 as original
from finrate_completed100_parallel7_v1 import identity, same_process, owned_group_exists
from finrate_completed100_retry_v1 import retry_flags, seed_case, link_existing
from finrate_final_evaluation import single_executor
from finrate_poc.runtime import ROOT, DiskLow, check_disk, cpu_only, file_digest, write_json

FIRST = ROOT / 'result/finrate_completed100_v1'
CONTROL = FIRST / 'stream_v2'
RETRY = ROOT / 'result/finrate_completed100_retry_v2'
CONDITIONS = [c for c in original.CONDITIONS if c != 'naive_llm']
JUDGE_WORKERS = 7
MODULE = 'finrate_completed100_stream_v2'


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    original.verify()
    registration = read(CONTROL / 'registration.json')
    for name, sha in registration['code_sha256'].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError('Streaming code changed: ' + name)
    if file_digest(FIRST / 'manifest.json') != registration['original_manifest_sha256']:
        raise RuntimeError('Original manifest changed')


def live_child_pids(pid):
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:
        return []
    return [int(c) for c in children if (p := identity(int(c))) and p['state'] not in {'Z', 'X'}]


def call_logs_saved(folder):
    # A request without saved events/response may still be inside command();
    # let that call finish instead of interrupting or issuing it again.
    for request in folder.rglob('request.json'):
        if (request.parent / 'response.json').exists():
            continue
        events = request.parent / 'events.jsonl'
        try:
            saved = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        # A successful call must finish writing its accepted response first.
        # Terminal failed calls can be replayed as failures without resampling.
        if not any(row.get('type') == 'turn.failed' for row in saved):
            return False
    return True


def current_call_folder(condition, status, qs):
    stage = status.get('state')
    if stage == 'inference':
        folder = FIRST / 'answers' / status['q_id'] / condition
        return CONTROL / 'no_active_call' if (folder / 'prediction.json').exists() else folder
    n = status.get('completed', len(qs))
    if stage in {'reference_grading', 'source_grounding'} and n < len(qs):
        qid = qs[n]['q_id']
        if stage == 'source_grounding':
            folder = FIRST / 'source_grounding' / qid / condition
            return CONTROL / 'no_active_call' if (folder / 'assessment.json').exists() else folder
        folder = FIRST / 'answers' / qid / condition
        return CONTROL / 'no_active_call' if (folder / 'assessment.json').exists() else folder / 'blinded_judge'
    return CONTROL / 'no_active_call'


def try_checkpoint(saved, folder_for_status, status_path):
    """Pause a process only when no child call or unsaved call result exists."""
    p = identity(saved['pid'])
    if not same_process(saved, p):
        return 'exited', None
    os.kill(saved['pid'], signal.SIGSTOP)
    keep_paused = False
    try:
        for _ in range(100):
            p = identity(saved['pid'])
            if not p or p['state'] in {'Z', 'X'}:
                return 'exited', None
            if p['state'] in {'T', 't'}:
                break
            time.sleep(.002)
        else:
            return 'busy', None
        status = read(status_path)
        if live_child_pids(saved['pid']) or not call_logs_saved(folder_for_status(status)):
            return 'busy', None
        keep_paused = True
        return 'paused', status
    finally:
        if not keep_paused and same_process(saved, identity(saved['pid'])):
            os.kill(saved['pid'], signal.SIGCONT)


def stop_pid(saved):
    if same_process(saved, identity(saved['pid'])):
        os.kill(saved['pid'], signal.SIGTERM)
        try:
            os.kill(saved['pid'], signal.SIGCONT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while (p := identity(saved['pid'])) and p['state'] not in {'Z', 'X'}:
        if time.monotonic() > deadline:
            raise RuntimeError('Process did not exit after checkpoint: ' + str(saved['pid']))
        time.sleep(.02)


def guard(status):
    done = threading.Event()
    registered = read(CONTROL / 'registration.json')['old_groups']

    def watch():
        while not done.wait(2):
            try:
                check_disk()
            except DiskLow as exc:
                try:
                    status.parent.mkdir(parents=True, exist_ok=True)
                    status.write_text(json.dumps({'state': 'stopped_low_disk', 'reason': str(exc)}) + '\n')
                finally:
                    if os.getpid() == os.getpgrp():
                        for row in registered:
                            if owned_group_exists(row['pgid'], row['latest_start_ticks']):
                                try:
                                    os.killpg(row['pgid'], signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                    os._exit(75)
    threading.Thread(target=watch, daemon=True).start()
    return done


def handoff():
    reg = read(CONTROL / 'registration.json')
    for name, saved in reg['superseded_coordinators'].items():
        if not same_process(saved, identity(saved['pid'])):
            raise RuntimeError('Coordinator identity changed before handoff: ' + name)
    waiter_root = ROOT / 'result/finrate_completed100_retry_v1'
    waiter_status = read(waiter_root / 'status.json')
    if waiter_status['state'] != 'waiting_for_initial_run' or waiter_status['retry_started']:
        raise RuntimeError('Prior retry is no longer waiting')
    write_json(CONTROL / 'before_handoff.json', {
        'primary': read(FIRST / 'status.json'), 'retry': waiter_status,
        'workers': {c: read(FIRST / 'workers' / c / 'status.json') for c in CONDITIONS},
        'time': time.time()}, immutable=True)
    for saved in reg['superseded_coordinators'].values():
        stop_pid(saved)
    write_json(waiter_root / 'status.json', {'state': 'superseded', 'replacement': str(CONTROL),
               'reason': 'User requested concurrent grading and removed naive.', 'retry_started': False})
    qs = original.inputs()
    pending = dict(reg['workers']); transferred = {}
    while pending:
        check_disk()
        for c, saved in list(pending.items()):
            path = FIRST / 'workers' / c / 'status.json'
            state, status = try_checkpoint(saved, lambda s, c=c: current_call_folder(c, s, qs), path)
            if state == 'busy':
                continue
            if state == 'exited':
                status = read(path)
                if status.get('state') != 'completed':
                    raise RuntimeError('Original worker exited without completion: ' + c)
            else:
                # No running child; the frozen function can replay saved calls
                # and tool observations without regenerating accepted responses.
                stop_pid(saved)
            transferred[c] = {'identity': saved, 'status': status, 'handoff_state': state}
            write_json(CONTROL / 'handoff' / (c + '.json'), transferred[c], immutable=True)
            del pending[c]
        write_json(FIRST / 'status.json', {'state': 'switching_to_streaming_grading',
                   'waiting_for_call_to_finish': list(pending), 'transferred': list(transferred),
                   'conditions': 6, 'excluded_conditions': ['naive_llm'], 'pid': os.getpid()})
        if pending:
            time.sleep(.05)
    write_json(CONTROL / 'handoff_complete.json', {'workers': transferred, 'time': time.time()}, immutable=True)


def tasks(qs):
    return [(stage, q['q_id'], c) for q in qs for stage in ['reference', 'grounding'] for c in CONDITIONS]


def artifact_path(base, task):
    stage, qid, c = task
    return base / ('answers' if stage == 'reference' else 'source_grounding') / qid / c / 'assessment.json'


def slot_tasks(qs, slot):
    return [task for n, task in enumerate(tasks(qs)) if n % JUDGE_WORKERS == slot]


def generate(base, condition, status):
    original.BASE = base
    with single_executor(base / 'locks' / (condition + '.lock')):
        original.infer(condition, status)
    write_json(status, {'state': 'completed', 'condition': condition, 'pid': os.getpid(), 'time': time.time()})


def judge(base, slot, status):
    qs = original.inputs(); gold = {q['q_id']: q for q in read(original.DATA / 'cohort/gold.json')}
    db = original.connect(original.DATA / 'corpus.sqlite', True)
    corpus = {r['id']: {'_id': r['id'], 'title': r['title'], 'text': r['text']}
              for r in db.execute('SELECT id,title,text FROM sources')}; db.close()
    pending = slot_tasks(qs, slot)
    model = original.RecoveredLuna(max_calls=len(pending))
    while pending:
        remaining = []; processed = False
        for task in pending:
            check_disk()
            stage, qid, c = task
            prediction = base / 'answers' / qid / c / 'prediction.json'
            dest = artifact_path(base, task)
            if dest.exists():
                saved = read(dest)
                sha = saved['prediction_sha256'] if stage == 'reference' else saved['identity']['prediction_sha256']
                if not prediction.exists() or sha != file_digest(prediction):
                    raise RuntimeError('Existing judgment input changed')
                continue
            if not prediction.exists():
                remaining.append(task); continue
            write_json(status, {'state': 'grading', 'stage': stage, 'condition': c, 'q_id': qid,
                               'slot': slot, 'remaining': len(pending), 'pid': os.getpid()})
            q = gold[qid]
            if stage == 'reference':
                pred = read(prediction)
                row = {'q_id': qid, 'task': q['task'], 'condition': c,
                       'prediction_sha256': file_digest(prediction), 'generation_status': pred['status'],
                       **original.grade_one(q, pred, model, prediction.parent / 'blinded_judge')}
                write_json(dest, row, immutable=True)
            else:
                original.assess_source(q['question'], prediction, corpus, model, dest.parent)
            processed = True
        pending = remaining
        if pending and not processed:
            write_json(status, {'state': 'waiting_for_predictions', 'slot': slot,
                               'remaining': len(pending), 'pid': os.getpid()})
            time.sleep(2)
    write_json(status, {'state': 'completed', 'slot': slot, 'pid': os.getpid(), 'time': time.time()})


def progress(base):
    rows = {}
    for c in CONDITIONS:
        preds = [read(p) for p in (base / 'answers').glob('*/' + c + '/prediction.json')]
        grades = [read(p) for p in (base / 'answers').glob('*/' + c + '/assessment.json')]
        source_count = len(list((base / 'source_grounding').glob('*/' + c + '/assessment.json')))
        summary = original.summarize(grades) if grades else None
        rows[c] = {'generation_processed': len(preds), 'generation_failed': sum(p['status'] == 'failed' for p in preds),
                   'reference_graded': len(grades), 'source_processed': source_count, 'score_so_far': summary}
        write_json(base / 'workers' / c / 'score_progress.json', {
            'graded': len(grades), 'total': 100, 'summary': summary, 'streaming': True})
    return rows


def collect(base):
    original.verify(); qs = original.inputs(); summaries = {t: {} for t in ['all', *original.TASKS]}
    all_rows = []; grounding_states = {}; costs = {}
    for c in CONDITIONS:
        rows = [read(base / 'answers' / q['q_id'] / c / 'assessment.json') for q in qs]
        source_rows = [{'q_id': q['q_id'], 'task': q['task'], 'condition': c,
                        **read(base / 'source_grounding' / q['q_id'] / c / 'assessment.json')} for q in qs]
        all_rows.extend(rows); grounding_states[c] = dict(Counter(r['state'] for r in source_rows))
        for task in summaries:
            summaries[task][c] = original.summarize([r for r in rows if task == 'all' or r['task'] == task])
        report = {'condition': c, 'rows': rows, 'summary': {t: summaries[t][c] for t in summaries},
                  'scope': 'Fixed balanced100, six completed companies; same-family judge, failures retained, not primary heldout.'}
        write_json(base / 'reports' / (c + '.json'), report, immutable=True)
        write_json(base / 'reports' / (c + '_grounding.json'), {'rows': source_rows, 'states': grounding_states[c]}, immutable=True)
        usage = Counter(); calls = unknown = 0
        for q in qs:
            u = original.online_usage(base / 'answers' / q['q_id'] / c)
            usage.update(u['provider_usage']); calls += u['logged_response_calls']; unknown += u['requests_with_unknown_usage']
        costs[c] = {'provider_usage': dict(usage), 'logged_response_calls': calls, 'requests_with_unknown_usage': unknown}
    if len(all_rows) != 600 or len({(r['q_id'], r['condition']) for r in all_rows}) != 600:
        raise RuntimeError('Expected fixed 100 questions x six conditions')
    result = {'summary': summaries, 'rows': all_rows, 'inference_costs': costs,
              'source_grounding_states': grounding_states, 'excluded_conditions': ['naive_llm'],
              'scope': 'Six methods, six companies, balanced50LT+50EC; naive excluded by user instruction.',
              'all_failures_retained': True, 'bank_construction_cost_separate': True}
    write_json(base / 'reports/reference.json', result, immutable=True)
    lines = ['Six-method comparison: 100 questions (50LT+50EC), six companies. Naive excluded.']
    for c, r in summaries['all'].items():
        lines.append(f"{c}: correct {r['correct']}/100; keypoints {r['key_point_lower']:.2%}..{r['key_point_upper']:.2%}; generation failures {r['generation_failures']}; unassessed {r['unassessed']}")
    (base / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    return result


def run_phase(base, phase):
    folder = CONTROL / phase
    jobs = [('generate', c, None) for c in CONDITIONS] + [('judge', None, n) for n in range(JUDGE_WORKERS)]
    active = {}; failed = []
    for stage, condition, slot in jobs:
        name = condition if condition else str(slot)
        work = folder / stage / name; work.mkdir(parents=True, exist_ok=True)
        argv = [original.PYTHON, '-u', '-m', MODULE, stage, '--phase', phase]
        argv += ['--condition', condition] if condition else ['--slot', str(slot)]
        with (work / 'stdout.log').open('a') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env={**os.environ, 'PYTHONPATH': 'method'}, stdout=log, stderr=subprocess.STDOUT)
        write_json(work / 'launch.json', {'pid': child.pid, 'pgid': os.getpgrp(), 'argv': argv, 'time': time.time()}, immutable=True)
        active[stage + '/' + name] = child
    last_report = 0
    while active:
        check_disk()
        for name, child in list(active.items()):
            rc = child.poll()
            if rc is None:
                continue
            if rc:
                failed.append({'worker': name, 'returncode': rc})
            del active[name]
        if failed:
            write_json(base / 'status.json', {'state': 'needs_review', 'failed': failed,
                       'active': list(active), 'pid': os.getpid()})
            # Remaining tasks could otherwise wait forever on missing predictions.
            for child in active.values():
                child.terminate()
            raise RuntimeError('Streaming worker failed; originals preserved')
        if time.monotonic() - last_report > 3 or not active:
            snapshot = {'state': 'running', 'phase': phase, 'active_workers': list(active),
                        'progress': progress(base), 'conditions': 6, 'questions': 100,
                        'excluded_conditions': ['naive_llm'], 'grading_workers': JUDGE_WORKERS,
                        'pid': os.getpid(), 'time': time.time(), 'control': str(CONTROL)}
            write_json(base / 'status.json', snapshot); write_json(CONTROL / 'status.json', snapshot)
            last_report = time.monotonic()
        if active:
            time.sleep(1)
    result = collect(base)
    for c in CONDITIONS:
        write_json(base / 'workers' / c / 'status.json', {'state': 'completed', 'condition': c, 'time': time.time()})
    write_json(base / 'status.json', {'state': 'completed', 'phase': phase, 'conditions': 6,
               'questions': 100, 'excluded_conditions': ['naive_llm'], 'pid': os.getpid(), 'time': time.time()})
    return result


def prepare_retry():
    rows = []; hashes = {}
    for q in original.inputs():
        for c in CONDITIONS:
            source = FIRST / 'answers' / q['q_id'] / c
            ground = FIRST / 'source_grounding' / q['q_id'] / c
            pred, grade, audit = read(source / 'prediction.json'), read(source / 'assessment.json'), read(ground / 'assessment.json')
            sha = file_digest(source / 'prediction.json')
            if grade['prediction_sha256'] != sha or audit['identity']['prediction_sha256'] != sha:
                raise RuntimeError('Retry inputs differ from first-attempt assessments')
            flags = retry_flags(pred, grade, audit)
            rows.append({'q_id': q['q_id'], 'condition': c, **flags})
            for path in [source / 'prediction.json', source / 'assessment.json', ground / 'assessment.json']:
                hashes[str(path.relative_to(FIRST))] = file_digest(path)
    selection = {'rows': rows, 'original_artifact_sha256': hashes, 'additional_attempts_per_failure': 1,
                 'excluded_conditions': ['naive_llm'], 'selection_uses_correctness_score': False,
                 'counts': {c: {s: sum(r[s] for r in rows if r['condition'] == c)
                               for s in ['generation', 'reference', 'grounding']} for c in CONDITIONS}}
    write_json(RETRY / 'selection.json', selection, immutable=True)
    for name in ['manifest.json', 'retrieval_manifest.json', 'retrieval']:
        link_existing(FIRST / name, RETRY / name)
    for row in rows:
        qid, c = row['q_id'], row['condition']
        seed_case(FIRST / 'answers' / qid / c, RETRY / 'answers' / qid / c,
                  FIRST / 'source_grounding' / qid / c, RETRY / 'source_grounding' / qid / c, row)
    return selection


def pipeline():
    handoff()
    first = run_phase(FIRST, 'initial')
    write_json(CONTROL / 'status.json', {'state': 'cooldown_before_retry', 'seconds': 60, 'pid': os.getpid()})
    time.sleep(60)
    selected = prepare_retry()
    second = run_phase(RETRY, 'retry')
    for name, sha in selected['original_artifact_sha256'].items():
        if file_digest(FIRST / name) != sha:
            raise RuntimeError('First-attempt artifact changed during retry: ' + name)
    costs = {}
    for c in CONDITIONS:
        before, extra = first['inference_costs'][c], second['inference_costs'][c]
        total = Counter(before['provider_usage']); total.update(extra['provider_usage'])
        costs[c] = {'first_attempt': before, 'retry_increment': extra, 'cumulative_provider_usage': dict(total)}
    write_json(RETRY / 'reports/retry_comparison.json', {'first_attempt_summary': first['summary'],
               'after_one_retry_summary': second['summary'], 'inference_costs': costs,
               'retry_selection_counts': selected['counts'], 'excluded_conditions': ['naive_llm'],
               'same_100_questions': True, 'successful_answers_resampled': 0,
               'original_artifacts_unchanged': True}, immutable=True)
    write_json(CONTROL / 'status.json', {'state': 'completed', 'conditions': 6, 'questions': 100,
               'initial_report': str(FIRST / 'reports/reference.json'),
               'retry_report': str(RETRY / 'reports/retry_comparison.json'), 'time': time.time()})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['pipeline', 'generate', 'judge'])
    parser.add_argument('--condition', choices=CONDITIONS); parser.add_argument('--slot', type=int)
    parser.add_argument('--phase', choices=['initial', 'retry'], default='initial'); args = parser.parse_args()
    cpu_only(); os.environ['OPENBLAS_NUM_THREADS'] = '2'; check_disk(); os.nice(10)
    if args.stage == 'pipeline' and os.getpgrp() != os.getpid():
        raise RuntimeError('Dedicated streaming process group required')
    base = FIRST if args.phase == 'initial' else RETRY
    name = args.condition if args.stage == 'generate' else str(args.slot)
    status = CONTROL / ('status.json' if args.stage == 'pipeline' else f'{args.phase}/{args.stage}/{name}/status.json')
    stop = guard(status)
    try:
        verify()
        with single_executor(CONTROL / 'locks' / (args.phase + '_' + args.stage + '_' + name)):
            if args.stage == 'pipeline':
                pipeline()
            elif args.stage == 'generate':
                generate(base, args.condition, status)
            else:
                if args.slot not in range(JUDGE_WORKERS):
                    raise ValueError('Invalid grading slot')
                judge(base, args.slot, status)
    except Exception as exc:
        write_json(status, {'state': 'failed', 'error': repr(exc), 'pid': os.getpid(), 'time': time.time()})
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
