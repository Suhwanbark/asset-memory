"""One-release held-out evaluation, kept separate from development workers.

Preparation does not execute this module. Freeze requires complete development
and ablation reports plus an explicit recorded review decision. Inference
verifies the release and acquires the cohort-wide lock BEFORE reading held-out
questions. Gold is opened only after every prediction is recorded. Interrupted
requests without saved responses are failures, never silently regenerated.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import time

from finrate_agent_v3 import predict as legacy_agent_predict
from finrate_agent_v6 import predict as agent_predict
from finrate_agent_v7 import TemporalRetriever, predict as period_agent_predict
from finrate_luna_v7 import LunaV7
from finrate_dev_controls import judgement_prompt
from finrate_history_ablation import NoHistoryEdges
from finrate_memory_audit import audit as audit_bank
from finrate_source_grading import assess
from finrate_v5_development import verify_frozen
from finrate_poc.agent import predict as static_predict
from finrate_poc.bank import Bank
from finrate_poc.cohort import approve_gate
from finrate_poc.data import documents, metadata
from finrate_poc.model import Luna, parse_json
from finrate_poc.retrieval import Index
from finrate_poc.runtime import ROOT, check_disk, cpu_only, digest, file_digest, watchdog, write_json
from finrate_poc.schemas import JUDGEMENT

CONDITIONS = ['naive_llm', 'embedding', 'hybrid', 'raw_agent', 'agentic', 'history_disabled']


def atomic_identity(path, value):
    """An exclusive create prevents simultaneous releases from both winning."""
    check_disk(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise ValueError('The cohort is already locked to a different evaluation release')
        return
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


@contextmanager
def single_executor(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('An evaluation worker already holds this cohort') from exc
        try: yield
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


class OnceLuna(Luna):
    def invoke(self, prompt, folder, *, schema=None):
        folder = Path(folder)
        if (folder / 'request.json').exists() and not (folder / 'response.json').exists():
            raise RuntimeError('Interrupted or failed prior request; automatic regeneration is prohibited')
        return super().invoke(prompt, folder, schema=schema)


def prediction_grid(folder, ids, conditions=CONDITIONS):
    expected = {(q, c) for q in ids for c in conditions}
    found = {(p.parent.parent.name, p.parent.name) for p in Path(folder).glob('*/*/prediction.json')}
    if found != expected: raise ValueError('Prediction grid is incomplete or contains non-frozen rows')
    hashes = {}
    for q, condition in sorted(expected):
        p = Path(folder) / q / condition / 'prediction.json'; record = json.loads(p.read_text())
        if record.get('q_id') != q or record.get('condition') != condition:
            raise ValueError('Prediction identity mismatch')
        if record.get('status') not in {'completed', 'failed'}: raise ValueError('Nonterminal prediction')
        hashes[str(p.relative_to(folder))] = file_digest(p)
    return hashes


def complete_assessment_rows(rows, ids, conditions, *, implicit_condition=None):
    expected = {(q, c) for q in ids for c in conditions}
    observed = [(r['q_id'], r.get('condition', implicit_condition)) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Development assessment grid is incomplete, duplicated or substituted')


def freeze(base, version, review_path):
    base = Path(base); out = base / version; cohort = base / 'cohort'
    method = json.loads((out / 'method_manifest.json').read_text()); verify_frozen(ROOT, method)
    selected = json.loads((cohort / 'cohort_manifest.json').read_text())
    policy = json.loads((ROOT / 'data/finrate_luna_protocol_v1.json').read_text())
    if method['policy'] != policy or selected['policy'] != policy: raise ValueError('Frozen policy differs')
    approve_gate(json.loads((base / 'cutoff_audit.json').read_text()), policy)
    source_manifest = json.loads((ROOT / 'data/finrate_public_v1/source_manifest.json').read_text())
    for row in source_manifest['files']:
        if file_digest(ROOT / 'data/finrate_public_v1' / row['path']) != row['sha256']:
            raise ValueError('Public dataset bytes differ from the pinned source manifest')
    dev_ids = selected['selected_ids']['dev']
    prediction_grid(out / 'dev', dev_ids, CONDITIONS[:-1])
    prediction_grid(out / 'history_ablation/dev', dev_ids, ['history_disabled'])
    required = [out / 'dev_report.json', out / 'source_grounding.json', out / 'memory_manifest.json',
                out / 'history_ablation/dev_report.json', out / 'history_ablation/source_grounding.json',
                out / 'history_ablation/method_manifest.json',
                base / 'development_gold_audit_v1/report.json', base / 'development_gold_audit_v1/manifest.json',
                base / 'development_gold_source_review_v1/reviewed_findings.json']
    if any(not p.is_file() for p in required): raise ValueError('Development assessments are incomplete')
    complete_assessment_rows(json.loads((out / 'dev_report.json').read_text())['rows'], dev_ids, CONDITIONS[:-1])
    complete_assessment_rows(json.loads((out / 'source_grounding.json').read_text()), dev_ids, CONDITIONS[:-1])
    complete_assessment_rows(json.loads((out / 'history_ablation/dev_report.json').read_text())['rows'],
                             dev_ids, ['history_disabled'])
    complete_assessment_rows(json.loads((out / 'history_ablation/source_grounding.json').read_text()),
                             dev_ids, ['history_disabled'], implicit_condition='history_disabled')
    complete_assessment_rows(json.loads((base / 'development_gold_audit_v1/report.json').read_text())['rows'],
                             dev_ids, ['reference_audit'], implicit_condition='reference_audit')
    review_path = Path(review_path).resolve(); review = json.loads(review_path.read_text())
    checks = {'development_manifest_sha256': file_digest(out / 'method_manifest.json'),
              'development_report_sha256': file_digest(out / 'dev_report.json'),
              'ablation_report_sha256': file_digest(out / 'history_ablation/dev_report.json'),
              'reference_audit_sha256': file_digest(base / 'development_gold_audit_v1/report.json'),
              'source_review_sha256': file_digest(base / 'development_gold_source_review_v1/reviewed_findings.json'),
              'decision': 'freeze_for_one_heldout_run', 'all_fixed_dev_questions_retained': True,
              'no_heldout_results_used': True}
    if any(review.get(k) != v for k, v in checks.items()) or not review.get('rationale') or not review.get('limitations'):
        raise ValueError('A concrete review of the current development results is required')
    code = dict(method['code_sha256'])
    for name in ['finrate_history_ablation.py', 'finrate_agent_v6.py', 'finrate_final_evaluation.py']:
        p = ROOT / 'method' / name; code[str(p.relative_to(ROOT))] = file_digest(p)
    files = [cohort / 'cohort_manifest.json', cohort / 'evaluation_inputs.json', cohort / 'gold/evaluation.json',
             ROOT / 'data/finrate_luna_protocol_v1.json', ROOT / 'data/finrate_public_v1/corpus/corpus.zip',
             ROOT / 'data/finrate_public_v1/source_manifest.json',
             out / 'method_manifest.json', review_path, *required, *list((base / 'index').glob('*'))]
    banks = {}
    memory_version = method.get('offline_memory_version', version)
    online_version = method['online_code_version']
    if online_version not in {'v3_unchanged', 'v6_compact_provenance', 'v7_period_procedure'}:
        raise ValueError('Unsupported frozen online policy')
    if online_version == 'v7_period_procedure':
        for name in ['finrate_agent_v7.py', 'finrate_luna_v7.py']:
            p = ROOT / 'method' / name; code[str(p.relative_to(ROOT))] = file_digest(p)
        files.extend(out / 'fact_index' / name for name in ['manifest.json', 'passages.json', 'vectors.npy'])
        if any(not (out / 'fact_index' / name).is_file() for name in ['manifest.json', 'passages.json', 'vectors.npy']):
            raise ValueError('Frozen fact index is missing')
    for asset in selected['assets']:
        folder = base / 'asset_memory' / memory_version / asset
        completed = json.loads((folder / 'completed.json').read_text()); checked = audit_bank(folder / 'bank.sqlite')
        if file_digest(folder / 'bank.sqlite') != completed['bank_sha256'] or checked['event_hash'] != completed['event_hash']:
            raise ValueError('Completed memory differs from its recorded history')
        banks[asset] = {'path': str((folder / 'bank.sqlite').relative_to(ROOT)), 'sha256': completed['bank_sha256']}
        files.extend(folder / name for name in ['bank.sqlite', 'completed.json', 'build_manifest.json',
                                                'span_method_manifest.json', 'structural_audit.json'])
    release = {'method_version': version, 'offline_memory_version': memory_version,
               'online_code_version': online_version, 'code_sha256': code,
               'input_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in files if p.is_file()},
               'conditions': CONDITIONS, 'primary_conditions': policy['conditions'],
               'policy': policy, 'assets': selected['assets'], 'banks': banks,
               'evaluation_ids': selected['selected_ids']['evaluation'],
               'review_sha256': file_digest(review_path), 'evaluation_count': len(selected['selected_ids']['evaluation']),
               'execution_rule': 'One frozen release for this cohort. Resume saved responses; never regenerate interrupted requests or failed predictions.',
               'scoring_rule': 'All questions remain. Generation failures are zero; judge failures are unassessed with lower/upper score bounds.',
               'cutoff_limitation': 'User-authorized original Fin-RATE; pretraining exposure is uncontrolled.',
               'compute_limitation': 'Shared text/step/evidence ceilings are not equal actual compute. Report provider usage and offline build/update costs separately.'}
    # Hashing held-out files above checks immutable bytes without parsing or
    # printing their question/answer contents. The first semantic read is gated.
    write_json(out / 'method_frozen_for_evaluation.json', release, immutable=True)
    return release


def open_release(base, version):
    base = Path(base); path = base / version / 'method_frozen_for_evaluation.json'
    if not path.is_file(): raise ValueError('Freeze a reviewed method before accessing held-out inputs')
    release = json.loads(path.read_text()); verify_frozen(ROOT, release)
    if release['method_version'] != version or release['conditions'] != CONDITIONS:
        raise ValueError('Unsupported or mismatched release identity')
    approve_gate(json.loads((base / 'cutoff_audit.json').read_text()), release['policy'])
    return release, file_digest(path)


def locked_inputs(base, version):
    base = Path(base); release, release_hash = open_release(base, version)
    atomic_identity(base / 'evaluation_lock.json', {'method_version': version, 'release_sha256': release_hash})
    # No held-out text is read until release validation and cohort lock succeed.
    inputs = json.loads((base / 'cohort/evaluation_inputs.json').read_text())
    if any(set(q) != {'q_id', 'question', 'task'} for q in inputs): raise ValueError('Gold leakage in inference inputs')
    if [q['q_id'] for q in inputs] != release['evaluation_ids'] or not inputs:
        raise ValueError('Evaluation inputs differ from the frozen cohort')
    return release, release_hash, inputs


def infer(base, version, status, common):
    base = Path(base); out = base / version / 'evaluation'
    with single_executor(base / 'evaluation_execution.lock'):
        release, release_hash, inputs = locked_inputs(base, version)
        write_json(out / 'run_manifest.json', {'release_sha256': release_hash,
                   'conditions': CONDITIONS, 'q_ids': release['evaluation_ids'], 'gold_available': False}, immutable=True)
        docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] in release['assets']]
        index = Index.build(docs, base / 'index')
        is_v7 = release['online_code_version'] == 'v7_period_procedure'
        model = LunaV7() if is_v7 else OnceLuna(); banks = {}
        try:
            for asset, row in release['banks'].items(): banks[asset] = Bank(ROOT / row['path'], readonly=True)
            flat = {a: NoHistoryEdges(bank) for a, bank in banks.items()}
            temporal = TemporalRetriever.build(docs, index, banks, base / version / 'fact_index') if is_v7 else None
            for q in inputs:
                for condition in CONDITIONS:
                    folder = out / q['q_id'] / condition
                    if (folder / 'prediction.json').exists(): continue
                    check_disk(); write_json(status, {**common, 'state': 'inference', 'q_id': q['q_id'], 'condition': condition})
                    try:
                        if condition in {'naive_llm', 'embedding', 'hybrid'}:
                            result = static_predict(q['question'], condition, index, docs, {}, model, folder,
                                evidence_budget=release['policy']['evidence_tokens'],
                                token_ceiling=release['policy']['online_token_ceiling'], steps=release['policy']['agent_max_steps'])
                        elif is_v7:
                            result = period_agent_predict(q['question'], release['assets'], temporal, model, folder,
                                release['policy'], memory=condition != 'raw_agent', history=condition == 'agentic')
                        else:
                            selected_banks = flat if condition == 'history_disabled' else banks if condition == 'agentic' else {}
                            if release['online_code_version'] == 'v6_compact_provenance': predictor = agent_predict
                            elif release['online_code_version'] == 'v3_unchanged': predictor = legacy_agent_predict
                            else: raise ValueError('Unsupported frozen online policy')
                            result = predictor(q['question'], release['assets'], index, docs, selected_banks,
                                               model, folder, release['policy'], memory=condition != 'raw_agent')
                        row = {'q_id': q['q_id'], 'condition': condition, 'status': 'completed', **result}
                    except Exception as exc:
                        row = {'q_id': q['q_id'], 'condition': condition, 'status': 'failed',
                               'error': type(exc).__name__ + ': ' + str(exc)}
                    write_json(folder / 'prediction.json', row, immutable=True)
                    print('EVALUATION_PREDICTION', q['q_id'], condition, row['status'], flush=True)
            hashes = prediction_grid(out, release['evaluation_ids'])
            open_release(base, version)
            write_json(out / 'inference_complete.json', {'release_sha256': release_hash,
                       'prediction_sha256': hashes, 'gold_available': False}, immutable=True)
        finally:
            for bank in banks.values(): bank.close()


def grade_candidate(q, prediction, model, folder):
    if prediction['status'] == 'failed':
        return {'label': 'FAILURE', 'correct': False, 'key_points_met': [False]*len(q['key_points']),
                'key_point_fraction': 0.0, 'generation_failure': prediction['error']}
    try:
        judged = parse_json(model.invoke(judgement_prompt(q, prediction), folder, schema=JUDGEMENT))
        if judged['label'] not in {'CORRECT', 'PARTIAL', 'INCORRECT', 'FAILURE'}: raise ValueError('Invalid judge label')
        points = judged['key_points_met']
        if len(points) != len(q['key_points']) or any(type(x) is not bool for x in points):
            raise ValueError('Judge key-point count/type mismatch')
        return {**judged, 'correct': judged['label'] == 'CORRECT',
                'key_point_fraction': sum(points)/max(1, len(points))}
    except Exception as exc:
        return {'label': 'UNASSESSED', 'correct': None, 'key_point_fraction': None,
                'judge_failure': type(exc).__name__ + ': ' + str(exc)}


def online_usage(folder):
    total = Counter(); calls = 0; unknown = 0; event_only = 0
    for path in Path(folder).rglob('request.json'):
        if 'blinded_judge' in path.parts: continue
        response = path.parent / 'response.json'
        if response.exists():
            calls += 1
            usages = json.loads(response.read_text()).get('usage', [])
        else:
            # A policy-rejected response can still have complete provider
            # usage in its trace. Its prediction remains a failure.
            usages = []
            events = path.parent / 'events.jsonl'
            if events.exists():
                for line in events.read_text().splitlines():
                    try: event = json.loads(line)
                    except json.JSONDecodeError: continue
                    if event.get('type') == 'turn.completed':
                        usages.append(event.get('usage', {}))
        reported = [u for u in usages if isinstance(u, dict)
                    and type(u.get('input_tokens')) is int and type(u.get('output_tokens')) is int]
        if not reported: unknown += 1; continue
        if not response.exists(): event_only += 1
        for usage in reported:
            total.update({k: v for k, v in usage.items() if type(v) is int})
    return {'logged_response_calls': calls, 'requests_with_unknown_usage': unknown,
            'known_usage_requests_without_saved_response': event_only, 'provider_usage': dict(total)}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['condition']].append(row)
    result = {}
    for condition, items in groups.items():
        n = len(items); unknown = sum(r['correct'] is None for r in items)
        known_correct = sum(r['correct'] is True for r in items)
        known_points = sum(r['key_point_fraction'] or 0 for r in items)
        usage = Counter()
        for row in items: usage.update(row['online_usage']['provider_usage'])
        result[condition] = {'n': n, 'strict_correct': known_correct, 'unassessed': unknown,
            'correct_rate_lower': known_correct/n, 'correct_rate_upper': (known_correct+unknown)/n,
            'mean_key_point_fraction_lower': known_points/n, 'mean_key_point_fraction_upper': (known_points+unknown)/n,
            'generation_failures': sum('generation_failure' in r for r in items),
            'labels': dict(Counter(r['label'] for r in items)), 'online_provider_usage': dict(usage),
            'logged_response_calls': sum(r['online_usage']['logged_response_calls'] for r in items),
            'known_usage_requests_without_saved_response': sum(r['online_usage'].get('known_usage_requests_without_saved_response', 0) for r in items),
            'requests_with_unknown_usage': sum(r['online_usage']['requests_with_unknown_usage'] for r in items)}
    return result


def grade(base, version, status, common):
    base = Path(base); out = base / version / 'evaluation'
    with single_executor(base / 'evaluation_execution.lock'):
        release, release_hash = open_release(base, version)
        completion_path = out / 'inference_complete.json'
        if not completion_path.exists(): raise ValueError('Complete all predictions before opening held-out gold')
        completion = json.loads(completion_path.read_text())
        if completion != {'release_sha256': release_hash,
                          'prediction_sha256': prediction_grid(out, release['evaluation_ids']), 'gold_available': False}:
            raise ValueError('Completed inference artifacts changed')
        atomic_identity(base / 'evaluation_lock.json', {'method_version': version, 'release_sha256': release_hash})
        gold = json.loads((base / 'cohort/gold/evaluation.json').read_text())
        if [q['q_id'] for q in gold] != release['evaluation_ids']: raise ValueError('Gold cohort differs')
        corpus = {d['_id']: d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] in release['assets']}
        model = LunaV7() if release['online_code_version'] == 'v7_period_procedure' else OnceLuna(); rows = []
        for q in gold:
            for condition in CONDITIONS:
                folder = out / q['q_id'] / condition; prediction = json.loads((folder / 'prediction.json').read_text())
                result_path = folder / 'assessment.json'
                identity = {'prediction_sha256': file_digest(folder / 'prediction.json'), 'release_sha256': release_hash}
                if result_path.exists():
                    row = json.loads(result_path.read_text())
                    if row['identity'] != identity: raise ValueError('Assessed prediction changed')
                else:
                    write_json(status, {**common, 'state': 'grading', 'q_id': q['q_id'], 'condition': condition})
                    judged = grade_candidate(q, prediction, model, folder / 'blinded_judge')
                    grounding = assess(q['question'], folder / 'prediction.json', corpus, model,
                                       out / 'source_grounding' / q['q_id'] / condition)
                    row = {'identity': identity, 'q_id': q['q_id'], 'condition': condition,
                           'assets': q['assets'], **judged, 'source_grounding': grounding,
                           'online_usage': online_usage(folder)}
                    write_json(result_path, row, immutable=True)
                rows.append(row)
        report = {'split': 'evaluation', 'method_version': version, 'release_sha256': release_hash,
                  'summary': summarize(rows), 'rows': rows, 'all_fixed_questions_and_failures_retained': True,
                  'limitations': ['Small fixed toy cohort; no population-level superiority claim.',
                    'Single Luna judge and original Fin-RATE gold; source checks do not replace independent human annotation.',
                    release['cutoff_limitation'], release['compute_limitation']],
                  'unknown_usage_rule': 'Missing response usage is unknown, never zero. Cached input is part of input totals.'}
        write_json(out / 'report.json', report, immutable=True)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'infer', 'grade', 'run'])
    parser.add_argument('--version', default='v6'); parser.add_argument('--review')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.version): raise ValueError('Invalid version')
    cpu_only(); check_disk(); os.nice(10); base = ROOT / 'result/finrate_luna_v1'
    if args.stage == 'freeze':
        if not args.review: raise ValueError('Provide the recorded development review')
        freeze(base, args.version, args.review); print('FROZEN_FOR_EVALUATION', args.version); return
    status = base / args.version / 'evaluation/status.json'; stop = watchdog(status)
    common = {'pid': os.getpid(), 'started': time.time(), 'cpu_only': True,
              'disk_stop_bytes': 10_000_000_000, 'version': args.version}
    try:
        if args.stage in {'infer', 'run'}: infer(base, args.version, status, common)
        if args.stage in {'grade', 'run'}: grade(base, args.version, status, common)
        write_json(status, {**common, 'state': 'completed_' + args.stage, 'finished': time.time()})
    except Exception as exc:
        write_json(status, {**common, 'state': 'evaluation_failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    finally: stop.set()


if __name__ == '__main__': main()
