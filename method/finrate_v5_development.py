"""Frozen DEV orchestration for v5 span-selected memory and v3 online tools.

Asset workers never open question or gold files. The online worker runs every
fixed development question after all banks pass provenance audits. Evaluation
is deliberately unavailable in this runner.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from finrate_agent_v3 import predict
from finrate_batched_memory import batches
from finrate_dev_controls import grade_dev
from finrate_memory_audit import audit
from finrate_source_grading import assess
from finrate_spanned_memory import build_asset, POLICY
from finrate_poc.bank import Bank
from finrate_poc.cohort import approve_gate
from finrate_poc.data import documents, metadata
from finrate_poc.model import Luna
from finrate_poc.retrieval import Index
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json

CONDITIONS = ['naive_llm', 'embedding', 'hybrid', 'raw_agent', 'agentic']
BASE = ROOT / 'result/finrate_luna_v1'
OUT = BASE / 'v5'


def copy_frozen_tree(source, target):
    source, target = Path(source), Path(target)
    hashes = {str(p.relative_to(source)): file_digest(p) for p in source.rglob('*') if p.is_file()}
    if not hashes: raise ValueError('Cannot reuse an empty artifact tree')
    for name, expected in hashes.items():
        check_disk(); destination = target / name
        if destination.exists():
            if file_digest(destination) != expected: raise ValueError('Reused artifact changed: ' + str(destination))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
        if file_digest(destination) != expected: raise ValueError('Artifact copy mismatch')
    actual = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
    if actual != set(hashes): raise ValueError('Unexpected file in reused artifact tree')
    return hashes


def verify_frozen(root, manifest):
    for relative, expected in manifest['code_sha256'].items():
        if file_digest(Path(root) / relative) != expected:
            raise ValueError('Frozen code changed: ' + relative)
    for relative, expected in manifest['input_sha256'].items():
        if file_digest(Path(root) / relative) != expected:
            raise ValueError('Frozen input changed: ' + relative)


def prepare():
    smoke = json.loads((BASE / 'spanned_smoke_v5/report.json').read_text())
    if not smoke['parent_unchanged'] or not smoke['structural_audit']['passed']:
        raise ValueError('Incremental source-span smoke has not passed')
    policy = json.loads((ROOT / 'data/finrate_luna_protocol_v1.json').read_text())
    approve_gate(json.loads((BASE / 'cutoff_audit.json').read_text()), policy)
    cohort = json.loads((BASE / 'cohort/cohort_manifest.json').read_text())
    if cohort['policy'] != policy: raise ValueError('Cohort policy changed')
    old = json.loads((BASE / 'v3/method_manifest.json').read_text())
    verify_frozen(ROOT, {'code_sha256': old['code_sha256'], 'input_sha256': {}})
    if old['policy'] != policy: raise ValueError('Reused baseline policy differs')
    raw_status = json.loads((BASE / 'v3/raw_agent/status.json').read_text())
    if raw_status['state'] != 'development_ready_for_review': raise ValueError('Raw control is incomplete')
    paths = [*(ROOT / 'method/finrate_poc').glob('*.py'),
             *[ROOT / 'method' / name for name in ['finrate_batched_memory.py', 'finrate_spanned_memory.py',
                'finrate_agent_v3.py', 'finrate_v5_development.py', 'finrate_dev_controls.py',
                'finrate_memory_audit.py', 'finrate_source_grading.py']]]
    inputs = [ROOT / 'data/finrate_luna_protocol_v1.json', ROOT / 'data/finrate_public_v1/source_manifest.json',
              BASE / 'cohort/cohort_manifest.json', BASE / 'cohort/dev_inputs.json']
    manifest = {'method_version': 'v5', 'memory_policy': POLICY, 'online_code_version': 'v3_unchanged',
                'code_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in paths},
                'input_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in inputs},
                'policy': policy, 'assets': cohort['assets'], 'conditions': CONDITIONS,
                'evaluation_allowed': False, 'baseline_predictions_reused': 80,
                'comparison_limit': 'Same sources/model/question set and online ceilings; actual offline/online usage differs and must be reported.',
                'change_reason': 'Reduce per-window construction overhead and quotation transcription failures; no memory-agent QA outcomes used.'}
    write_json(OUT / 'method_manifest.json', manifest, immutable=True)
    for p in paths:
        dest = OUT / 'code_snapshot' / p.relative_to(ROOT)
        if dest.exists() and file_digest(dest) != file_digest(p): raise ValueError('Snapshot differs')
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    questions = json.loads((BASE / 'cohort/dev_inputs.json').read_text()); reused = []
    for q in questions:
        for condition in CONDITIONS[:-1]:
            origin = BASE / 'v3/dev' / q['q_id'] / condition
            if not (origin / 'prediction.json').exists(): raise ValueError('Missing baseline prediction')
            destination = OUT / 'dev' / q['q_id'] / condition
            hashes = copy_frozen_tree(origin, destination)
            source_origin = (BASE / 'v3/source_grounding' if condition == 'raw_agent'
                             else BASE / 'source_grounding_v1') / q['q_id'] / condition
            source_hashes = copy_frozen_tree(source_origin, OUT / 'source_grounding' / q['q_id'] / condition)
            reused.append({'q_id': q['q_id'], 'condition': condition,
                           'origin': str(origin.relative_to(BASE)), 'file_sha256': hashes,
                           'source_grounding_origin': str(source_origin.relative_to(BASE)),
                           'source_grounding_sha256': source_hashes})
    write_json(OUT / 'baseline_reuse.json', {'rows': reused, 'all_failures_retained': True}, immutable=True)
    print('PREPARED', len(reused), 'unchanged baseline predictions', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'asset', 'agentic'])
    parser.add_argument('--asset')
    args = parser.parse_args(); cpu_only(); check_disk(); os.nice(10)
    if args.stage == 'prepare': prepare(); return
    manifest = json.loads((OUT / 'method_manifest.json').read_text()); verify_frozen(ROOT, manifest)
    if args.stage == 'asset' and args.asset not in manifest['assets']: raise ValueError('Unknown frozen asset')
    label = args.asset if args.stage == 'asset' else 'agentic'
    status = OUT / 'workers' / label / 'status.json'; stop = watchdog(status)
    common = {'pid': os.getpid(), 'started': time.time(), 'stage': args.stage, 'asset': args.asset,
              'cpu_only': True, 'disk_stop_bytes': 10_000_000_000, 'evaluation_touched': False}
    banks = {}
    try:
        if args.stage == 'asset':
            write_json(status, {**common, 'state': 'building_asset'})
            docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] == args.asset]
            folder = BASE / 'asset_memory/v5' / args.asset
            model = Luna()
            completion = build_asset(docs, folder, model)
            checked = audit(folder / 'bank.sqlite'); verify_frozen(ROOT, manifest)
            if checked['sources'] != len(docs): raise ValueError('Incomplete source corpus')
            write_json(folder / 'structural_audit.json', checked, immutable=True)
            write_json(status, {**common, 'state': 'asset_ready', 'finished': time.time(),
                                'new_model_calls': model.calls, 'completion': completion})
            return
        deadline = time.time() + 24*3600
        while True:
            check_disk()
            missing = [a for a in manifest['assets'] if not (BASE / 'asset_memory/v5' / a / 'structural_audit.json').exists()]
            if not missing: break
            for a in missing:
                other = OUT / 'workers' / a / 'status.json'
                if other.exists() and json.loads(other.read_text())['state'] in {'worker_failed', 'stopped_low_disk'}:
                    raise RuntimeError('Asset worker stopped: ' + a)
            if time.time() >= deadline: raise TimeoutError('Incomplete banks after 24h wait')
            write_json(status, {**common, 'state': 'waiting_for_asset_memories', 'missing_assets': missing})
            time.sleep(15)
        verify_frozen(ROOT, manifest); verified = {}
        for asset in manifest['assets']:
            folder = BASE / 'asset_memory/v5' / asset
            completion = json.loads((folder / 'completed.json').read_text()); checked = audit(folder / 'bank.sqlite')
            if file_digest(folder / 'bank.sqlite') != completion['bank_sha256'] or checked['event_hash'] != completion['event_hash']:
                raise ValueError('Completed bank changed')
            verified[asset] = {'completion': completion, 'audit': checked,
                              'method_manifest_sha256': file_digest(folder / 'span_method_manifest.json')}
            banks[asset] = Bank(folder / 'bank.sqlite', readonly=True)
        write_json(OUT / 'memory_manifest.json', verified, immutable=True)
        docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] in manifest['assets']]
        corpus = {d['_id']: d for d in docs}; index = Index.build(docs, BASE / 'index'); model = Luna()
        questions = json.loads((BASE / 'cohort/dev_inputs.json').read_text())
        if any(set(q) != {'q_id', 'question', 'task'} for q in questions): raise ValueError('Unexpected inference fields')
        write_json(status, {**common, 'state': 'running_development'})
        for q in questions:
            folder = OUT / 'dev' / q['q_id'] / 'agentic'
            if (folder / 'prediction.json').exists(): continue
            try:
                response = predict(q['question'], manifest['assets'], index, docs, banks, model,
                                   folder, manifest['policy'], memory=True)
                record = {'q_id': q['q_id'], 'condition': 'agentic', 'status': 'completed', **response}
            except Exception as exc:
                record = {'q_id': q['q_id'], 'condition': 'agentic', 'status': 'failed',
                          'error': type(exc).__name__ + ': ' + str(exc)}
            write_json(folder / 'prediction.json', record, immutable=True)
            print('V5_PREDICTION', q['q_id'], record['status'], flush=True)
        write_json(status, {**common, 'state': 'grading_original_gold'})
        report = grade_dev(BASE / 'cohort', OUT / 'dev', CONDITIONS, OUT / 'dev_report.json', model)
        write_json(status, {**common, 'state': 'grading_source_grounding'})
        grounding = []
        for q in questions:
            for condition in CONDITIONS:
                row = assess(q['question'], OUT / 'dev' / q['q_id'] / condition / 'prediction.json', corpus,
                             model, OUT / 'source_grounding' / q['q_id'] / condition)
                grounding.append({'q_id': q['q_id'], 'condition': condition, **row})
        write_json(OUT / 'source_grounding.json', grounding, immutable=True)
        write_json(status, {**common, 'state': 'development_ready_for_review', 'finished': time.time(),
                            'summary': report['summary'], 'source_assessments': len(grounding)})
    except Exception as exc:
        write_json(status, {**common, 'state': 'worker_failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    finally:
        for bank in banks.values(): bank.close()
        stop.set()


if __name__ == '__main__': main()
