"""v6 compact memory presentation; reuse unchanged v5 banks and baselines."""
import argparse
import json
import os
import shutil
import time

from finrate_agent_v6 import predict
from finrate_dev_controls import grade_dev
from finrate_history_ablation import NoHistoryEdges
from finrate_memory_audit import audit
from finrate_source_grading import assess
from finrate_v5_development import BASE, copy_frozen_tree, verify_frozen
from finrate_poc.bank import Bank
from finrate_poc.data import documents, metadata
from finrate_poc.model import Luna
from finrate_poc.retrieval import Index
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json

OUT = BASE / 'v6'
MAIN_CONDITIONS = ['naive_llm', 'embedding', 'hybrid', 'raw_agent', 'agentic']


def prepare():
    previous = json.loads((BASE / 'v5/method_manifest.json').read_text()); verify_frozen(ROOT, previous)
    code = dict(previous['code_sha256'])
    for name in ['finrate_agent_v6.py', 'finrate_v6_development.py', 'finrate_history_ablation.py']:
        p = ROOT / 'method' / name; code[str(p.relative_to(ROOT))] = file_digest(p)
    manifest = {**previous, 'method_version': 'v6', 'offline_memory_version': 'v5',
        'online_code_version': 'v6_compact_provenance', 'code_sha256': code,
        'parent_method_manifest_sha256': file_digest(BASE / 'v5/method_manifest.json'),
        'change_reason': 'Remove duplicate extraction quotations and audit hashes only from memory presented to the model. Preserve complete provenance in the bank and tool ledger. No memory-agent QA outputs used.',
        'ablation': 'Same compact presentation with all history edges removed; all observations and dates retained.',
        'raw_agent_reuse_reason': 'Raw tool results and planner/final prompts unchanged; byte-equivalence tested.'}
    write_json(OUT / 'method_manifest.json', manifest, immutable=True)
    for name in code:
        p = ROOT / name; dest = OUT / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and file_digest(dest) != file_digest(p): raise ValueError('Code snapshot differs')
        shutil.copyfile(p, dest)
    questions = json.loads((BASE / 'cohort/dev_inputs.json').read_text()); reused = []
    for q in questions:
        for condition in MAIN_CONDITIONS[:-1]:
            origin = BASE / 'v5/dev' / q['q_id'] / condition
            hashes = copy_frozen_tree(origin, OUT / 'dev' / q['q_id'] / condition)
            source = BASE / 'v5/source_grounding' / q['q_id'] / condition
            source_hashes = copy_frozen_tree(source, OUT / 'source_grounding' / q['q_id'] / condition)
            reused.append({'q_id': q['q_id'], 'condition': condition, 'origin': str(origin.relative_to(BASE)),
                           'file_sha256': hashes, 'source_grounding_origin': str(source.relative_to(BASE)),
                           'source_grounding_sha256': source_hashes})
    write_json(OUT / 'baseline_reuse.json', {'rows': reused, 'all_failures_retained': True}, immutable=True)
    print('PREPARED_V6', len(reused), 'unchanged baseline predictions', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--condition', choices=['agentic', 'history_disabled'])
    args = parser.parse_args(); cpu_only(); check_disk(); os.nice(10)
    if args.stage == 'prepare': prepare(); return
    if not args.condition: raise ValueError('Condition required')
    manifest = json.loads((OUT / 'method_manifest.json').read_text()); verify_frozen(ROOT, manifest)
    out = OUT if args.condition == 'agentic' else OUT / 'history_ablation'
    status = OUT / 'workers' / args.condition / 'status.json'; stop = watchdog(status)
    started = time.time(); banks = {}
    common = {'pid': os.getpid(), 'started': started, 'condition': args.condition, 'cpu_only': True,
              'disk_stop_bytes': 10_000_000_000, 'evaluation_touched': False}
    try:
        if args.condition == 'history_disabled':
            write_json(out / 'method_manifest.json', {
                'parent_manifest_sha256': file_digest(OUT / 'method_manifest.json'),
                'condition': args.condition, 'intervention': 'history(seed) returns only the original seed and no edges.',
                'all_observations_dates_and_grep_results_retained': True,
                'same_compact_presentation': True, 'evaluation_allowed': False}, immutable=True)
        while True:
            check_disk()
            missing = [a for a in manifest['assets'] if not (BASE / 'asset_memory/v5' / a / 'structural_audit.json').exists()]
            if not missing: break
            for asset in missing:
                p = BASE / 'v5/workers' / asset / 'status.json'
                if p.exists() and json.loads(p.read_text())['state'] in {'worker_failed', 'stopped_low_disk'}:
                    raise RuntimeError('Asset worker stopped: ' + asset)
            if time.time()-started > 24*3600: raise TimeoutError('Incomplete banks after 24h')
            write_json(status, {**common, 'state': 'waiting_for_asset_memories', 'missing_assets': missing})
            time.sleep(15)
        verify_frozen(ROOT, manifest); verified = {}
        for asset in manifest['assets']:
            folder = BASE / 'asset_memory/v5' / asset
            completion = json.loads((folder / 'completed.json').read_text()); checked = audit(folder / 'bank.sqlite')
            if file_digest(folder / 'bank.sqlite') != completion['bank_sha256'] or checked['event_hash'] != completion['event_hash']:
                raise ValueError('Completed bank changed')
            verified[asset] = {'completion': completion, 'audit': checked,
                              'path': str((folder / 'bank.sqlite').relative_to(ROOT))}
            banks[asset] = Bank(folder / 'bank.sqlite', readonly=True)
        write_json(out / 'memory_manifest.json', verified, immutable=True)
        views = banks if args.condition == 'agentic' else {a: NoHistoryEdges(b) for a, b in banks.items()}
        docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] in manifest['assets']]
        corpus = {d['_id']: d for d in docs}; index = Index.build(docs, BASE / 'index'); model = Luna()
        questions = json.loads((BASE / 'cohort/dev_inputs.json').read_text())
        if any(set(q) != {'q_id', 'question', 'task'} for q in questions): raise ValueError('Unexpected inference fields')
        write_json(status, {**common, 'state': 'running_development'})
        for q in questions:
            folder = out / 'dev' / q['q_id'] / args.condition
            if (folder / 'prediction.json').exists(): continue
            try:
                result = predict(q['question'], manifest['assets'], index, docs, views, model,
                                 folder, manifest['policy'], memory=True)
                row = {'q_id': q['q_id'], 'condition': args.condition, 'status': 'completed', **result}
            except Exception as exc:
                row = {'q_id': q['q_id'], 'condition': args.condition, 'status': 'failed',
                       'error': type(exc).__name__ + ': ' + str(exc)}
            write_json(folder / 'prediction.json', row, immutable=True)
            print('V6_PREDICTION', q['q_id'], args.condition, row['status'], flush=True)
        conditions = MAIN_CONDITIONS if args.condition == 'agentic' else ['history_disabled']
        write_json(status, {**common, 'state': 'grading_original_gold'})
        report = grade_dev(BASE / 'cohort', out / 'dev', conditions, out / 'dev_report.json', model)
        write_json(status, {**common, 'state': 'grading_source_grounding'}); rows = []
        for q in questions:
            for condition in conditions:
                row = assess(q['question'], out / 'dev' / q['q_id'] / condition / 'prediction.json',
                             corpus, model, out / 'source_grounding' / q['q_id'] / condition)
                rows.append({'q_id': q['q_id'], 'condition': condition, **row})
        write_json(out / 'source_grounding.json', rows, immutable=True)
        write_json(status, {**common, 'state': 'development_ready_for_review', 'finished': time.time(),
                            'summary': report['summary'], 'source_assessments': len(rows)})
    except Exception as exc:
        write_json(status, {**common, 'state': 'worker_failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    finally:
        for bank in banks.values(): bank.close()
        stop.set()


if __name__ == '__main__': main()
