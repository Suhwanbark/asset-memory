"""DEV edge-deletion ablation: identical observations/dates, no history edges.

The v3 planner, action schema and source/memory-grep results are unchanged.
history(seed) returns just the seed, as it would in a graph with no edges.
This isolates link traversal, not all temporal information: dates and past
observations remain accessible through the common grep tools.
"""
import json
import os
import time

from finrate_agent_v3 import predict
from finrate_dev_controls import grade_dev
from finrate_memory_audit import audit
from finrate_source_grading import assess
from finrate_v5_development import BASE, OUT, verify_frozen
from finrate_poc.bank import Bank
from finrate_poc.data import documents, metadata
from finrate_poc.model import Luna
from finrate_poc.retrieval import Index
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json


class NoHistoryEdges:
    def __init__(self, bank):
        self.bank = bank
        self.db = bank.db

    def history(self, fact_id, *, limit=20):
        if not 1 <= limit <= 50: raise ValueError('Invalid history limit')
        return {'facts': [self.bank.read_fact(fact_id)], 'links': []}


def main():
    cpu_only(); check_disk(); os.nice(10)
    out = OUT / 'history_ablation'; status = out / 'status.json'
    stop = watchdog(status); started = time.time(); banks = {}
    common = {'pid': os.getpid(), 'started': started, 'cpu_only': True,
              'disk_stop_bytes': 10_000_000_000, 'evaluation_touched': False}
    try:
        manifest = json.loads((OUT / 'method_manifest.json').read_text()); verify_frozen(ROOT, manifest)
        own_hash = file_digest(__file__)
        prior = out / 'method_manifest.json'
        before_results = (json.loads(prior.read_text())['declared_before_memory_QA_results'] if prior.exists()
                          else not any((OUT / 'dev').glob('*/agentic/prediction.json')))
        write_json(out / 'method_manifest.json', {
            'parent_manifest_sha256': file_digest(OUT / 'method_manifest.json'),
            'ablation_code_sha256': own_hash, 'condition': 'history_disabled',
            'intervention': 'Remove history edges only; history(seed) returns its original seed observation.',
            'unchanged': 'All original sources, facts, dates, grep results, planner/action schema, answer prompt and compute ceilings.',
            'limitation': 'Tests graph-traversal benefit, not the value of all temporal metadata; actual calls/tokens and stochastic outputs may differ.',
            'declared_before_memory_QA_results': before_results,
            'evaluation_allowed': False}, immutable=True)
        while True:
            check_disk()
            missing = [a for a in manifest['assets'] if not (BASE / 'asset_memory/v5' / a / 'structural_audit.json').exists()]
            if not missing: break
            for a in missing:
                path = OUT / 'workers' / a / 'status.json'
                if path.exists() and json.loads(path.read_text())['state'] in {'worker_failed', 'stopped_low_disk'}:
                    raise RuntimeError('Asset worker stopped: ' + a)
            if time.time()-started > 24*3600: raise TimeoutError('Incomplete banks after 24h')
            write_json(status, {**common, 'state': 'waiting_for_asset_memories', 'missing_assets': missing})
            time.sleep(15)
        verify_frozen(ROOT, manifest)
        if file_digest(__file__) != own_hash: raise ValueError('Ablation code changed while waiting')
        hashes = {}
        for asset in manifest['assets']:
            folder = BASE / 'asset_memory/v5' / asset
            completion = json.loads((folder / 'completed.json').read_text())
            hashes[asset] = file_digest(folder / 'bank.sqlite')
            if hashes[asset] != completion['bank_sha256']: raise ValueError('Bank changed')
            audit(folder / 'bank.sqlite')
            banks[asset] = Bank(folder / 'bank.sqlite', readonly=True)
        write_json(out / 'bank_hashes.json', hashes, immutable=True)
        views = {asset: NoHistoryEdges(bank) for asset, bank in banks.items()}
        docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] in manifest['assets']]
        corpus = {d['_id']: d for d in docs}; index = Index.build(docs, BASE / 'index'); model = Luna()
        questions = json.loads((BASE / 'cohort/dev_inputs.json').read_text())
        if any(set(q) != {'q_id', 'question', 'task'} for q in questions): raise ValueError('Unexpected inference fields')
        write_json(status, {**common, 'state': 'running_development'})
        for q in questions:
            folder = out / 'dev' / q['q_id'] / 'history_disabled'
            if (folder / 'prediction.json').exists(): continue
            try:
                result = predict(q['question'], manifest['assets'], index, docs, views, model, folder,
                                 manifest['policy'], memory=True)
                record = {'q_id': q['q_id'], 'condition': 'history_disabled', 'status': 'completed', **result}
            except Exception as exc:
                record = {'q_id': q['q_id'], 'condition': 'history_disabled', 'status': 'failed',
                          'error': type(exc).__name__ + ': ' + str(exc)}
            write_json(folder / 'prediction.json', record, immutable=True)
            print('ABLATION_PREDICTION', q['q_id'], record['status'], flush=True)
        write_json(status, {**common, 'state': 'grading_original_gold'})
        report = grade_dev(BASE / 'cohort', out / 'dev', ['history_disabled'], out / 'dev_report.json', model)
        write_json(status, {**common, 'state': 'grading_source_grounding'})
        rows = []
        for q in questions:
            row = assess(q['question'], out / 'dev' / q['q_id'] / 'history_disabled/prediction.json', corpus,
                         model, out / 'source_grounding' / q['q_id'] / 'history_disabled')
            rows.append({'q_id': q['q_id'], **row})
        write_json(out / 'source_grounding.json', rows, immutable=True)
        for asset, expected in hashes.items():
            if file_digest(BASE / 'asset_memory/v5' / asset / 'bank.sqlite') != expected:
                raise ValueError('Ablation mutated its source bank')
        write_json(status, {**common, 'state': 'development_ready_for_review', 'finished': time.time(),
                            'summary': report['summary'], 'source_assessments': len(rows)})
    except Exception as exc:
        write_json(status, {**common, 'state': 'worker_failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    finally:
        for bank in banks.values(): bank.close()
        stop.set()


if __name__ == '__main__': main()
