"""Run the corrected memory-on/off online agents on the frozen DEV set only."""
import argparse
import json
import os
from pathlib import Path
import time

from finrate_agent_v3 import predict
from finrate_dev_controls import grade_dev
from finrate_memory_audit import audit
from finrate_source_grading import assess
from finrate_poc.bank import Bank
from finrate_poc.cohort import approve_gate
from finrate_poc.data import documents, metadata
from finrate_poc.model import Luna
from finrate_poc.retrieval import Index
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['raw_agent', 'agentic'], required=True)
    args = parser.parse_args()
    cpu_only(); check_disk(); os.nice(10)
    base = ROOT / 'result/finrate_luna_v1'; out = base / 'v3'
    status = out / args.condition / 'status.json'
    stop = watchdog(status)
    started = time.time(); deadline = started + 24 * 3600
    common = {'pid': os.getpid(), 'started': started, 'condition': args.condition,
              'cpu_only': True, 'disk_stop_bytes': 10_000_000_000, 'evaluation_touched': False}
    banks = {}
    try:
        policy = json.loads((ROOT / 'data/finrate_luna_protocol_v1.json').read_text())
        approve_gate(json.loads((base / 'cutoff_audit.json').read_text()), policy)
        cohort = base / 'cohort'; manifest = json.loads((cohort / 'cohort_manifest.json').read_text())
        if policy != manifest['policy']: raise ValueError('Frozen policy changed')
        inputs = json.loads((cohort / 'dev_inputs.json').read_text())
        if any(set(q) != {'q_id', 'question', 'task'} for q in inputs): raise ValueError('Unexpected inference fields')
        code_paths = [*(ROOT / 'method/finrate_poc').glob('*.py'),
                      *[ROOT / 'method' / n for n in ['finrate_agent_v3.py', 'finrate_v3_development.py',
                          'finrate_dev_controls.py', 'finrate_memory_audit.py', 'finrate_source_grading.py']]]
        code = {str(p.relative_to(ROOT)): file_digest(p) for p in code_paths}
        write_json(out / 'method_manifest.json', {
            'online_method_version': 'v3', 'offline_memory_version': 'v2', 'code_sha256': code,
            'policy': policy, 'cohort_sha256': file_digest(cohort / 'cohort_manifest.json'),
            'shared_online_changes': 'Explicit bounded regex, whitespace-aligned source spans, date/asset validation, structured actions, bounded planner feedback. Same on/off source tools and answer prompt.',
            'primary_conditions': policy['conditions'], 'control': 'raw_agent',
            'baseline_reuse': 'All 60 static v2 responses and blinded grades retained, including one format failure.',
            'evaluation_allowed': False}, immutable=True)
        if args.condition == 'agentic':
            while True:
                check_disk()
                missing = [a for a in manifest['assets'] if not (base / 'asset_memory/v2' / a / 'completed.json').exists()]
                if not missing: break
                write_json(status, {**common, 'state': 'waiting_for_asset_memories', 'missing_assets': missing})
                if time.time() >= deadline:
                    write_json(status, {**common, 'state': 'ended_waiting_for_asset_memories', 'missing_assets': missing}); return
                time.sleep(30)
            verified = {}
            for asset in manifest['assets']:
                folder = base / 'asset_memory/v2' / asset
                result = audit(folder / 'bank.sqlite')
                completion = json.loads((folder / 'completed.json').read_text())
                if result['event_hash'] != completion['event_hash']: raise ValueError('Completed bank history differs')
                verified[asset] = {'bank_sha256': file_digest(folder / 'bank.sqlite'),
                                   'build_manifest_sha256': file_digest(folder / 'build_manifest.json'), 'audit': result}
                banks[asset] = Bank(folder / 'bank.sqlite', readonly=True)
            write_json(out / 'memory_manifest.json', verified, immutable=True)
        if any(file_digest(ROOT / p) != h for p, h in code.items()):
            raise ValueError('Code changed while waiting; use a new method version')
        write_json(status, {**common, 'state': 'building_or_loading_shared_cpu_index'})
        docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] in manifest['assets']]
        corpus = {d['_id']: d for d in docs}
        index = Index.build(docs, base / 'index'); model = Luna()
        write_json(status, {**common, 'state': 'running_development'})
        for q in inputs:
            folder = out / 'dev' / q['q_id'] / args.condition
            if (folder / 'prediction.json').exists(): continue
            try:
                response = predict(q['question'], manifest['assets'], index, docs, banks, model, folder, policy,
                                   memory=args.condition == 'agentic')
                record = {'q_id': q['q_id'], 'condition': args.condition, 'status': 'completed', **response}
            except Exception as exc:
                record = {'q_id': q['q_id'], 'condition': args.condition, 'status': 'failed',
                          'error': type(exc).__name__ + ': ' + str(exc)}
            write_json(folder / 'prediction.json', record, immutable=True)
            print('V3_PREDICTION', q['q_id'], args.condition, record['status'], flush=True)
        write_json(status, {**common, 'state': 'grading_original_gold'})
        report = grade_dev(cohort, out / 'dev', [args.condition], out / f'{args.condition}_report.json', model)
        write_json(status, {**common, 'state': 'grading_source_grounding'})
        grounding = []
        for q in inputs:
            row = assess(q['question'], out / 'dev' / q['q_id'] / args.condition / 'prediction.json',
                         corpus, model, out / 'source_grounding' / q['q_id'] / args.condition)
            grounding.append({'q_id': q['q_id'], **row})
        write_json(out / f'{args.condition}_source_grounding.json', grounding, immutable=True)
        write_json(status, {**common, 'state': 'development_ready_for_review', 'finished': time.time(),
                            'summary': report['summary'], 'source_assessments': len(grounding)})
    except Exception as exc:
        write_json(status, {**common, 'state': 'development_failed',
                            'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    finally:
        for bank in banks.values(): bank.close()
        stop.set()


if __name__ == '__main__': main()
