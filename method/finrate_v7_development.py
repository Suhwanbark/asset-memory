"""Full fixed DEV rerun with v7 transport/procedure; never reads heldout inputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from finrate_agent_v7 import TemporalRetriever, predict
from finrate_final_evaluation import grade_candidate, online_usage, summarize, single_executor, prediction_grid
from finrate_luna_v7 import LunaV7
from finrate_memory_audit import audit
from finrate_source_grading import assess
from finrate_v5_development import BASE, verify_frozen
from finrate_poc.agent import predict as static_predict
from finrate_poc.bank import Bank
from finrate_poc.cohort import approve_gate
from finrate_poc.data import documents, metadata
from finrate_poc.retrieval import Index
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json

OUT = BASE / 'v7'
MAIN = ['naive_llm', 'embedding', 'hybrid', 'raw_agent', 'agentic']
CONDITIONS = [*MAIN, 'history_disabled']


def load(assets):
    docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] in assets]
    banks = {}; verified = {}
    for asset in assets:
        folder = BASE / 'asset_memory/v5' / asset
        completion = json.loads((folder / 'completed.json').read_text()); checked = audit(folder / 'bank.sqlite')
        if file_digest(folder / 'bank.sqlite') != completion['bank_sha256'] or checked['event_hash'] != completion['event_hash']:
            raise ValueError('Immutable v5 bank changed')
        banks[asset] = Bank(folder / 'bank.sqlite', readonly=True)
        verified[asset] = {'completion': completion, 'audit': checked, 'path': str((folder / 'bank.sqlite').relative_to(ROOT))}
    return docs, banks, verified


def prepare():
    parent = json.loads((BASE / 'v6/method_manifest.json').read_text()); verify_frozen(ROOT, parent)
    approve_gate(json.loads((BASE / 'cutoff_audit.json').read_text()), parent['policy'])
    docs, banks, verified = load(parent['assets'])
    try:
        index = Index.build(docs, BASE / 'index')
        started = time.monotonic()
        TemporalRetriever.build(docs, index, banks, OUT / 'fact_index')
        print('FACT_INDEX_READY', round(time.monotonic()-started, 2), flush=True)
        write_json(OUT / 'memory_manifest.json', verified, immutable=True)
        code = dict(parent['code_sha256'])
        for name in ['finrate_agent_v7.py', 'finrate_luna_v7.py', 'finrate_v7_development.py', 'finrate_final_evaluation.py']:
            p = ROOT / 'method' / name; code[str(p.relative_to(ROOT))] = file_digest(p)
        manifest = {**parent, 'method_version': 'v7', 'online_code_version': 'v7_period_procedure',
            'code_sha256': code, 'parent_method_manifest_sha256': file_digest(BASE / 'v6/method_manifest.json'),
            'development_review_sha256': file_digest(BASE / 'v6/development_review.json'),
            'fact_index_sha256': {p.name: file_digest(p) for p in sorted((OUT / 'fact_index').iterdir())},
            'change_reason': 'Completed v6 DEV shows no exercised graph intervention and frequent early finish. Mandatory plan/retrieve/review/synthesize, period-balanced retrieval and source-verified memory navigation.',
            'online_procedure': 'Question-only asset/year/topic plan (with shared source orientation); mandatory temporal collection; gap review and optional second retrieval; final source-grounded JSON.',
            'transport': 'Exact Luna low; native shell/unified_exec/sleep disabled, read-only empty working directory, final JSON schema, no resampling failed requests.',
            'static_baselines': 'Rerun all60 static DEV rows with the same restricted transport/schema as agents. Same corpus, MiniLM, static ranking, final prompt and6000 evidence ceiling.',
            'ablation': 'History-disabled has the same fact index, metadata, direct lookup and three-stage procedure; skip only graph traversal/hints. Independent stochastic model plans remain a limitation.',
            'raw_agent_reuse_reason': None,
            'fixed_before_v7_predictions': True, 'evaluation_allowed': False}
        write_json(OUT / 'method_manifest.json', manifest, immutable=True)
        for name in code:
            p = ROOT / name; dest = OUT / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and file_digest(dest) != file_digest(p): raise ValueError('Snapshot changed')
            shutil.copyfile(p, dest)
        write_json(OUT / 'history_ablation/method_manifest.json', {
            'parent_manifest_sha256': file_digest(OUT / 'method_manifest.json'),
            'condition': 'history_disabled', 'intervention': 'Graph traversal and edge hints disabled; direct facts/dates/retrieval/procedure retained.',
            'independent_stochastic_plans': True, 'evaluation_allowed': False}, immutable=True)
    finally:
        for bank in banks.values(): bank.close()


def run(label):
    manifest = json.loads((OUT / 'method_manifest.json').read_text()); verify_frozen(ROOT, manifest)
    for name, h in manifest['fact_index_sha256'].items():
        if file_digest(OUT / 'fact_index' / name) != h: raise ValueError('Fact index changed')
    conditions = MAIN[:3] if label == 'static' else [label]
    out = OUT / 'history_ablation' if label == 'history_disabled' else OUT
    status = OUT / 'workers' / label / 'status.json'; stop = watchdog(status); banks = {}
    common = {'pid': os.getpid(), 'started': time.time(), 'label': label, 'cpu_only': True,
              'disk_stop_bytes': 10_000_000_000, 'evaluation_touched': False}
    try:
        with single_executor(OUT / 'workers' / label / 'execution.lock'):
            write_json(status, {**common, 'state': 'loading_frozen_inputs'})
            docs, banks, _ = load(manifest['assets']); corpus = {d['_id']: d for d in docs}
            index = Index.build(docs, BASE / 'index')
            retriever = TemporalRetriever.build(docs, index, banks, OUT / 'fact_index') if label != 'static' else None
            model = LunaV7(); questions = json.loads((BASE / 'cohort/dev_inputs.json').read_text())
            if any(set(q) != {'q_id', 'question', 'task'} for q in questions): raise ValueError('Unexpected inference input fields')
            for q in questions:
                for condition in conditions:
                    folder = out / 'dev' / q['q_id'] / condition
                    if (folder / 'prediction.json').exists(): continue
                    write_json(status, {**common, 'state': 'running_development', 'q_id': q['q_id'], 'condition': condition})
                    try:
                        if condition in MAIN[:3]:
                            result = static_predict(q['question'], condition, index, docs, {}, model, folder,
                                evidence_budget=manifest['policy']['evidence_tokens'],
                                token_ceiling=manifest['policy']['online_token_ceiling'], steps=manifest['policy']['agent_max_steps'])
                        else:
                            result = predict(q['question'], manifest['assets'], retriever, model, folder, manifest['policy'],
                                             memory=condition != 'raw_agent', history=condition == 'agentic')
                        row = {'q_id': q['q_id'], 'condition': condition, 'status': 'completed', **result}
                    except Exception as exc:
                        row = {'q_id': q['q_id'], 'condition': condition, 'status': 'failed', 'error': type(exc).__name__+': '+str(exc)}
                    write_json(folder / 'prediction.json', row, immutable=True)
                    print('V7_PREDICTION', q['q_id'], condition, row['status'], flush=True)
            prediction_grid(out / 'dev', [q['q_id'] for q in questions], conditions) if label == 'history_disabled' else None
            gold = json.loads((BASE / 'cohort/gold/dev.json').read_text()); rows = []; grounding = []
            if {q['q_id'] for q in gold} != {q['q_id'] for q in questions}: raise ValueError('DEV cohort mismatch')
            for q in gold:
                for condition in conditions:
                    folder = out / 'dev' / q['q_id'] / condition; p = folder / 'prediction.json'
                    write_json(status, {**common, 'state': 'grading_development', 'q_id': q['q_id'], 'condition': condition})
                    assessment = folder / 'assessment.json'
                    if assessment.exists():
                        row = json.loads(assessment.read_text())
                        if row['prediction_sha256'] != file_digest(p): raise ValueError('Prediction changed')
                    else:
                        row = {'q_id': q['q_id'], 'condition': condition, 'prediction_sha256': file_digest(p),
                               **grade_candidate(q, json.loads(p.read_text()), model, folder / 'blinded_judge'),
                               'online_usage': online_usage(folder)}
                        write_json(assessment, row, immutable=True)
                    rows.append(row)
                    checked = assess(q['question'], p, corpus, model, out / 'source_grounding' / q['q_id'] / condition)
                    grounding.append({'q_id': q['q_id'], 'condition': condition, **checked})
                    print('V7_GRADED', q['q_id'], condition, row['label'], checked['state'], flush=True)
            report = {'split': 'dev', 'rows': rows, 'summary': summarize(rows), 'conditions': conditions,
                      'all_fixed_questions_and_failures_retained': True, 'evaluation_touched': False}
            dest = out if label == 'history_disabled' else OUT / 'workers' / label
            write_json(dest / 'dev_report.json', report, immutable=True)
            write_json(dest / 'source_grounding.json', grounding, immutable=True)
            verify_frozen(ROOT, manifest)
            write_json(status, {**common, 'state': 'development_ready_for_review', 'finished': time.time(),
                                'summary': report['summary']})
    except Exception as exc:
        write_json(status, {**common, 'state': 'worker_failed', 'error': type(exc).__name__+': '+str(exc)})
        raise
    finally:
        for bank in banks.values(): bank.close()
        stop.set()


def aggregate():
    manifest = json.loads((OUT / 'method_manifest.json').read_text()); verify_frozen(ROOT, manifest)
    rows = []; grounding = []
    for label in ['static', 'raw_agent', 'agentic']:
        folder = OUT / 'workers' / label
        rows.extend(json.loads((folder / 'dev_report.json').read_text())['rows'])
        grounding.extend(json.loads((folder / 'source_grounding.json').read_text()))
    ids = json.loads((BASE / 'cohort/cohort_manifest.json').read_text())['selected_ids']['dev']
    prediction_grid(OUT / 'dev', ids, MAIN)
    from finrate_final_evaluation import complete_assessment_rows
    complete_assessment_rows(rows, ids, MAIN); complete_assessment_rows(grounding, ids, MAIN)
    write_json(OUT / 'dev_report.json', {'split': 'dev', 'rows': rows, 'summary': summarize(rows),
        'conditions': MAIN, 'all_fixed_questions_and_failures_retained': True, 'evaluation_touched': False}, immutable=True)
    write_json(OUT / 'source_grounding.json', grounding, immutable=True)
    print('V7_MAIN_GRID_READY', len(rows), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'run', 'aggregate'])
    p.add_argument('--label', choices=['static', 'raw_agent', 'agentic', 'history_disabled']); args = p.parse_args()
    cpu_only(); check_disk(); os.nice(10)
    if args.stage == 'prepare': prepare()
    elif args.stage == 'aggregate': aggregate()
    elif args.label: run(args.label)
    else: raise ValueError('Worker label required')


if __name__ == '__main__': main()
