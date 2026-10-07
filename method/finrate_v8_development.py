"""Fixed full DEV with shared planning, immutable source core and cost components."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from finrate_agent_v7 import TemporalRetriever
from finrate_agent_v8 import prepare_common, predict
from finrate_accounting_v8 import components, summary, unique_inference_accounting
from finrate_final_evaluation import grade_candidate, complete_assessment_rows, prediction_grid, single_executor
from finrate_luna_v7 import LunaV7
from finrate_source_grading import assess
from finrate_v5_development import BASE, copy_frozen_tree, verify_frozen
from finrate_v7_development import load
from finrate_poc.retrieval import Index
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json

OUT = BASE/'v8'
STATIC = ['naive_llm', 'embedding', 'hybrid']
AGENTS = ['raw_agent', 'agentic', 'history_disabled']
CONDITIONS = [*STATIC, *AGENTS]


def prepare():
    previous = BASE/'v7'; parent = json.loads((previous/'method_manifest.json').read_text())
    verify_frozen(ROOT, parent)
    if json.loads((previous/'completion_audits/status.json').read_text())['state'] != 'audits_ready_for_review':
        raise ValueError('Complete v7 before preparing v8')
    copy_frozen_tree(previous/'fact_index', OUT/'fact_index')
    write_json(OUT/'memory_manifest.json', json.loads((previous/'memory_manifest.json').read_text()), immutable=True)
    inputs = json.loads((BASE/'cohort/dev_inputs.json').read_text()); reuse = []
    for q in inputs:
        for condition in STATIC:
            source = previous/'dev'/q['q_id']/condition
            files = copy_frozen_tree(source, OUT/'dev'/q['q_id']/condition)
            grounding = copy_frozen_tree(previous/'source_grounding'/q['q_id']/condition,
                                         OUT/'source_grounding'/q['q_id']/condition)
            reuse.append({'q_id': q['q_id'], 'condition': condition, 'origin': str(source.relative_to(ROOT)),
                          'file_sha256': files, 'source_grounding_sha256': grounding})
    write_json(OUT/'baseline_reuse.json', {'rows': reuse, 'all_failures_and_unassessed_retained': True,
        'reason': 'Static retrieval, exact model/transport/schema and final prompt are unchanged from v7.'}, immutable=True)
    old_rows = json.loads((previous/'dev_report.json').read_text())['rows']
    rows = [{**r, **components(OUT/'dev'/r['q_id']/r['condition']), 'reused_baseline': True}
            for r in old_rows if r['condition'] in STATIC]
    source_rows = [r for r in json.loads((previous/'source_grounding.json').read_text()) if r['condition'] in STATIC]
    write_json(OUT/'static_report.json', {'rows': rows, 'summary': summary(rows), 'new_model_calls': 0}, immutable=True)
    write_json(OUT/'static_source_grounding.json', source_rows, immutable=True)
    code = dict(parent['code_sha256'])
    for name in ['finrate_retrieval_v8.py', 'finrate_agent_v8.py', 'finrate_accounting_v8.py',
                 'finrate_v8_development.py', 'finrate_final_evaluation_v8.py', 'finrate_v8_report.py']:
        p = ROOT/'method'/name; code[str(p.relative_to(ROOT))] = file_digest(p)
    files = [OUT/'baseline_reuse.json', OUT/'development_plan.json', previous/'development_review.json',
             *list((OUT/'fact_index').iterdir())]
    frozen_inputs = {**parent['input_sha256'], **{str(p.relative_to(ROOT)): file_digest(p) for p in files}}
    manifest = {**parent, 'method_version': 'v8', 'online_code_version': 'v8_shared_core',
        'code_sha256': code, 'input_sha256': frozen_inputs,
        'parent_method_manifest_sha256': file_digest(previous/'method_manifest.json'),
        'development_review_sha256': file_digest(previous/'development_review.json'),
        'change_reason': 'v7 did not establish memory benefit; all20 query plans differed across conditions. Share fresh source-only planning/review and preserve identical4800-token core, routing memory only to extra source spans.',
        'online_procedure': 'Two shared source-only planning/review calls per question, then one independent final-answer call per agent. Common source core retained in every condition; extra capacity up to6000tokens.',
        'ablation': 'Skip graph expansion only; same shared plan/core, same direct fact lookup, source fallback, final prompt/schema and budgets.',
        'static_baselines': 'All60 v7 static predictions/grades/source assessments reused byte-for-byte including the naive UNASSESSED row.',
        'cost_rule': 'Count shared planning once physically; show standalone planning+answer attribution for each agent separately from actual totals.',
        'fixed_before_v8_predictions': True, 'evaluation_allowed': False}
    write_json(OUT/'method_manifest.json', manifest, immutable=True)
    for name in code:
        p = ROOT/name; dest = OUT/'code_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and file_digest(dest) != file_digest(p): raise ValueError('Code snapshot changed')
        shutil.copyfile(p, dest)
    write_json(OUT/'history_ablation/method_manifest.json', {
        'parent_manifest_sha256': file_digest(OUT/'method_manifest.json'), 'condition': 'history_disabled',
        'intervention': 'Remove graph expansion only, retain the identical shared plan/core and direct fact lookup.',
        'evaluation_allowed': False}, immutable=True)
    print('V8_PREPARED', len(reuse), 'unchanged static rows', flush=True)


def wait_shared(q, status, common):
    folder = OUT/'shared_planning'/q['q_id']; deadline = time.monotonic()+2*3600
    while not (folder/'shared.json').exists():
        check_disk()
        upstream = OUT/'workers/shared_planning/status.json'
        if upstream.exists():
            state = json.loads(upstream.read_text())['state']
            if 'failed' in state or 'low_disk' in state: raise RuntimeError('Shared planning worker stopped: '+state)
        if time.monotonic() > deadline: raise TimeoutError('Shared question plan missing after2hours')
        write_json(status, {**common, 'state': 'waiting_for_shared_plan', 'q_id': q['q_id']})
        time.sleep(10)
    return folder


def run(label):
    manifest = json.loads((OUT/'method_manifest.json').read_text()); verify_frozen(ROOT, manifest)
    status = OUT/'workers'/label/'status.json'; stop = watchdog(status); banks = {}
    common = {'pid': os.getpid(), 'started': time.time(), 'label': label, 'cpu_only': True,
              'disk_stop_bytes': 10_000_000_000, 'evaluation_touched': False}
    try:
        with single_executor(OUT/'workers'/label/'execution.lock'):
            write_json(status, {**common, 'state': 'loading_frozen_inputs'})
            docs, banks, _ = load(manifest['assets']); corpus = {d['_id']: d for d in docs}
            index = Index.build(docs, BASE/'index')
            temporal = TemporalRetriever.build(docs, index, banks, OUT/'fact_index')
            model = LunaV7(max_calls=150); questions = json.loads((BASE/'cohort/dev_inputs.json').read_text())
            if any(set(q) != {'q_id', 'question', 'task'} for q in questions): raise ValueError('Unexpected inference fields')
            if label == 'shared_planning':
                records = []
                for q in questions:
                    write_json(status, {**common, 'state': 'planning_development', 'q_id': q['q_id']})
                    shared = prepare_common(q['question'], manifest['assets'], temporal, model,
                                            OUT/'shared_planning'/q['q_id'], manifest['policy'])
                    records.append({'q_id': q['q_id'], 'status': shared['status']})
                    print('V8_SHARED_PLAN', q['q_id'], shared['status'], flush=True)
                verify_frozen(ROOT, manifest)
                write_json(status, {**common, 'state': 'shared_planning_ready', 'finished': time.time(), 'rows': records})
                return
            out = OUT/'history_ablation' if label == 'history_disabled' else OUT
            for q in questions:
                folder = out/'dev'/q['q_id']/label
                if (folder/'prediction.json').exists(): continue
                shared_folder = wait_shared(q, status, common)
                write_json(status, {**common, 'state': 'running_development', 'q_id': q['q_id']})
                try:
                    result = predict(q['question'], manifest['assets'], temporal, model, folder, shared_folder,
                                     manifest['policy'], memory=label != 'raw_agent', history=label == 'agentic')
                    row = {'q_id': q['q_id'], 'condition': label, 'status': 'completed', **result}
                except Exception as exc:
                    row = {'q_id': q['q_id'], 'condition': label, 'status': 'failed', 'error': type(exc).__name__+': '+str(exc)}
                write_json(folder/'prediction.json', row, immutable=True)
                print('V8_PREDICTION', q['q_id'], label, row['status'], flush=True)
            rows = []; grounding = []; gold = json.loads((BASE/'cohort/gold/dev.json').read_text())
            if {q['q_id'] for q in gold} != {q['q_id'] for q in questions}: raise ValueError('DEV cohort differs')
            for q in gold:
                folder = out/'dev'/q['q_id']/label; p = folder/'prediction.json'
                write_json(status, {**common, 'state': 'grading_development', 'q_id': q['q_id']})
                assessment = folder/'assessment.json'
                if assessment.exists():
                    row = json.loads(assessment.read_text())
                    if row['prediction_sha256'] != file_digest(p): raise ValueError('Prediction changed')
                else:
                    row = {'q_id': q['q_id'], 'condition': label, 'prediction_sha256': file_digest(p),
                           **grade_candidate(q, json.loads(p.read_text()), model, folder/'blinded_judge'),
                           **components(folder, OUT/'shared_planning'/q['q_id']), 'reused_baseline': False}
                    write_json(assessment, row, immutable=True)
                rows.append(row)
                checked = assess(q['question'], p, corpus, model, out/'source_grounding'/q['q_id']/label)
                grounding.append({'q_id': q['q_id'], 'condition': label, **checked})
                print('V8_GRADED', q['q_id'], label, row['label'], checked['state'], flush=True)
            dest = out if label == 'history_disabled' else OUT/'workers'/label
            write_json(dest/'dev_report.json', {'split': 'dev', 'conditions': [label], 'rows': rows,
                'summary': summary(rows), 'all_fixed_questions_and_failures_retained': True, 'evaluation_touched': False}, immutable=True)
            write_json(dest/'source_grounding.json', grounding, immutable=True)
            verify_frozen(ROOT, manifest)
            write_json(status, {**common, 'state': 'development_ready_for_review', 'finished': time.time(), 'summary': summary(rows)})
    except Exception as exc:
        write_json(status, {**common, 'state': 'worker_failed', 'error': type(exc).__name__+': '+str(exc)})
        raise
    finally:
        for bank in banks.values(): bank.close()
        stop.set()


def aggregate():
    verify_frozen(ROOT, json.loads((OUT/'method_manifest.json').read_text()))
    rows = json.loads((OUT/'static_report.json').read_text())['rows']
    grounding = json.loads((OUT/'static_source_grounding.json').read_text())
    for label in ['raw_agent', 'agentic']:
        folder = OUT/'workers'/label
        rows.extend(json.loads((folder/'dev_report.json').read_text())['rows'])
        grounding.extend(json.loads((folder/'source_grounding.json').read_text()))
    ids = json.loads((BASE/'cohort/cohort_manifest.json').read_text())['selected_ids']['dev']
    main = [*STATIC, 'raw_agent', 'agentic']
    prediction_grid(OUT/'dev', ids, main)
    complete_assessment_rows(rows, ids, main); complete_assessment_rows(grounding, ids, main)
    write_json(OUT/'dev_report.json', {'split': 'dev', 'conditions': main, 'rows': rows,
        'summary': summary(rows), 'all_fixed_questions_and_failures_retained': True, 'evaluation_touched': False}, immutable=True)
    write_json(OUT/'source_grounding.json', grounding, immutable=True)
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'run', 'aggregate'])
    p.add_argument('--label', choices=['shared_planning', *AGENTS]); args = p.parse_args()
    cpu_only(); check_disk(); os.nice(10)
    if args.stage == 'prepare': prepare()
    elif args.stage == 'aggregate': aggregate()
    elif args.label: run(args.label)
    else: raise ValueError('Worker label required')


if __name__ == '__main__': main()
