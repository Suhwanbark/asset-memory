"""Read-only paired development analysis; never reads held-out files or invokes a model.

Original-gold agreement, source grounding, tool use, and actual online transport
usage are separate measurements. Reference-quality flags never change the grid.
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import statistics

from finrate_poc.runtime import ROOT, check_disk, file_digest, write_json


def paired_comparison(rows, ids, conditions):
    expected = {(q, c) for q in ids for c in conditions}
    lookup = {}
    for row in rows:
        key = (row['q_id'], row['condition'])
        if key in lookup: raise ValueError('Duplicate comparison row')
        lookup[key] = row
    if set(lookup) != expected: raise ValueError('Comparison must contain the complete fixed grid')
    if not ids or len(ids) != len(set(ids)): raise ValueError('Question IDs must be nonempty and unique')
    result = []
    for left, right in combinations(conditions, 2):
        cells = Counter(); kp = Counter(); differences = []
        for q in ids:
            a = lookup[q, left]; b = lookup[q, right]
            cells.update(['both_correct' if a['correct'] and b['correct'] else
                          'left_only_correct' if a['correct'] else
                          'right_only_correct' if b['correct'] else 'neither_correct'])
            diff = b['key_point_fraction'] - a['key_point_fraction']
            differences.append(diff)
            kp.update(['right_better' if diff > 1e-12 else 'left_better' if diff < -1e-12 else 'tie'])
        result.append({'left': left, 'right': right, 'n': len(ids),
            'correctness_pairs': {k: cells[k] for k in ['both_correct', 'left_only_correct', 'right_only_correct', 'neither_correct']},
            'key_point_pairs': {k: kp[k] for k in ['left_better', 'right_better', 'tie']},
            'right_minus_left_mean_key_point_fraction': statistics.mean(differences),
            'right_minus_left_correct_rate': (cells['right_only_correct']-cells['left_only_correct'])/len(ids)})
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['condition']].append(row)
    result = {}
    for condition, items in groups.items():
        usage = Counter(); tools = Counter(); claims = Counter()
        for row in items:
            usage.update(row['provider_usage']); tools.update(row['tool_counts']); claims.update(row['source_claim_verdicts'])
        result[condition] = {'n': len(items), 'correct': sum(r['correct'] for r in items),
            'correct_rate': sum(r['correct'] for r in items)/len(items),
            'mean_key_point_fraction': statistics.mean(r['key_point_fraction'] for r in items),
            'labels': dict(Counter(r['label'] for r in items)),
            'generation_failures': sum(r['prediction_status'] == 'failed' for r in items),
            'source_assessment_states': dict(Counter(r['source_assessment_state'] for r in items)),
            'source_grounding': dict(Counter(r['source_grounding'] for r in items)),
            'source_claim_verdicts': dict(claims),
            'unsupported_change_inference_questions': sum(r['unsupported_change_inference'] is True for r in items),
            'tool_counts': dict(tools), 'tool_errors': sum(r['tool_errors'] for r in items),
            'questions_with_memory_evidence': sum(r['memory_evidence_items'] > 0 for r in items),
            'questions_with_history_edges_returned': sum(r['history_edges_returned'] > 0 for r in items),
            'actual_response_calls': sum(r['actual_response_calls'] for r in items),
            'online_provider_usage': dict(usage)}
    return result


def analyze(version='v6', *, baseline_only=False):
    check_disk()
    if version != 'v6': raise ValueError('This analyzer targets the v6 development comparison')
    base = ROOT/'result/finrate_luna_v1'; out = base/version; files = {}
    def read(path):
        files[str(path.relative_to(ROOT))] = file_digest(path)
        return json.loads(path.read_text())
    gold = read(base/'cohort/gold/dev.json'); inputs = read(base/'cohort/dev_inputs.json')
    ids = [q['q_id'] for q in inputs]; questions = {q['q_id']: q for q in gold}
    if set(ids) != set(questions): raise ValueError('Development input/gold grid differs')
    conditions = ['naive_llm', 'embedding', 'hybrid', 'raw_agent']
    if baseline_only:
        reports = [read(base/'controls/raw_agent_v1/static_report.json'), read(base/'v3/raw_agent_report.json')]
    else:
        conditions += ['agentic', 'history_disabled']
        reports = [read(out/'dev_report.json'), read(out/'history_ablation/dev_report.json')]
    graded = []
    for report in reports:
        if report['split'] != 'dev' or report['evaluation_touched']: raise ValueError('Development reports only')
        graded.extend(report['rows'])
    # Fail before analyzing a convenient subset if any method/question is absent.
    paired_comparison(graded, ids, conditions)
    quality = read(base/'development_gold_source_review_v1/reviewed_findings.json')
    flags = defaultdict(list)
    for kind in ['verified_reporting_year_mismatches', 'direct_reference_contradictions']:
        for flag in quality[kind]: flags[flag['q_id']].append(kind)
    if set(flags)-set(ids): raise ValueError('Quality review references a different development set')
    rows = []
    for grade in graded:
        qid = grade['q_id']; condition = grade['condition']; q = questions[qid]
        location = out/'history_ablation' if condition == 'history_disabled' else out
        folder = location/'dev'/qid/condition
        prediction_path = folder/'prediction.json'; prediction = read(prediction_path)
        if prediction['q_id'] != qid or prediction['condition'] != condition: raise ValueError('Prediction identity mismatch')
        points = grade['key_points_met']
        if len(points) != len(q['key_points']) or any(type(v) is not bool for v in points):
            raise ValueError('Invalid key-point judgement')
        if grade['correct'] != (grade['label'] == 'CORRECT') or abs(grade['key_point_fraction']-sum(points)/max(1,len(points))) > 1e-12:
            raise ValueError('Reported score differs from its judgement')
        if prediction['status'] == 'failed' and (grade['label'] != 'FAILURE' or any(points)):
            raise ValueError('Generation failure must remain in the denominator with zero credit')
        assessment = read(location/'source_grounding'/qid/condition/'assessment.json')
        identity = assessment['identity']
        if identity['prediction_sha256'] != file_digest(prediction_path) or identity['question'] != q['question']:
            raise ValueError('Source assessment refers to a different prediction/question')
        tools = Counter(); errors = edges = 0; ledger = folder/'investigation/ledger.jsonl'
        if ledger.exists():
            files[str(ledger.relative_to(ROOT))] = file_digest(ledger)
            for line in ledger.read_text().splitlines():
                item = json.loads(line); tool = item['action']['tool']; tools.update([tool]); errors += 'error' in item
                if tool == 'memory_history': edges += len(item.get('result', {}).get('links', []))
        rows.append({**grade, 'assets': q['assets'], 'reference_quality_flags': flags[qid],
            'prediction_status': prediction['status'], 'source_assessment_state': assessment['state'],
            'source_grounding': assessment.get('overall_grounding', 'not_assessed'),
            'source_claim_verdicts': dict(Counter(c['verdict'] for c in assessment.get('claims', []))),
            'unsupported_change_inference': assessment.get('unsupported_change_inference'),
            'tool_counts': dict(tools), 'tool_errors': errors, 'history_edges_returned': edges,
            'memory_evidence_items': sum(str(e.get('id','')).startswith('fact_') for e in prediction.get('evidence', []))})
    assets = defaultdict(list)
    for q in gold: assets[','.join(sorted(q['assets']))].append(q['q_id'])
    return {'split': 'dev', 'version': version, 'baseline_only': baseline_only, 'conditions': conditions,
        'n_questions': len(ids), 'all_fixed_questions_retained': True, 'evaluation_touched': False,
        'new_model_calls': 0, 'code_sha256': file_digest(__file__), 'input_file_sha256': files,
        'summary': summarize(rows), 'paired_comparisons': paired_comparison(rows, ids, conditions),
        'by_asset': {a: {'q_ids': group, 'summary': summarize([r for r in rows if r['q_id'] in group]),
                        'paired_comparisons': paired_comparison([r for r in rows if r['q_id'] in group], group, conditions)}
                     for a, group in assets.items()},
        'rows': rows, 'limitations': [
            'Original-gold agreement and source grounding remain separate; reference flags do not alter scores or exclude questions.',
            'A single same-family model judged answers; source grounding does not establish answer completeness.',
            'Claims have different lengths and counts; claim verdict counts are not a common-denominator accuracy metric.',
            'Matched text/step ceilings do not mean equal actual computation. Input includes cached input.',
            'Reported online usage excludes offline memory/index construction, judges, and abandoned development runs.',
            'Tool invocation or returned links do not prove useful reasoning or causal benefit; compare the fixed controls.',
            'Questions share only three assets and may share evidence; no independence assumption or significance claim is made.',
            'This is development-only exploratory analysis with model exposure uncontrolled.']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--version', default='v6')
    parser.add_argument('--baseline-only', action='store_true'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = analyze(args.version, baseline_only=args.baseline_only)
    write_json(args.output, result, immutable=True)
    print(json.dumps({'conditions': result['conditions'], 'n_questions': result['n_questions'],
        'summary': result['summary'], 'paired_comparisons': result['paired_comparisons']}, ensure_ascii=False))


if __name__ == '__main__': main()
