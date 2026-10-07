"""Full fixed-DEV comparisons, shared-input checks and nonduplicated costs."""
from collections import Counter
import json

from finrate_accounting_v8 import summary, unique_inference_accounting
from finrate_final_evaluation import complete_assessment_rows
from finrate_v8_development import BASE, OUT, CONDITIONS, AGENTS, aggregate
from finrate_poc.runtime import check_disk, file_digest, write_json


def report():
    check_disk()
    for label in ['shared_planning', *AGENTS]:
        state = json.loads((OUT/'workers'/label/'status.json').read_text())['state']
        expected = 'shared_planning_ready' if label == 'shared_planning' else 'development_ready_for_review'
        if state != expected: raise ValueError('Complete all v8 workers before reporting')
    rows = aggregate() + json.loads((OUT/'history_ablation/dev_report.json').read_text())['rows']
    ids = json.loads((BASE/'cohort/cohort_manifest.json').read_text())['selected_ids']['dev']
    complete_assessment_rows(rows, ids, CONDITIONS)
    source = json.loads((OUT/'source_grounding.json').read_text()) + json.loads((OUT/'history_ablation/source_grounding.json').read_text())
    complete_assessment_rows(source, ids, CONDITIONS)
    grid = {(r['q_id'], r['condition']): r for r in rows}; inputs = {}; prompts = {}; mechanisms = {}; shared_checks = []
    for condition in CONDITIONS:
        checks = []
        for q in ids:
            folder = (OUT/'history_ablation' if condition == 'history_disabled' else OUT)/'dev'/q/condition
            p = folder/'prediction.json'; prediction = json.loads(p.read_text()); inputs[str(p.relative_to(OUT))] = file_digest(p)
            req = folder/'answer/request.json'
            prompts[q, condition] = json.loads(req.read_text())['prompt_sha256'] if req.exists() else None
            trace_path = folder/'retrieval.json'; trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
            if trace_path.exists(): inputs[str(trace_path.relative_to(OUT))] = file_digest(trace_path)
            memory_ids = {r['evidence_id'] for r in trace.get('selected_memory_routes', [])}
            checks.append({'q_id': q, 'status': prediction['status'],
                           'memory_routed_extra_count': len(memory_ids),
                           'graph_routed_extra_count': len(trace.get('selected_graph_extra_ids', [])),
                           'graph_expansion_calls': len(trace.get('traversals', [])),
                           'core_preserved': trace.get('all_core_evidence_preserved'),
                           'evidence_tokens': prediction.get('evidence_tokens'),
                           'standalone_text_tokens': prediction.get('online_text_tokens'),
                           'budget_exceeded': prediction.get('online_budget_exceeded')})
        selected = [s for s in source if s['condition'] == condition]
        mechanisms[condition] = {'rows': checks,
            'questions_with_memory_routed_extra': sum(r['memory_routed_extra_count'] > 0 for r in checks),
            'questions_with_graph_routed_extra': sum(r['graph_routed_extra_count'] > 0 for r in checks),
            'source_states': dict(Counter(s['state'] for s in selected)),
            'source_grounding': dict(Counter(s.get('overall_grounding', 'unassessed') for s in selected)),
            'abstention': dict(Counter(s.get('abstention', 'unassessed') for s in selected))}
    for q in ids:
        p = OUT/'shared_planning'/q/'shared.json'; common = json.loads(p.read_text()); h = file_digest(p)
        inputs[str(p.relative_to(OUT))] = h; core = {e['id']: e for e in common.get('core', [])}; checked = []
        for c in AGENTS:
            root = OUT/'history_ablation' if c == 'history_disabled' else OUT
            prediction = json.loads((root/'dev'/q/c/'prediction.json').read_text())
            if prediction['status'] != 'completed': continue
            supplied = {e['id']: e for e in prediction['evidence']}
            if prediction['shared_plan_sha256'] != h or any(supplied.get(i) != e for i, e in core.items()):
                raise ValueError('Agent conditions did not retain identical shared planning/core')
            checked.append(c)
        shared_checks.append({'q_id': q, 'shared_status': common['status'], 'verified_conditions': checked})
    pairs = []
    for left in ['embedding', 'hybrid', 'raw_agent', 'history_disabled']:
        right = 'agentic'; correct = Counter(); points = Counter(); delta = []; same = []; same_disagreement = []
        for q in ids:
            a, b = grid[q, left], grid[q, right]
            if a['correct'] is None or b['correct'] is None: correct['unassessed_pair'] += 1
            else: correct['both' if a['correct'] and b['correct'] else 'left_only' if a['correct'] else 'right_only' if b['correct'] else 'neither'] += 1
            x, y = a['key_point_fraction'], b['key_point_fraction']
            if x is None or y is None: points['unassessed_pair'] += 1
            else:
                points['right_better' if y > x else 'left_better' if x > y else 'tie'] += 1; delta.append(y-x)
            if prompts[q, left] and prompts[q, left] == prompts[q, right]:
                same.append(q)
                if a['correct'] != b['correct'] or x != y: same_disagreement.append(q)
        pairs.append({'left': left, 'right': right, 'n': len(ids), 'correctness_pairs': dict(correct),
                      'key_point_pairs': dict(points), 'assessed_pairs': len(delta),
                      'mean_paired_key_point_difference_among_assessed': sum(delta)/len(delta) if delta else None,
                      'identical_final_prompt_questions': same,
                      'score_disagreements_despite_identical_final_prompt': same_disagreement})
    for p in [OUT/'dev_report.json', OUT/'source_grounding.json', OUT/'history_ablation/dev_report.json',
              OUT/'history_ablation/source_grounding.json', OUT/'method_manifest.json', OUT/'baseline_reuse.json']:
        inputs[str(p.relative_to(OUT))] = file_digest(p)
    result = {'version': 'v8', 'split': 'dev', 'n_questions': len(ids), 'conditions': CONDITIONS,
              'all_fixed_questions_and_failures_retained': True, 'summary': summary(rows), 'rows': rows,
              'unique_inference_accounting': unique_inference_accounting(rows), 'paired_comparisons': pairs,
              'shared_input_checks': shared_checks, 'mechanisms': mechanisms, 'input_sha256': inputs,
              'analysis_code_sha256': file_digest(__file__), 'new_model_calls': 0, 'evaluation_touched': False,
              'limitations': ['Exploratory small fixed DEV; no population superiority or significance claim.',
                'Shared plans/core remove query-plan differences, but independent answers/judges still introduce stochastic variation.',
                'Identical final prompts cannot support a memory information-effect claim even when sampled scores differ.',
                'Graph-routed extras may also be discoverable through direct retrieval; compare actual inputs with the edge-disabled control.',
                'Original-gold agreement and source grounding remain separate; gold quality flags do not change the cohort.',
                'Standalone cost attribution repeats shared components; actual new physical call totals count common planning once.',
                'Offline bank/index costs and earlier development are excluded from online totals; exposure is uncontrolled.']}
    write_json(OUT/'paired_development_analysis.json', result, immutable=True)
    print(json.dumps({'summary': result['summary'], 'pairs': pairs}), flush=True)
    return result


if __name__ == '__main__': report()
