"""Read completed full-DEV artifacts; no question/gold reads or model calls."""
from collections import Counter
import json
from pathlib import Path

from finrate_final_evaluation import complete_assessment_rows, summarize
from finrate_v7_development import BASE, OUT, CONDITIONS, aggregate
from finrate_poc.runtime import check_disk, file_digest, write_json


def report():
    check_disk()
    for label in ['static', 'raw_agent', 'agentic', 'history_disabled']:
        state = json.loads((OUT / 'workers' / label / 'status.json').read_text())
        if state['state'] != 'development_ready_for_review': raise ValueError('Complete all four workers first')
    aggregate()
    main = json.loads((OUT / 'dev_report.json').read_text())
    ablation = json.loads((OUT / 'history_ablation/dev_report.json').read_text())
    rows = main['rows'] + ablation['rows']
    ids = json.loads((BASE / 'cohort/cohort_manifest.json').read_text())['selected_ids']['dev']
    complete_assessment_rows(rows, ids, CONDITIONS)
    sources = json.loads((OUT / 'source_grounding.json').read_text()) + json.loads(
        (OUT / 'history_ablation/source_grounding.json').read_text())
    complete_assessment_rows(sources, ids, CONDITIONS)
    grid = {(r['q_id'], r['condition']): r for r in rows}
    mechanisms = {}; inputs = {}
    for condition in CONDITIONS:
        found = []; item_counts = Counter(); visible_edge_relations = Counter(); transport_errors = Counter()
        for q in ids:
            folder = OUT / 'history_ablation' if condition == 'history_disabled' else OUT
            p = folder / 'dev' / q / condition / 'prediction.json'
            inputs[str(p.relative_to(OUT))] = file_digest(p)
            prediction = json.loads(p.read_text())
            traces = []
            for path in sorted(p.parent.glob('retrieval_*.json')):
                traces.append(json.loads(path.read_text())); inputs[str(path.relative_to(OUT))] = file_digest(path)
            memory_count = sum(e['id'].startswith('fact_') for e in prediction.get('evidence', []))
            visible = prediction.get('visible_history_links', [])
            visible_edge_relations.update(l['relation'] for l in visible)
            for event_path in p.parent.rglob('events.jsonl'):
                if 'blinded_judge' in event_path.parts: continue
                for line in event_path.read_text().splitlines():
                    try: event = json.loads(line)
                    except json.JSONDecodeError: continue
                    if event.get('type') == 'item.completed':
                        item = event.get('item', {}); item_counts[item.get('type')] += 1
                        if item.get('type') == 'error':
                            category = ('websocket_to_https_fallback' if item.get('message', '').startswith(
                                'Falling back from WebSockets to HTTPS transport.') else 'other_error')
                            transport_errors[category] += 1
            found.append({'q_id': q, 'status': prediction['status'], 'memory_evidence_count': memory_count,
                          'visible_edge_count': len(visible),
                          'history_lookup_count': sum(len(t['history_lookups']) for t in traces if t['history_enabled']),
                          'returned_history_edge_count': sum(len(h['returned_links']) for t in traces for h in t['history_lookups']),
                          'invalid_citations': len(prediction.get('invalid_citation_ids', [])),
                          'budget_exceeded': prediction.get('online_budget_exceeded'),
                          'evidence_tokens': prediction.get('evidence_tokens')})
        source_rows = [s for s in sources if s['condition'] == condition]
        mechanisms[condition] = {'rows': found,
            'questions_with_memory_evidence': sum(r['memory_evidence_count'] > 0 for r in found),
            'questions_with_visible_history_edges': sum(r['visible_edge_count'] > 0 for r in found),
            'visible_edge_relations': dict(visible_edge_relations),
            'online_completed_item_types': dict(item_counts),
            'online_transport_error_events': dict(transport_errors),
            'source_assessment_states': dict(Counter(s['state'] for s in source_rows)),
            'source_grounding': dict(Counter(s.get('overall_grounding', 'unassessed') for s in source_rows)),
            'abstention': dict(Counter(s.get('abstention', 'unassessed') for s in source_rows))}
    pairs = []
    for left, right in [('embedding', 'agentic'), ('hybrid', 'agentic'),
                        ('raw_agent', 'agentic'), ('history_disabled', 'agentic')]:
        points = Counter(); correctness = Counter(); deltas = []
        for q in ids:
            a, b = grid[q, left], grid[q, right]
            if a['correct'] is None or b['correct'] is None:
                correctness['unassessed_pair'] += 1
            else:
                correctness['both' if a['correct'] and b['correct'] else 'left_only' if a['correct']
                            else 'right_only' if b['correct'] else 'neither'] += 1
            x, y = a['key_point_fraction'], b['key_point_fraction']
            if x is None or y is None: points['unassessed_pair'] += 1
            else:
                points['right_better' if y > x else 'left_better' if x > y else 'tie'] += 1
                deltas.append(y-x)
        pairs.append({'left': left, 'right': right, 'n': len(ids), 'correctness_pairs': dict(correctness),
                      'key_point_pairs': dict(points), 'assessed_key_point_pairs': len(deltas),
                      'mean_paired_key_point_difference_among_assessed': sum(deltas)/len(deltas) if deltas else None})
    for p in [OUT/'dev_report.json', OUT/'source_grounding.json', OUT/'history_ablation/dev_report.json',
              OUT/'history_ablation/source_grounding.json', OUT/'method_manifest.json']:
        inputs[str(p.relative_to(OUT))] = file_digest(p)
    result = {'split': 'dev', 'version': 'v7', 'n_questions': len(ids), 'conditions': CONDITIONS,
              'all_fixed_questions_and_failures_retained': True, 'summary': summarize(rows),
              'paired_comparisons': pairs, 'mechanisms': mechanisms, 'rows': rows,
              'input_sha256': inputs, 'analysis_code_sha256': file_digest(__file__),
              'questions_or_gold_read_by_reporter': False, 'new_model_calls': 0, 'evaluation_touched': False,
              'limitations': ['Exploratory fixed toy DEV; no significance or population superiority claim.',
                'Same-family Luna judge and original gold with separately documented quality flags.',
                'Grounding is separate from completeness; abstention can be grounded but incomplete.',
                'Independent stochastic plans; edge exposure alone does not establish causal utility.',
                'Shared ceilings do not equal actual compute; bank and index construction excluded from online totals.',
                'v5 bank costs and semantic audits remain applicable; model pretraining exposure is uncontrolled.']}
    write_json(OUT/'paired_development_analysis.json', result, immutable=True)
    print(json.dumps({'summary': result['summary'], 'paired_comparisons': pairs}), flush=True)
    return result


if __name__ == '__main__': report()
