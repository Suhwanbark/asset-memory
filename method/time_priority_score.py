"""Offline scoring; never changes data, gold, prompts or model responses."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from time_priority_resources import ROOT, disk_check
from time_priority_infer import parse_terminal_answer

DATA = ROOT / 'data/time_priority_v1'
OUT = ROOT / 'result/time_priority_v1'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rate(rows):
    return {'correct': sum(rows), 'n': len(rows), 'accuracy': sum(rows) / len(rows) if rows else None}

def age_group(case, winner):
    dates = [i['published'] for i in case['items']]
    if len(set(dates)) == 1:
        return 'same_date'
    win = next(i for i in case['items'] if i['id'] == winner)
    return 'older_wins' if win['published'] == min(dates) else 'newer_wins'

def score(version='v1'):
    disk_check()
    frozen = json.loads((DATA / 'freeze.json').read_text())
    assert digest(DATA / 'cases.json') == frozen['cases_sha256']
    assert digest(DATA / 'gold.json') == frozen['gold_sha256']
    cases = json.loads((DATA / 'cases.json').read_text())['cases']
    gold = {g['id']: g for g in json.loads((DATA / 'gold.json').read_text())['labels']}
    output = OUT if version=='v1' else OUT/version
    report = {'dataset': frozen, 'prompt_version': version, 'unit_note': '50 authored questions in 16 source clusters; two order variants are repeated measurements, not 100 independent questions.',
              'main_n': 38, 'control_n': 12, 'models': {}, 'baseline_predictions': []}
    simple = defaultdict(list)
    for c in cases:
        target = gold[c['id']]['winner']
        items = c['items']
        dates = [i['published'] for i in items]
        v = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        mat = v.fit_transform([c['question']] + [i['text'] for i in items])
        sim = (mat[1:] @ mat[:1].T).toarray().ravel().tolist()
        lexical = .5 if sim[0] == sim[1] else float(items[int(sim[1] > sim[0])]['id'] == target)
        latest = .5 if dates[0] == dates[1] else float(items[int(dates[1] > dates[0])]['id'] == target)
        oldest = .5 if dates[0] == dates[1] else 1 - latest
        row = {'case_id': c['id'], 'stratum': c['stratum'], 'latest_expected_correct': latest,
               'oldest_expected_correct': oldest, 'lexical_expected_correct': lexical, 'lexical_similarities': sim}
        report['baseline_predictions'].append(row)
        for kind, value in [('always_latest', latest), ('always_oldest', oldest), ('lexical_TFIDF', lexical), ('uniform_random', .5)]:
            for group in ['all', 'controls' if c['stratum'].endswith('_control') else 'main', c['stratum']]:
                simple[kind + '/' + group].append(value)
    report['simple_baselines'] = {k: rate(v) for k, v in simple.items()}
    for model in json.loads((DATA / 'model_manifest.json').read_text()):
        name = model['id'].split('/')[-1]
        folder = output / name
        all_rows = []
        per_case = {}
        grouped = defaultdict(list)
        for c in cases:
            rows = []
            for order in ['AB', 'BA']:
                path = folder / (c['id'] + '__' + order + '.json')
                if not path.exists():
                    continue
                r = json.loads(path.read_text())
                assert r['dataset_sha256'] == frozen['cases_sha256']
                assert r['revision'] == model['revision']
                assert hashlib.sha256(r['prompt'].encode()).hexdigest() == r['prompt_sha256']
                # Normalization only broadens whitespace acceptance; do not reject
                # an already valid answer line because explanatory text follows it.
                answer = r['answer'] or (parse_terminal_answer(r['response']) if version=='v2' else None)
                chosen = r['shown_order'][['A','B'].index(answer)] if answer else None
                ok = chosen == gold[c['id']]['winner']
                row = {'case_id': c['id'], 'cluster': c['cluster'], 'stratum': c['stratum'],
                       'order': order, 'correct': ok, 'valid': answer in ['A', 'B'],
                       'raw_answer': r['answer'], 'answer': answer, 'format_normalized': answer != r['answer'],
                       'chosen_item': chosen, 'gold_item': gold[c['id']]['winner'],
                       'response': r['response'], 'path': str(path), 'sha256': digest(path),
                       'input_tokens': r['input_tokens'], 'output_tokens': r['output_tokens'],
                       'possibly_length_limited': r['output_tokens'] >= r['max_new_tokens'],
                       'likelihood_correct': r['likelihood_choice'] == gold[c['id']]['winner'] if r['likelihood_choice'] else None}
                rows.append(row)
                all_rows.append(row)
                for group in ['all', 'controls' if c['stratum'].endswith('_control') else 'main', c['stratum'], age_group(c, gold[c['id']]['winner'])]:
                    grouped[group].append(ok)
            per_case[c['id']] = rows
        complete = [rows for rows in per_case.values() if len(rows) == 2]
        paired = defaultdict(list)
        pairs = defaultdict(list)
        for c in cases:
            if c['pair_id']:
                pairs[c['pair_id']].append(c)
        pair_details = []
        for pair_id, pair in pairs.items():
            if any(len(per_case[c['id']]) != 2 for c in pair):
                continue
            outcomes = [r for c in pair for r in per_case[c['id']]]
            ok = all(r['correct'] for r in outcomes)
            kind = pair[0]['change']
            paired[kind].append(ok)
            pair_details.append({'id': pair_id, 'kind': kind, 'questions': [c['id'] for c in pair], 'all_orders_all_questions_correct': ok})
        report['models'][name] = {
            'id': model['id'], 'revision': model['revision'], 'completed_responses': len(all_rows),
            'completed_questions_both_orders': len(complete),
            'scores': {k: rate(v) for k, v in grouped.items()},
            'both_orders_correct': rate([all(r['correct'] for r in rows) for rows in complete]),
            'order_invariance': rate([all(r['valid'] for r in rows) and rows[0]['chosen_item'] == rows[1]['chosen_item'] for rows in complete]),
            'invalid_responses': sum(not r['valid'] for r in all_rows),
            'format_normalization_count': sum(r['format_normalized'] for r in all_rows),
            'raw_strict_label_accuracy': rate([r['raw_answer'] in ['A','B'] and (c['items'][['A','B'].index(r['raw_answer']) if r['order']=='AB' else 1-['A','B'].index(r['raw_answer'])]['id']==gold[c['id']]['winner']) for c in cases for r in per_case[c['id']]]),
            'possibly_length_limited': sum(r['possibly_length_limited'] for r in all_rows),
            'likelihood_accuracy_diagnostic_only': rate([r['likelihood_correct'] for r in all_rows if r['likelihood_correct'] is not None]),
            'paired_scores': {k: rate(v) for k, v in paired.items()},
            'pair_details': pair_details,
            'mean_input_tokens': statistics.mean([r['input_tokens'] for r in all_rows]) if all_rows else None,
            'mean_output_tokens': statistics.mean([r['output_tokens'] for r in all_rows]) if all_rows else None,
            'rows': all_rows,
        }
    disk_check()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    summary = {k: {j: v for j, v in m.items() if j not in ['rows', 'pair_details']} for k, m in report['models'].items()}
    print(json.dumps({'models': summary, 'baselines': {k: v for k, v in report['simple_baselines'].items() if k.endswith('/all') or k.endswith('/main')}}, indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--version', choices=['v1','v2','v3'], default='v1')
    score(parser.parse_args().version)
