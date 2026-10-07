import json
from pathlib import Path
import tempfile
import unittest

from finrate_batched_memory import batches, build_asset, prior_candidates
from finrate_memory_audit import audit
from finrate_poc.bank import Bank
from finrate_poc.runtime import file_digest


class FixtureModel:
    def __init__(self): self.sources = []
    def invoke(self, prompt, folder, **kwargs):
        payload = json.loads(prompt.rsplit('\n', 1)[1])
        if 'windows' in payload:
            self.sources.append([w['source_id'] for w in payload['windows']])
            return json.dumps({'facts': [{'subject': payload['asset'], 'topic': 'manufacturer',
                'predicate': 'arrangement', 'value': w['text'], 'period': None, 'modality': 'reported',
                'source_id': w['source_id'], 'quote': w['text']} for w in payload['windows'] if w['text']]})
        return json.dumps({'links': [{'old_id': payload['past'][0]['id'], 'new_id': payload['new'][0]['id'],
            'relation': 'updates', 'rationale': 'The newer observation describes an additional manufacturer.',
            'evidence_quote': payload['new'][0]['quote']}]})


class BatchedMemoryTests(unittest.TestCase):
    def test_packing_preserves_all_characters_and_never_crosses_asset_or_day(self):
        docs = [{'_id': 'a', 'title': 'A_2025-02-01_10-K', 'text': ('Alpha\n  beta. ' * 50)},
                {'_id': 'b', 'title': 'A_2025-02-02_10-K', 'text': 'Next day information.'},
                {'_id': 'c', 'title': 'B_2025-02-01_10-K', 'text': 'Another company.'},
                {'_id': 'empty', 'title': 'A_2025-02-02_10-K', 'text': ''}]
        packets = list(batches(docs, budget=35))
        restored = {d['_id']: [] for d in docs}
        for p in packets:
            self.assertLessEqual(p['source_tokens'], 35)
            for w in p['windows']:
                d = next(d for d in docs if d['_id'] == w['source_id'])
                self.assertTrue(d['title'].startswith(p['asset'] + '_' + p['published']))
                self.assertEqual(d['text'][w['start']:w['end']], w['text'])
                restored[d['_id']].append(w)
        for d in docs:
            spans = sorted(restored[d['_id']], key=lambda w: w['start'])
            self.assertEqual(''.join(w['text'] for w in spans), d['text'])

    def test_same_day_batch_and_next_quarter_delta_keep_parent_history(self):
        docs = [{'_id': 'a', 'title': 'A_2025-02-01_10-K', 'text': 'We use one external manufacturer.'},
                {'_id': 'b', 'title': 'A_2025-02-01_10-K', 'text': 'The manufacturer supplies all components.'}]
        delta = [{'_id': 'c', 'title': 'A_2025-05-01_10-Q', 'text': 'We qualified a second manufacturer.'}]
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / 'q1'; second = Path(tmp) / 'q2'; model = FixtureModel()
            result = build_asset(docs, first, model)
            self.assertEqual(model.sources, [['a', 'b']])
            before = file_digest(first / 'bank.sqlite')
            next_model = FixtureModel(); updated = build_asset(delta, second, next_model, parent=first)
            self.assertEqual(next_model.sources, [['c']])
            self.assertEqual(updated['total_sources'], 3)
            self.assertEqual(file_digest(first / 'bank.sqlite'), before)
            report = audit(second / 'bank.sqlite')
            self.assertEqual([c['quarter'] for c in report['checkpoints']], ['2025-Q1', '2025-Q2'])
            self.assertGreater(report['links'], 0)
            self.assertEqual(build_asset(delta, second, next_model, parent=first), updated)
            self.assertEqual(next_model.sources, [['c']])

    def test_new_source_is_excluded_before_prior_candidate_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = Bank(Path(tmp) / 'bank.sqlite')
            for source_id, day in [('old', '2025-02-01'), ('new', '2025-05-01')]:
                bank.register_source({'id': source_id, 'published': day, 'title': source_id,
                                      'text': 'We use one external manufacturer.'})
            def fact(source, n):
                return {'subject': 'A', 'topic': 'manufacturer', 'predicate': 'arrangement',
                        'value': str(n), 'period': None, 'modality': 'reported', 'source_id': source,
                        'quote': 'We use one external manufacturer.'}
            old = bank.add_fact(fact('old', 0))
            for i in range(12): current = bank.read_fact(bank.add_fact(fact('new', i)))
            candidates = prior_candidates(bank, current, {'new'}, limit=8)
            self.assertEqual([f['id'] for f in candidates], [old])
            bank.close()

    def test_wrong_asset_or_already_closed_quarter_cannot_extend_parent(self):
        d = {'_id': 'a', 'title': 'A_2025-02-01_10-K', 'text': 'We use one external manufacturer.'}
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / 'parent'; build_asset([d], parent, FixtureModel())
            for new in [{**d, '_id': 'b', 'title': 'B_2025-05-01_10-Q'},
                        {**d, '_id': 'b', 'title': 'A_2025-03-01_10-Q'}]:
                with self.assertRaises(ValueError): build_asset([new], Path(tmp) / 'new', FixtureModel(), parent=parent)


if __name__ == '__main__': unittest.main()
