"""Historical continuity and failure isolation, using synthetic implementation fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from finrate_memory_incremental import append_snapshot
from finrate_poc.bank import Bank
from finrate_poc.memory import build_asset_memory
from finrate_poc.runtime import file_digest


class SourceFixtureModel:
    def __init__(self):
        self.source_calls = []
        self.link_calls = 0

    def invoke(self, prompt, folder, **kwargs):
        obj = json.loads(prompt.rsplit('\n', 1)[1])
        if 'candidate history' in prompt:
            self.link_calls += 1
            return json.dumps({'links': [{'old_id': obj['past'][0]['id'], 'new_id': obj['new'][0]['id'],
                'relation': 'updates', 'rationale': 'A second manufacturer was qualified.',
                'evidence_quote': obj['new'][0]['quote']}]})
        self.source_calls.append(obj['source_id'])
        return json.dumps({'facts': [{'subject': obj['asset'], 'topic': 'manufacturer',
            'predicate': 'arrangement', 'value': obj['window'], 'period': None, 'modality': 'reported',
            'source_id': obj['source_id'], 'quote': obj['window']}]})


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.parent_doc = {'_id': 's1', 'title': 'A_2025-02-01_10-K',
                           'text': 'We use one external manufacturer.'}
        self.new_doc = {'_id': 's2', 'title': 'A_2025-05-01_10-Q',
                        'text': 'We qualified a second external manufacturer.'}
        build_asset_memory([self.parent_doc], self.base / 'initial', SourceFixtureModel())
        self.parent = self.base / 'initial/v1/A'
        self.parent_hash = file_digest(self.parent / 'bank.sqlite')

    def tearDown(self):
        self.temp.cleanup()

    def test_later_quarter_inherits_history_without_reextracting_parent(self):
        model = SourceFixtureModel()
        target = self.base / 'q2'
        report = append_snapshot(self.parent, target, [self.new_doc], model)
        self.assertEqual(model.source_calls, ['s2'])
        self.assertEqual(model.link_calls, 1)
        self.assertEqual(report['inherited_sources'], 1)
        self.assertEqual(file_digest(self.parent / 'bank.sqlite'), self.parent_hash)
        bank = Bank(target / 'bank.sqlite', readonly=True)
        try:
            facts = bank.grep('manufacturer')
            self.assertEqual(len(bank.history(facts[0]['id'])['facts']), 2)
            self.assertEqual([r[0] for r in bank.db.execute('SELECT label FROM checkpoints ORDER BY label')],
                             ['2025-Q1', '2025-Q2'])
            self.assertEqual(bank.verify_event_chain(), report['event_hash'])
        finally:
            bank.close()
        # Completed restart performs zero additional model calls.
        self.assertEqual(append_snapshot(self.parent, target, [self.new_doc], model), report)
        self.assertEqual(model.source_calls, ['s2'])

    def test_rejects_duplicate_source_wrong_asset_and_closed_quarter_before_inference(self):
        for doc in [self.parent_doc, {**self.new_doc, 'title': 'B_2025-05-01_10-Q'},
                    {**self.new_doc, 'title': 'A_2025-03-01_10-Q'}]:
            model = SourceFixtureModel()
            with self.assertRaises(ValueError):
                append_snapshot(self.parent, self.base / 'invalid', [doc], model)
            self.assertFalse(model.source_calls)
        self.assertEqual(file_digest(self.parent / 'bank.sqlite'), self.parent_hash)

    def test_failed_append_does_not_publish_or_change_parent(self):
        class BrokenModel:
            def invoke(self, *args, **kwargs):
                raise RuntimeError('simulated transport failure')
        target = self.base / 'failed'
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            append_snapshot(self.parent, target, [self.new_doc], BrokenModel())
        self.assertFalse((target / 'completed.json').exists())
        self.assertFalse((target / 'bank.sqlite').exists())
        self.assertEqual(file_digest(self.parent / 'bank.sqlite'), self.parent_hash)
        # Restart rebuilds only the delta staging snapshot from the same parent.
        report = append_snapshot(self.parent, target, [self.new_doc], SourceFixtureModel())
        self.assertEqual(report['total_sources'], 2)

    def test_completed_snapshot_rejects_changed_bytes(self):
        target = self.base / 'q2'
        append_snapshot(self.parent, target, [self.new_doc], SourceFixtureModel())
        bank = Bank(target / 'bank.sqlite')
        bank.db.execute("UPDATE facts SET body='{}'")
        bank.db.commit(); bank.close()
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            append_snapshot(self.parent, target, [self.new_doc], SourceFixtureModel())


if __name__ == '__main__':
    unittest.main()
