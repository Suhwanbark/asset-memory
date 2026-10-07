import json
from pathlib import Path
import tempfile
import unittest

from finrate_memory_audit import audit, snapshot_audit
from finrate_poc.bank import Bank


class MaterializedAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name) / 'bank.sqlite'
        bank = Bank(self.path)
        source = {'id': 's', 'title': 'A_2025-02-01_10-K', 'published': '2025-02-01',
                  'text': 'We use one external manufacturer.'}
        bank.register_source(source)
        bank.add_fact({'subject': 'A', 'topic': 'manufacturing', 'predicate': 'arrangement',
                       'value': source['text'], 'period': None, 'modality': 'reported',
                       'source_id': 's', 'quote': source['text']})
        bank.checkpoint('2025-Q1'); bank.close()

    def tearDown(self): self.tmp.cleanup()

    def test_snapshot_passes_and_source_is_unchanged(self):
        from finrate_poc.runtime import file_digest
        before = file_digest(self.path)
        report = snapshot_audit(self.path, self.path.parent / 'snapshot.sqlite')
        self.assertTrue(report['passed']); self.assertEqual(report['facts'], 1)
        self.assertEqual(file_digest(self.path), before)

    def test_fact_table_tampering_detected_even_with_valid_event_chain(self):
        bank = Bank(self.path)
        bank.db.execute("UPDATE facts SET body='{}'"); bank.db.commit()
        bank.verify_event_chain(); bank.close()  # Hash-chain-only validation misses this.
        with self.assertRaisesRegex(ValueError, 'Fact table content'):
            audit(self.path)

    def test_checkpoint_count_tampering_detected(self):
        bank = Bank(self.path)
        body = json.loads(bank.db.execute('SELECT body FROM checkpoints').fetchone()[0])
        body['facts'] = 100
        bank.db.execute('UPDATE checkpoints SET body=?', (json.dumps(body),)); bank.db.commit(); bank.close()
        with self.assertRaisesRegex(ValueError, 'Checkpoint counts'):
            audit(self.path)


if __name__ == '__main__': unittest.main()
