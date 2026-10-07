import json
from pathlib import Path
import tempfile
import unittest

from finrate_quarter_replay import snapshot, table_hashes
from finrate_memory_audit import audit
from finrate_poc.bank import Bank
from finrate_poc.runtime import file_digest, write_json


def add_quarter(bank, index, previous=None):
    day = ['2025-02-01', '2025-05-01', '2025-08-01'][index-1]
    text = f'The company operates {index} production sites.'
    source_id = f's{index}'
    bank.register_source({'id': source_id, 'published': day, 'title': 'A_'+day+'_10-Q', 'text': text})
    fact_id = bank.add_fact({'subject': 'A', 'topic': 'production sites', 'predicate': 'operates',
        'value': str(index), 'period': day, 'modality': 'reported', 'source_id': source_id, 'quote': text})
    if previous:
        bank.link(old_id=previous, new_id=fact_id, relation='updates', rationale='Updated site count.', evidence_quote=text)
    bank.checkpoint(f'2025-Q{index}')
    return fact_id


def complete_fixture(folder):
    bank = Bank(folder/'bank.sqlite'); ids = []; previous = None
    for index in range(1, 4): previous = add_quarter(bank, index, previous); ids.append(previous)
    event_hash = bank.verify_event_chain(); bank.close()
    write_json(folder/'completed.json', {'asset': 'A', 'bank_sha256': file_digest(folder/'bank.sqlite'),
        'final_quarter': '2025-Q3', 'event_hash': event_hash})
    write_json(folder/'build_manifest.json', {'sources': [{'id': f's{i}'} for i in range(1, 4)], 'batch_count': 3})
    write_json(folder/'span_method_manifest.json', {'fixture': True})
    for index, day in enumerate(['2025-02-01', '2025-05-01', '2025-08-01']):
        (folder/'calls'/f'{index:04d}_{day}').mkdir(parents=True)
    return ids


class QuarterReplayTests(unittest.TestCase):
    def test_snapshot_removes_only_future_and_append_recovers_identical_logical_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp)/'full'; parent = Path(tmp)/'parent'; ids = complete_fixture(full)
            original_hash = file_digest(full/'bank.sqlite'); original_tables = table_hashes(full/'bank.sqlite')
            completion = snapshot(full, parent, '2025-Q2')
            self.assertEqual(completion['total_sources'], 2)
            self.assertEqual(completion['facts'], 2); self.assertEqual(completion['links'], 1)
            self.assertTrue(audit(parent/'bank.sqlite')['passed'])
            self.assertEqual(file_digest(full/'bank.sqlite'), original_hash)
            bank = Bank(parent/'bank.sqlite')
            self.assertEqual([r[0] for r in bank.db.execute('SELECT label FROM checkpoints ORDER BY label')], ['2025-Q1','2025-Q2'])
            self.assertEqual(add_quarter(bank, 3, ids[1]), ids[2]); bank.close()
            self.assertEqual(table_hashes(parent/'bank.sqlite'), original_tables)
            self.assertEqual(file_digest(full/'bank.sqlite'), original_hash)

    def test_unknown_quarter_does_not_create_a_misleading_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp)/'full'; destination = Path(tmp)/'missing'; complete_fixture(full)
            with self.assertRaisesRegex(ValueError, 'Unknown closed-quarter'): snapshot(full, destination, '2024-Q4')
            self.assertFalse(destination.exists())


if __name__ == '__main__': unittest.main()
