from pathlib import Path
import tempfile
import unittest

from finrate_agent_v3 import Tools
from finrate_history_ablation import NoHistoryEdges
from finrate_poc.bank import Bank
from finrate_poc.runtime import file_digest


class HistoryAblationTests(unittest.TestCase):
    def test_removes_edges_but_keeps_all_dated_grep_results_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bank.sqlite'; bank = Bank(path); docs = []
            ids = []
            for source_id, day, quote in [('s1', '2025-02-01', 'One manufacturer supplies all goods.'),
                                          ('s2', '2025-05-01', 'Two manufacturers now supply goods.')]:
                title = 'A_' + day + '_10-Q'
                docs.append({'_id': source_id, 'title': title, 'text': quote})
                bank.register_source({'id': source_id, 'published': day, 'title': title, 'text': quote})
                ids.append(bank.add_fact({'subject': 'A', 'topic': 'manufacturers', 'predicate': 'arrangement',
                    'value': quote, 'period': day, 'modality': 'reported', 'source_id': source_id, 'quote': quote}))
            bank.link(old_id=ids[0], new_id=ids[1], relation='updates', rationale='Additional manufacturer',
                      evidence_quote=docs[1]['text'])
            bank.close(); original_hash = file_digest(path); bank = Bank(path, readonly=True)
            full = Tools(docs, {'A': bank}, ['A'], memory=True)
            flat = Tools(docs, {'A': NoHistoryEdges(bank)}, ['A'], memory=True)
            for tool, args in [('memory_grep', {'asset': 'A', 'pattern': 'manufacturer'}),
                               ('source_grep', {'asset': 'A', 'pattern': 'manufacturer'}),
                               ('memory_grep', {'asset': 'A', 'pattern': 'manufacturer', 'before': '2025-03-01'})]:
                self.assertEqual(full.call(tool, args), flat.call(tool, args))
            args = {'asset': 'A', 'fact_id': ids[1]}
            self.assertEqual(len(full.call('memory_history', args)['evidence']), 2)
            result = flat.call('memory_history', args)
            self.assertEqual([f['id'] for f in result['evidence']], [ids[1]])
            self.assertEqual(result['links'], [])
            self.assertEqual(file_digest(path), original_hash)
            bank.close()


if __name__ == '__main__': unittest.main()
