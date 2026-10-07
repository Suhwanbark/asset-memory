import json
import unittest

from finrate_retrieval_v8 import CoreAugmentRetriever
from finrate_agent_v7 import TemporalRetriever
from finrate_poc.model import token_count
import test_finrate_agent_v7 as fixtures


class CommonCoreTests(unittest.TestCase):
    def test_memory_variants_preserve_identical_source_core(self):
        t, bank, plan = fixtures.RetrievalTests().fixture(); r = CoreAugmentRetriever(t)
        raw, raw_trace = r.collect(plan, memory=False)
        full, full_trace = r.collect(plan, memory=True, history=True)
        flat, flat_trace = r.collect(plan, memory=True, history=False)
        for evidence, trace in [(raw, raw_trace), (full, full_trace), (flat, flat_trace)]:
            self.assertEqual(trace['shared_core_ids'], raw_trace['shared_core_ids'])
            actual = {e['id']: e for e in evidence}
            for core in r.core: self.assertEqual(actual[core['id']], core)
            self.assertLessEqual(token_count(json.dumps(evidence, ensure_ascii=False)), 6000)
            round_trip = json.loads(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
            self.assertEqual(json.dumps(evidence, ensure_ascii=False), json.dumps(round_trip, ensure_ascii=False))
            for e in evidence:
                self.assertEqual(e['text'], t.docs[e['source_id']]['text'][e['start']:e['end']])
                self.assertFalse(e['id'].startswith('fact_'))
        self.assertTrue(full_trace['traversals']); self.assertFalse(flat_trace['traversals'])
        # With no new passages available, no fabricated memory benefit/content.
        self.assertEqual(raw, full); self.assertEqual(raw, flat)

    def test_invalid_memory_quote_cannot_route_unverified_evidence(self):
        t, bank, plan = fixtures.RetrievalTests().fixture(); t.facts['fact_d2020']['quote'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'routing span'):
            CoreAugmentRetriever(t).collect(plan, memory=True, history=True)

    def test_history_can_add_a_new_span_without_displacing_common_evidence(self):
        docs = [{'_id': f'd{i}', 'title': f'AAA_{2020+i//4}-02-20_10-K_item1a',
                 'text': f'Observation {i}: '+('risk ' * 600)} for i in range(12)]
        passages = [{'id': d['_id']+':0:'+str(len(d['text'])), 'source_id': d['_id'],
                     'title': d['title'], 'text': d['text'], 'start': 0, 'end': len(d['text'])} for d in docs]
        facts = [{'id': 'fact_'+d['_id'], 'source_id': d['_id'], 'quote_start': 0,
                  'quote_end': len(d['text']), 'quote': d['text'], 'topic': 'Risk', 'period': None} for d in docs]
        link = {'old_id': 'fact_d0', 'new_id': 'fact_d11', 'relation': 'updates'}
        class LinkedBank(fixtures.FakeBank):
            def history(self, seed, **kwargs):
                current = next(f for f in self.facts if f['id'] == seed)
                return {'facts': [self.facts[0], self.facts[11]], 'links': [link]} if seed in {
                    'fact_d0', 'fact_d11'} else {'facts': [current], 'links': []}
        bank = LinkedBank(facts, [link])
        fact_rows = [{'id': f['id'], 'source_id': f['source_id'], 'text': 'risk'} for f in facts]
        t = TemporalRetriever(docs, fixtures.FakeIndex(passages), {'AAA': bank}, fixtures.FakeIndex(fact_rows))
        r = CoreAugmentRetriever(t); plan = {'assets': ['AAA'], 'years': [2020, 2021],
            'queries': ['risk'], 'annual_preferred': True, 'period_interpretation': 'fixture'}
        flat, flat_trace = r.collect(plan, memory=True, history=False)
        full, full_trace = r.collect(plan, memory=True, history=True)
        graph_ids = set(full_trace['selected_graph_extra_ids'])
        self.assertTrue(graph_ids)
        self.assertTrue(graph_ids - {e['id'] for e in flat})
        self.assertEqual(flat_trace['shared_core_ids'], full_trace['shared_core_ids'])
        self.assertLessEqual(token_count(json.dumps(full, ensure_ascii=False)), 6000)


if __name__ == '__main__': unittest.main()
