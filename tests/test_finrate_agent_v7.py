import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finrate_agent_v7 import TemporalRetriever, diverse, interleave, validate_plan
from finrate_luna_v7 import LunaV7
from finrate_poc.model import token_count
from finrate_source_grading import prepare


class FakeIndex:
    def __init__(self, rows): self.passages = rows
    def search(self, question, **kwargs): return self.passages


class FakeDB:
    def __init__(self, facts): self.facts = facts
    def execute(self, sql): return [{'id': f['id'], 'body': json.dumps(f)} for f in self.facts]


class FakeBank:
    def __init__(self, facts, links): self.db = FakeDB(facts); self.facts = facts; self.links = links; self.calls = 0
    def history(self, seed, **kwargs):
        self.calls += 1
        return {'facts': self.facts, 'links': self.links}


class RetrievalTests(unittest.TestCase):
    def fixture(self):
        docs = [{'_id': 'd'+str(y), 'title': f'AAA_{y}-02-20_10-K_item3',
                 'text': 'Header '+f'Legal proceeding observation {y}. '*20} for y in (2020, 2021, 2022, 2024)]
        rows = [{'id': d['_id']+':0:80', 'source_id': d['_id'], 'title': d['title'],
                 'start': 0, 'end': 80, 'text': d['text'][:80]} for d in reversed(docs)]
        facts = [{'id': 'fact_'+d['_id'], 'source_id': d['_id'], 'quote_start': 0, 'quote_end': 80,
                  'quote': d['text'][:80], 'observed_at': d['title'][4:14], 'topic': 'Legal', 'period': None}
                 for d in docs[:3]]
        links = [{'old_id': facts[0]['id'], 'new_id': facts[1]['id'], 'relation': 'continues'}]
        bank = FakeBank(facts, links)
        fr = [{'id': f['id'], 'source_id': f['source_id'], 'text': 'Legal', 'start': 0, 'end': 80} for f in facts]
        retriever = TemporalRetriever(docs, FakeIndex(rows), {'AAA': bank}, FakeIndex(fr))
        plan = {'assets': ['AAA'], 'years': [2020, 2021], 'queries': ['Legal'],
                'annual_preferred': True, 'period_interpretation': 'Requested years'}
        return retriever, bank, plan

    def test_old_periods_survive_latest_first_ranking_and_spans_are_exact(self):
        r, b, plan = self.fixture(); evidence, trace = r.collect(plan, memory=False, history=False, budget=6000)
        self.assertEqual({e['source_id'] for e in evidence}, {'d2020', 'd2021', 'd2022'})
        self.assertEqual(b.calls, 0)
        for e in evidence: self.assertEqual(e['text'], r.docs[e['source_id']]['text'][e['start']:e['end']])
        payload, audit = prepare('question', {'status': 'completed', 'answer': {'answer': 'x', 'claims': []},
                                             'evidence': evidence}, r.docs)
        self.assertFalse(audit['gold_fields_available'])

    def test_edge_intervention_keeps_direct_observations_and_exercises_graph(self):
        r, b, plan = self.fixture(); full, yes = r.collect(plan, memory=True, history=True, budget=6000)
        flat, no = r.collect(plan, memory=True, history=False, budget=6000)
        self.assertEqual(yes['seed_ids'], no['seed_ids'])
        self.assertTrue(yes['visible_history_links']); self.assertEqual(no['visible_history_links'], [])
        self.assertGreater(b.calls, 0)
        self.assertEqual({e['id'] for e in full}, {e['id'] for e in flat})
        self.assertLessEqual(token_count(json.dumps(full, ensure_ascii=False)), 6000)

    def test_source_mismatch_fails_closed(self):
        r, b, plan = self.fixture(); r.facts['fact_d2020']['quote'] = 'made up'
        with self.assertRaisesRegex(ValueError, 'quote changed'):
            r.collect(plan, memory=True, history=True, budget=6000)

    def test_overlap_removal_and_period_interleave(self):
        self.assertEqual(interleave({2021: ['c', 'd'], 2020: ['a', 'b']}, limit=4), ['a', 'c', 'b', 'd'])
        rows = [{'id': str(a), 'source_id': 'd', 'start': a, 'end': a+100} for a in (0, 1, 100)]
        self.assertEqual([e['start'] for e in diverse(rows)], [0, 100])

    def test_invalid_plan_not_silently_substituted(self):
        r, b, plan = self.fixture(); plan['assets'] = ['UNKNOWN']
        with self.assertRaises(ValueError): validate_plan(plan, ['AAA'])


class TransportTests(unittest.TestCase):
    def test_native_flags_schema_cache_and_unfinished_request(self):
        events = '\n'.join(json.dumps(e) for e in [
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"answer":"ok","claims":[]}'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 11, 'output_tokens': 3}}])
        with tempfile.TemporaryDirectory() as tmp, patch('finrate_luna_v7.command', return_value=(0, events, '')) as call:
            m = LunaV7(); p = Path(tmp)/'call'; m.invoke('Test', p); m.invoke('Test', p)
            self.assertEqual(call.call_count, 1)
            argv = call.call_args.args[0]
            for name in ['shell_tool', 'unified_exec', 'sleep_tool']:
                self.assertEqual(argv[argv.index(name)-1], '--disable')
            self.assertIn('--output-schema', argv)
            self.assertEqual(argv[argv.index('-m')+1], 'gpt-5.6-luna')
            (p/'response.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'no automatic regeneration'): m.invoke('Test', p)

    def test_rejected_native_call_keeps_usage_trace(self):
        events = json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution', 'command': 'true'}})
        with tempfile.TemporaryDirectory() as tmp, patch('finrate_luna_v7.command', return_value=(0, events, '')):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected tool'): LunaV7().invoke('Test', tmp)
            self.assertTrue((Path(tmp)/'events.jsonl').exists())
            self.assertFalse((Path(tmp)/'response.json').exists())


if __name__ == '__main__': unittest.main()
