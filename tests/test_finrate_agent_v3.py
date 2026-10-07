import json
from pathlib import Path
import tempfile
import unittest
import regex

from finrate_agent_v3 import Tools, original_match_span, predict


class AgentContractTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {'_id': 'old', 'title': 'A_2020-04-01_10-K', 'text': 'Legal\n  Proceedings: no pending claims.'},
            {'_id': 'new', 'title': 'A_2021-04-01_10-K', 'text': 'There is one pending lawsuit.'},
            {'_id': 'b', 'title': 'B_2021-04-01_10-K', 'text': 'Legal proceedings are reported.'}]

    def test_regex_or_and_linebreaks_return_original_spans(self):
        tool = Tools(self.docs, {}, ['A', 'B'], memory=False)
        result = tool.call('source_grep', {'asset': 'A', 'pattern': 'Legal Proceedings|lawsuit'})
        self.assertEqual([e['source_id'] for e in result['evidence']], ['new', 'old'])
        for e in result['evidence']:
            d = next(d for d in self.docs if d['_id'] == e['source_id'])
            self.assertEqual(e['text'], d['text'][e['start']:e['end']])
        span = original_match_span(self.docs[0]['text'], regex.compile('Legal Proceedings'))
        self.assertEqual(self.docs[0]['text'][span[0]:span[1]], 'Legal\n  Proceedings')

    def test_date_and_asset_filters_are_shared_source_metadata_filters(self):
        tool = Tools(self.docs, {}, ['A', 'B'], memory=False)
        result = tool.call('source_grep', {'asset': 'A', 'pattern': 'proceedings|lawsuit',
                                           'after': '2020-01-01', 'before': '2020-12-31'})
        self.assertEqual([e['source_id'] for e in result['evidence']], ['old'])

    def test_bad_argument_and_regex_fail_explicitly(self):
        tool = Tools(self.docs, {}, ['A', 'B'], memory=False)
        with self.assertRaisesRegex(ValueError, 'Unknown argument'):
            tool.call('source_grep', {'optional asset': 'A', 'pattern': 'lawsuit'})
        with self.assertRaisesRegex(ValueError, 'Invalid regex'):
            tool.call('source_grep', {'asset': 'A', 'pattern': '('})
        with self.assertRaisesRegex(ValueError, 'empty string'):
            tool.call('source_grep', {'asset': 'A', 'pattern': '^'})

    def test_memory_tools_unavailable_in_control(self):
        tool = Tools(self.docs, {}, ['A', 'B'], memory=False)
        for name in ['memory_grep', 'memory_history']:
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                tool.call(name, {'asset': 'A', 'pattern': 'lawsuit', 'fact_id': 'x'})

    def test_common_final_answer_and_structured_planning(self):
        class Index:
            def search(self, *args, **kwargs): return []
        class Model:
            def __init__(self): self.calls = []
            def invoke(self, prompt, folder, **kwargs):
                self.calls.append((prompt, kwargs))
                if len(self.calls) == 1:
                    return json.dumps({'reason': 'find evidence', 'tool': 'source_grep',
                        'arguments': {'asset': 'A', 'pattern': 'lawsuit|proceedings', 'before': None,
                                      'after': None, 'source_id': None, 'fact_id': None, 'start': None}})
                return '{"answer":"One lawsuit was reported in the later filing.", "claims":[]}'
        model = Model()
        with tempfile.TemporaryDirectory() as tmp:
            result = predict('What changed?', ['A', 'B'], Index(), self.docs, {}, model, Path(tmp),
                             {'online_token_ceiling': 24000, 'evidence_tokens': 6000, 'agent_max_steps': 1}, memory=False)
        self.assertIn('schema', model.calls[0][1])
        self.assertEqual(model.calls[-1][1], {})
        from finrate_poc.model import answer_prompt
        self.assertEqual(model.calls[-1][0], answer_prompt('What changed?', result['evidence']))
        self.assertEqual(result['model_calls'], 2)
        self.assertFalse(result['online_budget_exceeded'])


if __name__ == '__main__': unittest.main()
