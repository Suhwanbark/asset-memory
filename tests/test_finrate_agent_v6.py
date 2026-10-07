import inspect
import json
from pathlib import Path
import tempfile
import unittest

import finrate_agent_v3 as old
import finrate_agent_v6 as new
from finrate_poc.model import token_count


class ModelFixture:
    def __init__(self): self.requests = []
    def invoke(self, prompt, folder, **kwargs):
        self.requests.append((prompt, kwargs))
        if kwargs:
            return json.dumps({'reason': 'Enough source evidence', 'tool': 'finish', 'arguments': {
                k: None for k in ['asset', 'pattern', 'before', 'after', 'source_id', 'fact_id', 'start']}})
        return json.dumps({'answer': 'The company uses one manufacturer.',
                           'claims': [{'text': 'One manufacturer.', 'evidence_ids': ['s:0:33']}]})


class CompactProvenanceTests(unittest.TestCase):
    def test_online_policy_diff_is_only_the_tool_constructor(self):
        before = inspect.getsource(old.predict)
        after = inspect.getsource(new.predict)
        self.assertEqual(after, before.replace('tools = Tools(', 'tools = PresentationTools('))

    def test_removes_only_redundancy_and_retains_full_audit_result(self):
        quote = 'The company uses one manufacturer. ' * 30
        fact = {'id': 'fact_a', 'source_id': 's', 'subject': 'A', 'topic': 'manufacturer',
                'predicate': 'uses', 'value': 'one manufacturer', 'modality': 'reported',
                'period': '2025-Q1', 'observed_at': '2025-05-01', 'quote': quote,
                'quote_start': 10, 'quote_end': 10+len(quote), 'extraction_quote': quote,
                'quote_alignment': 'exact', 'source_sha256': 'a'*64}
        link = {'old_id': 'fact_old', 'new_id': 'fact_a', 'relation': 'continues',
                'rationale': 'Same reported dependency', 'evidence_quote': quote,
                'proposed_evidence_quote': quote, 'quote_alignment': 'exact'}
        result = {'evidence': [fact], 'links': [link], 'message': ''}
        compact = new.present(result)
        self.assertEqual(compact['audit_full_result'], result)
        self.assertEqual(compact['evidence'][0], {k: v for k, v in fact.items()
                         if k not in {'extraction_quote', 'quote_alignment', 'source_sha256'}})
        self.assertEqual(compact['links'][0]['evidence_quote'], quote)
        self.assertLess(token_count(json.dumps(compact['evidence'])), token_count(json.dumps(result['evidence'])))
        source_only = {'evidence': [{'id': 's:0:33', 'source_id': 's', 'text': quote}], 'links': []}
        self.assertIs(new.present(source_only), source_only)

    def test_raw_agent_prompts_and_predictions_are_byte_identical(self):
        text = 'The company uses one manufacturer.'
        evidence = {'id': 's:0:33', 'source_id': 's', 'title': 'A_2025-05-01_10-Q',
                    'start': 0, 'end': len(text), 'text': text}
        class Index:
            def search(self, *args, **kwargs): return [evidence]
        docs = [{'_id': 's', 'title': evidence['title'], 'text': text}]
        policy = {'online_token_ceiling': 24000, 'evidence_tokens': 6000, 'agent_max_steps': 4}
        with tempfile.TemporaryDirectory() as tmp:
            left, right = ModelFixture(), ModelFixture()
            a = old.predict('Which manufacturer?', ['A'], Index(), docs, {}, left, Path(tmp)/'old', policy, memory=False)
            b = new.predict('Which manufacturer?', ['A'], Index(), docs, {}, right, Path(tmp)/'new', policy, memory=False)
            self.assertEqual(left.requests, right.requests)
            self.assertEqual(a, b)


if __name__ == '__main__': unittest.main()
