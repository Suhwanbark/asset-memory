import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finrate_dev_controls import judgement_prompt, raw_agent_predict
from finrate_poc.scoring import grade


class ControlTests(unittest.TestCase):
    def test_grading_prompt_matches_frozen_main_grader_exactly(self):
        question = {'q_id': 'q', 'question': 'What changed?', 'answer': 'Added a manufacturer.',
                    'key_points': ['one additional manufacturer']}
        prediction = {'status': 'completed', 'answer': {'answer': 'One added.', 'claims': []},
                      'evidence': [{'id': 'p', 'source_id': 's', 'text': 'One manufacturer was added.'}],
                      'online_text_tokens': 80, 'model_calls': 1, 'invalid_citation_ids': [],
                      'online_budget_exceeded': False}
        prompts = []
        class Judge:
            def invoke(self, prompt, folder, **kwargs):
                prompts.append(prompt)
                return json.dumps({'label': 'CORRECT', 'key_points_met': [True],
                    'citation_support': 'supported', 'temporal_error': False, 'entity_error': False,
                    'reason': 'Supported.'})
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base / 'cohort/gold').mkdir(parents=True)
            (base / 'cohort/gold/dev.json').write_text(json.dumps([question]))
            for condition in ['naive_llm', 'embedding', 'hybrid', 'agentic']:
                folder = base / 'v2/dev/q' / condition; folder.mkdir(parents=True)
                (folder / 'prediction.json').write_text(json.dumps(prediction))
            with patch('finrate_poc.scoring.Luna', Judge):
                grade(base / 'cohort', base, split='dev', version='v2')
        self.assertEqual(prompts, [judgement_prompt(question, prediction)] * 4)

    def test_raw_agent_uses_same_source_tools_without_memory(self):
        docs = [{'_id': 's', 'title': 'A_2025-05-01_10-Q',
                 'text': 'A qualified a second manufacturer this quarter.'}]
        class EmptyIndex:
            def search(self, question, **kwargs):
                self.arguments = kwargs
                return []
        class Model:
            def __init__(self): self.prompts = []
            def invoke(self, prompt, folder):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return '{"reason":"find the change", "tool":"source_grep", "arguments":{"asset":"A","pattern":"manufacturer"}}'
                if len(self.prompts) == 2:
                    return '{"reason":"found evidence", "tool":"finish", "arguments":{}}'
                return json.dumps({'answer': 'Added a manufacturer.',
                    'claims': [{'text': 'Added one.', 'evidence_ids': [f's:0:{len(docs[0]["text"])}']}]})
        policy = {'online_token_ceiling': 24000, 'evidence_tokens': 6000, 'agent_max_steps': 4}
        model = Model(); index = EmptyIndex()
        with tempfile.TemporaryDirectory() as tmp:
            result = raw_agent_predict('What changed?', ['A'], index, docs, model, Path(tmp), policy)
        self.assertEqual(index.arguments, {'method': 'hybrid', 'k': 60})
        self.assertEqual(result['model_calls'], 3)
        self.assertFalse(result['invalid_citation_ids'])
        self.assertEqual(result['evidence'][0]['source_id'], 's')
        self.assertTrue(all('memory_grep' not in p and 'memory_history' not in p for p in model.prompts))
        self.assertIn(docs[0]['text'], model.prompts[-1])

    def test_raw_agent_rejects_hallucinated_memory_tool(self):
        class EmptyIndex:
            def search(self, *args, **kwargs): return []
        class Model:
            def __init__(self): self.calls = 0
            def invoke(self, prompt, folder):
                self.calls += 1
                if self.calls == 1:
                    return '{"tool":"memory_grep", "arguments":{"asset":"A","pattern":"x"}}'
                return '{"answer":"Insufficient evidence.", "claims":[]}'
        with tempfile.TemporaryDirectory() as tmp:
            result = raw_agent_predict('Question?', ['A'], EmptyIndex(), [], Model(), Path(tmp),
                                       {'online_token_ceiling': 24000, 'evidence_tokens': 6000, 'agent_max_steps': 1})
            ledger = json.loads((Path(tmp) / 'investigation/ledger.jsonl').read_text())
        self.assertIn('unavailable', ledger['error'])
        self.assertFalse(result['evidence'])


if __name__ == '__main__': unittest.main()
