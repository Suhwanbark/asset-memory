import json
from pathlib import Path
import tempfile
import unittest

from finrate_source_grading import assess, prepare


class GroundingInputTests(unittest.TestCase):
    def setUp(self):
        text = 'Amounts in thousands. March 31, 2020. Total assets were $12.'
        self.corpus = {'s': {'_id': 's', 'title': 'A_2020-04-01_10-Q', 'text': text}}
        start = text.index('Total')
        self.prediction = {'status': 'completed', 'condition': 'agentic', 'reference': 'SECRET GOLD',
            'answer': {'answer': 'Assets were $12 thousand.',
                       'claims': [{'text': 'Assets were $12 thousand.', 'evidence_ids': ['fact_hidden']}]},
            'evidence': [{'id': 'fact_hidden', 'source_id': 's', 'quote_start': start, 'quote_end': len(text),
                          'quote': text[start:], 'value': 'DERIVED MODEL CLAIM', 'period': '2099'}]}

    def test_judge_sees_original_context_not_gold_or_derived_memory_truth(self):
        payload, audit = prepare('What were assets?', self.prediction, self.corpus)
        encoded = json.dumps(payload)
        for forbidden in ['SECRET GOLD', 'DERIVED MODEL CLAIM', '2099', 'agentic', 'fact_hidden']:
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(payload['candidate']['claims'][0]['evidence_ids'], ['E001'])
        self.assertIn('Amounts in thousands.', encoded)
        self.assertFalse(audit['gold_fields_available'])

    def test_changed_quote_cannot_be_passed_off_as_original_source(self):
        self.prediction['evidence'][0]['quote'] = 'Total assets were $99.'
        with self.assertRaisesRegex(ValueError, 'exact source span'):
            prepare('Question?', self.prediction, self.corpus)

    def test_additional_context_obeys_predeclared_cap(self):
        payload, audit = prepare('Question?', self.prediction, self.corpus, evidence_cap=90)
        self.assertLessEqual(audit['source_text_tokens'], 90)
        self.assertEqual(len(payload['supplied_original_evidence']), 1)

    def test_no_evidence_is_not_a_claim_of_factual_error(self):
        class NeverCall:
            def invoke(self, *args, **kwargs): raise AssertionError('No source to judge')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'prediction.json'
            path.write_text(json.dumps({'status': 'completed', 'answer': {'answer': 'A fact.'}, 'evidence': []}))
            result = assess('Question?', path, {}, NeverCall(), Path(tmp) / 'audit')
        self.assertEqual(result['overall_grounding'], 'not_assessable')
        self.assertEqual(result['state'], 'no_external_evidence')


if __name__ == '__main__': unittest.main()
