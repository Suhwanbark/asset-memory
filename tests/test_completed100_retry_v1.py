import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import finrate_completed100_retry_v1 as retry
from finrate_poc.runtime import file_digest


class RetryTests(unittest.TestCase):
    def test_selection_uses_failure_not_answer_quality(self):
        for label in ['CORRECT', 'PARTIAL', 'INCORRECT', 'FAILURE']:
            flags = retry.retry_flags({'status': 'completed'}, {'label': label}, {'state': 'assessed'})
            self.assertEqual(flags, {'generation': False, 'reference': False, 'grounding': False})
        flags = retry.retry_flags({'status': 'failed'}, {'label': 'FAILURE'}, {'state': 'generation_failed'})
        self.assertTrue(all(flags.values()))

    def test_judge_errors_do_not_regenerate_answer(self):
        flags = retry.retry_flags({'status': 'completed'}, {'label': 'UNASSESSED'}, {'state': 'assessment_failed'})
        self.assertEqual(flags, {'generation': False, 'reference': True, 'grounding': True})

    def test_waits_for_every_initial_worker_and_completed_report_stage(self):
        workers = {c: {'state': 'completed'} for c in retry.CONDITIONS}
        self.assertTrue(retry.primary_ready({'state': 'completed'}, workers, []))
        self.assertFalse(retry.primary_ready({'state': 'completed'}, workers, ['raw_agent']))
        self.assertFalse(retry.primary_ready({'state': 'running'}, workers, []))
        self.assertFalse(retry.primary_ready({'state': 'completed'}, {}, []))
        workers['bank_agent'] = {'state': 'inference'}
        self.assertFalse(retry.primary_ready({'state': 'completed'}, workers, []))

    def test_frozen_functions_reuse_wrong_answer_and_retry_only_failed_answer(self):
        def save(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj))

        calls = []

        class FakeModel:
            def __init__(self, **kwargs):
                pass

            def invoke(self, prompt, folder, *, schema=None):
                calls.append(prompt)
                if schema is None:
                    return json.dumps({'answer': 'Retry response', 'claims': []})
                return json.dumps({'label': 'CORRECT', 'key_points_met': [True],
                                   'citation_support': 'no_citations', 'temporal_error': False,
                                   'entity_error': False, 'reason': 'test'})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); old = root / 'original'; new = root / 'retry'; data = root / 'data'
            qs = [{'q_id': 'q1', 'task': 'LT-QA', 'question': 'Existing wrong answer'},
                  {'q_id': 'q2', 'task': 'EC-QA', 'question': 'Previously failed generation'}]
            gold = [{**q, 'answer': 'REFERENCE_ONLY', 'key_points': ['GOLD_ONLY']} for q in qs]
            save(data / 'cohort/gold.json', gold)
            save(data / 'corpus_manifest.json', {'sources': 1762})
            save(new / 'retrieval_manifest.json', {'files': {}})
            hashes = {}
            for n, q in enumerate(qs):
                source = old / 'answers' / q['q_id'] / 'naive_llm'
                target = new / 'answers' / q['q_id'] / 'naive_llm'
                g_old = old / 'source_grounding' / q['q_id'] / 'naive_llm'
                g_new = new / 'source_grounding' / q['q_id'] / 'naive_llm'
                pred = {'status': 'completed', 'answer': {'answer': 'Wrong', 'claims': []}, 'evidence': []} if n == 0 else {'status': 'failed', 'error': 'capacity'}
                save(source / 'prediction.json', pred)
                grade = {'q_id': q['q_id'], 'task': q['task'], 'condition': 'naive_llm',
                         'prediction_sha256': file_digest(source / 'prediction.json'),
                         'generation_status': pred['status'], 'label': 'INCORRECT' if n == 0 else 'FAILURE',
                         'correct': False, 'key_point_fraction': 0.0}
                grounding = {'state': 'no_external_evidence' if n == 0 else 'generation_failed'}
                save(source / 'assessment.json', grade)
                save(g_old / 'assessment.json', grounding)
                retry.seed_case(source, target, g_old, g_new, retry.retry_flags(pred, grade, grounding))
            hashes = {p: file_digest(p) for p in old.rglob('*.json')}
            with patch.object(retry.experiment, 'BASE', new), patch.object(retry.experiment, 'DATA', data), \
                 patch.object(retry.experiment, 'verify'), patch.object(retry.experiment, 'inputs', return_value=qs), \
                 patch.object(retry.experiment, 'RecoveredLuna', FakeModel):
                retry.experiment.infer('naive_llm', new / 'status.json')
                retry.experiment.grade('naive_llm', new / 'status.json')
            self.assertEqual(len(calls), 2)  # One generation and one reference judgment.
            self.assertIn('Previously failed generation', calls[0])
            self.assertNotIn('REFERENCE_ONLY', calls[0])
            self.assertNotIn('GOLD_ONLY', calls[0])
            self.assertIn('REFERENCE_ONLY', calls[1])
            self.assertEqual({p: file_digest(p) for p in hashes}, hashes)
            self.assertTrue((new / 'answers/q1/naive_llm/prediction.json').is_symlink())
            self.assertEqual(retry.read(new / 'answers/q1/naive_llm/assessment.json')['label'], 'INCORRECT')
            self.assertFalse((new / 'answers/q2/naive_llm/prediction.json').is_symlink())
            report = retry.read(new / 'reports/naive_llm.json')
            self.assertEqual(report['summary']['all']['correct'], 1)
            self.assertEqual(report['summary']['all']['n'], 2)


if __name__ == '__main__':
    unittest.main()
