import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finrate_final_evaluation import (OnceLuna, atomic_identity, complete_assessment_rows,
    grade_candidate, locked_inputs, prediction_grid, single_executor, summarize, infer, grade, CONDITIONS,
    online_usage)


class EvaluationIntegrityTests(unittest.TestCase):
    def test_rejected_response_usage_is_retained_without_recounting_saved_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, value in [('accepted', 100), ('rejected', 25), ('blinded_judge', 999)]:
                call = folder / name; call.mkdir(); (call / 'request.json').write_text('{}')
                usage = {'input_tokens': value, 'output_tokens': 10, 'cached_input_tokens': 5}
                (call / 'events.jsonl').write_text(json.dumps({'type': 'turn.completed', 'usage': usage})+'\n')
                if name == 'accepted': (call / 'response.json').write_text(json.dumps({'usage': [usage]}))
            unknown = folder / 'interrupted'; unknown.mkdir(); (unknown / 'request.json').write_text('{}')
            report = online_usage(folder)
        self.assertEqual(report['provider_usage']['input_tokens'], 125)
        self.assertEqual(report['provider_usage']['output_tokens'], 20)
        self.assertEqual(report['provider_usage']['cached_input_tokens'], 10)
        self.assertEqual(report['logged_response_calls'], 1)
        self.assertEqual(report['known_usage_requests_without_saved_response'], 1)
        self.assertEqual(report['requests_with_unknown_usage'], 1)

    def test_completed_event_without_usage_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder / 'request.json').write_text('{}')
            (folder / 'events.jsonl').write_text(json.dumps({'type': 'turn.completed'})+'\n')
            report = online_usage(folder)
        self.assertEqual(report['requests_with_unknown_usage'], 1)
        self.assertEqual(report['known_usage_requests_without_saved_response'], 0)
        self.assertEqual(report['provider_usage'], {})

    def test_no_release_prevents_reading_heldout_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base / 'cohort').mkdir()
            protected = base / 'cohort/evaluation_inputs.json'
            protected.write_text('must never be parsed')
            original = Path.read_text; touched = []
            def checked(path, *args, **kwargs):
                if path == protected: touched.append(path); raise AssertionError('Question read before release')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'read_text', checked):
                with self.assertRaisesRegex(ValueError, 'Freeze a reviewed method'): locked_inputs(base, 'v5')
            self.assertEqual(touched, [])

    def test_conflicting_release_prevents_question_read_even_with_valid_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base / 'cohort').mkdir()
            protected = base / 'cohort/evaluation_inputs.json'; protected.write_text('must not be read')
            atomic_identity(base / 'evaluation_lock.json', {'method_version': 'first', 'release_sha256': 'old'})
            with patch('finrate_final_evaluation.open_release', return_value=({'evaluation_ids': ['q']}, 'new')):
                with self.assertRaisesRegex(ValueError, 'different evaluation release'): locked_inputs(base, 'second')
            self.assertEqual(protected.read_text(), 'must not be read')

    def test_same_release_can_resume_but_only_one_live_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'release.json'; identity = {'release': 'fixed'}
            atomic_identity(lock, identity); atomic_identity(lock, identity)
            with self.assertRaises(ValueError): atomic_identity(lock, {'release': 'replacement'})
            with single_executor(Path(tmp) / 'process.lock'):
                with self.assertRaises(RuntimeError):
                    with single_executor(Path(tmp) / 'process.lock'): pass
            with single_executor(Path(tmp) / 'process.lock'): pass

    def test_interrupted_inference_is_not_regenerated_but_saved_response_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder / 'request.json').write_text('{}')
            with patch('finrate_poc.model.Luna.invoke', return_value='cached') as invoke:
                with self.assertRaisesRegex(RuntimeError, 'regeneration is prohibited'): OnceLuna().invoke('prompt', folder)
                invoke.assert_not_called()
                (folder / 'response.json').write_text('{}')
                self.assertEqual(OnceLuna().invoke('prompt', folder), 'cached')
                invoke.assert_called_once()

    def test_full_prediction_grid_retains_failures_and_refuses_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); row = base / 'q/naive_llm/prediction.json'; row.parent.mkdir(parents=True)
            row.write_text(json.dumps({'q_id': 'q', 'condition': 'naive_llm', 'status': 'failed', 'error': 'format'}))
            self.assertEqual(len(prediction_grid(base, ['q'], ['naive_llm'])), 1)
            with self.assertRaises(ValueError): prediction_grid(base, ['q'], ['naive_llm', 'hybrid'])
            complete_assessment_rows([{'q_id': 'q', 'condition': 'naive_llm'}], ['q'], ['naive_llm'])
            with self.assertRaises(ValueError):
                complete_assessment_rows([{'q_id': 'q', 'condition': 'naive_llm'}]*2, ['q'], ['naive_llm'])

    def test_judge_failure_is_unknown_instead_of_a_candidate_error(self):
        q = {'question': 'fixture', 'answer': 'reference', 'key_points': ['a']}
        candidate = {'status': 'completed', 'answer': {'answer': 'candidate', 'claims': []}, 'evidence': []}
        class FailedJudge:
            def invoke(self, *args, **kwargs): raise RuntimeError('transport stopped')
        unknown = grade_candidate(q, candidate, FailedJudge(), Path('unused'))
        failed = grade_candidate(q, {'status': 'failed', 'error': 'bad JSON'}, FailedJudge(), Path('unused'))
        self.assertIsNone(unknown['correct']); self.assertFalse(failed['correct'])
        rows = []
        for judgement in [unknown, failed, {'correct': True, 'key_point_fraction': 1.0, 'label': 'CORRECT'}]:
            rows.append({'condition': 'hybrid', **judgement, 'online_usage': {
                'provider_usage': {}, 'logged_response_calls': 0, 'requests_with_unknown_usage': 0}})
        report = summarize(rows)['hybrid']
        self.assertEqual(report['n'], 3); self.assertEqual(report['unassessed'], 1)
        self.assertEqual(report['generation_failures'], 1)
        self.assertAlmostEqual(report['correct_rate_lower'], 1/3)
        self.assertAlmostEqual(report['correct_rate_upper'], 2/3)

    def test_fixture_end_to_end_opens_gold_only_after_six_condition_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); cohort = base / 'cohort'; (cohort / 'gold').mkdir(parents=True)
            q = {'q_id': 'fixture_q', 'task': 'LT-QA', 'question': 'fixture question'}
            (cohort / 'evaluation_inputs.json').write_text(json.dumps([q]))
            gold = cohort / 'gold/evaluation.json'
            gold.write_text(json.dumps([{**q, 'answer': 'reference', 'key_points': ['a'], 'assets': ['A']}]))
            release = {'evaluation_ids': ['fixture_q'], 'assets': ['A'], 'banks': {'A': {'path': 'unused'}},
                       'online_code_version': 'v6_compact_provenance',
                       'conditions': CONDITIONS, 'policy': {'evidence_tokens': 6000, 'online_token_ceiling': 24000,
                       'agent_max_steps': 4}, 'cutoff_limitation': 'fixture', 'compute_limitation': 'fixture'}
            class FakeBank:
                db = None
                def close(self): pass
            candidate = {'answer': {'answer': 'candidate', 'claims': []}, 'evidence': []}
            source = {'_id': 's', 'title': 'A_2025-02-01_10-K', 'text': 'fixture source'}
            original = Path.read_text
            def protected(path, *args, **kwargs):
                if path == gold: raise AssertionError('Gold accessed during inference')
                return original(path, *args, **kwargs)
            with patch('finrate_final_evaluation.open_release', return_value=(release, 'fixed_hash')), \
                 patch('finrate_final_evaluation.documents', return_value=[source]), \
                 patch('finrate_final_evaluation.Index.build', return_value=None), \
                 patch('finrate_final_evaluation.Bank', side_effect=lambda *a, **k: FakeBank()), \
                 patch('finrate_final_evaluation.static_predict', return_value=candidate) as static, \
                 patch('finrate_final_evaluation.agent_predict', return_value=candidate) as agent:
                with patch.object(Path, 'read_text', protected):
                    infer(base, 'v5', base / 'status.json', {})
                self.assertEqual(static.call_count, 3); self.assertEqual(agent.call_count, 3)
                self.assertEqual([c.kwargs['memory'] for c in agent.call_args_list], [False, True, True])
                with patch('finrate_final_evaluation.grade_candidate', return_value={
                    'label': 'CORRECT', 'correct': True, 'key_point_fraction': 1.0}), \
                     patch('finrate_final_evaluation.assess', return_value={'state': 'fixture'}):
                    report = grade(base, 'v5', base / 'status.json', {})
                self.assertEqual(len(report['rows']), 6)
                self.assertEqual(set(report['summary']), set(CONDITIONS))

    def test_v7_uses_same_restricted_transport_and_explicit_edge_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'cohort').mkdir()
            (base/'cohort/evaluation_inputs.json').write_text(json.dumps([
                {'q_id': 'fixture', 'question': 'fixture question', 'task': 'LT-QA'}]))
            release = {'evaluation_ids': ['fixture'], 'assets': ['A'], 'banks': {},
                       'online_code_version': 'v7_period_procedure',
                       'policy': {'evidence_tokens': 6000, 'online_token_ceiling': 24000, 'agent_max_steps': 4}}
            candidate = {'answer': {'answer': 'fixture', 'claims': []}, 'evidence': []}
            model = object(); temporal = object()
            with patch('finrate_final_evaluation.open_release', return_value=(release, 'fixture_hash')), \
                 patch('finrate_final_evaluation.documents', return_value=[]), \
                 patch('finrate_final_evaluation.Index.build', return_value=None), \
                 patch('finrate_final_evaluation.TemporalRetriever.build', return_value=temporal), \
                 patch('finrate_final_evaluation.LunaV7', return_value=model), \
                 patch('finrate_final_evaluation.static_predict', return_value=candidate) as static, \
                 patch('finrate_final_evaluation.period_agent_predict', return_value=candidate) as agent:
                infer(base, 'v7', base/'status.json', {})
                self.assertEqual(static.call_count, 3); self.assertEqual(agent.call_count, 3)
                self.assertTrue(all(c.args[5] is model for c in static.call_args_list))
                self.assertTrue(all(c.args[2] is temporal and c.args[3] is model for c in agent.call_args_list))
                self.assertEqual([(c.kwargs['memory'], c.kwargs['history']) for c in agent.call_args_list],
                                 [(False, False), (True, True), (True, False)])
                self.assertTrue((base/'v7/evaluation/inference_complete.json').exists())


if __name__ == '__main__': unittest.main()
