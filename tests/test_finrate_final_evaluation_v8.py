import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import finrate_final_evaluation_v8 as evaluation


class EvaluationV8Tests(unittest.TestCase):
    def test_no_release_means_no_heldout_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'cohort').mkdir()
            (base/'cohort/evaluation_inputs.json').write_text('not JSON; must never be read')
            with self.assertRaisesRegex(ValueError, 'Freeze a reviewed method'):
                evaluation.locked_inputs(base, 'v8')

    def test_six_condition_grid_before_gold_and_one_shared_question_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'cohort/gold').mkdir(parents=True)
            q = {'q_id': 'fixture', 'question': 'fixture question', 'task': 'LT-QA'}
            (base/'cohort/evaluation_inputs.json').write_text(json.dumps([q]))
            gold = base/'cohort/gold/evaluation.json'; gold.write_text(json.dumps([
                {**q, 'answer': 'reference', 'key_points': ['point'], 'assets': ['A']}]))
            release = {'evaluation_ids': ['fixture'], 'assets': ['A'], 'banks': {},
                       'online_code_version': 'v8_shared_core', 'policy': {'evidence_tokens': 6000,
                       'online_token_ceiling': 24000, 'agent_max_steps': 4},
                       'cutoff_limitation': 'fixture', 'compute_limitation': 'fixture'}
            candidate = {'answer': {'answer': 'fixture', 'claims': []}, 'evidence': []}
            original_read = Path.read_text
            def forbid_gold(path, *args, **kwargs):
                if path == gold: raise AssertionError('Premature heldout gold read')
                return original_read(path, *args, **kwargs)
            with patch.object(evaluation, 'open_release', return_value=(release, 'fixture_release')), \
                 patch.object(evaluation, 'documents', return_value=[]), \
                 patch.object(evaluation.Index, 'build', return_value=None), \
                 patch.object(evaluation.TemporalRetriever, 'build', return_value=object()), \
                 patch.object(evaluation, 'static_predict', return_value=candidate) as static, \
                 patch.object(evaluation, 'shared_agent_predict', return_value=candidate) as agent:
                with patch.object(Path, 'read_text', forbid_gold):
                    evaluation.infer(base, 'v8', base/'status.json', {})
                self.assertEqual(static.call_count, 3); self.assertEqual(agent.call_count, 3)
                self.assertEqual({c.args[5] for c in agent.call_args_list}, {base/'v8/evaluation/shared_planning/fixture'})
                self.assertEqual([(c.kwargs['memory'], c.kwargs['history']) for c in agent.call_args_list],
                                 [(False, False), (True, True), (True, False)])
                with patch.object(evaluation, 'grade_candidate', return_value={
                    'label': 'CORRECT', 'correct': True, 'key_point_fraction': 1}), \
                     patch.object(evaluation, 'assess', return_value={'state': 'fixture'}):
                    result = evaluation.grade(base, 'v8', base/'status.json', {})
                self.assertEqual(len(result['rows']), 6)
                self.assertEqual(result['unique_inference_accounting']['shared_planning_unique_questions'], 1)


if __name__ == '__main__': unittest.main()
