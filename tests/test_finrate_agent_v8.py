import json
from pathlib import Path
import tempfile
import unittest

from finrate_agent_v8 import prepare_common, predict
from finrate_accounting_v8 import combined_usage, summary, unique_inference_accounting
import test_finrate_agent_v7 as fixtures

POLICY = {'evidence_tokens': 6000, 'online_token_ceiling': 24000, 'agent_max_steps': 4}


class FakeModel:
    def __init__(self): self.calls = []
    def invoke(self, prompt, folder, **kwargs):
        name = Path(folder).name; self.calls.append(name)
        if name == 'plan':
            value = {'assets': ['AAA'], 'years': [2020, 2021], 'queries': ['legal proceedings'],
                     'annual_preferred': True, 'period_interpretation': 'requested years'}
        elif name == 'review': value = {'missing_evidence': 'none', 'queries': []}
        else: value = {'answer': 'fixture', 'claims': []}
        return json.dumps(value)


class SharedProcedureTests(unittest.TestCase):
    def test_three_conditions_share_two_planning_calls_and_exact_source_core(self):
        temporal, _, _ = fixtures.RetrievalTests().fixture(); model = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); shared = base/'shared'; results = []
            for condition, memory, history in [('raw', False, False), ('ours', True, True), ('flat', True, False)]:
                results.append(predict('fixture question', ['AAA'], temporal, model, base/condition, shared,
                                       POLICY, memory=memory, history=history))
            self.assertEqual(model.calls, ['plan', 'review', 'answer', 'answer', 'answer'])
            self.assertEqual(len({r['shared_plan_sha256'] for r in results}), 1)
            self.assertEqual(results[0]['shared_core_ids'], results[1]['shared_core_ids'])
            self.assertEqual(results[0]['shared_core_ids'], results[2]['shared_core_ids'])
            for r in results:
                self.assertEqual(r['model_calls'], 1); self.assertEqual(r['shared_planning_model_calls'], 2)
                self.assertEqual(r['standalone_model_calls'], 3)
                self.assertFalse(r['online_budget_exceeded'])

    def test_failed_shared_request_is_not_resampled_for_other_conditions(self):
        temporal, _, _ = fixtures.RetrievalTests().fixture()
        class Broken:
            calls = 0
            def invoke(self, *args, **kwargs): self.calls += 1; raise RuntimeError('fixture transport failure')
        model = Broken()
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(3):
                result = prepare_common('fixture', ['AAA'], temporal, model, tmp, POLICY)
                self.assertEqual(result['status'], 'failed')
            with self.assertRaisesRegex(RuntimeError, 'Shared planning failed'):
                predict('fixture', ['AAA'], temporal, model, Path(tmp)/'answer_condition', tmp, POLICY, memory=False, history=False)
            self.assertEqual(model.calls, 1)

    def test_changed_question_and_tampered_core_are_rejected_before_answer(self):
        temporal, _, _ = fixtures.RetrievalTests().fixture(); model = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            common = prepare_common('fixture', ['AAA'], temporal, model, tmp, POLICY)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                prepare_common('another question', ['AAA'], temporal, model, tmp, POLICY)
            common['core'][0]['text'] = 'not from the source'
            (Path(tmp)/'shared.json').write_text(json.dumps(common))
            with self.assertRaisesRegex(ValueError, 'different common source core'):
                predict('fixture', ['AAA'], temporal, model, Path(tmp)/'condition', tmp, POLICY, memory=True, history=True)
            self.assertEqual(model.calls, ['plan', 'review'])


class CostTests(unittest.TestCase):
    def test_shared_cost_counted_once_and_reused_baselines_not_new_calls(self):
        def usage(n, calls):
            return {'provider_usage': {'input_tokens': n, 'output_tokens': calls}, 'logged_response_calls': calls,
                    'requests_with_unknown_usage': 0, 'known_usage_requests_without_saved_response': 0}
        rows = []
        for c in ['raw_agent', 'agentic', 'history_disabled']:
            local, shared = usage(10, 1), usage(100, 2)
            rows.append({'condition': c, 'correct': True, 'key_point_fraction': 1, 'label': 'CORRECT',
                         'online_usage': local, 'shared_planning_usage': shared,
                         'standalone_online_usage': combined_usage(local, shared),
                         'shared_planning_path': '/fixture/shared/q', 'reused_baseline': False})
        rows.append({'condition': 'naive_llm', 'correct': None, 'key_point_fraction': None, 'label': 'UNASSESSED',
                     'online_usage': usage(8, 1), 'shared_planning_usage': combined_usage(),
                     'standalone_online_usage': usage(8, 1), 'shared_planning_path': None, 'reused_baseline': True})
        counted = unique_inference_accounting(rows)
        self.assertEqual(counted['new_physical_inference_usage']['provider_usage']['input_tokens'], 130)
        self.assertEqual(counted['new_physical_inference_usage']['logged_response_calls'], 5)
        self.assertEqual(counted['previously_executed_reused_baseline_usage']['provider_usage']['input_tokens'], 8)
        self.assertEqual(summary(rows)['agentic']['standalone_attributed_provider_usage']['input_tokens'], 110)
        self.assertEqual(summary(rows)['naive_llm']['unassessed'], 1)
        rows[1]['shared_planning_usage'] = usage(101, 2)
        with self.assertRaisesRegex(ValueError, 'Shared usage differs'): unique_inference_accounting(rows)


if __name__ == '__main__': unittest.main()
