"""Offline integrity checks for the direct, non-memory priority POC."""
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'method'))
from time_priority_resources import FLOOR, disk_check, gpu_check
from time_priority_infer import user_prompt, parse_answer, parse_terminal_answer

DATA = ROOT / 'data/time_priority_v1'

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((DATA / 'cases.json').read_text())['cases']
        cls.gold = {x['id']: x for x in json.loads((DATA / 'gold.json').read_text())['labels']}

    def test_frozen_hashes(self):
        manifest = json.loads((DATA / 'freeze.json').read_text())
        for name in ['cases', 'gold']:
            self.assertEqual(hashlib.sha256((DATA / (name + '.json')).read_bytes()).hexdigest(), manifest[name + '_sha256'])

    def test_sample_units_and_labels(self):
        self.assertEqual(len(self.cases), 50)
        self.assertEqual(len({c['id'] for c in self.cases}), 50)
        self.assertEqual(len({c['cluster'] for c in self.cases}), 16)
        self.assertEqual(set(self.gold), {c['id'] for c in self.cases})
        self.assertEqual(sum(not c['stratum'].endswith('_control') for c in self.cases), 38)
        for c in self.cases:
            self.assertIn(self.gold[c['id']]['winner'], [i['id'] for i in c['items']])
            self.assertTrue(self.gold[c['id']]['warrant'])

    def test_source_spans_hashes_and_cutoff(self):
        for c in self.cases:
            for item in c['items']:
                self.assertLessEqual(item['published'], c['as_of'])
                self.assertEqual(item['published'], max(e['published'] for e in item['evidence']))
                for e in item['evidence']:
                    folder = DATA / 'sources' / e['source_id']
                    if not folder.exists():
                        folder = ROOT / 'data/temporal_evidence_poc_v1/sources' / e['source_id']
                    meta = json.loads((folder / 'meta.json').read_text())
                    text = Path(meta['text_path']).read_text()
                    self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), e['source_text_sha256'])
                    self.assertEqual(text[slice(*e['span'])], e['quote'])
                    self.assertLessEqual(e['published'], c['as_of'])

    def test_clock_pairs(self):
        pairs = {}
        for c in self.cases:
            if c['change'].startswith('clock_only'):
                pairs.setdefault(c['pair_id'], []).append(c)
        self.assertEqual(len(pairs), 11)  # 3 switching and 8 persistent-effect pairs
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            a, b = pair
            self.assertNotEqual(a['as_of'], b['as_of'])
            self.assertEqual(a['question'], b['question'])
            self.assertEqual(a['items'], b['items'])
            eq = self.gold[a['id']]['winner'] == self.gold[b['id']]['winner']
            self.assertEqual(eq, a['change'] == 'clock_only_stable')

    def test_prompt_order_and_no_labels(self):
        for c in self.cases:
            p, a = user_prompt(c, False)
            q, b = user_prompt(c, True)
            self.assertEqual(a, b[::-1])
            self.assertNotEqual(p, q)
            self.assertNotIn(self.gold[c['id']]['warrant'], p)
            self.assertNotIn('gold', p.lower())
            self.assertNotIn(c['id'], p)
            for x in c['items']:
                self.assertIn(x['text'], p)
                self.assertIn(x['text'], q)

class ResourceTests(unittest.TestCase):
    def test_response_parsers(self):
        self.assertEqual(parse_answer('A\nA short explanation.', 'v1'), 'A')
        self.assertEqual(parse_answer('This is directly relevant.\nAnswer: B', 'v2'), 'B')
        self.assertEqual(parse_answer('One sentence.\n**Answer: A**', 'v2'), 'A')
        self.assertIsNone(parse_answer('This mentions A but gives no final label.', 'v2'))
        self.assertEqual(parse_terminal_answer('One sentence. Answer: B'), 'B')
        self.assertEqual(parse_terminal_answer('One sentence.\nAnswer: B'), 'B')
        self.assertIsNone(parse_terminal_answer('Item B matters.'))

    def test_disk_floor(self):
        with patch('time_priority_resources.shutil.disk_usage', return_value=SimpleNamespace(free=FLOOR - 1)):
            with self.assertRaises(RuntimeError):
                disk_check()
        with patch('time_priority_resources.shutil.disk_usage', return_value=SimpleNamespace(free=FLOOR)):
            self.assertEqual(disk_check(), FLOOR)

    def test_gpu_busy_refused(self):
        for row, apps in [('GPU-test, 800, 0', ''), ('GPU-test, 1, 15', ''), ('GPU-test, 1, 0', 'GPU-test, 123')]:
            with patch('time_priority_resources.subprocess.check_output', side_effect=[row, apps]):
                with self.assertRaises(RuntimeError):
                    gpu_check('GPU-test')

    def test_gpu_idle_allowed(self):
        with patch('time_priority_resources.subprocess.check_output', side_effect=['GPU-test, 1, 0', '']):
            self.assertEqual(gpu_check('GPU-test'), 'GPU-test, 1, 0')

class CompletedResultTests(unittest.TestCase):
    def test_result_coverage_and_hashes(self):
        from time_priority_infer import parse_answer, parse_terminal_answer
        frozen = json.loads((DATA / 'freeze.json').read_text())
        for version in ['v1', 'v2', 'v3']:
            output = ROOT / 'result/time_priority_v1'
            if version != 'v1':
                output /= version
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(len(report['models']), 2)
            for model in report['models'].values():
                self.assertEqual(model['completed_responses'], 100)
                self.assertEqual(model['completed_questions_both_orders'], 50)
                self.assertEqual(model['scores']['main']['n'], 76)
                self.assertEqual(model['scores']['controls']['n'], 24)
                self.assertEqual(model['scores']['all']['correct'], sum(r['correct'] for r in model['rows']))
                for row in model['rows']:
                    path = Path(row['path'])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['sha256'])
                    raw = json.loads(path.read_text())
                    self.assertEqual(raw['dataset_sha256'], frozen['cases_sha256'])
                    self.assertLessEqual(raw['parameter_count'], 3_000_000_000)
                    self.assertTrue(all(math.isfinite(p) for p in raw['first_token_AB_probabilities']))
                    self.assertAlmostEqual(sum(raw['first_token_AB_probabilities']), 1, places=5)
                    self.assertEqual(parse_answer(raw['response'], version), raw['answer'])
                    expected = raw['answer'] or (parse_terminal_answer(raw['response']) if version == 'v2' else None)
                    self.assertEqual(expected, row['answer'])

    def test_sequential_single_gpu_resource_log(self):
        records = [json.loads(l) for l in (ROOT / 'result/time_priority_v1/resource_events.jsonl').read_text().splitlines()]
        starts = [r for r in records if r['event'] == 'start' and r.get('gpu')]
        self.assertEqual(len(starts), 6)
        self.assertEqual(len({r['gpu'] for r in starts}), 1)
        last_end = 0
        for start in starts:
            self.assertGreater(start['time'], last_end)
            self.assertGreaterEqual(start['free'], FLOOR)
            exit_record = next(r for r in records if r.get('pid') == start['pid'] and r['event'] == 'exit')
            self.assertEqual(exit_record['code'], 0)
            self.assertGreaterEqual(exit_record['free'], FLOOR)
            last_end = exit_record['time']

    def test_control_and_clock_outcomes(self):
        report = json.loads((ROOT / 'result/time_priority_v1/v3/report.json').read_text())
        for model in report['models'].values():
            self.assertEqual(model['scores']['reading_control']['correct'], 8)
            self.assertEqual(model['paired_scores']['clock_only_switch']['n'], 3)
            self.assertEqual(model['paired_scores']['clock_only_switch']['correct'], 0)
            self.assertEqual(model['invalid_responses'], 0)

if __name__ == '__main__':
    unittest.main()
