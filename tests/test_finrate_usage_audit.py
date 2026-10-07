import json
from pathlib import Path
import shutil
import tempfile
import unittest
from finrate_usage_audit import collect


class UsageTests(unittest.TestCase):
    def fixture(self, folder):
        folder.mkdir(parents=True)
        events = [{'type': 'thread.started', 'thread_id': 'unique-transport'},
                  {'type': 'turn.completed', 'usage': {'input_tokens': 100, 'cached_input_tokens': 40, 'output_tokens': 20}}]
        (folder / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
        (folder / 'request.json').write_text(json.dumps({'model': 'gpt-5.6-luna', 'prompt_sha256': 'fixed'}))

    def test_response_cache_copies_do_not_multiply_actual_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); self.fixture(base / 'v1/dev/q/embedding/answer')
            shutil.copytree(base / 'v1', base / 'v2'); shutil.copytree(base / 'v1', base / 'v3')
            r = collect(base)
        self.assertEqual(r['unique_logged_transport_invocations'], 1)
        self.assertEqual(r['response_trace_copies_ignored'], 2)
        self.assertEqual(r['total_provider_usage']['input_tokens'], 100)
        self.assertEqual(r['total_provider_usage']['cached_input_tokens'], 40)

    def test_unfinished_requests_are_explicit_and_eval_traces_are_not_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); pending = base / 'pending'; pending.mkdir()
            (pending / 'request.json').write_text('{}')
            evaluation = base / 'v3/evaluation/q/answer'; evaluation.mkdir(parents=True)
            (evaluation / 'events.jsonl').write_text('not valid json; should not be read')
            r = collect(base)
        self.assertEqual(r['request_only_pending_or_interrupted'], ['pending/request.json'])
        self.assertFalse(r['event_parse_warnings'])
        self.assertFalse(r['evaluation_responses_read'])


if __name__ == '__main__': unittest.main()
