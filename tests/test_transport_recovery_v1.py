import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finrate_autonomous_v10.transport import IsolatedLuna
from finrate_poc.data import MODEL
from finrate_poc.runtime import digest, file_digest
from finrate_transport_recovery_v1 import NOTICE, RecoveredLuna, recover

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        request = {"model": MODEL, "reasoning": "low", "prompt_sha256": digest("p"),
                   "response_schema_sha256": digest(json.dumps(SCHEMA, sort_keys=True)),
                   "transport": "codex_exec_ephemeral_v9_isolated_policy"}
        (self.root / "request.json").write_text(json.dumps(request))
        self.events = [
            {"type": "item.completed", "item": {"type": "error", "message": NOTICE + " 403"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}},
            {"type": "turn.completed", "usage": {"input_tokens": 15, "output_tokens": 4}},
        ]

    def save(self):
        (self.root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in self.events))

    def test_original_response_recovered_without_model_call_or_trace_changes(self):
        self.save()
        hashes = {p.name: file_digest(p) for p in self.root.iterdir()}
        with patch('finrate_autonomous_v10.transport.command', side_effect=AssertionError('No new call')):
            model = RecoveredLuna()
            self.assertEqual(model.invoke("p", self.root, schema=SCHEMA), '{"ok":true}')
            self.assertEqual(model.invoke("p", self.root, schema=SCHEMA), '{"ok":true}')
            self.assertEqual(model.calls, 0)
        for name, sha in hashes.items():
            self.assertEqual(file_digest(self.root / name), sha)
        result = json.loads((self.root / "response.json").read_text())
        self.assertEqual(result['usage'][0]['input_tokens'], 15)
        self.assertEqual(result['recovery']['new_model_calls'], 0)

    def test_native_tool_even_without_completed_event_rejected(self):
        self.events.insert(0, {"type": "item.started", "item": {"type": "command_execution"}})
        self.save()
        with self.assertRaisesRegex(ValueError, 'Native tool'):
            recover(self.root, "p", SCHEMA)
        self.assertFalse((self.root / 'response.json').exists())

    def test_terminal_failure_rejected_even_with_final_text(self):
        self.events.append({"type": "turn.failed", "error": {"message": "at capacity"}})
        self.save()
        with self.assertRaisesRegex(ValueError, 'Terminal'):
            recover(self.root, "p", SCHEMA)

    def test_changed_prompt_or_schema_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'identity'):
            recover(self.root, "changed", SCHEMA)
        with self.assertRaisesRegex(ValueError, 'identity'):
            recover(self.root, "p", {"type": "string"})

    def test_malformed_final_message_rejected(self):
        self.events[1]['item']['text'] = '{"ok":"not a boolean"}'
        self.save()
        with self.assertRaises(Exception):
            recover(self.root, "p", SCHEMA)
        self.assertFalse((self.root / 'response.json').exists())

    def test_multiple_answers_rejected(self):
        self.events.insert(2, self.events[1])
        self.save()
        with self.assertRaisesRegex(ValueError, 'one original'):
            recover(self.root, "p", SCHEMA)

    def test_fresh_call_fallback_uses_original_answer(self):
        self.save()
        with patch.object(IsolatedLuna, 'invoke', side_effect=RuntimeError('Unexpected tool use: response rejected and preserved')) as delegate:
            self.assertEqual(RecoveredLuna().invoke('p', self.root, schema=SCHEMA), '{"ok":true}')
            self.assertEqual(delegate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
