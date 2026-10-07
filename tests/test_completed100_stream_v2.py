import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import finrate_completed100_stream_v2 as stream


class StreamingTests(unittest.TestCase):
    def test_every_task_has_one_owner_and_naive_is_excluded(self):
        qs = [{'q_id': 'q' + str(n)} for n in range(100)]
        all_tasks = [task for slot in range(7) for task in stream.slot_tasks(qs, slot)]
        self.assertEqual(len(all_tasks), 1200)
        self.assertEqual(len(set(all_tasks)), 1200)
        self.assertFalse(any(t[2] == 'naive_llm' for t in all_tasks))
        self.assertEqual(set(all_tasks), set(stream.tasks(qs)))

    def test_successful_event_without_response_is_not_safe_to_interrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / 'request.json').write_text('{}')
            self.assertFalse(stream.call_logs_saved(p))
            (p / 'events.jsonl').write_text('{"type":"turn.completed"}\n')
            self.assertFalse(stream.call_logs_saved(p))
            (p / 'response.json').write_text('{}')
            self.assertTrue(stream.call_logs_saved(p))
            (p / 'response.json').unlink()
            (p / 'events.jsonl').write_text('{"type":"turn.failed"}\n')
            self.assertTrue(stream.call_logs_saved(p))

    def test_real_handoff_waits_for_child_and_preserves_completed_response(self):
        script = '''
import json,pathlib,subprocess,sys,time
p=pathlib.Path(sys.argv[1])
(p/'request.json').write_text('{}')
(p/'status.json').write_text('{"state":"inference"}')
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.6)'])
(p/'ready').write_text('ready')
child.wait()
(p/'response.json').write_text('{"answer":"saved"}')
time.sleep(60)
'''
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            proc = subprocess.Popen([stream.original.PYTHON, '-c', script, temp], start_new_session=True)
            try:
                until = time.monotonic() + 5
                while not (p / 'ready').exists():
                    self.assertLess(time.monotonic(), until); time.sleep(.01)
                saved = stream.identity(proc.pid)
                state, _ = stream.try_checkpoint(saved, lambda _: p, p / 'status.json')
                self.assertEqual(state, 'busy')
                while not (p / 'response.json').exists():
                    self.assertLess(time.monotonic(), until); time.sleep(.01)
                state, _ = stream.try_checkpoint(saved, lambda _: p, p / 'status.json')
                self.assertEqual(state, 'paused')
                stream.stop_pid(saved)
                self.assertEqual(proc.wait(timeout=5), -signal.SIGTERM)
                self.assertEqual(json.loads((p / 'response.json').read_text()), {'answer': 'saved'})
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)

    def test_available_answer_is_graded_before_later_answer_arrives(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            qs = [{'q_id': 'q1', 'task': 'LT-QA', 'question': 'one'},
                  {'q_id': 'q2', 'task': 'EC-QA', 'question': 'two'}]
            gold = [{**q, 'answer': 'reference', 'key_points': ['point']} for q in qs]
            (base / 'cohort').mkdir(); (base / 'cohort/gold.json').write_text(json.dumps(gold))
            first = base / 'answers/q1/embedding/prediction.json'; first.parent.mkdir(parents=True)
            first.write_text(json.dumps({'status': 'completed', 'answer': {'answer': 'one'}, 'evidence': []}))
            second = base / 'answers/q2/embedding/prediction.json'
            calls = []

            class FakeDB:
                def execute(self, *args): return []
                def close(self): pass

            def grade(q, pred, model, folder):
                calls.append(q['q_id'])
                if q['q_id'] == 'q1':
                    self.assertFalse(second.exists())
                    second.parent.mkdir(parents=True)
                    second.write_text(first.read_text())
                return {'label': 'CORRECT', 'correct': True, 'key_point_fraction': 1.0}

            assigned = [('reference', 'q1', 'embedding'), ('reference', 'q2', 'embedding')]
            with patch.object(stream.original, 'inputs', return_value=qs), \
                 patch.object(stream.original, 'DATA', base), patch.object(stream.original, 'connect', return_value=FakeDB()), \
                 patch.object(stream.original, 'RecoveredLuna'), patch.object(stream.original, 'grade_one', side_effect=grade), \
                 patch.object(stream, 'slot_tasks', return_value=assigned):
                stream.judge(base, 0, base / 'status.json')
                stream.judge(base, 0, base / 'status.json')
            self.assertEqual(calls, ['q1', 'q2'])  # Existing grades are not sampled twice.


if __name__ == '__main__':
    unittest.main()
