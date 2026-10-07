import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import finrate_completed100_parallel7_v1 as scheduler


class HandoffTests(unittest.TestCase):
    def test_recycled_pid_and_zombie_are_not_adopted(self):
        saved = scheduler.identity(os.getpid())
        self.assertTrue(scheduler.same_process(saved, saved))
        self.assertFalse(scheduler.same_process(saved, {**saved, 'start_ticks': saved['start_ticks'] + 1}))
        self.assertFalse(scheduler.same_process(saved, {**saved, 'state': 'Z'}))

    def test_real_parent_handoff_preserves_running_child_and_request(self):
        # An isolated dummy scheduler exercises actual Linux process semantics;
        # it does not invoke a model or touch any experiment worker.
        code = '''
import json, os, pathlib, subprocess, sys, time
b = pathlib.Path(sys.argv[1])
argv = [sys.executable, '-c', 'import time; time.sleep(60)']
child = subprocess.Popen(argv)
f = b / 'workers' / 'naive_llm'
f.mkdir(parents=True)
(f / 'launch.json').write_text(json.dumps({'pid': child.pid, 'pgid': os.getpgrp(), 'argv': argv}))
(f / 'status.json').write_text(json.dumps({'state': 'inference'}))
(b / 'status.json').write_text(json.dumps({'state': 'running'}))
(b / 'launch.json').write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp(), 'argv': pathlib.Path('/proc/self/cmdline').read_bytes().decode().strip('\\0').split('\\0')}))
time.sleep(60)
'''
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'manifest.json').write_text('{}')
            (base / 'retrieval_manifest.json').write_text('{}')
            request = base / 'in_flight_request.json'
            request.write_text('{"must_remain": true}')
            before = request.read_bytes()
            parent = subprocess.Popen([scheduler.experiment.PYTHON, '-c', code, temp], start_new_session=True)
            try:
                deadline = time.monotonic() + 10
                while not (base / 'launch.json').exists():
                    if parent.poll() is not None or time.monotonic() > deadline:
                        self.fail('Dummy scheduler did not start')
                    time.sleep(.05)
                child = json.loads((base / 'workers/naive_llm/launch.json').read_text())
                with patch.object(scheduler, 'BASE', base), patch.object(scheduler, 'RUN', base / 'parallel7'), \
                     patch.object(scheduler.experiment, 'verify'), \
                     patch.object(scheduler, 'expected_argv', return_value=child['argv']):
                    adopted, completed = scheduler.handoff()
                self.assertEqual(list(adopted), ['naive_llm'])
                self.assertEqual(completed, [])
                self.assertTrue(scheduler.alive(adopted['naive_llm']))
                self.assertEqual(parent.wait(timeout=5), -signal.SIGTERM)
                self.assertEqual(request.read_bytes(), before)
                self.assertEqual(len(list((base / 'workers').glob('*/launch.json'))), 1)
            finally:
                try:
                    os.killpg(parent.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                parent.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
