"""Exercise low-disk termination in a disposable process group, not real disk."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


class WatchdogProcessGroupTests(unittest.TestCase):
    def test_low_disk_terminates_only_own_group_including_model_child(self):
        code = '''
import json, os, subprocess, sys, time
from pathlib import Path
from finrate_poc import runtime
folder = Path(sys.argv[1])
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
(folder/'identity.json').write_text(json.dumps({'worker': os.getpid(), 'group': os.getpgrp(), 'child': child.pid}))
def synthetic_low_disk(): raise runtime.DiskLow('synthetic guard test; no disk filled')
runtime.check_disk = synthetic_low_disk
runtime.watchdog(folder/'status.json', interval=0.02)
time.sleep(30)
'''
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
            worker = subprocess.Popen([sys.executable, '-c', code, tmp], start_new_session=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                _, stderr = worker.communicate(timeout=5)
                self.assertEqual(worker.returncode, -signal.SIGTERM, stderr)
                identity = json.loads((Path(tmp)/'identity.json').read_text())
                self.assertEqual(identity['worker'], identity['group'])
                status = json.loads((Path(tmp)/'status.json').read_text())
                self.assertEqual(status['state'], 'stopped_low_disk')
                self.assertEqual(status['pid'], worker.pid)
                self.assertIsNone(sentinel.poll(), 'Unrelated process group must stay alive')
                state = None
                for _ in range(20):
                    path = Path('/proc')/str(identity['child'])/'stat'
                    try: state = path.read_text().rsplit(')', 1)[1].split()[0]
                    except FileNotFoundError: state = None
                    if state in {None, 'Z', 'X'}: break
                    time.sleep(.02)
                self.assertIn(state, {None, 'Z', 'X'}, 'Model child must not remain running')
            finally:
                if worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGKILL); worker.wait(timeout=5)
                sentinel.terminate(); sentinel.wait(timeout=5)


if __name__ == '__main__': unittest.main()
