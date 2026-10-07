import io
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'method'))
import recover_finrate_sec_v1 as recovery


class SafetyTest(unittest.TestCase):
    def setUp(self):
        recovery.STOP.clear()
        recovery.STOP_REASON=None
        recovery.NEXT_START=0

    def tearDown(self):
        recovery.STOP.clear()
        recovery.STOP_REASON=None

    def test_access_denial_opens_circuit_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'http_errors').mkdir()
            row=dict(document_id='1_a',source_url='https://www.sec.gov/Archives/test.htm',
                     cik=1,accession='a',original_filename='test.htm')
            exc=HTTPError(row['source_url'],403,'Forbidden',{},io.BytesIO(b'undeclared automated tool'))
            with patch.object(recovery,'STORE',root),patch.object(recovery,'RESULT',root), \
                 patch.object(recovery,'UA','Research researcher@example.org'), \
                 patch.object(recovery.urllib.request,'urlopen',side_effect=exc) as request:
                result=recovery.download(row)
                self.assertEqual(request.call_count,1)
                self.assertTrue(recovery.STOP.is_set())
                self.assertEqual(result['acquisition_status'],'source_unavailable')
                self.assertEqual(result['attempts'][0]['status'],'403')
                self.assertFalse(list(root.rglob('*.part')))

    def test_disk_floor_prevents_network_request(self):
        class Disk:
            free=9*1024**3
        with patch.object(recovery.shutil,'disk_usage',return_value=Disk()):
            with self.assertRaises(RuntimeError):recovery.rate_limit()
        self.assertEqual(recovery.STOP_REASON,'free_disk_below_10_GiB')

    def test_shared_limiter_does_not_burst(self):
        starts=[]
        def invoke():
            recovery.rate_limit()
            starts.append(time.monotonic())
        with patch.object(recovery,'REQUESTS_PER_SECOND',20):
            threads=[threading.Thread(target=invoke) for _ in range(6)]
            for thread in threads:thread.start()
            for thread in threads:thread.join()
        starts.sort()
        self.assertEqual(len(starts),6)
        self.assertGreaterEqual(starts[-1]-starts[0],.24)
        self.assertTrue(all(b-a>=.045 for a,b in zip(starts,starts[1:])))


if __name__=='__main__':unittest.main()
