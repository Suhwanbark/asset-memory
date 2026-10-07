"""Execute representation-independent frozen DEV arms while banks/index build.

The main pipeline later reuses the same immutable predictions, never resamples
them. No reference grading is started before the complete comparison grid.
"""
import json
import os
import time

from finrate_poc.runtime import check_disk,cpu_only,watchdog,write_json
from finrate_autonomous_v10.config import BASE
from finrate_autonomous_v10.evaluate import infer,verify_online


def main():
    cpu_only();check_disk();os.nice(10)
    status=BASE/'early_dev/status.json';stop=watchdog(status)
    try:
        while not (BASE/'development_reference_audit/report.json').exists():
            check_disk()
            audit=json.loads((BASE/'development_reference_audit/status.json').read_text())
            if audit['state']=='failed':raise RuntimeError('Inspect failed reference diagnostic first')
            write_json(status,{'state':'waiting_for_reference_diagnostic_slot','pid':os.getpid()});time.sleep(10)
        verify_online()
        for condition in ['naive_llm','raw_agent']:
            write_json(status,{'state':'inference','condition':condition,'pid':os.getpid()})
            target=BASE/'workers'/f'dev_{condition}/status.json'
            infer('dev',condition,target)
            write_json(target,{'state':'completed','split':'dev','condition':condition,'pid':os.getpid()})
        write_json(status,{'state':'completed','pid':os.getpid(),'finished':time.time(),
                           'conditions':['naive_llm','raw_agent'],'reference_grading_started':False})
    except Exception as exc:
        write_json(status,{'state':'failed','error':str(exc),'pid':os.getpid()});raise
    finally:stop.set()


if __name__=='__main__':main()
