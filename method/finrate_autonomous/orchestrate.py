"""Supervise the authorized background stages without silently accepting failures."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time

from finrate_poc.runtime import ROOT,check_disk,cpu_only,digest,file_digest,watchdog,write_json
from .config import BASE,POLICY,SEED
from .export import export_all

PYTHON='/home/hob/anaconda3/envs/su/bin/python'


def launch(label,module,args):
    root=BASE/'workers'/label;root.mkdir(parents=True,exist_ok=True)
    old=root/'launch.json'
    if old.exists():raise RuntimeError('A recorded worker exists; inspect before relaunch: '+label)
    argv=[PYTHON,'-u','-m',module,*args]
    with (root/'stdout.log').open('a') as log:
        # Inherit this supervisor's dedicated process group: the watchdog stops
        # the supervisor, all its workers and model subprocesses together.
        child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
          env={**os.environ,'PYTHONPATH':'method','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    record={'pid':child.pid,'pgid':os.getpgrp(),'argv':argv,'started':time.time(),'label':label,
      'cpu_only':True,'min_free_disk_bytes':POLICY['min_free_disk_bytes']}
    write_json(old,record,immutable=True);return child,record


def supervise(jobs,status,*,concurrency=2):
    queue=deque(jobs);active=[];completed=[];failed=[]
    while queue or active:
        check_disk()
        while queue and len(active)<concurrency and not failed:
            label,module,args=queue.popleft();p,record=launch(label,module,args);active.append((p,record))
        remaining=[]
        for p,record in active:
            code=p.poll()
            if code is None:remaining.append((p,record));continue
            item={**record,'returncode':code,'finished':time.time()}
            (completed if code==0 else failed).append(item)
            write_json(BASE/'workers'/record['label']/'exit.json',item,immutable=True)
        active=remaining
        write_json(status,{'state':'needs_review' if failed else 'running','pid':os.getpid(),
          'active':[r for _,r in active],'completed':completed,'failed':failed,'queued':[j[0] for j in queue]})
        if failed and not active:raise RuntimeError('A worker failed; unstarted jobs retained for review')
        if queue or active:time.sleep(5)
    return completed


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['banks','qa','grade','pipeline']);a=p.parse_args()
    cpu_only();check_disk();os.nice(5)
    status=BASE/'supervisors'/a.stage/'status.json';stop=watchdog(status)
    try:
        if a.stage in {'banks','pipeline'}:
            smoke=BASE/'bank_smoke_exactanchors_v1/NRT'
            if json.loads((smoke/'completed.json').read_text())['state']!='completed':raise ValueError('Complete real writer/update smoke first')
            if not json.loads((smoke/'audit/structural.json').read_text())['passed']:raise ValueError('Smoke structural audit failed')
            assets=json.loads((BASE/'corpus_manifest.json').read_text())['assets']
            assets=sorted(assets,key=lambda s:digest(SEED+'build'+s))
            jobs=[('bank_'+asset,'finrate_autonomous.bank',['full','--asset',asset]) for asset in assets]
            supervise(jobs,status,concurrency=2)
            export_all()
            write_json(status,{'state':'banks_and_reader_exports_completed','pid':os.getpid(),'finished':time.time()})
        if a.stage in {'qa','pipeline'}:
            # Intentionally bounded at development. A separate reviewed release
            # is required before heldout inference; no automatic best-score loop.
            jobs=[('dev_'+c,'finrate_autonomous.evaluate',['infer','--condition',c]) for c in POLICY['conditions']]
            while a.stage=='pipeline' and not (BASE/'index/manifest.json').exists():
                check_disk()
                index_status=json.loads((BASE/'workers/index/status.json').read_text())
                if index_status['state'] in {'failed','stopped_low_disk'}:raise RuntimeError('CPU index worker stopped')
                write_json(status,{'state':'waiting_for_cpu_index','pid':os.getpid(),'bank_exports_ready':True});time.sleep(10)
            if not (BASE/'index/manifest.json').exists() or not (BASE/'reader_banks/manifest.json').exists():
                raise ValueError('Complete all43-company representations before full comparison')
            supervise(jobs,status,concurrency=2)
            write_json(status,{'state':'development_predictions_completed','pid':os.getpid(),'finished':time.time()})
        if a.stage in {'grade','pipeline'}:
            supervise([('dev_grade','finrate_autonomous.evaluate',['grade'])],status,concurrency=1)
            supervise([('dev_ground','finrate_autonomous.evaluate',['ground'])],status,concurrency=1)
            from .diagnostics import collect
            collect('dev')
            write_json(status,{'state':'development_ready_for_review','pid':os.getpid(),'finished':time.time()})
    except Exception as e:
        prior=json.loads(status.read_text()) if status.exists() else {}
        write_json(status,{**prior,'state':'needs_review','error':str(e),'pid':os.getpid()});raise
    finally:stop.set()

if __name__=='__main__':main()
