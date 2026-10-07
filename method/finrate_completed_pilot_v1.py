"""User-requested interim comparison on the companies complete at request time.

The full43 experiment and its heldout remain unchanged. All pilot arms receive
the same two-company corpus. Questions are fixed before any pilot predictions.
"""
from collections import Counter,defaultdict,deque
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

from finrate_poc.runtime import ROOT,check_disk,cpu_only,digest,file_digest,watchdog,write_json
from finrate_poc.model import answer_prompt,fit_evidence,parse_json,token_count
from finrate_poc.retrieval import CpuEmbedding,EMBEDDING_REVISION
from finrate_final_evaluation import grade_candidate,online_usage,single_executor
from finrate_source_grading import assess as assess_source
from finrate_autonomous_v10.config import BASE as MAIN,CORPUS as MAIN_CORPUS,POLICY
from finrate_autonomous_v10.evaluate import verify_online
from finrate_autonomous_v10.agent import run
from finrate_autonomous_v10.workspace import Notebook,Workspace,connect,ANCHOR
from finrate_autonomous_v10.retrieval import FullIndex
from finrate_autonomous_v10.export import reader_records
import finrate_autonomous_v10.export as exporter

BASE=ROOT/'result/finrate_completed_pilot_v1'
ASSETS=['HUSA','REEMF']
SEED='finrate-completed-assets-20261005-v1'
PYTHON='/home/hob/anaconda3/envs/su/bin/python'
CONDITIONS=['naive_llm','raw_agent','bank_agent','embedding','hybrid','chunk_agent','flat_bank_agent']


def prepare():
    verify_online();check_disk();BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'manifest.json').exists():return
    completed={a:json.loads((MAIN/'banks'/a/'completed.json').read_text()) for a in ASSETS}
    for a,r in completed.items():
        if r['state'] not in {'completed','completed_with_budget_limits'} or not r['full_asset_history']:
            raise ValueError('Pilot company was not complete: '+a)
        if r['bank_sha256']!=file_digest(MAIN/'banks'/a/'bank.sqlite'):raise ValueError('Completed bank changed')
    source=connect(MAIN_CORPUS,True)
    source_assets=dict(source.execute('SELECT id,asset FROM sources'))
    # Reconstruct source-connected components from metadata only; keep every
    # primary heldout component excluded, not merely the40 chosen question IDs.
    qs=[]
    for task in POLICY['tasks']:
        qs.extend({**q,'task':task} for q in json.loads((ROOT/'data/finrate_public_v1/qa'/f'{task}.json').read_text()))
    parent={q['q_id']:q['q_id'] for q in qs};owners={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for q in qs:
        for sid in q['doc_ids']:
            if sid in owners:parent[find(q['q_id'])]=find(owners[sid])
            owners[sid]=q['q_id']
    groups=defaultdict(list)
    for q in qs:groups[find(q['q_id'])].append(q['q_id'])
    group_id={qid:min(ids) for ids in groups.values() for qid in ids}
    primary=json.loads((MAIN/'cohort/manifest.json').read_text())
    splits={g['id']:g['split'] for g in primary['groups']}
    if set(splits)!={min(ids) for ids in groups.values()}:raise ValueError('Component reconstruction differs')
    excluded=set(primary['excluded_prior_ids'])|set(sum(primary['selected_ids'].values(),[]))
    pools={task:[] for task in POLICY['tasks']}
    for q in qs:
        assets=sorted({source_assets[s] for s in q['doc_ids']})
        if set(assets)<=set(ASSETS) and splits[group_id[q['q_id']]]=='dev' and q['q_id'] not in excluded:
            pools[q['task']].append({**q,'assets':assets,'source_ids':q['doc_ids']})
    selected=[]
    for task,pool in pools.items():
        if len(pool)<10:raise ValueError('Insufficient predeclared10-per-task pilot pool; inspect before changing selection')
        selected.extend(sorted(pool,key=lambda q:digest(SEED+'question'+q['q_id']))[:10])
    selected.sort(key=lambda q:digest(SEED+'order'+q['q_id']))
    write_json(BASE/'cohort/inputs.json',[{k:q[k] for k in ['q_id','question','task']} for q in selected],immutable=True)
    write_json(BASE/'cohort/gold.json',selected,immutable=True)
    write_json(BASE/'cohort/manifest.json',{'seed':SEED,'assets':ASSETS,
      'pool_counts':{k:len(v) for k,v in pools.items()},'counts':dict(Counter(q['task'] for q in selected)),
      'ids':[q['q_id'] for q in selected],'rule':'Hash sample10LT+10EC from complete-company queries in primary DEV components; exclude all old/existing selected IDs.',
      'score_or_question_text_selection':False,'primary_heldout_components_excluded':True,
      'primary_fixed_dev20_eligible_at_request_time':0},immutable=True)
    target=BASE/'corpus.sqlite'
    if target.exists():raise ValueError('Incomplete pilot preparation exists; inspect before retry')
    dst=sqlite3.connect(target)
    dst.executescript('CREATE TABLE sources(id TEXT PRIMARY KEY,title TEXT,asset TEXT,published TEXT,quarter TEXT,text TEXT,sha256 TEXT); CREATE INDEX source_asset_date ON sources(asset,published,id); CREATE VIRTUAL TABLE source_fts USING fts5(id UNINDEXED,title,text,tokenize="unicode61");')
    rows=source.execute('SELECT * FROM sources WHERE asset IN (?,?) ORDER BY id',ASSETS).fetchall()
    with dst:
        dst.executemany('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',[tuple(r) for r in rows])
        dst.executemany('INSERT INTO source_fts VALUES(?,?,?)',[(r['id'],r['title'],r['text']) for r in rows])
    dst.close();source.close()
    write_json(BASE/'corpus_manifest.json',{'assets':ASSETS,'sources':len(rows),'sqlite_sha256':file_digest(target)},immutable=True)
    for a in ASSETS:
        path=BASE/'banks'/a;path.parent.mkdir(exist_ok=True);path.symlink_to(MAIN/'banks'/a,target_is_directory=True)
        book=Notebook(path/'bank.sqlite',readonly=True)
        if any(source_assets[m.group(1)] not in ASSETS for r in reader_records(book) for m in ANCHOR.finditer(r['body'])):
            raise ValueError('A pilot bank cites sources outside the shared pilot corpus')
        book.close()
    old_base=exporter.BASE
    try:exporter.BASE=BASE;exporter.export_all()
    finally:exporter.BASE=old_base
    record={'assets':ASSETS,'source_count':len(rows),'conditions':CONDITIONS,'policy':POLICY,
      'pilot_code_sha256':file_digest(__file__),'primary_online_manifest_sha256':file_digest(MAIN/'online_manifest.json'),
      'cohort_manifest_sha256':file_digest(BASE/'cohort/manifest.json'),'bank_sha256':{a:r['bank_sha256'] for a,r in completed.items()},
      'scope':'Separate availability-based two-company interim diagnostic. All representations use the same two-company sources, not a full43-corpus score.',
      'original_full43_cohort_and_results_unchanged':True,'heldout_used':False,'no_resampling_failed_calls':True}
    write_json(BASE/'manifest.json',record,immutable=True)
    print('PILOT_READY',ASSETS,len(rows),len(selected),flush=True)


def verify():
    verify_online();m=json.loads((BASE/'manifest.json').read_text())
    if file_digest(__file__)!=m['pilot_code_sha256']:raise ValueError('Frozen pilot code changed')


def index():
    verify();dest=BASE/'index';dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists():return
    original=connect(MAIN/'index/chunks.sqlite',True)
    rows=original.execute('SELECT rowid AS global_row,* FROM chunks WHERE asset IN (?,?) ORDER BY rowid',ASSETS).fetchall();original.close()
    path=dest/'chunks.sqlite'
    if path.exists():raise ValueError('Partial pilot index retained; inspect before retry')
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE chunks(id TEXT PRIMARY KEY,source_id TEXT,title TEXT,asset TEXT,published TEXT,start INTEGER,end INTEGER,text TEXT); CREATE INDEX chunk_source ON chunks(source_id); CREATE VIRTUAL TABLE chunk_fts USING fts5(id UNINDEXED,title,text,tokenize="unicode61");')
    with db:
        db.executemany('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)',[tuple(r)[1:] for r in rows])
        db.executemany('INSERT INTO chunk_fts VALUES(?,?,?)',[(r['id'],r['title'],r['text']) for r in rows])
    db.close();write_json(dest/'ids.json',[r['id'] for r in rows],immutable=True)
    import numpy as np
    completed=json.loads((MAIN/'index/vector_progress.json').read_text())['completed']
    full=np.load(MAIN/'index/vectors.npy',mmap_mode='r');vectors=np.zeros((len(rows),384),dtype='float32');missing=[]
    for i,r in enumerate(rows):
        if r['global_row']<=completed:vectors[i]=full[r['global_row']-1]
        else:missing.append(i)
    del full
    paused=None
    try:
        if missing and not (MAIN/'index/manifest.json').exists():
            worker=json.loads((MAIN/'workers/index/status.json').read_text());pid=worker['pid'];proc=Path('/proc')/str(pid)/'cmdline'
            if proc.exists():
                argv=proc.read_bytes().replace(b'\0',b' ').decode()
                if 'finrate_autonomous.prepare index' not in argv:raise ValueError('Global index PID identity differs')
                write_json(dest/'temporary_global_index_pause.json',{'pid':pid,'argv':argv,'time':time.time(),'reason':'Keep embedding CPU work to two threads while completing the pilot missing vectors.'},immutable=True)
                os.kill(pid,signal.SIGSTOP);paused=pid
        if missing:
            embed=CpuEmbedding()
            for start in range(0,len(missing),128):
                check_disk();positions=missing[start:start+128]
                vectors[positions]=embed.encode([rows[i]['title'][:150]+'\n'+rows[i]['text'] for i in positions])
                write_json(dest/'progress.json',{'reused':len(rows)-len(missing),'new_completed':start+len(positions),'new_total':len(missing)})
        check_disk();np.save(dest/'vectors.npy',vectors)
    finally:
        if paused:
            check_disk();proc=Path('/proc')/str(paused)/'cmdline'
            if proc.exists() and 'finrate_autonomous.prepare index' in proc.read_bytes().replace(b'\0',b' ').decode():
                os.kill(paused,signal.SIGCONT)
                write_json(dest/'global_index_resumed.json',{'pid':paused,'time':time.time()},immutable=True)
    write_json(dest/'manifest.json',{'chunks':len(rows),'sources':len({r['source_id'] for r in rows}),'assets':ASSETS,
      'embedding_revision':EMBEDDING_REVISION,'device':'cpu','chunk_tokens':192,'stride':160,
      'reused_completed_vectors':len(rows)-len(missing),'new_vectors':len(missing),
      'files':{p.name:file_digest(p) for p in [path,dest/'vectors.npy',dest/'ids.json']}},immutable=True)
    print('PILOT_INDEX_READY',len(rows),'new',len(missing),flush=True)


def infer(condition,status):
    verify();qs=json.loads((BASE/'cohort/inputs.json').read_text())
    model=IsolatedLuna(max_calls=len(qs)*POLICY['agent_max_turns'])
    ix=FullIndex(BASE/'index') if condition in {'embedding','hybrid'} else None
    try:
        for q in qs:
            folder=BASE/'dev'/q['q_id']/condition;dest=folder/'prediction.json'
            if dest.exists():continue
            write_json(status,{'state':'inference','condition':condition,'q_id':q['q_id'],'pid':os.getpid()})
            try:
                if condition in {'naive_llm','embedding','hybrid'}:
                    candidates=[] if condition=='naive_llm' else ix.search(q['question'],method=condition,k=POLICY['static_top_k'])
                    evidence=fit_evidence(candidates,POLICY['static_evidence_tokens'])
                    write_json(folder/'retrieval.json',{'candidate_ids':[e['id'] for e in candidates],'retained_ids':[e['id'] for e in evidence],
                      'candidate_pool':'all chunks from the same completed-company pilot corpus','assets':ASSETS,'gold_routing':False},immutable=True)
                    prompt=answer_prompt(q['question'],evidence,naive=condition=='naive_llm')
                    raw=model.invoke(prompt,folder/'answer');answer=parse_json(raw)
                    result={'status':'completed','answer':answer,'evidence':evidence,'model_calls':1,'online_text_tokens':token_count(prompt)+token_count(raw),
                      'invalid_citation_ids':[i for c in answer['claims'] for i in c['evidence_ids'] if i not in {e['id'] for e in evidence}]}
                else:
                    representation={'raw_agent':'raw','chunk_agent':'chunks','bank_agent':'bank','flat_bank_agent':'bank'}[condition]
                    name='flat.sqlite' if condition=='flat_bank_agent' else 'bank.sqlite'
                    book=Notebook(BASE/'reader_banks'/name,readonly=True) if representation=='bank' else None
                    w=Workspace(BASE/'corpus.sqlite',representation=representation,chunks=BASE/'index/chunks.sqlite' if representation=='chunks' else None,notebook=book)
                    initial={'question':q['question'],'available_companies':ASSETS,
                      'workspace':'Searchable financial materials. The same original corpus is also available under sources.','original_source_count':784}
                    try:result=run(initial,w,model,folder/'agent')
                    finally:
                        w.close()
                        if book:book.close()
                row={'q_id':q['q_id'],'task':q['task'],'condition':condition,**result}
            except Exception as exc:row={'q_id':q['q_id'],'task':q['task'],'condition':condition,'status':'failed','error':type(exc).__name__+': '+str(exc)}
            write_json(dest,row,immutable=True);print('PILOT_PREDICTION',q['q_id'],condition,row['status'],flush=True)
    finally:
        if ix:ix.close()
    write_json(status,{'state':'completed','condition':condition,'pid':os.getpid()})


def grade(status):
    verify();qs=json.loads((BASE/'cohort/gold.json').read_text())
    if any(not (BASE/'dev'/q['q_id']/c/'prediction.json').exists() for q in qs for c in CONDITIONS):raise ValueError('Complete all140pilot predictions first')
    model=IsolatedLuna(max_calls=len(qs)*len(CONDITIONS));rows=[]
    for q in qs:
        for c in CONDITIONS:
            folder=BASE/'dev'/q['q_id']/c;dest=folder/'assessment.json'
            if dest.exists():row=json.loads(dest.read_text())
            else:
                write_json(status,{'state':'grading','condition':c,'q_id':q['q_id'],'pid':os.getpid()})
                pred=json.loads((folder/'prediction.json').read_text())
                row={'q_id':q['q_id'],'task':q['task'],'condition':c,**grade_candidate(q,pred,model,folder/'blinded_judge'),'usage':online_usage(folder)}
                write_json(dest,row,immutable=True)
            rows.append(row)
    summary={}
    for task in ['all',*POLICY['tasks']]:
        summary[task]={}
        for c in CONDITIONS:
            group=[r for r in rows if r['condition']==c and (task=='all' or r['task']==task)]
            n=len(group);unknown=sum(r['correct'] is None for r in group);right=sum(r['correct'] is True for r in group);points=sum(r['key_point_fraction'] or 0 for r in group)
            summary[task][c]={'n':n,'correct':right,'unassessed':unknown,'accuracy_lower':right/n,'accuracy_upper':(right+unknown)/n,
              'key_point_lower':points/n,'key_point_upper':(points+unknown)/n,'labels':dict(Counter(r['label'] for r in group))}
    write_json(BASE/'dev/report.json',{'rows':rows,'summary':summary,'scope':'separate two-company interim DEV pilot','all_failures_retained':True},immutable=True)
    lines=['Completed-company interim pilot: HUSA + REEMF; LT10 + EC10; separate from full43 results.']
    for c,r in summary['all'].items():lines.append(f"{c}: correct {r['correct']}/{r['n']}; keypoints {r['key_point_lower']:.2%}..{r['key_point_upper']:.2%}; unassessed {r['unassessed']}")
    (BASE/'dev/RESULTS.txt').write_text('\n'.join(lines)+'\n')


def ground(status):
    qs=json.loads((BASE/'cohort/inputs.json').read_text());db=connect(BASE/'corpus.sqlite',True)
    corpus={r['id']:{'_id':r['id'],'title':r['title'],'text':r['text']} for r in db.execute('SELECT id,title,text FROM sources')};db.close()
    model=IsolatedLuna(max_calls=len(qs)*len(CONDITIONS));rows=[]
    for q in qs:
        for c in CONDITIONS:
            write_json(status,{'state':'source_grounding','condition':c,'q_id':q['q_id'],'pid':os.getpid()})
            r=assess_source(q['question'],BASE/'dev'/q['q_id']/c/'prediction.json',corpus,model,BASE/'dev/source_grounding'/q['q_id']/c)
            rows.append({'q_id':q['q_id'],'task':q['task'],'condition':c,**r})
    write_json(BASE/'dev/source_grounding.json',rows,immutable=True)


def pipeline(status):
    prepare();index();queue=deque(CONDITIONS);active=[];done=[];failed=[]
    while queue or active:
        check_disk()
        while queue and len(active)<2 and not failed:
            c=queue.popleft();folder=BASE/'workers'/c;folder.mkdir(parents=True,exist_ok=True)
            argv=[PYTHON,'-u','-m','finrate_completed_pilot_v1','infer','--condition',c]
            with (folder/'stdout.log').open('a') as log:p=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':'method'},stdout=log,stderr=subprocess.STDOUT)
            active.append((c,p));write_json(folder/'launch.json',{'pid':p.pid,'pgid':os.getpgrp(),'argv':argv},immutable=True)
        remaining=[]
        for c,p in active:
            rc=p.poll()
            if rc is None:remaining.append((c,p))
            elif rc:failed.append({'condition':c,'returncode':rc})
            else:done.append(c)
        active=remaining;write_json(status,{'state':'needs_review' if failed else 'inference','active':[c for c,_ in active],'completed_conditions':done,'failed':failed,'queued':list(queue),'pid':os.getpid()})
        if failed and not active:raise RuntimeError('Pilot condition failures retained: '+str(failed))
        if active or queue:time.sleep(5)
    grade(status);ground(status)
    write_json(status,{'state':'completed','pid':os.getpid(),'finished':time.time(),'predictions':140,'questions':20,'heldout_access':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','index','infer','grade','ground','pipeline']);p.add_argument('--condition',choices=CONDITIONS);args=p.parse_args()
    cpu_only();check_disk();os.nice(10)
    status=BASE/('workers/'+args.condition+'/status.json' if args.stage=='infer' else 'status.json');stop=watchdog(status)
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='index':index()
        elif args.stage=='infer':infer(args.condition,status)
        elif args.stage=='grade':grade(status)
        elif args.stage=='ground':ground(status)
        else:pipeline(status)
    except Exception as exc:
        write_json(status,{'state':'failed','error':type(exc).__name__+': '+str(exc),'pid':os.getpid()});raise
    finally:stop.set()


from finrate_autonomous_v10.transport import IsolatedLuna
if __name__=='__main__':main()
