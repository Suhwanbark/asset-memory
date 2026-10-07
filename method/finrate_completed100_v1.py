"""Fixed 100-question evaluation on the six completed company banks.

Source/DEV-component availability is the only eligibility filter. Selection is
50 LT + 50 EC by a fixed hash, excluding the earlier pilot20. Existing banks,
agent prompts/tools/budgets and old results stay unchanged. All failures remain.
"""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from finrate_poc.runtime import ROOT, check_disk, cpu_only, digest, file_digest, watchdog, write_json
from finrate_poc.model import answer_prompt, fit_evidence, parse_json, token_count
from finrate_poc.schemas import JUDGEMENT
from finrate_dev_controls import judgement_prompt
from finrate_final_evaluation import online_usage, single_executor
from finrate_source_grading import assess as assess_source
from finrate_autonomous_v10.config import BASE as MAIN, POLICY
from finrate_autonomous_v10.evaluate import verify_online
from finrate_autonomous_v10.workspace import Notebook, Workspace, connect, ANCHOR
from finrate_autonomous_v10.retrieval import FullIndex
from finrate_autonomous_v10.export import reader_records
from finrate_resume_agent_v1 import run
from finrate_transport_recovery_v1 import RecoveredLuna

BASE = ROOT / 'result/finrate_completed100_v1'
DATA = ROOT / 'data/finrate_completed100_v1'
PUBLIC = ROOT / 'data/finrate_public_v1'
COVERAGE = ROOT / 'result/finrate_bank_query_coverage_20261006_v1/report.json'
ASSETS = ['ALP.PQ', 'GEN', 'HUSA', 'LANC', 'NXPL', 'REEMF']
TASKS = ['LT-QA', 'EC-QA']
CONDITIONS = list(POLICY['conditions'])
SEED = 'finrate-completed-six-100-20261006-v1'
PYTHON = '/home/hob/anaconda3/envs/su/bin/python'
MODULE = 'finrate_completed100_v1'


def select_metadata(pools, *, per_task=50, seed=SEED):
    """Do not inspect question, reference, key points or previous outcomes."""
    selected = []
    for task in TASKS:
        pool = pools[task]
        if len(pool) < per_task or len({q['q_id'] for q in pool}) != len(pool):
            raise ValueError('Insufficient or duplicate eligible metadata')
        selected.extend(sorted(pool, key=lambda q: digest(seed+'question'+q['q_id']))[:per_task])
    if len({q['q_id'] for q in selected}) != len(selected):
        raise ValueError('Question identity repeated across tasks')
    return sorted(selected, key=lambda q: digest(seed+'order'+q['q_id']))


def select_cohort(source_assets):
    coverage = json.loads(COVERAGE.read_text())
    if coverage['completed_assets'] != ASSETS:
        raise ValueError('Completed asset scope changed')
    qs = []
    source_manifest = json.loads((PUBLIC/'source_manifest.json').read_text())
    for task in TASKS:
        f = PUBLIC/'qa'/f'{task}.json'
        expected = next(r['sha256'] for r in source_manifest['files'] if r['path']==f'qa/{task}.json')
        if file_digest(f) != expected:raise ValueError('Public QA bytes changed')
        qs.extend({**q, 'task': task} for q in json.loads(f.read_text()))
    parent = {q['q_id']:q['q_id'] for q in qs};owners = {}
    def find(x):
        while parent[x] != x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for q in qs:
        for sid in q['doc_ids']:
            if sid in owners:parent[find(q['q_id'])]=find(owners[sid])
            owners[sid]=q['q_id']
    groups=defaultdict(list)
    for q in qs:groups[find(q['q_id'])].append(q['q_id'])
    group_id={qid:min(ids) for ids in groups.values() for qid in ids}
    main=json.loads((MAIN/'cohort/manifest.json').read_text())
    splits={g['id']:g['split'] for g in main['groups']}
    if set(splits)!=set(group_id.values()):raise ValueError('Source component split changed')
    prior=json.loads((ROOT/'result/finrate_completed_pilot_v1/cohort/inputs.json').read_text())
    excluded=set(main['excluded_prior_ids'])|set(sum(main['selected_ids'].values(),[]))|{q['q_id'] for q in prior}
    pools={t:[] for t in TASKS};lookup={q['q_id']:q for q in qs}
    for q in qs:
        assets=sorted({source_assets[sid] for sid in q['doc_ids']})
        if set(assets)<=set(ASSETS) and splits[group_id[q['q_id']]]=='dev' and q['q_id'] not in excluded:
            pools[q['task']].append({'q_id':q['q_id'],'task':q['task'],'assets':assets,'source_ids':q['doc_ids']})
    if {t:len(v) for t,v in pools.items()}!={'LT-QA':268,'EC-QA':92}:
        raise ValueError('Eligible360 pool differs from the recorded coverage count')
    selected=select_metadata(pools)
    gold=[{**lookup[q['q_id']],'assets':q['assets'],'source_ids':q['source_ids']} for q in selected]
    write_json(DATA/'cohort/inputs.json',[{k:q[k] for k in ['q_id','task','question']} for q in gold],immutable=True)
    write_json(DATA/'cohort/gold.json',gold,immutable=True)
    selection={'seed':SEED,'pool_counts':{t:len(v) for t,v in pools.items()},'counts':dict(Counter(q['task'] for q in selected)),
      'ids':[q['q_id'] for q in selected],'question_metadata':selected,
      'question_asset_combinations':dict(Counter(' + '.join(q['assets']) for q in selected)),
      'rule':'Fixed hash sample50LT+50EC from eligible360; no question text, gold content, difficulty or performance criterion.',
      'previous_pilot20_excluded':True,'all_primary_heldout_components_excluded':True,
      'old_and_primary_selected_ids_excluded':True,'bank_building_uses_no_question_or_gold':True,
      'interpretation':'Balanced task sample; overall mean is not a population-weighted estimate of all384 questions.'}
    write_json(DATA/'cohort/manifest.json',selection,immutable=True)
    return selection


def build_index():
    import numpy as np
    target=DATA/'index';target.mkdir(exist_ok=True)
    manifest=json.loads((MAIN/'index/manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if file_digest(MAIN/'index'/name)!=sha:raise ValueError('Full index changed')
    db=connect(MAIN/'index/chunks.sqlite',True)
    slots=','.join('?' for _ in ASSETS)
    rows=db.execute(f'SELECT rowid AS global_row,* FROM chunks WHERE asset IN ({slots}) ORDER BY rowid',ASSETS).fetchall();db.close()
    if (target/'chunks.sqlite').exists():raise ValueError('Inspect partial subset index before restarting')
    db=sqlite3.connect(target/'chunks.sqlite')
    db.executescript('CREATE TABLE chunks(id TEXT PRIMARY KEY,source_id TEXT,title TEXT,asset TEXT,published TEXT,start INTEGER,end INTEGER,text TEXT); CREATE INDEX chunk_source ON chunks(source_id); CREATE VIRTUAL TABLE chunk_fts USING fts5(id UNINDEXED,title,text,tokenize="unicode61");')
    with db:
        db.executemany('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)',[tuple(r)[1:] for r in rows])
        db.executemany('INSERT INTO chunk_fts VALUES(?,?,?)',[(r['id'],r['title'],r['text']) for r in rows])
    db.close();ids=json.loads((MAIN/'index/ids.json').read_text())
    if any(ids[r['global_row']-1]!=r['id'] for r in rows):raise ValueError('Global vector/row mapping differs')
    full=np.load(MAIN/'index/vectors.npy',mmap_mode='r')
    vectors=full[np.asarray([r['global_row']-1 for r in rows])].copy();del full
    if vectors.shape!=(len(rows),384) or not np.isfinite(vectors).all():raise ValueError('Invalid subset vectors')
    check_disk();np.save(target/'vectors.npy',vectors)
    write_json(target/'ids.json',[r['id'] for r in rows],immutable=True)
    write_json(target/'manifest.json',{'chunks':len(rows),'sources':len({r['source_id'] for r in rows}),
      'assets':ASSETS,'embedding_revision':manifest['embedding_revision'],'device':'cpu','chunk_tokens':192,'stride':160,
      'reused_completed_vectors':len(rows),'new_document_vectors':0,
      'files':{name:file_digest(target/name) for name in ['chunks.sqlite','ids.json','vectors.npy']}},immutable=True)


def prepare():
    verify_online();check_disk()
    if (BASE/'manifest.json').exists():verify();return
    BASE.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    coverage=json.loads(COVERAGE.read_text());completed={}
    source=connect(MAIN/'corpus.sqlite',True);source_assets=dict(source.execute('SELECT id,asset FROM sources'))
    for a in ASSETS:
        root=MAIN/'banks'/a;done=json.loads((root/'completed.json').read_text())
        if done['state'] not in {'completed','completed_with_budget_limits'} or not done['full_asset_history']:
            raise ValueError('Incomplete company bank: '+a)
        if file_digest(root/'bank.sqlite')!=coverage['bank_sha256'][a] or done['bank_sha256']!=coverage['bank_sha256'][a]:
            raise ValueError('Completed bank bytes changed: '+a)
        observed=set()
        for p in (root/'updates').glob('*/report.json'):observed.update(json.loads(p.read_text())['visited_source_ids'])
        if any(source_assets[s] not in ASSETS for s in observed):raise ValueError('Bank observed out-of-scope originals')
        book=Notebook(root/'bank.sqlite',readonly=True)
        try:
            if any(source_assets[m.group(1)] not in ASSETS for row in reader_records(book) for m in ANCHOR.finditer(row['body'])):
                raise ValueError('Bank cites out-of-scope originals')
        finally:book.close()
        completed[a]=done
    selection=select_cohort(source_assets)
    path=DATA/'corpus.sqlite'
    if path.exists():raise ValueError('Inspect incomplete preparation before retry')
    slots=','.join('?' for _ in ASSETS)
    rows=source.execute(f'SELECT * FROM sources WHERE asset IN ({slots}) ORDER BY id',ASSETS).fetchall();source.close()
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE sources(id TEXT PRIMARY KEY,title TEXT,asset TEXT,published TEXT,quarter TEXT,text TEXT,sha256 TEXT); CREATE INDEX source_asset_date ON sources(asset,published,id); CREATE VIRTUAL TABLE source_fts USING fts5(id UNINDEXED,title,text,tokenize="unicode61");')
    with db:
        db.executemany('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',[tuple(r) for r in rows])
        db.executemany('INSERT INTO source_fts VALUES(?,?,?)',[(r['id'],r['title'],r['text']) for r in rows])
    db.close()
    write_json(DATA/'corpus_manifest.json',{'assets':ASSETS,'sources':len(rows),'sqlite_sha256':file_digest(path)},immutable=True)
    (DATA/'reader_banks').mkdir()
    for name in ['corpus.sqlite','corpus_manifest.json','cohort','index','reader_banks']:
        (BASE/name).symlink_to(DATA/name,target_is_directory=name in {'cohort','index','reader_banks'})
    (BASE/'banks').mkdir()
    for a in ASSETS:(BASE/'banks'/a).symlink_to(MAIN/'banks'/a,target_is_directory=True)
    import finrate_autonomous_v10.export as exporter
    old=exporter.BASE
    try:exporter.BASE=BASE;exporter.export_all()
    finally:exporter.BASE=old
    build_index()
    code=[Path(__file__),ROOT/'method/finrate_transport_recovery_v1.py',ROOT/'method/finrate_resume_agent_v1.py',
          ROOT/'tests/test_completed100_v1.py']
    input_files=[DATA/'corpus.sqlite',DATA/'corpus_manifest.json',*(DATA/'cohort').glob('*.json'),
                 *(DATA/'index').glob('*'),*(DATA/'reader_banks').glob('*')]
    manifest={'assets':ASSETS,'questions':100,'task_counts':selection['counts'],'conditions':CONDITIONS,'source_count':len(rows),
      'policy':POLICY,'code_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in code},
      'input_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in input_files if p.is_file()},
      'bank_sha256':{a:d['bank_sha256'] for a,d in completed.items()},
      'original_online_manifest_sha256':file_digest(MAIN/'online_manifest.json'),
      'coverage_sha256':file_digest(COVERAGE),'seed':SEED,'source_scope':'All sources from the same six completed companies in every evidence-based condition.',
      'selection_uses_performance':False,'heldout_used':False,'cutoff_filter':False,
      'transport_change':'Accept exact valid completed response after informational fallback notice; no regeneration of failed calls.',
      'grading_change':'Same reference judge prompt; require exactly one boolean per supplied key point in output schema.',
      'source_grounding':'Separate unchanged reference-free original-evidence verification.',
      'all_failures_retained':True,'new_bank_building':False,'worker_concurrency':2,'cpu_embedding_threads':2}
    write_json(BASE/'manifest.json',manifest,immutable=True)
    for p in code:
        target=BASE/'code_snapshot'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    print('PREPARED100',json.dumps({'sources':len(rows),'selection':selection['question_asset_combinations']}),flush=True)


def verify():
    verify_online();check_disk();m=json.loads((BASE/'manifest.json').read_text())
    for group in ['code_sha256','input_sha256']:
        for name,sha in m[group].items():
            if file_digest(ROOT/name)!=sha:raise ValueError('Frozen100 file changed: '+name)
    for a,sha in m['bank_sha256'].items():
        if file_digest(MAIN/'banks'/a/'bank.sqlite')!=sha:raise ValueError('Completed bank changed: '+a)
    if file_digest(MAIN/'online_manifest.json')!=m['original_online_manifest_sha256']:raise ValueError('Original method manifest changed')
    return m


def inputs():
    qs=json.loads((DATA/'cohort/inputs.json').read_text())
    if len(qs)!=100 or any(set(q)!={'q_id','task','question'} for q in qs):raise ValueError('Invalid inference input fields')
    return qs


def retrieve(status):
    verify();qs=inputs();index=FullIndex(DATA/'index');files={}
    try:
        for n,q in enumerate(qs):
            for method in ['embedding','hybrid']:
                path=BASE/'retrieval'/q['q_id']/f'{method}.json'
                if not path.exists():
                    write_json(status,{'state':'cpu_query_retrieval','completed_questions':n,'total_questions':100,'pid':os.getpid()})
                    start=time.monotonic();candidates=index.search(q['question'],method=method,k=POLICY['static_top_k'])
                    evidence=fit_evidence(candidates,POLICY['static_evidence_tokens'])
                    write_json(path,{'question_sha256':digest(q['question']),'method':method,'evidence':evidence,
                      'candidate_ids':[e['id'] for e in candidates],'retained_ids':[e['id'] for e in evidence],
                      'candidate_pool':'all23099chunks/sixcompanies','gold_routing':False,
                      'elapsed_s':time.monotonic()-start,'device':'cpu','threads':2},immutable=True)
                files[str(path.relative_to(BASE))]=file_digest(path)
    finally:index.close()
    write_json(BASE/'retrieval_manifest.json',{'files':files,'count':len(files),'query_specific_cost_is_online':True},immutable=True)


def infer(condition,status):
    verify();qs=inputs();model=RecoveredLuna(max_calls=len(qs)*POLICY['agent_max_turns'])
    retrieval=json.loads((BASE/'retrieval_manifest.json').read_text())
    original_count=json.loads((DATA/'corpus_manifest.json').read_text())['sources']
    for n,q in enumerate(qs):
        check_disk();folder=BASE/'answers'/q['q_id']/condition;dest=folder/'prediction.json'
        if dest.exists():continue
        write_json(status,{'state':'inference','condition':condition,'q_id':q['q_id'],'completed':n,'total':100,'pid':os.getpid()})
        started=time.monotonic()
        try:
            if condition in {'naive_llm','embedding','hybrid'}:
                evidence=[]
                if condition!='naive_llm':
                    p=BASE/'retrieval'/q['q_id']/f'{condition}.json'
                    if file_digest(p)!=retrieval['files'][str(p.relative_to(BASE))]:raise ValueError('Frozen retrieval changed')
                    r=json.loads(p.read_text())
                    if r['question_sha256']!=digest(q['question']):raise ValueError('Retrieval question mismatch')
                    evidence=r['evidence']
                prompt=answer_prompt(q['question'],evidence,naive=condition=='naive_llm')
                raw=model.invoke(prompt,folder/'answer');answer=parse_json(raw)
                result={'status':'completed','answer':answer,'evidence':evidence,'model_calls':1,
                  'online_text_tokens':token_count(prompt)+token_count(raw),
                  'invalid_citation_ids':[i for claim in answer['claims'] for i in claim['evidence_ids'] if i not in {e['id'] for e in evidence}]}
            else:
                representation={'raw_agent':'raw','chunk_agent':'chunks','bank_agent':'bank','flat_bank_agent':'bank'}[condition]
                name='flat.sqlite' if condition=='flat_bank_agent' else 'bank.sqlite'
                book=Notebook(DATA/'reader_banks'/name,readonly=True) if representation=='bank' else None
                workspace=Workspace(DATA/'corpus.sqlite',representation=representation,
                  chunks=DATA/'index/chunks.sqlite' if representation=='chunks' else None,notebook=book)
                initial={'question':q['question'],'available_companies':ASSETS,
                  'workspace':'Searchable financial materials. The same original corpus is also available under sources.',
                  'original_source_count':original_count}
                try:result=run(initial,workspace,model,folder/'agent')
                finally:
                    workspace.close()
                    if book:book.close()
        except Exception as exc:result={'status':'failed','error':type(exc).__name__+': '+str(exc)}
        write_json(dest,{'q_id':q['q_id'],'task':q['task'],'condition':condition,'elapsed_s':time.monotonic()-started,**result},immutable=True)
        print('PREDICTION100',condition,n+1,q['q_id'],result['status'],flush=True)


def grade_one(q,prediction,model,folder):
    if prediction['status']=='failed':
        return {'label':'FAILURE','correct':False,'key_points_met':[False]*len(q['key_points']),
                'key_point_fraction':0.0,'generation_failure':prediction['error']}
    try:
        schema=deepcopy(JUDGEMENT);schema['properties']['key_points_met'].update(minItems=len(q['key_points']),maxItems=len(q['key_points']))
        row=parse_json(model.invoke(judgement_prompt(q,prediction),folder,schema=schema))
        points=row['key_points_met']
        if row['label'] not in {'CORRECT','PARTIAL','INCORRECT','FAILURE'}:raise ValueError('Invalid judge label')
        if len(points)!=len(q['key_points']) or any(type(x) is not bool for x in points):raise ValueError('Judge key-point count/type mismatch')
        return {**row,'correct':row['label']=='CORRECT','key_point_fraction':sum(points)/len(points)}
    except Exception as exc:return {'label':'UNASSESSED','correct':None,'key_point_fraction':None,'judge_failure':type(exc).__name__+': '+str(exc)}


def summarize(rows):
    n=len(rows);right=sum(r['correct'] is True for r in rows);unknown=sum(r['correct'] is None for r in rows)
    points=sum(r['key_point_fraction'] or 0 for r in rows)
    return {'n':n,'correct':right,'unassessed':unknown,'accuracy_lower':right/n,'accuracy_upper':(right+unknown)/n,
      'key_point_lower':points/n,'key_point_upper':(points+unknown)/n,
      'generation_failures':sum(r['generation_status']=='failed' for r in rows),'labels':dict(Counter(r['label'] for r in rows))}


def grade(condition,status):
    verify();qs=json.loads((DATA/'cohort/gold.json').read_text())
    if any(not (BASE/'answers'/q['q_id']/condition/'prediction.json').exists() for q in qs):raise ValueError('Finish all100 predictions before grading this condition')
    model=RecoveredLuna(max_calls=len(qs));rows=[]
    for n,q in enumerate(qs):
        folder=BASE/'answers'/q['q_id']/condition;dest=folder/'assessment.json';pred=folder/'prediction.json'
        if dest.exists():row=json.loads(dest.read_text())
        else:
            write_json(status,{'state':'reference_grading','condition':condition,'completed':n,'total':100,'pid':os.getpid()})
            prediction=json.loads(pred.read_text())
            row={'q_id':q['q_id'],'task':q['task'],'condition':condition,'prediction_sha256':file_digest(pred),
              'generation_status':prediction['status'],**grade_one(q,prediction,model,folder/'blinded_judge')}
            write_json(dest,row,immutable=True)
        if row['prediction_sha256']!=file_digest(pred):raise ValueError('Prediction changed after grading')
        rows.append(row)
        write_json(BASE/'workers'/condition/'score_progress.json',{'graded':len(rows),'total':100,'summary':summarize(rows)})
    write_json(BASE/'reports'/f'{condition}.json',{'condition':condition,'rows':rows,
      'summary':{task:summarize([r for r in rows if task=='all' or r['task']==task]) for task in ['all',*TASKS]},
      'scope':'Fixed balanced100, six completed companies; same-family judge, failures retained, not primary heldout.'},immutable=True)


def ground(condition,status):
    verify();qs=inputs();db=connect(DATA/'corpus.sqlite',True)
    corpus={r['id']:{'_id':r['id'],'title':r['title'],'text':r['text']} for r in db.execute('SELECT id,title,text FROM sources')};db.close()
    model=RecoveredLuna(max_calls=len(qs));rows=[]
    for n,q in enumerate(qs):
        write_json(status,{'state':'source_grounding','condition':condition,'completed':n,'total':100,'pid':os.getpid()})
        pred=BASE/'answers'/q['q_id']/condition/'prediction.json'
        result=assess_source(q['question'],pred,corpus,model,BASE/'source_grounding'/q['q_id']/condition)
        rows.append({'q_id':q['q_id'],'task':q['task'],'condition':condition,**result})
    write_json(BASE/'reports'/f'{condition}_grounding.json',{'rows':rows,'states':dict(Counter(r['state'] for r in rows))},immutable=True)


def collect():
    verify();reports={c:json.loads((BASE/'reports'/f'{c}.json').read_text()) for c in CONDITIONS}
    rows=[r for report in reports.values() for r in report['rows']]
    if len(rows)!=700 or len({(r['q_id'],r['condition']) for r in rows})!=700:raise ValueError('Incomplete comparison grid')
    grounding={c:json.loads((BASE/'reports'/f'{c}_grounding.json').read_text()) for c in CONDITIONS}
    costs={}
    for c in CONDITIONS:
        usage=Counter();calls=unknown=0
        for q in inputs():
            u=online_usage(BASE/'answers'/q['q_id']/c);usage.update(u['provider_usage']);calls+=u['logged_response_calls'];unknown+=u['requests_with_unknown_usage']
        costs[c]={'provider_usage':dict(usage),'logged_response_calls':calls,'requests_with_unknown_usage':unknown}
    summary={task:{c:reports[c]['summary'][task] for c in CONDITIONS} for task in ['all',*TASKS]}
    write_json(BASE/'reports/reference.json',{'summary':summary,'rows':rows,'inference_costs':costs,
      'source_grounding_states':{c:grounding[c]['states'] for c in CONDITIONS},
      'scope':'Balanced50LT+50EC, six completed companies; same-family reference judge; no primary heldout release.',
      'all_failures_retained':True,'bank_construction_cost_separate':True},immutable=True)
    lines=['Completed-six-company comparison: fixed100 questions (50LT + 50EC).']
    for c,r in summary['all'].items():
        lines.append(f"{c}: correct {r['correct']}/{r['n']}; keypoints {r['key_point_lower']:.2%}..{r['key_point_upper']:.2%}; generation failures {r['generation_failures']}; unassessed {r['unassessed']}")
    (BASE/'RESULTS.txt').write_text('\n'.join(lines)+'\n')


def pipeline(status):
    verify()
    if not (BASE/'retrieval_manifest.json').exists():retrieve(status)
    order=['naive_llm','raw_agent','bank_agent','embedding','hybrid','chunk_agent','flat_bank_agent']
    queue=deque(order);active=[];done=[];failed=[]
    while queue or active:
        check_disk()
        while queue and len(active)<2 and not failed:
            c=queue.popleft();folder=BASE/'workers'/c;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'launch.json').exists():raise ValueError('Recorded condition worker exists; inspect before relaunch')
            argv=[PYTHON,'-u','-m',MODULE,'condition','--condition',c]
            with (folder/'stdout.log').open('a') as log:
                p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'method'})
            write_json(folder/'launch.json',{'pid':p.pid,'pgid':os.getpgrp(),'argv':argv,'started':time.time()},immutable=True)
            active.append((c,p))
        remaining=[]
        for c,p in active:
            rc=p.poll()
            if rc is None:remaining.append((c,p))
            elif rc:failed.append({'condition':c,'returncode':rc})
            else:done.append(c)
        active=remaining
        write_json(status,{'state':'needs_review' if failed else 'running','active':[c for c,_ in active],
          'completed_conditions':done,'queued':list(queue),'failed':failed,'pid':os.getpid(),'questions':100,'conditions':7})
        if failed and not active:raise RuntimeError('Worker failed; preserve unstarted conditions')
        if queue or active:time.sleep(5)
    collect();write_json(status,{'state':'completed','questions':100,'predictions':700,'finished':time.time(),'pid':os.getpid(),'heldout_used':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','retrieve','condition','collect','pipeline']);p.add_argument('--condition',choices=CONDITIONS)
    a=p.parse_args();cpu_only();check_disk();os.nice(10)
    status=BASE/('workers/'+a.condition+'/status.json' if a.stage=='condition' else 'status.json');stop=watchdog(status)
    try:
        with single_executor(BASE/'locks'/(a.condition+'.lock' if a.stage=='condition' else a.stage+'.lock')):
            if a.stage=='prepare':prepare()
            elif a.stage=='retrieve':retrieve(status)
            elif a.stage=='pipeline':pipeline(status)
            elif a.stage=='collect':collect()
            else:
                infer(a.condition,status);grade(a.condition,status);ground(a.condition,status)
                write_json(status,{'state':'completed','condition':a.condition,'finished':time.time(),'pid':os.getpid()})
    except Exception as exc:
        write_json(status,{'state':'failed','error':type(exc).__name__+': '+str(exc),'pid':os.getpid()});raise
    finally:stop.set()


if __name__=='__main__':main()
