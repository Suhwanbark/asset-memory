"""Matched autonomous QA arms; separate inference, grading, and one heldout release."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time

from .transport import IsolatedLuna
from finrate_final_evaluation import grade_candidate,online_usage,single_executor
from finrate_source_grading import assess as assess_source
from finrate_poc.model import answer_prompt,fit_evidence,parse_json,token_count
from finrate_poc.runtime import ROOT,check_disk,cpu_only,digest,file_digest,watchdog,write_json
from .agent import run
from .config import BASE,CORPUS,POLICY
from .retrieval import FullIndex
from .workspace import Notebook,Workspace,connect


def freeze_online():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'method'/name for name in ['finrate_final_evaluation.py','finrate_dev_controls.py','finrate_source_grading.py',
      'finrate_poc/runtime.py','finrate_poc/model.py','finrate_poc/data.py','finrate_poc/schemas.py','finrate_poc/retrieval.py']]
    # Reader agent and tools are shared byte-for-byte across representations.
    result={'policy':POLICY,'code_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in sorted(paths)},
      'transport_sha256':file_digest(ROOT/'method/finrate_luna_v7.py'),
      'cohort_manifest_sha256':file_digest(BASE/'cohort/manifest.json'),
      'corpus_manifest_sha256':file_digest(BASE/'corpus_manifest.json'),
      'question_selection_used_outcomes':False,'evaluation_allowed':False}
    write_json(BASE/'online_manifest.json',result,immutable=True);return result


def verify_online():
    result=json.loads((BASE/'online_manifest.json').read_text())
    for name,sha in result['code_sha256'].items():
        if file_digest(ROOT/name)!=sha:raise ValueError('Frozen online implementation changed: '+name)
    if result['policy']!=POLICY:raise ValueError('Frozen policy changed')
    return result


def inputs(split):
    verify_online()
    if split=='evaluation':
        release=json.loads((BASE/'release.json').read_text())
        if release['online_manifest_sha256']!=file_digest(BASE/'online_manifest.json'):raise ValueError('Release method changed')
        for name,sha in release['input_sha256'].items():
            if file_digest(BASE/name)!=sha:raise ValueError('Frozen evaluation input changed')
        lock=BASE/'evaluation_release_lock.json'
        write_json(lock,{'release_sha256':file_digest(BASE/'release.json')},immutable=True)
    qs=json.loads((BASE/'cohort'/f'{split}_inputs.json').read_text())
    if any(set(q)!={'q_id','question','task'} for q in qs):raise ValueError('Unexpected inference fields')
    expected=json.loads((BASE/'cohort/manifest.json').read_text())['selected_ids'][split]
    if [q['q_id'] for q in qs]!=expected:raise ValueError('Cohort order changed')
    return qs


def predict(q,condition,model,folder,index=None):
    if condition in {'naive_llm','embedding','hybrid'}:
        candidates=[] if condition=='naive_llm' else index.search(q['question'],method=condition,k=POLICY['static_top_k'])
        evidence=fit_evidence(candidates,POLICY['static_evidence_tokens'])
        write_json(folder/'retrieval.json',{'candidate_ids':[e['id'] for e in candidates],
          'retained_ids':[e['id'] for e in evidence],'candidate_pool':'all43companies/allchunks',
          'gold_routing':False,'top_k':POLICY['static_top_k'],'evidence_tokens':token_count(json.dumps(evidence,ensure_ascii=False))},immutable=True)
        prompt=answer_prompt(q['question'],evidence,naive=condition=='naive_llm')
        text=model.invoke(prompt,folder/'answer');answer=parse_json(text)
        invalid=[i for claim in answer['claims'] for i in claim['evidence_ids'] if i not in {e['id'] for e in evidence}]
        return {'status':'completed','answer':answer,'evidence':evidence,'invalid_citation_ids':invalid,
          'model_calls':1,'online_text_tokens':token_count(prompt)+token_count(text)}
    representation={'raw_agent':'raw','chunk_agent':'chunks','bank_agent':'bank','flat_bank_agent':'bank'}[condition]
    notebook=None
    if representation=='bank':
        manifest=json.loads((BASE/'reader_banks/manifest.json').read_text());name='flat.sqlite' if condition=='flat_bank_agent' else 'bank.sqlite'
        if file_digest(BASE/'reader_banks'/name)!=manifest['files'][name]:raise ValueError('Reader bank changed')
        notebook=Notebook(BASE/'reader_banks'/name,readonly=True)
    workspace=Workspace(CORPUS,representation=representation,
      chunks=BASE/'index/chunks.sqlite' if representation=='chunks' else None,notebook=notebook)
    available=json.loads((BASE/'corpus_manifest.json').read_text())['assets']
    initial={'question':q['question'],'available_companies':available,
      'workspace':'Searchable financial materials. The same original corpus is also available under sources.',
      'original_source_count':15311}
    try:return run(initial,workspace,model,folder/'agent')
    finally:
        workspace.close()
        if notebook:notebook.close()


def infer(split,condition,status):
    qs=inputs(split);root=BASE/split
    with single_executor(BASE/'workers'/f'{split}_{condition}.lock'):
        model=IsolatedLuna(max_calls=len(qs)*POLICY['agent_max_turns'])
        index=FullIndex(BASE/'index') if condition in {'embedding','hybrid'} else None
        try:
            for q in qs:
                check_disk();folder=root/q['q_id']/condition;dest=folder/'prediction.json'
                if dest.exists():continue
                write_json(status,{'state':'inference','split':split,'condition':condition,'q_id':q['q_id'],'pid':os.getpid()})
                try:result=predict(q,condition,model,folder,index)
                except Exception as e:result={'status':'failed','error':type(e).__name__+': '+str(e)}
                write_json(dest,{'q_id':q['q_id'],'task':q['task'],'condition':condition,**result},immutable=True)
                print('PREDICTION',split,q['q_id'],condition,result['status'],flush=True)
        finally:
            if index:index.close()
    verify_online()
    write_json(BASE/'workers'/f'{split}_{condition}_complete.json',{'condition':condition,'split':split,'n':len(qs)},immutable=True)


def prediction_grid(split):
    ids=json.loads((BASE/'cohort/manifest.json').read_text())['selected_ids'][split]
    paths=[BASE/split/q/c/'prediction.json' for q in ids for c in POLICY['conditions']]
    if any(not p.exists() for p in paths):raise ValueError('Finish all condition predictions before grading')
    return {str(p.relative_to(BASE)):file_digest(p) for p in paths}


def grade(split,status):
    verify_online();grid=prediction_grid(split)
    write_json(BASE/split/'inference_complete.json',{'predictions':grid,'gold_not_used_by_inference':True},immutable=True)
    if split=='evaluation':inputs(split)
    # Gold is first opened by this separate evaluator after all predictions.
    gold=json.loads((BASE/'cohort/gold'/f'{split}.json').read_text());rows=[];model=IsolatedLuna(max_calls=len(gold)*len(POLICY['conditions']))
    for q in gold:
        for condition in POLICY['conditions']:
            folder=BASE/split/q['q_id']/condition;dest=folder/'assessment.json'
            if dest.exists():row=json.loads(dest.read_text())
            else:
                write_json(status,{'state':'grading','split':split,'condition':condition,'q_id':q['q_id'],'pid':os.getpid()})
                pred=json.loads((folder/'prediction.json').read_text())
                row={'q_id':q['q_id'],'task':q['task'],'condition':condition,
                  'prediction_sha256':file_digest(folder/'prediction.json'),
                  **grade_candidate(q,pred,model,folder/'blinded_judge'),'usage':online_usage(folder)}
                write_json(dest,row,immutable=True)
            rows.append(row);print('GRADED',split,q['q_id'],condition,row['label'],flush=True)
    summary={}
    for task in ['all',*POLICY['tasks']]:
        summary[task]={}
        for condition in POLICY['conditions']:
            group=[r for r in rows if r['condition']==condition and (task=='all' or r['task']==task)]
            n=len(group);unknown=sum(r['correct'] is None for r in group);right=sum(r['correct'] is True for r in group)
            points=sum(r['key_point_fraction'] or 0 for r in group);usage=Counter()
            for r in group:usage.update(r['usage']['provider_usage'])
            summary[task][condition]={'n':n,'correct':right,'unassessed':unknown,
              'accuracy_lower':right/n,'accuracy_upper':(right+unknown)/n,
              'key_point_lower':points/n,'key_point_upper':(points+unknown)/n,
              'labels':dict(Counter(r['label'] for r in group)),'provider_usage':dict(usage)}
    report={'split':split,'rows':rows,'summary':summary,'all_failures_retained':True,
      'limitation':'Single Luna reference-based judge; source grounding and reference-quality audits are separate. Exposure uncontrolled.'}
    write_json(BASE/split/'report.json',report,immutable=True);return report


def ground(split,status):
    verify_online();prediction_grid(split)
    qs=inputs(split);db=connect(CORPUS,True)
    corpus={r['id']:{'_id':r['id'],'title':r['title'],'text':r['text']} for r in db.execute('SELECT id,title,text FROM sources')};db.close()
    model=IsolatedLuna(max_calls=len(qs)*len(POLICY['conditions']));rows=[]
    for q in qs:
        for condition in POLICY['conditions']:
            write_json(status,{'state':'source_grounding','split':split,'condition':condition,'q_id':q['q_id'],'pid':os.getpid()})
            result=assess_source(q['question'],BASE/split/q['q_id']/condition/'prediction.json',corpus,model,
                                BASE/split/'source_grounding'/q['q_id']/condition)
            rows.append({'q_id':q['q_id'],'task':q['task'],'condition':condition,**result})
            print('SOURCE_GROUNDED',split,q['q_id'],condition,result['state'],flush=True)
    write_json(BASE/split/'source_grounding.json',rows,immutable=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze_online','infer','grade','ground']);p.add_argument('--split',choices=['dev','evaluation'],default='dev');p.add_argument('--condition',choices=POLICY['conditions']);a=p.parse_args()
    cpu_only();check_disk();os.nice(10)
    if a.stage=='freeze_online':freeze_online();return
    status=BASE/'workers'/f'{a.split}_{a.condition or a.stage}'/'status.json';stop=watchdog(status)
    try:
        if a.stage=='infer':infer(a.split,a.condition,status)
        elif a.stage=='grade':grade(a.split,status)
        else:ground(a.split,status)
        write_json(status,{'state':'completed','stage':a.stage,'condition':a.condition,'split':a.split,'pid':os.getpid()})
    except Exception as e:
        write_json(status,{'state':'failed','error':str(e),'pid':os.getpid()});raise
    finally:stop.set()

if __name__=='__main__':main()
