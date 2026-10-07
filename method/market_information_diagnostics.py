"""Post-hoc case diagnostics; fixed packets, repeated calls, original runs preserved."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from market_information_sources import ROOT,save,sha
from market_information_eval import forecast_prompt,parse_output
from temporal_evidence_eval import invoke,tokens,MODEL

OUT=ROOT/'result/market_information_diagnostics_v1'
SPECS=[
    ('news_repeats','market_information_news_poc_v1','KXCBDECISIONJAPAN-25DEC18',['recency','llm_temporal'],False),
    ('excerpt_repeats','market_information_poc_v1','KXFEDDECISION-25SEP-H0',['recency','llm_temporal'],False),
    ('full_source_sensitivity','market_information_poc_v1','KXFEDDECISION-25SEP-H0',['recency','llm_temporal'],True),
    ('anchor_repeats','market_information_poc_v1','FEDDECISION-24NOV-H0',['anchor_without','anchor_with'],False),
]

def prepare():
    path=OUT/'manifest.json'
    if path.exists():return json.loads(path.read_text())
    tasks=[]
    for group,name,event,conditions,full in SPECS:
        d=ROOT/'data'/name;r=ROOT/'result'/name
        cases=[c for c in json.loads((d/'cases.json').read_text())['cases'] if c['event_id']==event]
        assert len(cases)==3
        metadata={e['id']:e for e in json.loads((d/'evidence_cards.json').read_text())['cards']}
        for c in cases:
            for condition in conditions:
                original=json.loads((r/c['id']/condition/'result.json').read_text())
                cards=[dict(e) for e in c['evidence'] if e['id'] in original['selected_ids']]
                provenance=[]
                if full:
                    for card in cards:
                        m=metadata[card['id']];source=Path(m['source_text_path'])
                        assert sha(source.read_bytes())==m['source_text_sha256']
                        card['text']=source.read_text();card['tokens']=tokens(card['text'])
                        provenance.append(dict(id=card['id'],path=str(source),sha256=m['source_text_sha256']))
                prompt=forecast_prompt(c,cards)
                assert tokens(prompt)<50000
                for repeat in range(1,6):
                    tasks.append(dict(id=f'{group}/{c["id"]}/{condition}/r{repeat}',group=group,dataset=name,
                        case_id=c['id'],event_id=event,condition=condition,repeat=repeat,
                        selected_ids=original['selected_ids'],body_tokens=sum(e['tokens'] for e in cards),
                        prompt=prompt,prompt_sha256=sha(prompt),source_provenance=provenance))
    obj=dict(created_at=datetime.now(timezone.utc).isoformat(),model=MODEL,repetitions=5,tasks=tasks,
             selection='Exploratory cases chosen after inspecting original outputs; not held-out confirmation.',
             interpretation='Repeated forecasting conditional on frozen selections. Full-source lengths differ; diagnostic only.')
    save(path,obj);return obj

def run_task(task):
    folder=OUT/'calls'/task['id']
    r=invoke(task['prompt'],folder)
    obj=parse_output(r['text'],'forecast');p=obj['probability']
    assert isinstance(p,(int,float)) and not isinstance(p,bool) and 0<=p<=1
    save(folder/'result.json',dict(task_id=task['id'],probability=p,reason=obj.get('reason'),
        parse_status=obj['_parse_status'],unsupported_citation_ids=sorted(set(obj.get('evidence_ids',[]))-set(task['selected_ids'])),
        prompt_sha256=r['prompt_sha256'],response_path=str(folder/'response.json')))
    print('DONE',task['id'],p,flush=True)

def report(manifest):
    labels={name:{x['case_id']:x for x in json.loads((ROOT/'data'/name/'labels.json').read_text())['labels']}
            for name in {t['dataset'] for t in manifest['tasks']}}
    groups={};missing=[]
    for t in manifest['tasks']:
        p=OUT/'calls'/t['id']/'result.json'
        if not p.exists():missing.append(t['id']);continue
        r=json.loads(p.read_text());response=json.loads(Path(r['response_path']).read_text())
        assert r['prompt_sha256']==sha(t['prompt'])==response['prompt_sha256']
        assert response['tool_calls']==[]
        y=labels[t['dataset']][t['case_id']]['outcome']
        key=(t['group'],t['case_id'],t['condition']);groups.setdefault(key,[]).append((r['probability'],(r['probability']-y)**2))
    summary=[]
    for (g,c,k),values in groups.items():
        ps,bs=zip(*values)
        summary.append(dict(group=g,case_id=c,condition=k,n=len(values),probabilities=list(ps),
            mean_probability=float(np.mean(ps)),mean_brier=float(np.mean(bs)),min_p=min(ps),max_p=max(ps)))
    paired=[]
    for g,c,_ in groups:
        treatment='anchor_with' if g=='anchor_repeats' else 'llm_temporal'
        reference='anchor_without' if g=='anchor_repeats' else 'recency'
        if any(x['group']==g and x['case_id']==c for x in paired):continue
        if (g,c,treatment) in groups and (g,c,reference) in groups:
            a=groups[g,c,treatment];b=groups[g,c,reference]
            paired.append(dict(group=g,case_id=c,treatment=treatment,reference=reference,
                mean_brier_improvement=float(np.mean([x[1] for x in b])-np.mean([x[1] for x in a]))))
    usage=[json.loads(p.read_text()) for p in (OUT/'calls').rglob('response.json')]
    obj=dict(complete=not missing,missing=missing,summary=summary,paired=paired,n_actual_model_calls=len(usage),
        input_tokens=sum(u.get('input_tokens',0) for x in usage for u in x.get('usage',[])),
        output_tokens=sum(u.get('output_tokens',0) for x in usage for u in x.get('usage',[])),
        limitations=manifest['selection']+' '+manifest['interpretation'])
    save(OUT/'report.json',obj);print(json.dumps(obj,indent=2));return obj

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--workers',type=int,default=2);a=p.parse_args();m=prepare()
    print('MANIFEST',len(m['tasks']),'calls',flush=True)
    if a.run:
        errors=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            fs={pool.submit(run_task,t):t['id'] for t in m['tasks']}
            for f in as_completed(fs):
                try:f.result()
                except Exception as exc:errors.append(dict(task=fs[f],error=repr(exc)));print('ERROR',fs[f],repr(exc),flush=True)
        save(OUT/'errors.json',errors)
    if a.report:report(m)
