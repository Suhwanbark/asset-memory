"""Offline outcome scoring; this module alone joins labels to forecasts."""
from __future__ import annotations
import json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from market_information_sources import ROOT,DATA,save,sha
from market_information_eval import OUT,CONDITIONS,DIAGNOSTICS

def brier(p,y):return (p-y)**2

def paired_comparison(rows,treatment,reference,split='main'):
    grouped=defaultdict(dict)
    for r in rows:
        if r['split']==split:grouped[r['case_id']][r['condition']]=r
    diffs=defaultdict(list);pairs=[]
    for cid,g in grouped.items():
        if treatment in g and reference in g:
            t,r=g[treatment],g[reference]
            delta=r['brier']-t['brier']
            diffs[t['event_id']].append(delta)
            pairs.append(dict(case_id=cid,event_id=t['event_id'],phase=t['phase'],improvement=delta,
                              treatment_p=t['probability'],reference_p=r['probability'],outcome=t['outcome']))
    event_ids=sorted(diffs);means=np.array([np.mean(diffs[e]) for e in event_ids])
    if not len(means):return {'treatment':treatment,'reference':reference,'n_events':0}
    rng=np.random.default_rng(3701)
    boot=np.mean(rng.choice(means,size=(10000,len(means)),replace=True),axis=1)
    # Event chronology for circular 3-meeting block sensitivity analysis.
    chronological=sorted(event_ids,key=lambda e:min(r['meeting_date'] for r in rows if r['event_id']==e))
    cm=np.array([np.mean(diffs[e]) for e in chronological]);blocks=[]
    for _ in range(10000):
        indices=[]
        while len(indices)<len(cm):
            start=int(rng.integers(len(cm)));indices.extend([(start+j)%len(cm) for j in range(min(3,len(cm)))])
        blocks.append(float(np.mean(cm[indices[:len(cm)]])))
    return dict(treatment=treatment,reference=reference,split=split,n_events=len(means),n_snapshots=len(pairs),
        mean_brier_improvement=float(means.mean()),event_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
        three_meeting_block_95ci=np.quantile(blocks,[.025,.975]).tolist(),
        improved_events=int((means>1e-9).sum()),worse_events=int((means<-1e-9).sum()),tied_events=int((abs(means)<=1e-9).sum()),
        event_differences={e:float(np.mean(diffs[e])) for e in chronological},pairs=sorted(pairs,key=lambda r:r['improvement'],reverse=True))

def report():
    cases={c['id']:c for c in json.loads((DATA/'cases.json').read_text())['cases']}
    labels={x['case_id']:x for x in json.loads((DATA/'labels.json').read_text())['labels']}
    rows=[]
    for path in sorted(OUT.glob('*/*/result.json')):
        r=json.loads(path.read_text());c=cases[r['case_id']];g=labels[r['case_id']]
        assert r['dataset_sha256']==json.loads((DATA/'freeze.json').read_text())['cases_sha256']
        response=json.loads(Path(r['forecast_response_path']).read_text())
        assert response['prompt_sha256']==r['prompt_sha256']
        rows.append(dict(r,outcome=g['outcome'],phase=g['phase']['phase'],meeting_date=c['meeting_date'],
                         lead_days=c['lead_days'],brier=brier(r['probability'],g['outcome']),result_path=str(path)))
    for cid,c in cases.items():
        g=labels[cid];p=g['phase']['p']
        rows.append(dict(case_id=cid,event_id=c['event_id'],condition='market_price',split=c['split'],
            probability=p,outcome=g['outcome'],brier=brier(p,g['outcome']),phase=g['phase']['phase'],
            meeting_date=c['meeting_date'],lead_days=c['lead_days'],body_tokens=0))
    summary=[]
    for split in ['smoke','main']:
        for condition in CONDITIONS+DIAGNOSTICS+['market_price']:
            rs=[r for r in rows if r['split']==split and r['condition']==condition]
            if not rs:continue
            es=defaultdict(list)
            for r in rs:es[r['event_id']].append(r['brier'])
            summary.append(dict(split=split,condition=condition,n_events=len(es),n_snapshots=len(rs),
                event_mean_brier=float(np.mean([np.mean(v) for v in es.values()])),
                snapshot_mean_brier=float(np.mean([r['brier'] for r in rs])),
                mean_body_tokens=float(np.mean([r['body_tokens'] for r in rs])),
                mean_selected_age_days=float(np.mean([r['selected_mean_age_days'] for r in rs if r.get('selected_mean_age_days') is not None])) if any(r.get('selected_mean_age_days') is not None for r in rs) else None))
    phase_summary=[]
    for phase in ['consensus','fluctuation','reversal','intermediate']:
        for condition in CONDITIONS+DIAGNOSTICS+['market_price']:
            rs=[r for r in rows if r['split']=='main' and r['phase']==phase and r['condition']==condition]
            es=defaultdict(list)
            for r in rs:es[r['event_id']].append(r['brier'])
            if rs:phase_summary.append(dict(phase=phase,condition=condition,n_snapshots=len(rs),n_events=len(es),
                event_mean_brier=float(np.mean([np.mean(v) for v in es.values()]))))
    comparisons=[paired_comparison(rows,t,r) for t,r in [('llm_temporal','recency'),('llm_priority','recency'),
         ('llm_temporal','relevance'),('llm_temporal','llm_priority'),('full_pool','recency'),('recency','no_evidence'),('anchor_with','anchor_without')]]
    have={(r['case_id'],r['condition']) for r in rows}
    failures=[json.loads(p.read_text()) for p in OUT.glob('*/*/failure.json')]
    failed={(r['case_id'],r['condition']) for r in failures}
    expected=CONDITIONS+(DIAGNOSTICS if (DATA/'anchor_manifest.json').exists() else [])
    missing=[dict(case_id=c,condition=k) for c in cases for k in expected if (c,k) not in have|failed]
    # If a selector fails, retain it as a task failure; never fabricate a forecast.
    # Bounds show sensitivity of its aggregate score to the unscored [0,1] loss.
    failure_bounds=[]
    for condition in CONDITIONS:
        relevant=[f for f in failures if f['condition']==condition and cases[f['case_id']]['split']=='main']
        if not relevant:continue
        bounds=[]
        for loss in [0.,1.]:
            es=defaultdict(list)
            for r in rows:
                if r['split']=='main' and r['condition']==condition:es[r['event_id']].append(r['brier'])
            for f in relevant:es[cases[f['case_id']]['event_id']].append(loss)
            bounds.append(float(np.mean([np.mean(v) for v in es.values()])))
        failure_bounds.append(dict(condition=condition,failed_snapshots=len(relevant),event_mean_brier_bounds=bounds))
    # Descriptive target pattern fixed before main scoring: older retained evidence + >=.02 Brier improvement.
    lookup={(r['case_id'],r['condition']):r for r in rows};patterns=[]
    for cid,c in cases.items():
        if c['split']!='main':continue
        r=lookup.get((cid,'recency'));t=lookup.get((cid,'llm_temporal'))
        if not r or not t:continue
        ages={e['id']:(__import__('datetime').datetime.fromisoformat(c['cutoff'])-__import__('datetime').datetime.fromisoformat(e['available_at'])).days for e in c['evidence']}
        older=[i for i in t['selected_ids'] if i not in r['selected_ids'] and ages[i]>max(ages[j] for j in r['selected_ids'])]
        if older:
            patterns.append(dict(case_id=cid,event_id=c['event_id'],phase=t['phase'],older_ids=older,
                temporal_selected=t['selected_ids'],recency_selected=r['selected_ids'],
                improvement=r['brier']-t['brier'],temporal_p=t['probability'],recency_p=r['probability'],outcome=t['outcome']))
    usage=[]
    for p in OUT.rglob('response.json'):
        x=json.loads(p.read_text());usage.append(x)
    obj=dict(dataset=json.loads((DATA/'freeze.json').read_text()),summary=summary,phase_summary=phase_summary,
        comparisons=comparisons,rows=rows,older_evidence_cases=sorted(patterns,key=lambda x:x['improvement'],reverse=True),
        older_evidence_descriptive_counts=dict(total=len(patterns),helped_at_least_002=sum(p['improvement']>=.02 for p in patterns),
            harmed_at_least_002=sum(p['improvement']<=-.02 for p in patterns),small_or_tied=sum(abs(p['improvement'])<.02 for p in patterns)),
        missing=missing,complete=not missing,all_tasks_successful=not missing and not failures,
        task_failures=failures,failure_score_bounds=failure_bounds,
        n_condition_results=sum(r['condition']!='market_price' for r in rows),
        n_used_source_documents=len({e['id'] for c in cases.values() for e in c['evidence']}),
        n_actual_model_calls=len(usage),
        citation_errors=[dict(case_id=r['case_id'],condition=r['condition'],ids=r['unsupported_citation_ids']) for r in rows if r.get('unsupported_citation_ids')],
        partial_format_responses=[dict(case_id=r['case_id'],condition=r['condition'],parse_status=r.get('parse_status'),selector_parse_status=r.get('selector_parse_status')) for r in rows if r.get('parse_status','complete')!='complete' or r.get('selector_parse_status','complete') not in ['complete','not_applicable']],
        transport_usage=dict(input_tokens=sum(u.get('input_tokens',0) for x in usage for u in x.get('usage',[])),
            output_tokens=sum(u.get('output_tokens',0) for x in usage for u in x.get('usage',[])),
            cached_input_tokens=sum(u.get('cached_input_tokens',0) for x in usage for u in x.get('usage',[]))),
        interpretation='Retrospective single-family, source-passage information selection PoC. Not intrinsic news utility or uncontaminated ex-ante alpha.')
    save(OUT/'report.json',obj)
    print(json.dumps({k:obj[k] for k in ['complete','n_actual_model_calls','summary','older_evidence_descriptive_counts']},indent=2))
    return obj

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset-name',default='market_information_poc_v1');a=p.parse_args()
    DATA=ROOT/'data'/a.dataset_name;OUT=ROOT/'result'/a.dataset_name
    report()
