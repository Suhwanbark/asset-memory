"""Recover valid original grades rejected solely for a transport-fallback notice.

No model call, replacement prediction, or original artifact mutation occurs.
Replay the exact original response through the original grade validator.
"""
from collections import Counter
import json

from finrate_completed_pilot_v1 import BASE,verify
from finrate_final_evaluation import grade_candidate
from finrate_poc.runtime import digest,file_digest,write_json


class Replay:
    def __init__(self,folder):
        self.request=json.loads((folder/'request.json').read_text())
        self.events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.startswith('{')]
        items=[e['item'] for e in self.events if e.get('type')=='item.completed']
        if any(e.get('type')=='turn.failed' for e in self.events):raise ValueError('Terminal failure cannot be recovered as success')
        if sum(e.get('type')=='turn.completed' for e in self.events)!=1:raise ValueError('Require exactly one completed model turn')
        for item in items:
            if item.get('type') in {'agent_message','reasoning'}:continue
            if item.get('type')!='error' or not item.get('message','').startswith('Falling back from WebSockets to HTTPS transport.'):
                raise ValueError('Not solely a recoverable fallback notice')
        messages=[i['text'] for i in items if i.get('type')=='agent_message']
        if len(messages)!=1:raise ValueError('Require one unambiguous original final answer')
        self.text=messages[0]

    def invoke(self,prompt,folder,*,schema):
        if digest(prompt)!=self.request['prompt_sha256']:raise ValueError('Original grading prompt differs')
        if digest(json.dumps(schema,sort_keys=True))!=self.request['response_schema_sha256']:raise ValueError('Original grading schema differs')
        return self.text


def main():
    verify();qs=json.loads((BASE/'cohort/gold.json').read_text());conditions=['naive_llm','raw_agent','bank_agent']
    output=BASE/'provisional_scores/transport_reviewed';rows=[];audits=[]
    for c in conditions:
        if not (BASE/'provisional_scores'/c/'report.json').exists():raise ValueError('Wait for all20 provisional grades in each condition')
        for q in qs:
            path=BASE/'provisional_scores'/c/q['q_id'];original=json.loads((path/'assessment.json').read_text());row=original
            if original['label']=='UNASSESSED':
                try:
                    replay=Replay(path/'judge');pred=json.loads((BASE/'dev'/q['q_id']/c/'prediction.json').read_text())
                    normalized=grade_candidate(q,pred,replay,path/'no_external_call')
                    if normalized['label']=='UNASSESSED':raise ValueError(normalized['judge_failure'])
                    row={k:original[k] for k in ['q_id','task','condition','prediction_sha256','generation_status']}
                    row.update(normalized);row['recovered_original_grade']=True
                    audits.append({'q_id':q['q_id'],'condition':c,'state':'recovered',
                      'events_sha256':file_digest(path/'judge/events.jsonl'),'request_sha256':file_digest(path/'judge/request.json'),
                      'same_original_prompt_and_schema':True,'no_new_model_call':True,'no_native_tool_call':True})
                except Exception as exc:audits.append({'q_id':q['q_id'],'condition':c,'state':'not_recovered','reason':str(exc)})
            rows.append(row)
    summary={}
    for task in ['all','LT-QA','EC-QA']:
        summary[task]={}
        for c in conditions:
            group=[r for r in rows if r['condition']==c and (task=='all' or r['task']==task)];n=len(group)
            right=sum(r['correct'] is True for r in group);unknown=sum(r['correct'] is None for r in group);points=sum(r['key_point_fraction'] or 0 for r in group)
            summary[task][c]={'n':n,'correct':right,'unassessed':unknown,'generation_failures':sum(r['generation_status']=='failed' for r in group),
              'accuracy_lower':right/n,'accuracy_upper':(right+unknown)/n,'key_point_lower':points/n,'key_point_upper':(points+unknown)/n,
              'labels':dict(Counter(r['label'] for r in group))}
    import numpy as np
    by={(r['q_id'],r['condition']):r for r in rows};diffs=[]
    for q in qs:
        a,b=by[q['q_id'],'bank_agent']['key_point_fraction'],by[q['q_id'],'raw_agent']['key_point_fraction']
        if a is not None and b is not None:diffs.append(a-b)
    paired={}
    if diffs:
        rng=np.random.default_rng(20261005);samples=rng.choice(diffs,size=(10000,len(diffs)),replace=True).mean(axis=1)
        paired={'n':len(diffs),'mean_keypoint_difference':float(np.mean(diffs)),
          'descriptive_paired_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),
          'wins':sum(x>0 for x in diffs),'ties':sum(x==0 for x in diffs),'losses':sum(x<0 for x in diffs)}
    report={'rows':rows,'summary':summary,'bank_minus_raw':paired,'recovery_audit':audits,
      'original_artifacts_unchanged':True,'no_new_model_calls_for_recovery':True,
      'provisional':True,'scope':'HUSA+REEMF, fixed LT10+EC10, common restricted corpus; scheduled primary grading and source grounding remain separate.'}
    write_json(output/'report.json',report,immutable=True)
    print(json.dumps({'summary':summary,'bank_minus_raw':paired,'recovered':sum(r['state']=='recovered' for r in audits)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
