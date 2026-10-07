"""User-requested provisional scores without changing frozen pilot execution.

Grade each entire20-question arm once after its predictions finish. Store these
separate from the scheduled primary grading; never silently replace scores.
"""
from collections import Counter
import argparse
import json
import os
import time

from finrate_completed_pilot_v1 import BASE,verify
from finrate_autonomous_v10.transport import IsolatedLuna
from finrate_final_evaluation import grade_candidate
from finrate_poc.runtime import check_disk,cpu_only,file_digest,watchdog,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('condition',choices=['embedding','hybrid','chunk_agent','flat_bank_agent']);a=p.parse_args()
    cpu_only();check_disk();os.nice(10);verify()
    root=BASE/'provisional_scores'/a.condition;status=root/'status.json';stop=watchdog(status)
    qs=json.loads((BASE/'cohort/gold.json').read_text());model=IsolatedLuna(max_calls=len(qs))
    try:
        write_json(root/'manifest.json',{'condition':a.condition,'n':len(qs),'code_sha256':file_digest(__file__),
          'gold_sha256':file_digest(BASE/'cohort/gold.json'),'model':'gpt-5.6-luna',
          'grading':'Same grade_candidate prompt/schema as primary; separate single provisional assessment, not a replacement or best-of selection.',
          'all_questions_and_failures_retained':True,'heldout_access':False},immutable=True)
        while any(not (BASE/'dev'/q['q_id']/a.condition/'prediction.json').exists() for q in qs):
            check_disk();write_json(status,{'state':'waiting_for_all20_predictions','condition':a.condition,'pid':os.getpid()});time.sleep(10)
        rows=[]
        for q in qs:
            check_disk();folder=root/q['q_id'];dest=folder/'assessment.json'
            if dest.exists():row=json.loads(dest.read_text())
            else:
                write_json(status,{'state':'grading','condition':a.condition,'q_id':q['q_id'],'graded':len(rows),'pid':os.getpid()})
                prediction=BASE/'dev'/q['q_id']/a.condition/'prediction.json'
                pred=json.loads(prediction.read_text())
                row={'q_id':q['q_id'],'task':q['task'],'condition':a.condition,'prediction_sha256':file_digest(prediction),
                  'generation_status':pred['status'],**grade_candidate(q,pred,model,folder/'judge')}
                write_json(dest,row,immutable=True)
            rows.append(row)
            write_json(root/'progress.json',{'graded':len(rows),'total':len(qs),'labels':dict(Counter(r['label'] for r in rows))})
        summary={}
        for task in ['all','LT-QA','EC-QA']:
            group=[r for r in rows if task=='all' or r['task']==task];n=len(group);right=sum(r['correct'] is True for r in group)
            unknown=sum(r['correct'] is None for r in group);points=sum(r['key_point_fraction'] or 0 for r in group)
            summary[task]={'n':n,'correct':right,'unassessed':unknown,'generation_failures':sum(r['generation_status']=='failed' for r in group),
              'accuracy_lower':right/n,'accuracy_upper':(right+unknown)/n,'key_point_lower':points/n,'key_point_upper':(points+unknown)/n,
              'labels':dict(Counter(r['label'] for r in group))}
        write_json(root/'report.json',{'condition':a.condition,'rows':rows,'summary':summary,'provisional':True,
          'interpretation':'Same fixed20 pilot questions and same-family reference judge; source grounding and scheduled primary grading remain separate.'},immutable=True)
        write_json(status,{'state':'completed','condition':a.condition,'pid':os.getpid(),'finished':time.time()})
    except Exception as exc:
        write_json(status,{'state':'failed','error':str(exc),'pid':os.getpid()});raise
    finally:stop.set()


if __name__=='__main__':main()
