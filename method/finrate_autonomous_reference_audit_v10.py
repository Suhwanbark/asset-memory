"""Audit every fixed DEV reference, independently of method predictions.

This diagnostic never changes the cohort, original labels, or primary scores.
It is deliberately outside the frozen inference package.
"""
from collections import Counter
import json
import os
import time

from finrate_gold_audit import SCHEMA,prompt
from finrate_poc.model import parse_json,token_count
from finrate_poc.runtime import check_disk,cpu_only,file_digest,watchdog,write_json
from finrate_autonomous_v10.config import BASE,CORPUS
from finrate_autonomous_v10.transport import IsolatedLuna
from finrate_autonomous_v10.workspace import connect


def main():
    cpu_only();check_disk();os.nice(10)
    root=BASE/'development_reference_audit';status=root/'status.json';stop=watchdog(status)
    gold=json.loads((BASE/'cohort/gold/dev.json').read_text())
    assert [q['q_id'] for q in gold]==json.loads((BASE/'cohort/manifest.json').read_text())['selected_ids']['dev']
    db=connect(CORPUS,True);model=IsolatedLuna(max_calls=len(gold));rows=[]
    write_json(root/'manifest.json',{'code_sha256':file_digest(__file__),
      'prompt_schema_code_sha256':file_digest(__import__('finrate_gold_audit').__file__),
      'gold_sha256':file_digest(BASE/'cohort/gold/dev.json'),
      'maximum_prompt_tokens':120000,'model':'gpt-5.6-luna','split':'dev',
      'method_predictions_or_scores_available':False,'heldout_access':False,
      'all_fixed_questions_retained':True,'gold_and_scores_unchanged':True},immutable=True)
    try:
        for q in gold:
            check_disk();folder=root/q['q_id'];dest=folder/'assessment.json'
            write_json(status,{'state':'auditing','q_id':q['q_id'],'pid':os.getpid()})
            if dest.exists():rows.append(json.loads(dest.read_text()));continue
            sources=[{'source_id':sid,**dict(db.execute('SELECT title,text FROM sources WHERE id=?',(sid,)).fetchone())} for sid in q['source_ids']]
            text=prompt(q,sources);size=token_count(text)
            base={'q_id':q['q_id'],'source_ids':q['source_ids'],'full_prompt_tokens':size,'sources_truncated':False}
            if size>120000:row={**base,'state':'unassessed_source_limit'}
            else:
                try:
                    response=parse_json(model.invoke(text,folder/'judge',schema=SCHEMA))
                    if [p['index'] for p in response['key_points']]!=list(range(len(q['key_points']))):
                        raise ValueError('Key point count/order mismatch')
                    by={s['source_id']:s['text'] for s in sources};verified=[];invalid=[]
                    for point in response['key_points']:
                        for evidence in point['evidence']:
                            sid=evidence['source_id'];quote=evidence['quote'];start=by.get(sid,'').find(quote) if quote else -1
                            if sid not in by or start<0:
                                invalid.append({'key_point_index':point['index'],**evidence})
                            else:verified.append({'key_point_index':point['index'],'source_id':sid,'start':start,'end':start+len(quote)})
                    row={**base,'state':'assessed',**response,'verified_exact_quotes':verified,
                         'invalid_quote_proposals':invalid,'requires_review_if_invalid_quotes':bool(invalid)}
                except Exception as exc:row={**base,'state':'assessment_failed','error':type(exc).__name__+': '+str(exc)}
            write_json(dest,row,immutable=True);rows.append(row)
            print('REFERENCE_AUDIT',q['q_id'],row['state'],row.get('overall',''),flush=True)
        report={'n':len(rows),'rows':rows,'states':dict(Counter(r['state'] for r in rows)),
                'overall':dict(Counter(r.get('overall',r['state']) for r in rows)),
                'limitation':'Same-family model concerns require source review; not confirmed benchmark errors or replacement scores.'}
        write_json(root/'report.json',report,immutable=True)
        write_json(status,{'state':'ready_for_review','n':len(rows),'pid':os.getpid(),'finished':time.time()})
    except Exception as exc:
        write_json(status,{'state':'failed','error':str(exc),'pid':os.getpid()});raise
    finally:db.close();stop.set()


if __name__=='__main__':main()
