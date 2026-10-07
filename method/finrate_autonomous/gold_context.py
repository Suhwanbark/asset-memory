"""DEV-only oracle-retrieval diagnostic, excluded from primary method comparison.

Only the question and original gold documents reach answer generation, never
the reference answer or key points. This does not access heldout material.
"""
from collections import Counter
import json
import os
import time

from finrate_final_evaluation import grade_candidate,online_usage
from finrate_poc.model import answer_prompt,parse_json,token_count
from finrate_poc.runtime import check_disk,cpu_only,file_digest,watchdog,write_json
from .config import BASE,CORPUS
from .transport import IsolatedLuna
from .workspace import connect


def main():
    cpu_only();check_disk();os.nice(10)
    root=BASE/'gold_context_diagnostic';status=root/'status.json';stop=watchdog(status)
    gold=json.loads((BASE/'cohort/gold/dev.json').read_text());source=connect(CORPUS,True)
    model=IsolatedLuna(max_calls=len(gold)*2);rows=[]
    manifest={'split':'dev','purpose':'Diagnose retrieval loss versus answering/reference limitations, not an deployable baseline',
      'gold_cohort_sha256':file_digest(BASE/'cohort/gold/dev.json'),'model':'gpt-5.6-luna',
      'maximum_prompt_tokens':120000,'all_gold_originals_or_explicit_context_failure':True,
      'reference_answer_and_key_points_not_in_generation':True,'heldout_access':False,
      'module_sha256':file_digest(__file__)}
    write_json(root/'manifest.json',manifest,immutable=True)
    try:
        for q in gold:
            folder=root/q['q_id'];prediction=folder/'prediction.json'
            write_json(status,{'state':'answering_gold_context','q_id':q['q_id'],'pid':os.getpid()})
            if not prediction.exists():
                evidence=[]
                for sid in q['source_ids']:
                    doc=source.execute('SELECT title,text,asset,published FROM sources WHERE id=?',(sid,)).fetchone()
                    evidence.append({'id':f"{sid}:0:{len(doc['text'])}",'source_id':sid,'title':doc['title'],
                      'asset':doc['asset'],'published':doc['published'],'start':0,'end':len(doc['text']),'text':doc['text']})
                prompt=answer_prompt(q['question'],evidence)
                try:
                    if token_count(prompt)>manifest['maximum_prompt_tokens']:raise ValueError('Full gold context exceeds diagnostic limit; no partial oracle input substituted')
                    answer=parse_json(model.invoke(prompt,folder/'answer'))
                    row={'q_id':q['q_id'],'condition':'gold_context_diagnostic','status':'completed','answer':answer,'evidence':evidence}
                except Exception as e:row={'q_id':q['q_id'],'condition':'gold_context_diagnostic','status':'failed','error':str(e)}
                write_json(prediction,row,immutable=True)
        # Complete all diagnostic predictions before reference-based scoring.
        for q in gold:
            folder=root/q['q_id'];dest=folder/'assessment.json'
            write_json(status,{'state':'grading_gold_context','q_id':q['q_id'],'pid':os.getpid()})
            if dest.exists():row=json.loads(dest.read_text())
            else:
                row={'q_id':q['q_id'],'task':q['task'],**grade_candidate(q,json.loads((folder/'prediction.json').read_text()),model,folder/'blinded_judge'),
                  'usage':online_usage(folder)}
                write_json(dest,row,immutable=True)
            rows.append(row);print('GOLD_CONTEXT',q['q_id'],row['label'],flush=True)
        result={'n':len(rows),'rows':rows,'labels':dict(Counter(r['label'] for r in rows)),
          'correct':sum(r['correct'] is True for r in rows),'unassessed':sum(r['correct'] is None for r in rows),
          'key_point_mean_lower':sum(r['key_point_fraction'] or 0 for r in rows)/len(rows),
          'caution':'Oracle retrieval is diagnostic, not necessarily a mathematical performance upper bound; reference defects and long-context failures remain possible.'}
        write_json(root/'report.json',result,immutable=True)
        write_json(status,{'state':'completed','n':len(rows),'pid':os.getpid(),'finished':time.time()})
    except Exception as e:
        write_json(status,{'state':'failed','error':str(e),'pid':os.getpid()});raise
    finally:source.close();stop.set()

if __name__=='__main__':main()
