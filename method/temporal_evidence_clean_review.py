"""Correct an audited article boundary, rerun that whole case, retain prior runs.

Final reviewed cohort excludes the VanEck historical artifact with uncertain
temporal provenance. Unchanged cases retain their v2 evaluations; the corrected
Amazon case receives a fresh memory and every evaluation condition.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import temporal_evidence_eval as engine

ROOT=engine.ROOT
DATA=engine.DATA
V3=ROOT/'result/temporal_evidence_poc_v3'

def main():
    patch=json.loads((DATA/'corporate_clean_case_patch.json').read_text())
    cases=engine.load_cases()
    target=next(c for c in cases if c['id']=='corporate_amazon_irobot')
    replacement=patch['new_doc_metadata']
    target['documents']=[replacement if d['id']=='corp_irbt_20240129' else d for d in target['documents']]
    target['source_revision']='v3: publisher article body excludes separate dynamic recommended-news module'
    engine.save(V3/'corrected_case.json',target)
    engine.OUT=V3
    engine.BASE_OUT=V3
    engine.INSTRUCTION_VERSION='v2'
    engine.run_case(target,['full_history','naive_rag_top10','concise_summary'])
    # Gold semantics/probe wording are unchanged; source boundary alone corrected.
    raw_gold=[]
    for name in ['fed_gold.json','corporate_gold.json']:
        raw_gold+=json.loads((DATA/name).read_text())['labels']
    for g in raw_gold:
        if g['case_id']==target['id']:
            g['source_ids']=[replacement['id'] if s=='corp_irbt_20240129' else s for s in g['source_ids']]
    engine.save(V3/'reviewed_gold.json',{'labels':[g for g in raw_gold if g['case_id']!='corporate_vaneck_spot_bitcoin']})
    gold={(g['case_id'],g['probe_id']):g for g in raw_gold}
    included=[c for c in cases if c['id']!='corporate_vaneck_spot_bitcoin']
    aggregate=defaultdict(Counter); rows=[]; result_refs=[]; by_condition=defaultdict(list)
    for case in included:
        for condition in ['full_history','naive_rag_top10','concise_summary']:
            runroot=V3 if case['id']==target['id'] else ROOT/'result/temporal_evidence_poc_v2'
            path=runroot/case['id']/condition/'result.json'
            result=json.loads(path.read_text())
            result_refs.append(dict(case_id=case['id'],condition=condition,path=str(path),sha256=engine.sha(path.read_text())))
            by_condition[condition].append(result)
            for pred in result['judgments']:
                g=gold[(case['id'],pred['id'])]
                correct=pred['label']==g['label']
                for key in [condition,condition+'|'+g['category'],condition+'|'+case['event_family']]:
                    aggregate[key]['n']+=1;aggregate[key]['correct']+=int(correct)
                    aggregate[key]['gold_'+g['label']]+=1
                    aggregate[key][g['label']+'->'+pred['label']]+=1
                rows.append(dict(case_id=case['id'],condition=condition,probe_id=pred['id'],gold=g['label'],
                                 predicted=pred['label'],category=g['category'],correct=correct,
                                 reason=pred['reason'],gold_explanation=g['explanation']))
    summary={k:dict(v,accuracy=v['correct']/v['n']) for k,v in aggregate.items()}
    for condition,results in by_condition.items():
        summary[condition].update(case_count=len(results),
             mean_source_tokens=statistics.mean(x['source_tokens'] for x in results),
             mean_delivered_tokens=statistics.mean(x['delivered_tokens'] for x in results),
             mean_prompt_tokens=statistics.mean(x['prompt_tokens'] for x in results))
    artifact=dict(created_at=datetime.now(timezone.utc).isoformat(),event_count=len(included),
         probe_count=len([g for g in raw_gold if g['case_id']!='corporate_vaneck_spot_bitcoin']),
         case_ids=[c['id'] for c in included],summary=summary,rows=rows,
         errors=[r for r in rows if not r['correct']],result_references=result_refs,
         exclusions=[dict(case_id='corporate_vaneck_spot_bitcoin',reason='Launch PDF created/modified after historical cutoff; contemporaneous artifact provenance not established. Exclusion independent of model outcome; all original conditions were correct.')],
         correction=dict(case_id=target['id'],reason='Remove separately identified dynamic 2026 recommended-news module outside 2024 article; regenerate summary and reevaluate all three conditions.',source_patch=str(DATA/'corporate_clean_case_patch.json')),
         inference_scope='Official-event evidence validity, not a 29-market forecasting benchmark.',
         method='All unchanged v2 cases retained; corrected case rerun with same v2 instructions; no selective semantic retries.')
    engine.save(V3/'reviewed_report.json',artifact)
    print(json.dumps({k:v for k,v in summary.items() if '|' not in k},indent=2))

if __name__=='__main__': main()
