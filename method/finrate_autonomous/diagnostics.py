"""Actual retrieval behavior and question-blind bank costs, without heldout tuning."""
from collections import Counter,defaultdict
import json

from finrate_final_evaluation import online_usage
from finrate_poc.runtime import digest,write_json
from .config import BASE,POLICY,SEED


def collect(split='dev'):
    report=json.loads((BASE/split/'report.json').read_text());rows=[]
    for judged in report['rows']:
        folder=BASE/split/judged['q_id']/judged['condition'];p=json.loads((folder/'prediction.json').read_text())
        counts=Counter();source_ids=set();note_ids=set()
        for path in folder.glob('agent/turns/*/tool_*.json'):
            event=json.loads(path.read_text());action=event['action']
            try:args=json.loads(action['arguments_json'])
            except ValueError:continue
            scope=args.get('scope','workspace');counts[action['tool']+'_'+scope]+=1
            result=event['result'];items=result.get('items',[result])
            for item in items:
                if item.get('source_id'):source_ids.add(item['source_id'])
                elif item.get('id') and scope=='workspace' and judged['condition'] in {'bank_agent','flat_bank_agent'}:note_ids.add(item['id'])
        rows.append({'q_id':judged['q_id'],'task':judged['task'],'condition':judged['condition'],
          'status':p['status'],'correct':judged['correct'],'key_point_fraction':judged['key_point_fraction'],
          'tool_counts':dict(counts),'observed_original_sources':len(source_ids),'memory_items_observed':len(note_ids),
          'invalid_citation_count':len(p.get('invalid_citation_ids',[])),
          'finish_selected_by_model':p.get('finish_selected_by_model'),
          'model_calls':p.get('model_calls'),'text_tokens':p.get('online_text_tokens')})
    import numpy as np
    rng=np.random.default_rng(int(digest(SEED+'bootstrap')[:8],16));paired={}
    for task in ['all',*POLICY['tasks']]:
        for control in ['raw_agent','chunk_agent','flat_bank_agent','hybrid']:
            by=defaultdict(dict)
            for r in rows:
                if task=='all' or r['task']==task:by[r['q_id']][r['condition']]=r
            differences=[];unknown=0
            for q,arms in by.items():
                a,b=arms['bank_agent']['key_point_fraction'],arms[control]['key_point_fraction']
                if a is None or b is None:unknown+=1;continue
                differences.append(a-b)
            values=np.asarray(differences)
            if len(values):
                boot=rng.choice(values,size=(10000,len(values)),replace=True).mean(axis=1)
                paired[task+'/'+control]={'n_assessed_pairs':len(values),'unassessed_pairs':unknown,
                  'mean_key_point_difference':float(values.mean()),'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    result={'split':split,'rows':rows,'paired':paired,
      'limitations':'Small exploratory sample; paired intervals are descriptive, not a multiplicity-adjusted confirmation. Source grounding is separate from reference matching.'}
    if (BASE/'reader_banks/manifest.json').exists():
        result['bank_build_costs']={asset:online_usage(BASE/'banks'/asset) for asset in json.loads((BASE/'reader_banks/manifest.json').read_text())['assets']}
    write_json(BASE/split/'diagnostics.json',result,immutable=True);return result


if __name__=='__main__':collect()
