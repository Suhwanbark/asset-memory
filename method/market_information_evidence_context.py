"""Offline provenance: market state when each evidence item became available.

Does not modify frozen inference cases or expose future prices. Missing histories
remain missing. This annotation was added after forecasting and is exploratory.
"""
import json
from collections import Counter
from market_information_sources import ROOT,RAW,dt,save
from market_information_dataset import usable_candles,phase_label

def annotate(name):
    d=ROOT/'data'/name
    cases=json.loads((d/'cases.json').read_text())['cases']
    labels={x['case_id']:x for x in json.loads((d/'labels.json').read_text())['labels']}
    rows=[];cache={}
    for c in cases:
        ticker=c['id'].split('__')[0]
        prefix='news_candles_' if 'news' in name else 'candles_'
        path=RAW/(prefix+ticker+'.json')
        candles=json.loads(path.read_text())['candlesticks']
        for e in c['evidence']:
            key=(ticker,e['id'])
            if key not in cache:
                at=dt(e['available_at'])
                state=phase_label(usable_candles(candles,at),at)
                history=state.pop('history',[])
                assert all(x['t']<=at.timestamp() for x in history)
                cache[key]=state
            rows.append(dict(case_id=c['id'],evidence_id=e['id'],available_at=e['available_at'],
                at_availability=cache[key],at_forecast_phase=labels[c['id']]['phase']['phase']))
    obj=dict(annotation_timing='Post-forecast exploratory annotation; not supplied to original models.',
        limitation='Daily traded probabilities only; date-only sources use next-UTC-day availability. Does not infer phase before archived market history.',
        n_unique_market_evidence_pairs=len(cache),phase_counts_unique=dict(Counter(s['phase'] for s in cache.values())),rows=rows)
    save(d/'evidence_market_context.json',obj)
    print(name,obj['phase_counts_unique'])

if __name__=='__main__':
    for name in ['market_information_poc_v1','market_information_news_poc_v1']:annotate(name)
