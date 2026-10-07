"""Freeze real event/cutoff inputs separately from outcome/phase labels."""
from __future__ import annotations
import json
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from market_information_sources import DATA, ROOT, RAW, dt, save, sha

ENC=tiktoken.get_encoding('o200k_base')
CARD_TOKENS=256
TOP_K=2
MAX_CANDIDATES=24
LOOKBACK_DAYS=90
LEADS=[21,7,2]
PASSAGE_QUERY='Federal Reserve FOMC monetary policy interest rates rate cuts hikes inflation price stability labor employment unemployment economic outlook growth restrictive policy'

def make_excerpt(doc):
    text=Path(doc['text_path']).read_text()
    assert sha(text)==doc['text_sha256']
    ps=[re.sub(r'\s+',' ',p).strip() for p in doc['paragraphs'] if len(p.strip())>=60
        and not re.match(r'^[\w\-]+, .{0,220}?\((?:19\d\d|20\d\d|forthcoming)\)',p)]
    if not ps:return None
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    x=vectorizer.fit_transform(ps)
    scores=(x@vectorizer.transform([PASSAGE_QUERY]).T).toarray().ravel()
    # Deterministic topic-based passages, before model/outcome inspection.
    selected=[]
    for idx in np.argsort(-scores,kind='stable'):
        selected.append(int(idx))
        if len(ENC.encode('\n[...]\n'.join(ps[i] for i in sorted(selected))))>=CARD_TOKENS:break
    excerpt='\n[...]\n'.join(ps[i] for i in sorted(selected))
    ids=ENC.encode(excerpt)
    if len(ids)<CARD_TOKENS:return None
    excerpt=ENC.decode(ids[:CARD_TOKENS])
    if len(ENC.encode(excerpt))!=CARD_TOKENS:return None
    return dict(id=doc['id'],published_date=doc['published_date'],available_at=doc['available_at'],
                title=doc['title'],url=doc['url'],source_type=doc['source_type'],
                text=excerpt,tokens=CARD_TOKENS,excerpt_sha256=sha(excerpt),
                source_text_path=doc['text_path'],source_text_sha256=doc['text_sha256'],
                passage_indices=sorted(selected))

def usable_candles(candles,cutoff):
    rows=[]
    for c in candles:
        if c['end_period_ts']>cutoff.timestamp():continue
        price=c.get('price',{}).get('close',c.get('price',{}).get('close_dollars'))
        if price is None:continue
        p=float(price);v=float(c.get('volume',c.get('volume_fp')) or 0)
        if not 0<=p<=1:raise ValueError('Kalshi dollars probability out of range')
        rows.append(dict(t=c['end_period_ts'],p=p,volume=v))
    return sorted(rows,key=lambda r:r['t'])

def phase_label(rows,cutoff):
    if not rows:return {'phase':'missing','p':None}
    recent=[r for r in rows if r['t']>=(cutoff-timedelta(days=7)).timestamp()]
    stale=(cutoff.timestamp()-rows[-1]['t'])/86400
    if len(recent)<3 or stale>2:return {'phase':'insufficient_activity','p':rows[-1]['p'],'staleness_days':stale}
    prices=[r['p'] for r in recent]
    active=sum(r['volume']>0 for r in recent)>=3
    spread=max(prices)-min(prices)
    extreme=min(prices)>=.85 or max(prices)<=.15
    phase='consensus' if active and extreme and spread<=.10 else ('fluctuation' if active and spread>=.15 else 'intermediate')
    # A reversal is observed strictly from the past, after a prior consensus.
    prior_consensus=False
    for i in range(6,len(rows)):
        window=rows[i-6:i+1]
        if window[-1]['t']>=recent[0]['t']:break
        wp=[r['p'] for r in window]
        if sum(r['volume']>0 for r in window)>=3 and max(wp)-min(wp)<=.10 and (min(wp)>=.85 or max(wp)<=.15):
            if abs(rows[-1]['p']-wp[-1])>=.20:prior_consensus=True
    if prior_consensus and phase=='fluctuation':phase='reversal'
    return dict(phase=phase,p=rows[-1]['p'],seven_day_range=spread,active_days=sum(r['volume']>0 for r in recent),
                last_market_time=rows[-1]['t'],staleness_days=stale,history=rows)

def build():
    docs=[]
    for name in ['speech_sources.json','official_sources.json']:
        docs+=json.loads((DATA/name).read_text())['documents']
    excluded_docs=[];cards=[]
    for doc in docs:
        if doc.get('error'):excluded_docs.append({'id':doc['id'],'reason':doc['error']});continue
        card=make_excerpt(doc)
        if card:cards.append(card)
        else:excluded_docs.append({'id':doc['id'],'reason':'cannot form exact 256-token evidence card'})
    cards.sort(key=lambda c:(c['available_at'],c['id']))
    universe=json.loads((DATA/'market_universe.json').read_text())
    cases=[];labels=[];exclusions=[]
    for market in universe['markets']:
        ticker=market['ticker'];path=RAW/('candles_'+ticker+'.json')
        if not path.exists():exclusions.append({'ticker':ticker,'reason':'missing candles'});continue
        candles=json.loads(path.read_text()).get('candlesticks',[])
        if not candles:exclusions.append({'ticker':ticker,'reason':'no historical candles'});continue
        close=dt(market['close_time'])
        for lead in LEADS:
            cutoff=(close-timedelta(days=lead)).replace(hour=0,minute=0,second=0,microsecond=0)
            cid=ticker+f'__d{lead:02d}'
            if cutoff<dt(market['open_time']):
                exclusions.append({'case_id':cid,'reason':'market not open at cutoff'});continue
            available=[c for c in cards if cutoff-timedelta(days=LOOKBACK_DAYS)<=dt(c['available_at'])<=cutoff]
            if len(available)<6:
                exclusions.append({'case_id':cid,'reason':'fewer than six evidence cards','n':len(available)});continue
            if len(available)>MAX_CANDIDATES:
                indices=np.linspace(0,len(available)-1,MAX_CANDIDATES).round().astype(int)
                available=[available[i] for i in indices]
            phase=phase_label(usable_candles(candles,cutoff),cutoff)
            if phase['phase'] in ['missing','insufficient_activity']:
                exclusions.append({'case_id':cid,'reason':'inadequate historical price/activity','phase':phase['phase']});continue
            # Prevent metadata/schema showing future resolution and price.
            public=[{k:c[k] for k in ['id','published_date','available_at','title','source_type','text','tokens']} for c in available]
            # Same deterministic shuffled presentation for every selector.
            public.sort(key=lambda c:sha(cid+'|'+c['id']))
            question=f"Will the Federal Reserve leave its target federal funds rate range unchanged at the scheduled {close.date().isoformat()} FOMC meeting?"
            case=dict(id=cid,event_id=ticker,meeting_date=close.date().isoformat(),cutoff=cutoff.isoformat(),
                question=question,original_market_question=market['title'],lead_days=lead,
                evidence=public,select_k=TOP_K,body_token_budget=TOP_K*CARD_TOKENS)
            cases.append(case)
            labels.append(dict(case_id=cid,event_id=ticker,outcome=int(market['result']=='yes'),
                outcome_provenance='Kalshi finalized contract result; no manual news-utility gold',
                market_ticker=ticker,close_time=market['close_time'],settlement_ts=market.get('settlement_ts'),
                phase=phase,source_ids=[c['id'] for c in available]))
    cases.sort(key=lambda c:(c['meeting_date'],-c['lead_days']))
    events=sorted({c['event_id'] for c in cases},key=lambda e:min(c['meeting_date'] for c in cases if c['event_id']==e))
    # Earliest two events are engineering smoke. Remaining untouched events are main PoC.
    for c in cases:c['split']='smoke' if c['event_id'] in events[:2] else 'main'
    obj=dict(cases=cases,design=dict(card_tokens=CARD_TOKENS,select_k=TOP_K,lookback_days=LOOKBACK_DAYS,
             max_candidates=MAX_CANDIDATES,leads=LEADS,passage_query=PASSAGE_QUERY))
    case_path=DATA/'cases.json'
    if (DATA/'freeze.json').exists():
        frozen=json.loads((DATA/'freeze.json').read_text())
        if sha(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')!=frozen['cases_sha256']:
            raise RuntimeError('Dataset already frozen; use a new version')
    save(case_path,obj);save(DATA/'labels.json',{'labels':labels});save(DATA/'evidence_cards.json',{'cards':cards})
    save(DATA/'exclusions.json',{'markets_snapshots':exclusions,'documents':excluded_docs})
    manifest=dict(cases_sha256=sha(case_path.read_bytes()),labels_sha256=sha((DATA/'labels.json').read_bytes()),
        evidence_sha256=sha((DATA/'evidence_cards.json').read_bytes()),code_sha256=sha(Path(__file__).read_bytes()),
        n_events=len(events),n_snapshots=len(cases),n_source_documents=len(cards),
        phase_counts=dict(Counter(l['phase']['phase'] for l in labels)),split_counts=dict(Counter(c['split'] for c in cases)),
        main_event_count=len(events[2:]),sources='Official Kalshi API + official Federal Reserve dated archives',
        unused_archive='SWM-Bench inspected for follow-up; its news/attributions are NOT in this primary dataset',
        limitations=['Retrospective model knowledge contamination','One monetary-policy event family','Official-source passages, not complete heterogeneous news stream',
                    'Shared source windows and serially correlated monetary policy','Exact evidence body budget; metadata token counts may differ'])
    save(DATA/'freeze.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    import argparse,shutil
    p=argparse.ArgumentParser();p.add_argument('--preinference-revision');a=p.parse_args()
    if a.preinference_revision:
        assert not list((ROOT/'result/market_information_poc_v1').glob('*/*/result.json')), 'Cannot revise after forecasting'
        backup=DATA/'preinference_revisions'/a.preinference_revision
        backup.mkdir(parents=True,exist_ok=False)
        for name in ['cases.json','labels.json','evidence_cards.json','exclusions.json','freeze.json']:
            if (DATA/name).exists():shutil.move(str(DATA/name),str(backup/name))
        save(backup/'revision.json',{'reason':a.preinference_revision,'model_forecasts_existed':False})
    build()
