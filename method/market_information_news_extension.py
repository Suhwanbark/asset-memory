"""Prepare a separately labeled exploratory news replication; no utility gold."""
import json,re
from collections import defaultdict,Counter
from datetime import timedelta
from pathlib import Path
import numpy as np
import tiktoken
from market_information_sources import DATA,ROOT,RAW,API,fetch_json,save,sha,dt
from market_information_dataset import usable_candles,phase_label

DEST=ROOT/'data/market_information_news_poc_v1'
ENC=tiktoken.get_encoding('o200k_base')
ALLOWED=re.compile(r'^KX(?:CPI-|U3-|PAYROLLS-|GDP-|TESLA-|TESLAPROD-|INXY-|NASDAQ100Y-|TNOTED-|EGGS-|CBDECISIONJAPAN-)')

def build():
    candidates={c['market_id']:c for c in json.loads((DATA/'news_extension_candidates.json').read_text())['candidates']}
    market_rows=json.loads((DATA/'news_extension_market_audit.json').read_text())['markets']
    by_event=defaultdict(list);exclusions=[]
    for r in market_rows:
        if 'metadata' not in r or not ALLOWED.match(r['market_id']):continue
        by_event[r['event_id']].append(r)
    cases=[];labels=[];cards_all={}
    for event,rs in sorted(by_event.items()):
        # No outcome-dependent contract choice: first eligible lexicographic ticker.
        accepted=False
        for row in sorted(rs,key=lambda x:x['market_id']):
            m=row['metadata'];ticker=m['ticker'];close=dt(m['close_time']);cut=close-timedelta(days=2)
            raw_news=candidates[ticker]['news']
            news=[]
            for n in raw_news:
                pub=dt(n['published_at']);text=re.sub(r'\s+',' ',n['title']+'\n'+(n.get('description') or '')).strip()
                if not cut-timedelta(days=120)<=pub<=cut or len(text)<50:continue
                encoded=ENC.encode(text)[:256];text=ENC.decode(encoded)
                cid='news_'+sha(n['url'])[:16]
                source_path=DEST/'sources'/(cid+'.json');save(source_path,n)
                item=dict(id=cid,title=n['title'],published_date=pub.date().isoformat(),available_at=pub.isoformat(),
                    source_type='archived_news_title_description',text=text,tokens=len(encoded),url=n['url'],
                    source_text_path=str(source_path),source_text_sha256=sha(source_path.read_bytes()),excerpt_sha256=sha(text))
                news.append(item)
            if len(news)<8 or len({n['published_date'] for n in news})<3:
                exclusions.append(dict(market_id=ticker,reason='fewer than8 preclose news /3 publication dates'));continue
            news.sort(key=lambda n:(n['available_at'],n['id']))
            params={'start_ts':int(max(dt(m['open_time']),close-timedelta(days=125)).timestamp()),'end_ts':int(close.timestamp()),'period_interval':1440}
            data=fetch_json(API+f'/historical/markets/{ticker}/candlesticks','news_candles_'+ticker+'.json',params)
            market_cases=[];market_labels=[]
            # Standard calendar leads fixed independently of news content or performance.
            for lead in [21,7,2]:
                cutoff=(close-timedelta(days=lead)).replace(hour=0,minute=0,second=0,microsecond=0)
                ns=[n for n in news if dt(n['available_at'])<=cutoff and dt(n['available_at'])>=cutoff-timedelta(days=120)]
                if len(ns)<6 or cutoff<dt(m['open_time']):continue
                if len(ns)>24:ns=[ns[i] for i in np.linspace(0,len(ns)-1,24).round().astype(int)]
                phase=phase_label(usable_candles(data['candlesticks'],cutoff),cutoff)
                if phase['phase'] in ['missing','insufficient_activity']:continue
                cid=ticker+f'__d{lead:02d}';public=[{k:n[k] for k in ['id','title','published_date','available_at','source_type','text','tokens']} for n in ns]
                public.sort(key=lambda n:sha(cid+'|'+n['id']))
                q=m['title']+' Outcome definition: '+m.get('yes_sub_title','')
                market_cases.append(dict(id=cid,event_id=event,meeting_date=close.date().isoformat(),cutoff=cutoff.isoformat(),
                    question=q,original_market_question=m['title'],lead_days=lead,evidence=public,select_k=2,
                    body_token_budget=512,budget_kind='at_most',split='main'))
                market_labels.append(dict(case_id=cid,event_id=event,outcome=int(m['result']=='yes'),close_time=m['close_time'],
                    settlement_ts=m.get('settlement_ts'),market_ticker=ticker,phase=phase,source_ids=[n['id'] for n in ns]))
                for n in ns:cards_all[n['id']]=n
            if len(market_cases)<2:
                exclusions.append(dict(market_id=ticker,reason='fewer than two eligible standard cutoffs'));continue
            cases.extend(market_cases);labels.extend(market_labels);accepted=True
            break
        if not accepted:exclusions.append(dict(event_id=event,reason='no eligible contract'))
    obj={'cases':cases,'design':{'selection':'non-FOMC fixed-time macro/financial/company quantity targets only; >=2 calendar cutoffs',
        'body_budget':'at most 512 tokens / exactly2 original news snippets; no padding or invented expansion','news_archive_revision':'6358edee035750dd635de4e6c3fea1668cd0d8fe'}}
    save(DEST/'cases.json',obj);save(DEST/'labels.json',{'labels':labels});save(DEST/'evidence_cards.json',{'cards':list(cards_all.values())})
    save(DEST/'exclusions.json',{'exclusions':exclusions})
    freeze=dict(cases_sha256=sha((DEST/'cases.json').read_bytes()),labels_sha256=sha((DEST/'labels.json').read_bytes()),
        evidence_sha256=sha((DEST/'evidence_cards.json').read_bytes()),code_sha256=sha(Path(__file__).read_bytes()),
        n_events=len({c['event_id'] for c in cases}),n_snapshots=len(cases),n_source_documents=len(cards_all),
        phase_counts=dict(Counter(x['phase']['phase'] for x in labels)),main_event_count=len({c['event_id'] for c in cases}),
        limitations=['Exploratory SWM-Bench news-reuse extension; upstream samples around price moves',
                    'Archived headline/description only; full historical article versions not verified',
                    'Variable evidence lengths within common cap','Retrospective contamination','Small cross-domain sample'],
        not_pooled_with_primary=True)
    save(DEST/'freeze.json',freeze)
    print(json.dumps(freeze,indent=2));print([c['event_id'] for c in cases if c['lead_days']==2])

if __name__=='__main__':build()
