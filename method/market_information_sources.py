"""Read-only public source collection. No model calls, trading, or credentials."""
from __future__ import annotations
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/market_information_poc_v1'
RAW = DATA / 'raw'
REVISION = '6358edee035750dd635de4e6c3fea1668cd0d8fe'
API = 'https://api.elections.kalshi.com/trade-api/v2'

def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def fetch(url, name, params=None):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    meta = path.with_suffix(path.suffix + '.meta.json')
    if path.exists() and meta.exists():
        m = json.loads(meta.read_text())
        assert m['requested_url'] == url and m['params'] == params
        assert sha(path.read_bytes()) == m['sha256']
        return path, m
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Less than 20 GiB free disk')
    for attempt in range(3):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(2 * (attempt+1))
            continue
        break
    path.write_bytes(r.content)
    m = dict(requested_url=url, params=params, response_url=r.url,
             retrieved_utc=datetime.now(timezone.utc).isoformat(), status=r.status_code,
             bytes=len(r.content), sha256=sha(r.content))
    save(meta, m)
    return path, m

def fetch_json(url, name, params=None):
    p, m = fetch(url, name, params)
    if m['status'] != 200:
        raise RuntimeError(f"HTTP {m['status']} for {url}; saved to {p}")
    return json.loads(p.read_text())

def collect_archive():
    base = 'https://huggingface.co/datasets/ulab-ai/swm-bench/resolve/' + REVISION
    paths = ['README.md', 'Qwen3.5-397B-attributed-data/train.jsonl',
             'Qwen3.5-397B-attributed-data/test_kalshi.jsonl',
             'Qwen3.5-397B-attributed-data/valid_subset150.jsonl']
    for rel in paths:
        p, m = fetch(base + '/' + rel, 'swm_' + rel.split('/')[-1])
        print('ARCHIVE', p.name, m['status'], m['bytes'], flush=True)
    save(DATA/'archive_provenance.json', dict(
        repository='ulab-ai/swm-bench', revision=REVISION, license='Apache-2.0 as stated in dataset card',
        allowed_fields=['market_id','event_id','question','description','categories','news'],
        prohibited_inference_fields=['attributions','target','future','z_score','outcome'],
        warning='Upstream dataset samples toward volatile market moves. News are archived titles/descriptions, not independently verified contemporaneous full articles. Do not use upstream posterior attribution scores as utility gold.'))

def dt(s):
    from dateutil.parser import isoparse
    return isoparse(s)

def collect_fed_markets():
    markets={}
    for route in ['historical/markets','markets']:
        params={'limit':1000,'series_ticker':'KXFEDDECISION'}
        if route=='markets': params['status']='settled'
        x=fetch_json(API+'/'+route, 'kalshi_'+route.replace('/','_')+'.json',params)
        if x.get('cursor'):raise RuntimeError('Unexpected pagination; implement before using universe')
        for m in x['markets']:
            if m['ticker'].endswith('-H0') and m.get('result') in ['yes','no']:
                markets[m['ticker']]=m
    # One no-change contract per meeting date. Dedup is independent of outcomes.
    by_date={}; exclusions=[]
    for m in sorted(markets.values(),key=lambda m:m['ticker']):
        date=m['close_time'][:10]
        if date in by_date:
            exclusions.append({'ticker':m['ticker'],'reason':'duplicate meeting date','kept':by_date[date]['ticker']})
            continue
        by_date[date]=m
    selected=sorted(by_date.values(),key=lambda m:m['close_time'])
    save(DATA/'market_universe.json',{'series':'KXFEDDECISION / historical FEDDECISION alias',
        'selection':'All available finalized no-change H0 contracts; one per meeting close date; no outcome-based sampling',
        'markets':selected,'exclusions':exclusions})
    def candles(m):
        end=dt(m['close_time']);start=max(dt(m['open_time']),end-timedelta(days=100))
        params={'start_ts':int(start.timestamp()),'end_ts':int(end.timestamp()),'period_interval':1440}
        route=(f"historical/markets/{m['ticker']}/candlesticks" if dt(m.get('settlement_ts',m['close_time']))<dt('2026-07-30T00:00:00Z')
               else f"series/KXFEDDECISION/markets/{m['ticker']}/candlesticks")
        x=fetch_json(API+'/'+route,'candles_'+m['ticker']+'.json',params)
        print('CANDLES',m['ticker'],len(x.get('candlesticks',[])),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(candles,m) for m in selected]):f.result()
    return selected

def collect_speeches():
    links={}
    for year in range(2023,2027):
        url=f'https://www.federalreserve.gov/newsevents/{year}-speeches.htm'
        path,meta=fetch(url,f'fed_speeches_{year}.html')
        if meta['status']!=200:raise RuntimeError(meta)
        soup=BeautifulSoup(path.read_bytes(),'html.parser')
        for a in soup.select('a[href]'):
            m=re.search(r'/speech/([^/]+?(\d{8})[a-z])\.htm$',a['href'])
            if not m:continue
            title=a.get_text(' ',strip=True)
            # Public topic selection, fixed without reading market outcomes/model responses.
            if not re.search(r'econom|outlook|monetary|inflation|labor|employment|policy|cut or skip|progress|balance|path|landing',title,re.I):continue
            date=datetime.strptime(m[2],'%Y%m%d').replace(tzinfo=timezone.utc)
            links[m[1]]={'id':'fed_'+m[1],'title':title,'published_date':date.date().isoformat(),
                'available_at':(date+timedelta(days=1)).isoformat(),
                'url':'https://www.federalreserve.gov'+a['href'],'source_type':'official_speech'}
    def one(item):
        path,meta=fetch(item['url'],item['id']+'.html')
        if meta['status']!=200:return dict(item,error=f"HTTP{meta['status']}")
        s=BeautifulSoup(path.read_bytes(),'html.parser')
        article=s.select_one('#article')
        if not article:return dict(item,error='article container missing')
        candidates=article.select('div.col-xs-12.col-sm-8.col-md-8')
        body=max(candidates,key=lambda d:len(d.get_text())) if candidates else article
        paragraphs=[]
        for p in body.find_all('p'):
            txt=p.get_text(' ',strip=True)
            if len(txt)<70 or 'Return to text' in txt or re.match(r'^\d+\.',txt):continue
            paragraphs.append(txt)
        text='\n\n'.join(paragraphs)
        if len(text)<200:return dict(item,error='body too short')
        out=DATA/'sources'/item['id'];out.mkdir(parents=True,exist_ok=True)
        (out/'source.txt').write_text(text)
        return dict(item,text_path=str(out/'source.txt'),text_sha256=sha(text),raw_path=str(path),
            raw_sha256=meta['sha256'],paragraphs=paragraphs,
            temporal_provenance='Official date-stamped archive fetched now; date-only publication delayed to next UTC day. Historical body immutability not independently guaranteed.')
    docs=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(one,x) for x in links.values()]
        for i,f in enumerate(as_completed(fs)):
            docs.append(f.result())
            if (i+1)%25==0:print('SPEECHES',i+1,'/',len(fs),flush=True)
    save(DATA/'speech_sources.json',{'documents':sorted(docs,key=lambda x:x['id'])})
    return docs

def collect_existing_official():
    old=ROOT/'data/temporal_evidence_poc_v1/sources'
    docs=[]
    for p in sorted(old.glob('fed_*/meta.json')):
        m=json.loads(p.read_text())
        text_path=p.parent/'source.txt';text=text_path.read_text()
        assert sha(text)==m['text_sha256']
        date=dt(m['publication_date']+'T00:00:00Z')
        docs.append(dict(id=m['id'],title='FOMC '+m['kind'],published_date=m['publication_date'],
            available_at=(date+timedelta(days=1)).isoformat(),url=m['url'],source_type='official_'+m['kind'],
            text_path=str(text_path),text_sha256=sha(text),paragraphs=text.split('\n\n'),
            temporal_provenance='Previously archived official dated document; availability delayed to next UTC day.'))
    # Additional 2026 statements, before and including this year's observed meetings.
    for date in ['20260128','20260318','20260429','20260617','20260729','20260916']:
        url=f'https://www.federalreserve.gov/newsevents/pressreleases/monetary{date}a.htm'
        path,meta=fetch(url,'fed_'+date+'_statement.html')
        if meta['status']!=200:continue
        soup=BeautifulSoup(path.read_bytes(),'html.parser');article=soup.select_one('#article')
        if not article:continue
        candidates=article.select('div.col-xs-12.col-sm-8.col-md-8')
        body=max(candidates,key=lambda d:len(d.get_text())) if candidates else article
        ps=[p.get_text(' ',strip=True) for p in body.find_all('p') if len(p.get_text(' ',strip=True))>70]
        text='\n\n'.join(ps);out=DATA/'sources'/('fed_'+date+'_statement');out.mkdir(parents=True,exist_ok=True)
        (out/'source.txt').write_text(text)
        day=datetime.strptime(date,'%Y%m%d').replace(tzinfo=timezone.utc)
        docs.append(dict(id='fed_'+date+'_statement',title='FOMC statement',published_date=day.date().isoformat(),
            available_at=(day+timedelta(days=1)).isoformat(),url=url,source_type='official_statement',
            text_path=str(out/'source.txt'),text_sha256=sha(text),paragraphs=ps,
            temporal_provenance='Official date-stamped archive fetched now; next-UTC-day availability.'))
    save(DATA/'official_sources.json',{'documents':docs})
    print('OFFICIAL_PRIOR_RECORDS',len(docs),flush=True)
    return docs

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--collect',choices=['archive','markets','speeches','official'],default='archive');a=p.parse_args()
    {'archive':collect_archive,'markets':collect_fed_markets,'speeches':collect_speeches,'official':collect_existing_official}[a.collect]()
