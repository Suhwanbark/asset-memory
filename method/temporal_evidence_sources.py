"""Archive real FOMC histories and build a source-grounded temporal diagnostic.

This stratum is OFFICIAL EVENT DATA, not verified prediction-market contracts.
Fixed inclusion: all 24 regular 2023-2025 FOMC decisions, never model-selected.
"""
from __future__ import annotations
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/temporal_evidence_poc_v1'
BASE = 'https://www.federalreserve.gov'
CALENDAR = BASE + '/monetarypolicy/fomccalendars.htm'

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()

def download(url, source_id, publication_date, kind):
    folder = OUT / 'sources' / source_id
    meta_file = folder / 'meta.json'
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    if kind == 'minutes':
        body = soup.select_one('#article')
    else:
        nodes = soup.select('div.col-md-8')
        body = max(nodes, key=lambda x: len(x.get_text())) if nodes else None
    if body is None:
        raise ValueError(f'Missing main body {url}')
    for node in body.select('script,style,svg,noscript'):
        node.decompose()
    text = body.get_text('\n', strip=True).replace('\xa0', ' ')
    if len(text) < 1400:
        raise ValueError(f'Short or blocked source {url}: {len(text)}')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'source.html').write_bytes(response.content)
    (folder / 'source.txt').write_text(text)
    meta = dict(id=source_id, url=url, publication_date=publication_date, kind=kind,
                title='FOMC '+kind, text_path=str(folder/'source.txt'),
                html_sha256=sha(response.content), text_sha256=sha(text), characters=len(text),
                http_status=response.status_code, retrieved_at=datetime.now(timezone.utc).isoformat(),
                scope='complete official main document; no claim-based excerpt selection')
    write_json(meta_file, meta)
    return meta

def parse_range(text):
    flat = re.sub(r'\s+', ' ', text).replace('‑','-').replace('–','-')
    match = re.search(r'target range for the federal funds rate (?:by [\d /.-]+ percentage points? )?(?:at|to) ([\d /.-]+) to ([\d /.-]+) percent', flat)
    if not match:
        raise ValueError('Cannot parse official target range')
    return match.group(1).strip(), match.group(2).strip(), match.group(0)

def calendar_rows(html):
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for node in soup.select('.fomc-meeting'):
        links = [a['href'] for a in node.select('a[href]')]
        statements = [u for u in links if re.search(r'/monetary20\d{6}a\.htm$', u)]
        if not statements:
            continue
        stamp = re.search(r'monetary(\d{8})a', statements[0]).group(1)
        if not ('2022' <= stamp[:4] <= '2025'):
            continue
        minutes = [u for u in links if re.search(r'/fomcminutes\d{8}\.htm$', u)]
        released = re.search(r'Released ([A-Za-z]+ \d{2}, \d{4})', node.get_text(' ', strip=True))
        if not minutes or not released:
            continue
        rows.append(dict(stamp=stamp, date=datetime.strptime(stamp,'%Y%m%d').date().isoformat(),
                         statement_url=BASE+statements[0], minutes_url=BASE+minutes[0],
                         minutes_date=datetime.strptime(released.group(1),'%B %d, %Y').date().isoformat()))
    rows.sort(key=lambda x:x['date'])
    return rows

def make_case(rows, index, sources):
    row = rows[index]
    history_rows = rows[max(0,index-3):index+1]
    docs = [sources['fed_'+r['stamp']+'_statement'] for r in history_rows]
    docs.append(sources['fed_'+row['stamp']+'_minutes'])
    docs.sort(key=lambda x:(x['publication_date'], x['id']))
    current = parse_range(Path(sources['fed_'+row['stamp']+'_statement']['text_path']).read_text())
    prev_row = rows[index-1]
    prev_doc = sources['fed_'+prev_row['stamp']+'_statement']
    previous = parse_range(Path(prev_doc['text_path']).read_text())
    old_range = f'{previous[0]} to {previous[1]} percent'
    probe_specs = [
        ('current_prior_range', 'current', f'The current federal funds target range is {old_range}.',
         'supported' if previous[:2] == current[:2] else 'contradicted', 'policy_state', [docs[-2]['id'],docs[-1]['id']]),
        ('historical_prior_range', 'historical', f'On {prev_row["date"]}, the FOMC set or maintained its target range at {old_range}.',
         'supported', 'historical_record', [prev_doc['id']]),
        ('long_run_inflation', 'current', 'The FOMC still seeks inflation at a rate of 2 percent over the longer run.',
         'supported', 'persistent_objective', [docs[-2]['id']]),
        ('future_commitment', 'current', f'The policy target range will be exactly {old_range} immediately after the next scheduled FOMC meeting.',
         'insufficient', 'unconfirmed_future_commitment', [docs[-2]['id'],docs[-1]['id']]),
    ]
    # Opaque ids/order hide scoring categories, not the meaning of the claims.
    case_id = 'fomc_'+row['stamp']
    sorted_specs = sorted(probe_specs, key=lambda x:sha(case_id+x[0]))
    probes, gold = [], []
    for i, (name, scope, claim, label, category, evidence_ids) in enumerate(sorted_specs):
        pid = f'p{i+1}'
        probes.append(dict(id=pid, scope=scope, claim=claim))
        gold.append(dict(case_id=case_id, probe_id=pid, label=label, category=category,
                         source_ids=evidence_ids, explanation=(f'Current target {current[0]} to {current[1]}; previous {old_range}.'
                             if category=='policy_state' else
                             'Historical decision is explicitly recorded.' if category=='historical_record' else
                             'Current statement explicitly retains the 2 percent longer-run objective.' if category=='persistent_objective' else
                             'The supplied official documents do not establish this exact next-meeting outcome.')))
    case = dict(id=case_id, title=f'FOMC policy decision {row["date"]}', event_date=row['date'],
                cutoff=row['minutes_date']+'T23:59:59Z', source_stratum='official_event_not_market_linked',
                event_family='FOMC', market_reference=None,
                documents=docs, probes=probes,
                inclusion='all regularly scheduled 2023-2025 decisions with published minutes; fixed before inference',
                dependence='same monetary-policy institution and overlapping prior statement windows')
    return case,gold

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    response=requests.get(CALENDAR,timeout=60)
    response.raise_for_status()
    (OUT/'fed_calendar.html').write_bytes(response.content)
    rows=calendar_rows(response.text)
    requested=[]
    for row in rows:
        requested.append((row['statement_url'],'fed_'+row['stamp']+'_statement',row['date'],'statement'))
        if row['stamp'][:4] >= '2023':
            requested.append((row['minutes_url'],'fed_'+row['stamp']+'_minutes',row['minutes_date'],'minutes'))
    sources={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(download,*args) for args in requested]
        for f in concurrent.futures.as_completed(futures):
            doc=f.result();sources[doc['id']]=doc
            print('ARCHIVED',doc['id'],doc['characters'],flush=True)
    cases,gold=[],[]
    for i,row in enumerate(rows):
        if row['stamp'][:4]<'2023': continue
        case,labels=make_case(rows,i,sources);cases.append(case);gold+=labels
    assert len(cases)==24, f'Expected fixed full 24 meetings, got {len(cases)}'
    write_json(OUT/'fed_cases.json',dict(version='v1',cases=cases))
    write_json(OUT/'fed_gold.json',dict(version='v1',labels=gold))
    write_json(OUT/'fed_source_manifest.json',dict(calendar_url=CALENDAR,calendar_sha256=sha(response.content),sources=list(sources.values())))
    print('BUILT',len(cases),'events',len(gold),'probes',flush=True)

if __name__=='__main__': main()
