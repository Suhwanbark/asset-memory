"""Acquire SEC primary documents corresponding to ALL FIN-RATE title groups.

No QA/reference answers are read. SEC submissions establish candidate identity;
title groups are not assumed to be unique accessions. Download failures remain
in the catalog. Public copies retain their actual acquisition URL and hash.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'data/finrate_raw_v1'
RESULT = ROOT / 'result/finrate_raw_ingestion_v1'
UA = 'FinRATE academic archival research (local corpus reconstruction)'
LOCK = threading.Lock()
STOP = threading.Event()
NEXT_REQUEST = 0.0


def guard():
    if STOP.is_set() or shutil.disk_usage(STORE).free < 10 * 1024**3:
        STOP.set()
        raise RuntimeError('STOPPED: free disk below 10 GiB or stop requested')


def atomic_json(path, value):
    guard()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def limit_rate():
    global NEXT_REQUEST
    with LOCK:
        now = time.monotonic()
        wait = max(0, NEXT_REQUEST - now)
        # This worker only contacts the public Yahoo CDN, not SEC endpoints.
        NEXT_REQUEST = max(now, NEXT_REQUEST) + 0.05
    if wait:
        time.sleep(wait)
    guard()


def canonical_form(form):
    return form.split('/')[0].split(' ')[0]


def acceptance_date(value):
    if not value:
        return ''
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(
        ZoneInfo('America/New_York')).date().isoformat()


def rows(table):
    for i in range(len(table['accessionNumber'])):
        yield {k: v[i] for k, v in table.items() if isinstance(v, list) and len(v) > i}


def prepare():
    companies = json.loads((STORE / 'metadata/company_cik_candidates.json').read_text())['candidates']
    all_rows = {}
    for cik in sorted({c for cs in companies.values() for c in cs}):
        parent = STORE / 'metadata/sec_submissions'
        current = json.loads((parent / f'CIK{cik:010d}.json').read_text())
        tabs = [(parent / f'CIK{cik:010d}.json', current['filings']['recent'])]
        for hist in current['filings'].get('files', []):
            path = parent / hist['name']
            if path.exists():
                tabs.append((path, json.loads(path.read_text())))
        unique = {}
        for path, table in tabs:
            for r in rows(table):
                r.update(cik=cik, company_name=current['name'], metadata_path=str(path.relative_to(STORE)))
                r['accepted_date_et'] = acceptance_date(r.get('acceptanceDateTime', ''))
                unique[r['accessionNumber']] = r
        all_rows[cik] = list(unique.values())
    groups, docs = [], {}
    inventory = ROOT / 'result/finrate_corpus_inventory_20261007/filings.csv'
    for g in csv.DictReader(inventory.open(encoding='utf-8-sig')):
        key = '_'.join(g[k] for k in ('ticker', 'title_date', 'form_label'))
        pool = [r for cik in companies[g['ticker']] for r in all_rows[cik]
                if canonical_form(r['form']) == g['form_label']]
        exact = [r for r in pool if g['title_date'] in (r['filingDate'], r['accepted_date_et'])]
        candidates, basis = exact, 'official_filing_or_acceptance_ET_date_and_form'
        if not candidates:
            target = dt.date.fromisoformat(g['title_date'])
            candidates = [r for r in pool if abs((dt.date.fromisoformat(r['filingDate']) - target).days) <= 3]
            basis = 'within_3_days_and_form_REQUIRES_content_verification'
        ids = []
        for r in candidates:
            if not r.get('primaryDocument'):
                continue
            fid = f"{r['cik']:010d}_{r['accessionNumber'].replace('-', '')}"
            ids.append(fid)
            if fid not in docs:
                acc = r['accessionNumber'].replace('-', '')
                docs[fid] = dict(r, filing_id=fid, tickers=[],
                    source_url=f"https://www.sec.gov/Archives/edgar/data/{r['cik']}/{acc}/{r['primaryDocument']}",
                    mirror_url=f"https://cdn.yahoofinance.com/prod/sec-filings/{r['cik']:010d}/{acc}/{r['primaryDocument']}")
            if g['ticker'] not in docs[fid]['tickers']:
                docs[fid]['tickers'].append(g['ticker'])
        groups.append(dict(group_id=key, ticker=g['ticker'], title_date=g['title_date'],
                           form_label=g['form_label'], chunk_ids=g['chunk_ids_in_title_sequence'].split(';'),
                           candidates=sorted(set(ids)), identity_basis=basis))
    plan = dict(groups=groups, documents=docs, created_at=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(STORE / 'ingestion_plan.json', plan)
    print(json.dumps({'phase': 'plan', 'groups': len(groups), 'candidate_documents': len(docs),
                      'unmatched_groups': sum(not g['candidates'] for g in groups),
                      'ambiguous_groups': sum(len(g['candidates']) > 1 for g in groups)}), flush=True)
    return plan


def schema(db):
    db.executescript('''
    PRAGMA journal_mode=WAL;
    PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS store_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS filings(
      filing_id TEXT PRIMARY KEY,cik INTEGER,accession TEXT,ticker_aliases TEXT,
      company_name_at_metadata_snapshot TEXT,form TEXT,filed_at TEXT,accepted_at TEXT,
      report_date TEXT,is_amendment INTEGER,metadata_path TEXT);
    CREATE TABLE IF NOT EXISTS documents(
      document_id TEXT PRIMARY KEY,filing_id TEXT REFERENCES filings(filing_id),
      original_filename TEXT,source_url TEXT,acquired_from_url TEXT,
      acquisition_status TEXT,error TEXT,source_path TEXT,text_path TEXT,
      source_sha256 TEXT,text_sha256 TEXT,source_bytes INTEGER,text_characters INTEGER,
      media_type TEXT,sec_bytes_verified INTEGER DEFAULT 0,parser TEXT);
    CREATE TABLE IF NOT EXISTS title_groups(
      group_id TEXT PRIMARY KEY,ticker TEXT,title_date TEXT,form_label TEXT,
      chunk_count INTEGER,identity_basis TEXT,status TEXT);
    CREATE TABLE IF NOT EXISTS group_candidates(
      group_id TEXT REFERENCES title_groups(group_id),document_id TEXT REFERENCES documents(document_id),
      PRIMARY KEY(group_id,document_id));
    CREATE TABLE IF NOT EXISTS finrate_mapping(
      chunk_id TEXT PRIMARY KEY,group_id TEXT REFERENCES title_groups(group_id),
      document_id TEXT REFERENCES documents(document_id),mapping_status TEXT,
      matched_probes INTEGER,total_probes INTEGER);
    CREATE TABLE IF NOT EXISTS sections(
      section_id TEXT PRIMARY KEY,document_id TEXT REFERENCES documents(document_id),
      title TEXT,start_offset INTEGER,end_offset INTEGER,boundary_basis TEXT);
    CREATE TABLE IF NOT EXISTS acquisition_attempts(
      id INTEGER PRIMARY KEY,document_id TEXT,url TEXT,status TEXT,detail TEXT,attempted_at TEXT);
    CREATE INDEX IF NOT EXISTS filings_form_date ON filings(form,filed_at);
    CREATE INDEX IF NOT EXISTS mapping_group ON finrate_mapping(group_id);
    ''')


def initialize(db, plan):
    schema(db)
    with db:
        db.executemany('INSERT OR REPLACE INTO store_metadata VALUES (?,?)', [
            ('schema_version', '1'), ('body_scope', 'primary filing document only; exhibits/images not downloaded'),
            ('source_provenance', 'SEC metadata; public Yahoo Finance copies not byte-verified against SEC'),
            ('parser', 'finrate_raw_store_v2.HTMLParser; whole visible text, no item selection'),
            ('disk_guard_bytes', str(10 * 1024**3)), ('download_workers', '48'),
            ('max_yahoo_cdn_request_starts_per_second', '20'),
            ('max_sec_metadata_request_starts_per_second', str(1 / .15)),
            ('finrate_revision', '284b1b4f6cd1e58574f1f1d17c29b35c6aa72cce')])
        for fid, r in plan['documents'].items():
            db.execute('INSERT OR IGNORE INTO filings VALUES (?,?,?,?,?,?,?,?,?,?,?)', (
                fid,r['cik'],r['accessionNumber'],json.dumps(r['tickers']),r['company_name'],r['form'],
                r['filingDate'],r.get('acceptanceDateTime'),r.get('reportDate'),int(r['form'].endswith('/A')),r['metadata_path']))
            db.execute('INSERT OR IGNORE INTO documents(document_id,filing_id,original_filename,source_url,acquisition_status) VALUES (?,?,?,?,?)',
                       (fid,fid,r['primaryDocument'],r['source_url'],'pending'))
        for g in plan['groups']:
            db.execute('INSERT OR IGNORE INTO title_groups VALUES (?,?,?,?,?,?,?)', (
                g['group_id'],g['ticker'],g['title_date'],g['form_label'],len(g['chunk_ids']),g['identity_basis'],
                'candidate_identified' if g['candidates'] else 'no_metadata_match'))
            db.executemany('INSERT OR IGNORE INTO group_candidates VALUES (?,?)', [(g['group_id'],fid) for fid in g['candidates']])
            db.executemany('INSERT OR IGNORE INTO finrate_mapping VALUES (?,?,NULL,?,0,0)',
                           [(cid,g['group_id'],'unverified') for cid in g['chunk_ids']])


def download(r, reuse):
    guard()
    fid = r['filing_id']
    folder = STORE / 'filings' / f"{r['cik']:010d}" / r['accessionNumber'].replace('-', '')
    folder.mkdir(parents=True, exist_ok=True)
    ext = Path(r['primaryDocument']).suffix.lower()
    path = folder / ('primary' + (ext if ext in ('.html','.htm','.xml','.txt') else '.html'))
    attempts = []
    result = dict(document_id=fid, attempts=attempts)
    if r['accessionNumber'] in reuse:
        old, meta = reuse[r['accessionNumber']]
        raw = old.read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['html_sha256']:
            raise ValueError('Prior source hash mismatch: ' + str(old))
        if not path.exists():
            guard()
            shutil.copyfile(old, path)
        result.update(acquired_from_url=meta['html_download_url'],media_type='text/html',acquisition_status='downloaded')
    elif path.exists():
        result.update(acquired_from_url=r['mirror_url'],media_type='application/xml' if ext=='.xml' else 'text/html',acquisition_status='downloaded')
    else:
        url = r['mirror_url']
        for attempt in range(3):
            tmp = path.with_suffix(path.suffix + '.part')
            try:
                limit_rate()
                request = urllib.request.Request(url, headers={'User-Agent': UA})
                with urllib.request.urlopen(request, timeout=45) as response, tmp.open('wb') as out:
                    media = response.headers.get('Content-Type', '')
                    size = 0
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        guard()
                        out.write(block)
                        size += len(block)
                    if size < 50:
                        raise ValueError('Empty or tiny source document')
                prefix = tmp.read_bytes()[:2000].lower()
                if b'request rate threshold exceeded' in prefix or b'captcha' in prefix or b'<title>access denied' in prefix:
                    raise ValueError('Error/challenge page instead of source')
                tmp.replace(path)
                attempts.append(dict(url=url,status='200',detail=str(size)))
                result.update(acquired_from_url=url,media_type=media,acquisition_status='downloaded')
                break
            except Exception as exc:
                if tmp.exists():
                    tmp.unlink()
                code = getattr(exc, 'code', None)
                attempts.append(dict(url=url,status=str(code or type(exc).__name__),detail=str(exc)[:500]))
                result.update(acquisition_status='source_unavailable',error=str(exc)[:500])
                if STOP.is_set() or code in (400,401,403,404):
                    break
                if attempt < 2:
                    time.sleep(2**attempt)
    if result['acquisition_status'] == 'downloaded':
        raw = path.read_bytes()
        result.update(source_path=str(path.relative_to(STORE)),source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),error=None)
    return result


def parse_document(r):
    from finrate_raw_store_v2 import project
    guard()
    source = STORE / r['source_path']
    text, sections = project(source.read_bytes(), source.suffix)
    if not text.strip():
        raise ValueError('No readable source text: ' + str(source))
    path = source.parent / 'primary.txt'
    if path == source:
        path = source.parent / 'readable.txt'
    data = text.encode('utf-8')
    guard()
    tmp = path.with_suffix('.txt.part')
    tmp.write_bytes(data)
    tmp.replace(path)
    return dict(document_id=r['document_id'],text_path=str(path.relative_to(STORE)),
                text_sha256=hashlib.sha256(data).hexdigest(),text_characters=len(text),sections=sections)


def status(db, phase):
    value = dict(phase=phase,updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),
                 documents=dict(db.execute('SELECT acquisition_status,count(*) FROM documents GROUP BY acquisition_status')),
                 groups=dict(db.execute('SELECT status,count(*) FROM title_groups GROUP BY status')),
                 free_disk_gib=round(shutil.disk_usage(STORE).free/1024**3,3))
    atomic_json(RESULT / 'status.json', value)
    print(json.dumps(value), flush=True)


def run(args):
    STORE.mkdir(parents=True,exist_ok=True)
    RESULT.mkdir(parents=True,exist_ok=True)
    plan_path = STORE / 'ingestion_plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else prepare()
    db = sqlite3.connect(STORE / 'manifest.sqlite')
    initialize(db, plan)
    reuse = {}
    for meta_path in (ROOT / 'data/finrate_source_audit_20261007').glob('*.meta.json'):
        meta = json.loads(meta_path.read_text())
        old = meta_path.with_name(meta_path.name.replace('.meta.json','.mirror.html'))
        if old.exists() and meta.get('html_sha256'):
            reuse[meta['accession']] = (old,meta)
    work = [r for fid,r in plan['documents'].items() if db.execute(
        'SELECT acquisition_status FROM documents WHERE document_id=?',(fid,)).fetchone()[0] == 'pending']
    status(db,'downloading')
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download,r,reuse) for r in work]
        last = time.monotonic()
        for fut in cf.as_completed(futures):
            r = fut.result()
            with db:
                fields = [k for k in r if k not in ('document_id','attempts')]
                db.execute('UPDATE documents SET '+','.join(k+'=?' for k in fields)+' WHERE document_id=?',
                           [r[k] for k in fields]+[r['document_id']])
                for attempt in r['attempts']:
                    db.execute('INSERT INTO acquisition_attempts(document_id,url,status,detail,attempted_at) VALUES (?,?,?,?,?)',
                               (r['document_id'],attempt['url'],attempt['status'],attempt['detail'],dt.datetime.now(dt.timezone.utc).isoformat()))
            if time.monotonic()-last > 15:
                status(db,'downloading');last=time.monotonic()
    status(db,'parsing')
    db.row_factory = sqlite3.Row
    work = [dict(r) for r in db.execute("SELECT * FROM documents WHERE acquisition_status='downloaded'")]
    with cf.ProcessPoolExecutor(max_workers=args.parsers) as pool:
        futures = {pool.submit(parse_document,r):r['document_id'] for r in work}
        last = time.monotonic()
        for fut in cf.as_completed(futures):
            fid = futures[fut]
            try:
                r = fut.result()
                with db:
                    db.execute("UPDATE documents SET acquisition_status='ready',text_path=?,text_sha256=?,text_characters=?,parser='finrate_raw_store_v2' WHERE document_id=?",
                               (r['text_path'],r['text_sha256'],r['text_characters'],fid))
                    db.execute('DELETE FROM sections WHERE document_id=?',(fid,))
                    db.executemany('INSERT INTO sections VALUES (?,?,?,?,?,?)',
                                   [(f'{fid}:s{i}',fid,s['title'],s['start'],s['end'],s['basis']) for i,s in enumerate(r['sections'])])
            except Exception as exc:
                with db:
                    db.execute("UPDATE documents SET acquisition_status='parse_failed',error=? WHERE document_id=?",(str(exc),fid))
            if time.monotonic()-last > 15:
                status(db,'parsing');last=time.monotonic()
    with db:
        db.execute("UPDATE title_groups SET status=CASE WHEN NOT EXISTS(SELECT 1 FROM group_candidates g WHERE g.group_id=title_groups.group_id) THEN 'no_metadata_match' WHEN EXISTS(SELECT 1 FROM group_candidates g JOIN documents d ON d.document_id=g.document_id WHERE g.group_id=title_groups.group_id AND d.acquisition_status='ready') THEN 'source_ready_identity_pending' ELSE 'source_unavailable' END")
    status(db,'acquisition_finished_mapping_pending')
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=48)
    parser.add_argument('--parsers',type=int,default=8)
    args = parser.parse_args()
    run(args)
