"""Recover missing primary filings with a declared SEC identity and shared cap.

Preserves the first-pass reports and a database snapshot; never reads QA/gold.
The user supplies FINRATE_SEC_USER_AGENT. No browser spoofing or proxy rotation.
"""
from __future__ import annotations
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import threading
import time
import urllib.error
import urllib.request

from build_finrate_raw_corpus_v1 import ROOT, STORE, parse_document

RESULT = ROOT / 'result/finrate_sec_recovery_v2'
DIAG = ROOT / 'result/finrate_source_access_diagnosis_20261007'
WORKERS = 16
PARSERS = 8
REQUESTS_PER_SECOND = 8
MIN_FREE = 10 * 1024**3
STOP = threading.Event()
RATE_LOCK = threading.Lock()
NEXT_START = 0.0
STOP_REASON = None
UA = ''


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def stop(reason):
    global STOP_REASON
    if STOP_REASON is None:
        STOP_REASON = reason
    STOP.set()


def guard():
    if shutil.disk_usage(STORE).free < MIN_FREE:
        stop('free_disk_below_10_GiB')
    if STOP.is_set():
        raise RuntimeError(STOP_REASON or 'stop_requested')


def rate_limit():
    global NEXT_START
    # Sleep while holding the lock, so delayed threads cannot bunch requests.
    with RATE_LOCK:
        guard()
        delay = max(0, NEXT_START - time.monotonic())
        if STOP.wait(delay):
            guard()
        guard()
        NEXT_START = time.monotonic() + 1 / REQUESTS_PER_SECOND


def download(row):
    fid = row['document_id']
    url = row['source_url']
    folder = STORE / 'filings' / f"{row['cik']:010d}" / row['accession'].replace('-', '')
    ext = Path(row['original_filename']).suffix.lower()
    path = folder / ('primary' + (ext if ext in ('.html', '.htm', '.xml', '.txt') else '.html'))
    result = dict(document_id=fid, attempts=[])
    for attempt in range(3):
        tmp = path.with_suffix(path.suffix + '.part')
        started = None
        try:
            guard()
            if path.exists():
                raise FileExistsError('Uncommitted source exists; refusing to overwrite: ' + str(path))
            folder.mkdir(parents=True, exist_ok=True)
            rate_limit()
            started = now()
            req = urllib.request.Request(url, headers={
                'User-Agent': UA,
                'Accept': 'text/html, text/plain, application/xhtml+xml, */*',
                'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=45) as response, tmp.open('wb') as out:
                media = response.headers.get('Content-Type', '')
                final_url = response.geturl()
                size = 0
                digest = hashlib.sha256()
                prefix = bytearray()
                while True:
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    guard()
                    out.write(block)
                    digest.update(block)
                    size += len(block)
                    if len(prefix) < 8192:
                        prefix.extend(block[:8192-len(prefix)])
            lower = bytes(prefix).lower()
            if any(x in lower for x in (b'undeclared automated tool', b'request rate threshold exceeded',
                                       b'<title>access denied', b'<code>nosuchkey</code>')):
                stop('access_challenge_in_success_response')
                raise ValueError('Access challenge instead of source')
            if size < 50:
                raise ValueError('Empty or tiny source document')
            guard()
            tmp.replace(path)
            result['attempts'].append(dict(url=url, status='200', detail=str(size), attempted_at=started))
            result.update(acquisition_status='downloaded', source_path=str(path.relative_to(STORE)),
                          source_bytes=size, source_sha256=digest.hexdigest(), error=None,
                          acquired_from_url=final_url, media_type=media, sec_bytes_verified=1)
            return result
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            code = getattr(exc, 'code', None)
            detail = str(exc)[:500]
            if isinstance(exc, urllib.error.HTTPError):
                body = exc.read(16384)
                (RESULT / 'http_errors' / f'{fid}_{attempt}_{code}.response').write_bytes(body)
                exc.close()
            if started:
                result['attempts'].append(dict(url=url, status=str(code or type(exc).__name__),
                                               detail=detail, attempted_at=started))
            result.update(acquisition_status='source_unavailable', error=detail)
            if code in (401, 403, 429):
                stop(f'SEC_HTTP_{code}; circuit_open_no_automatic_retry')
            if STOP.is_set() or code in (400, 401, 403, 404, 429) or isinstance(exc, FileExistsError):
                break
            if attempt < 2:
                STOP.wait(2**attempt)
    return result


def save_download(db, result):
    with db:
        if result.get('attempts') or result.get('acquisition_status') == 'downloaded':
            fields = [k for k in result if k not in ('document_id', 'attempts')]
            db.execute('UPDATE documents SET ' + ','.join(k+'=?' for k in fields) + ' WHERE document_id=?',
                       [result[k] for k in fields] + [result['document_id']])
        for a in result['attempts']:
            db.execute('INSERT INTO acquisition_attempts(document_id,url,status,detail,attempted_at) VALUES (?,?,?,?,?)',
                       (result['document_id'], a['url'], a['status'], a['detail'], a['attempted_at']))
    with (RESULT / 'download_trace.jsonl').open('a') as out:
        out.write(json.dumps(result, ensure_ascii=False) + '\n')


def save_parse(db, result):
    fid = result['document_id']
    with db:
        db.execute("UPDATE documents SET acquisition_status='ready',text_path=?,text_sha256=?,text_characters=?,parser='finrate_raw_store_v2',error=NULL WHERE document_id=?",
                   (result['text_path'], result['text_sha256'], result['text_characters'], fid))
        db.execute('DELETE FROM sections WHERE document_id=?', (fid,))
        db.executemany('INSERT INTO sections VALUES (?,?,?,?,?,?)',
                       [(f'{fid}:s{i}',fid,s['title'],s['start'],s['end'],s['basis']) for i,s in enumerate(result['sections'])])


def status(db, phase, completed, total, started, extra=None):
    elapsed = time.monotonic() - started
    data = dict(phase=phase, pid=os.getpid(), updated_at=now(), stop_reason=STOP_REASON,
                candidate_tasks_finished=completed, candidate_tasks_total=total,
                elapsed_seconds=round(elapsed, 1), workers=WORKERS, parsers=PARSERS,
                requests_per_second_cap=REQUESTS_PER_SECOND,
                documents=dict(db.execute('SELECT acquisition_status,count(*) FROM documents GROUP BY 1')),
                direct_sec_ready=db.execute("SELECT count(*) FROM documents WHERE acquisition_status='ready' AND sec_bytes_verified=1").fetchone()[0],
                groups=dict(db.execute('SELECT status,count(*) FROM title_groups GROUP BY 1')),
                mapping_phase='pending_full_recheck',
                free_disk_gib=round(shutil.disk_usage(STORE).free / 1024**3, 3))
    if completed >= 20 and not STOP.is_set():
        data['download_eta_seconds_linear'] = round(elapsed / completed * (total-completed))
    if extra:
        data.update(extra)
    write_json(RESULT / 'status.json', data)
    print(json.dumps(data), flush=True)


def run():
    global UA
    UA = os.environ['FINRATE_SEC_USER_AGENT']
    if '@' not in UA:
        raise ValueError('A user-provided contact email is required in declared User-Agent')
    guard()
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / 'http_errors').mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: stop('termination_requested'))
    signal.signal(signal.SIGINT, lambda *_: stop('interrupt_requested'))
    db = sqlite3.connect(STORE / 'manifest.sqlite', timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    backup = RESULT / 'before_recovery.sqlite'
    if not backup.exists():
        dest = sqlite3.connect(backup)
        db.backup(dest)
        dest.close()
    initial = dict(db.execute('SELECT status,count(*) FROM title_groups GROUP BY 1'))
    work = [json.loads(line) for line in (DIAG / 'recovery_candidates.jsonl').read_text().splitlines()]
    counts = dict(db.execute('SELECT group_id,count(*) FROM group_candidates GROUP BY 1'))
    def priority(r):
        unique = any(counts[g['group_id']] == 1 for g in r['title_groups'])
        form_order = {'10-K':0,'10-Q':1,'8-K':2,'6-K':3}
        return (not unique, form_order.get(r['form'],4), r['filed_at'], r['document_id'])
    work.sort(key=priority)
    write_json(RESULT / 'run_manifest.json', dict(created_at=now(), workers=WORKERS, parsers=PARSERS,
        requests_per_second_cap=REQUESTS_PER_SECOND, free_disk_stop_bytes=MIN_FREE,
        user_agent_sha256=hashlib.sha256(UA.encode()).hexdigest(), declared_contact=True,
        stop_on_http=[401,403,429], source='official SEC primary documents', initial_group_statuses=initial,
        priority='unique metadata candidate, core filing form, filing date, document ID; no QA input',
        candidate_count=len(work), candidate_manifest_sha256=hashlib.sha256((DIAG/'recovery_candidates.jsonl').read_bytes()).hexdigest(),
        sec_bytes_verified_semantics='1 = fetched directly from SEC; 0 = mirror not compared with SEC bytes'))
    with db:
        db.executemany('INSERT OR REPLACE INTO store_metadata VALUES (?,?)',[
            ('source_provenance','SEC metadata; mixed SEC direct and Yahoo mirror primary documents; acquired_from_url records origin'),
            ('sec_bytes_verified_semantics','1: fetched directly from SEC; 0: public mirror without SEC byte comparison'),
            ('sec_recovery_result_directory',str(RESULT)),
            ('max_sec_primary_request_starts_per_second',str(REQUESTS_PER_SECOND))])
    # Reuse the already verified network smoke, without another request.
    smoke = json.loads((DIAG/'identified_smoke_result.json').read_text())['requests']
    for s in smoke:
        row = db.execute('SELECT * FROM documents WHERE document_id=?',(s['document_id'],)).fetchone()
        if row['acquisition_status'] == 'ready':
            continue
        source = Path(s['saved_path'])
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == s['sha256']
        folder = STORE / 'filings' / s['document_id'].split('_')[0] / s['document_id'].split('_')[1]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ('primary' + source.suffix)
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == s['sha256']
        else:
            guard()
            shutil.copyfile(source,path)
        result = dict(document_id=s['document_id'], acquisition_status='downloaded', source_path=str(path.relative_to(STORE)),
            source_bytes=len(raw), source_sha256=s['sha256'], error=None, acquired_from_url=s['url'],
            media_type=s['content_type'], sec_bytes_verified=1,
            attempts=[dict(url=s['url'],status='200',detail='reused identified smoke bytes',attempted_at=s['started_at_utc'])])
        save_download(db,result)
        save_parse(db,parse_document(result))
    pending = [r for r in work if db.execute('SELECT acquisition_status FROM documents WHERE document_id=?',
                                             (r['document_id'],)).fetchone()[0] == 'source_unavailable']
    total = len(pending)
    iterator = iter(pending)
    completed = 0
    started = last = time.monotonic()
    status(db,'recovering_sec_sources',completed,total,started)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as net, cf.ProcessPoolExecutor(
            max_workers=PARSERS,mp_context=mp.get_context('spawn')) as cpu:
        downloads = {}
        parses = {}
        exhausted = False
        while downloads or parses or not exhausted:
            if not STOP.is_set() and shutil.disk_usage(STORE).free < MIN_FREE:
                stop('free_disk_below_10_GiB')
            while not exhausted and not STOP.is_set() and len(downloads) < WORKERS and len(parses) < PARSERS*2:
                row = next(iterator,None)
                if row is None:
                    exhausted=True
                    break
                downloads[net.submit(download,row)] = row['document_id']
            if STOP.is_set():
                exhausted=True
            if not downloads and not parses:
                break
            done,_ = cf.wait(list(downloads)+list(parses), timeout=2,return_when=cf.FIRST_COMPLETED)
            for fut in done:
                if fut in downloads:
                    fid=downloads.pop(fut)
                    result=fut.result()
                    save_download(db,result)
                    completed+=1
                    if result.get('acquisition_status') == 'downloaded' and STOP_REASON != 'free_disk_below_10_GiB':
                        parses[cpu.submit(parse_document,result)] = fid
                else:
                    fid=parses.pop(fut)
                    try:
                        save_parse(db,fut.result())
                    except Exception as exc:
                        with db:
                            db.execute("UPDATE documents SET acquisition_status='parse_failed',error=? WHERE document_id=?",(str(exc),fid))
                        if shutil.disk_usage(STORE).free < MIN_FREE:
                            stop('free_disk_below_10_GiB')
            if time.monotonic()-last > 15:
                status(db,'recovering_sec_sources',completed,total,started)
                last=time.monotonic()
    status(db,'acquisition_stopped' if STOP.is_set() else 'acquisition_finished',completed,total,started)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    if STOP_REASON == 'free_disk_below_10_GiB' or shutil.disk_usage(STORE).free < MIN_FREE:
        return
    # Even if an access circuit opens, map/audit the successfully recovered files.
    from finalize_finrate_sec_recovery_v2 import run as finalize
    finalize()


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        RESULT.mkdir(parents=True,exist_ok=True)
        write_json(RESULT/'fatal_error.json',dict(at=now(),error=repr(exc),pid=os.getpid()))
        raise
