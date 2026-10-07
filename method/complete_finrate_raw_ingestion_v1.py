"""Finish a running acquisition, preserve source snapshots, and validate tools."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from build_finrate_raw_corpus_v1 import ROOT,STORE,RESULT,guard,atomic_json


def main(pid):
    while True:
        guard()
        try:os.kill(pid,0)
        except ProcessLookupError:break
        time.sleep(3)
    state=json.loads((RESULT/'status.json').read_text())
    if state['phase']!='acquisition_finished_mapping_pending':
        raise RuntimeError('Acquisition did not finish normally: '+str(state))
    guard()
    subprocess.run([sys.executable,str(ROOT/'method/finalize_finrate_raw_corpus_v1.py')],cwd=ROOT,check=True)
    from finrate_raw_store_v2 import RawStore
    store=RawStore()
    db=store.db
    trace=[]
    tested=0
    # One selected document per company: list, text read, and literal search
    # must return the exact stored text and consistent Unicode offsets.
    companies=[r[0] for r in db.execute('SELECT DISTINCT ticker FROM title_groups ORDER BY ticker')]
    for ticker in companies:
        listing=store.list_filings(ticker=ticker,limit=1)
        trace.append(dict(tool='list_filings',args=dict(ticker=ticker,limit=1),result=listing))
        if not listing:continue
        fid=listing[0]['document_id']
        result=store.read(fid,start=0,length=12000)
        row=db.execute('SELECT text_path FROM documents WHERE document_id=?',(fid,)).fetchone()
        text=(STORE/row['text_path']).read_text()
        assert result['text']==text[:result['end']]
        assert result['total_characters']==len(text)
        trace.append(dict(tool='read',args=dict(document_id=fid,start=0,length=12000),result=result))
        # Select a real visible token from the source, not QA/gold.
        query=next((word for word in text.split() if len(word)>=8),'')
        if query:
            hits=store.grep(query,document_id=fid,limit=3)
            assert hits['hits']
            for hit in hits['hits']:
                assert hit['text']==text[hit['start']:hit['end']]
                assert text[hit['match_start']:hit['match_end']].casefold()==query.casefold()
            trace.append(dict(tool='grep',args=dict(query=query,document_id=fid,limit=3),result=hits))
        # Full document read continuity, without duplicating all text in trace.
        pieces=[];position=0
        while True:
            part=store.read(fid,start=position,length=24000);pieces.append(part['text'])
            if part['next_start'] is None:break
            position=part['next_start']
        assert ''.join(pieces)==text
        tested+=1
    guard()
    with (RESULT/'tool_trace.jsonl').open('w') as f:
        for item in trace:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    atomic_json(RESULT/'tool_validation.json',dict(companies_with_selected_document=tested,calls_preserved=len(trace),full_read_roundtrips=tested,all_passed=True))
    snap=RESULT/'final_source_snapshot';snap.mkdir(exist_ok=True)
    hashes={}
    for filename in ['build_finrate_raw_corpus_v1.py','finrate_raw_store_v2.py','finalize_finrate_raw_corpus_v1.py','complete_finrate_raw_ingestion_v1.py']:
        source=ROOT/'method'/filename;shutil.copyfile(source,snap/filename);hashes[filename]=hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(RESULT/'final_source_hashes.json',hashes)
    report=json.loads((RESULT/'report.json').read_text());report['tool_validation_passed']=True;report['validated_companies']=tested
    atomic_json(RESULT/'report.json',report);atomic_json(RESULT/'status.json',report)
    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
    entry=(f"{now} — FIN-RATE 전체2472제목묶음 raw corpus 수집 후보 전수처리와 DB/도구 검증 완료. "
           f"SEC metadata 후보{report['metadata_candidate_documents']}개; acquisition={report['acquisition']}; "
           f"제목묶음 상태={report['group_statuses']}; 선택원문{report['selected_documents']}개. "
           "404 등 원문미확보/후보모호성을 분모에서 유지하므로 100% 원문구축 완료가 아님. "
           f"해시검증{report['source_hashes_checked']}건, SQLite integrity={report['integrity']}, "
           f"도구검증{tested}기업 및 full read roundtrip 통과. HTML/XML/TXT 원본과 전체text; "
           "sidecar structure/provenance·DOM span DB 없음. list_filings/grep/read 기본범위는 selected 원문; "
           "나머지는 candidate_not_selected로 격리. 본문probe는 identity heuristic이며 완전성 증거아님. "
           "결과 result/finrate_raw_ingestion_v1/report.json 및 company_coverage.json; data/finrate_raw_v1/manifest.sqlite. "
           "QA/gold/모델실험0; GPU사용0; 10GiB guard 유지.")
    for filename in ['context.md','process.md']:
        path=ROOT/filename;old=path.read_text();a,sep,b=old.partition('\n');path.write_text(a+'\n\n'+entry+'\n'+b)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);args=p.parse_args()
    try:main(args.pid)
    except Exception as exc:
        # Even if low disk prohibits normal artifacts, the tiny existing log
        # captures why the pipeline stopped; never label this complete.
        print('FINALIZATION FAILED:',repr(exc),flush=True)
        if shutil.disk_usage(STORE).free>=10*1024**3:
            atomic_json(RESULT/'finalization_error.json',dict(error=repr(exc),time=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise
