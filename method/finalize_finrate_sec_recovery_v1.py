"""Source-only candidate matching and acquisition audit; never reads QA/gold.

Probe overlap is a heuristic identity check, not a proof of full source coverage.
All alternative candidates and failed acquisitions remain in the manifest.
"""
from __future__ import annotations
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata
import zipfile

from build_finrate_raw_corpus_v1 import ROOT, STORE, guard, atomic_json
RESULT = ROOT / "result/finrate_sec_recovery_v1"


def normalize(text):
    text=re.sub(r'!?\[([^\]]*)\]\([^)]*\)',r'\1',text)
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def probes(text):
    n=normalize(text)
    if len(n)<256:
        return []
    starts=sorted({int(i*(len(n)-160)/15) for i in range(16)})
    return [n[s:s+160] for s in starts]


def run():
    guard()
    # Build mappings in a private database. Publish only after full validation.
    staging = RESULT/'mapping_work.sqlite'
    source_db=sqlite3.connect(STORE/'manifest.sqlite')
    staged_db=sqlite3.connect(staging)
    source_db.backup(staged_db)
    staged_db.close();source_db.close()
    plan=json.loads((STORE/'ingestion_plan.json').read_text())
    corpus={}
    whole_chunks={}
    with zipfile.ZipFile(ROOT/'data/finrate_public_v1/corpus/corpus.zip') as z:
        member=next(n for n in z.namelist() if n.endswith('corpus.jsonl'))
        for line in z.open(member):
            r=json.loads(line);corpus[r['_id']]=probes(r['text'])
            whole_chunks[r['_id']]=normalize(r['text'])
    db=sqlite3.connect(staging);db.row_factory=sqlite3.Row
    if 'selection_status' not in {r[1] for r in db.execute('PRAGMA table_info(documents)')}:
        db.execute("ALTER TABLE documents ADD COLUMN selection_status TEXT DEFAULT 'unreviewed'")
    db.executescript('''
    CREATE TABLE IF NOT EXISTS content_checks(
      group_id TEXT,chunk_id TEXT,document_id TEXT,matched_probes INTEGER,total_probes INTEGER,
      PRIMARY KEY(group_id,chunk_id,document_id));
    CREATE TABLE IF NOT EXISTS document_selection(
      group_id TEXT,document_id TEXT,selection_basis TEXT,
      PRIMARY KEY(group_id,document_id));
    ''')
    text_paths={r['document_id']:r['text_path'] for r in db.execute("SELECT * FROM documents WHERE acquisition_status='ready'")}
    source_hashes={r['document_id']:r['source_sha256'] for r in db.execute("SELECT * FROM documents WHERE acquisition_status='ready'")}
    results=[]
    with db:
        db.execute('DELETE FROM content_checks');db.execute('DELETE FROM document_selection')
        db.execute("UPDATE finrate_mapping SET document_id=NULL,mapping_status='unverified',matched_probes=0,total_probes=0")
    for gi,g in enumerate(plan['groups']):
        guard()
        ready=[fid for fid in g['candidates'] if fid in text_paths]
        norms={fid:normalize((STORE/text_paths[fid]).read_text()) for fid in ready}
        scores={cid:{fid:sum(probe in text for probe in corpus[cid]) for fid,text in norms.items()} for cid in g['chunk_ids']}
        selected={};matched_chunks=0;ambiguous_chunks=0
        # A single official date/form candidate is useful even when a chunk was
        # transformed or omitted. Explicitly distinguish metadata vs content.
        if len(g['candidates'])==1 and ready and not g['identity_basis'].startswith('within_3'):
            selected[ready[0]]='unique_official_date_and_form_candidate'
        with db:
            for cid,counts in scores.items():
                total=len(corpus[cid]);best=max(counts.values(),default=0)
                tops=[fid for fid,n in counts.items() if n==best]
                # Exact identical source bytes across co-registrants are one
                # content variant; retain all aliases as candidate metadata.
                variants=collections.defaultdict(list)
                for fid in tops:variants[source_hashes[fid]].append(fid)
                strong=total>=8 and best>=max(8,int(total*.75))
                winner=None;mapping='no_strong_content_match'
                if strong and len(variants)==1:
                    winner=sorted(next(iter(variants.values())))[0]
                    selected[winner]='strong_unique_content_variant_probe_match'
                    mapping='strong_content_probe_match_not_exact_offset';matched_chunks+=1
                elif strong:
                    mapping='ambiguous_content_match';ambiguous_chunks+=1
                # Source-only refinement: require the whole normalized chunk,
                # not merely equal sparse probes, to choose between candidates.
                if winner is None and len(whole_chunks[cid]) >= 1000:
                    exact=[fid for fid,text in norms.items() if whole_chunks[cid] in text]
                    exact_variants=collections.defaultdict(list)
                    for fid in exact:exact_variants[source_hashes[fid]].append(fid)
                    if len(exact_variants)==1:
                        winner=sorted(next(iter(exact_variants.values())))[0]
                        selected[winner]='unique_full_normalized_chunk_content_variant'
                        if mapping=='ambiguous_content_match':ambiguous_chunks-=1
                        mapping='unique_full_normalized_chunk_match_not_exact_offset'
                        matched_chunks+=1
                if winner is None and len(selected)==1 and len(g['candidates'])==1:
                    winner=next(iter(selected));mapping='metadata_candidate_only'
                elif not ready:
                    mapping='source_unavailable'
                db.execute('UPDATE finrate_mapping SET document_id=?,mapping_status=?,matched_probes=?,total_probes=? WHERE chunk_id=?',
                           (winner,mapping,best,total,cid))
                db.executemany('INSERT INTO content_checks VALUES (?,?,?,?,?)',[(g['group_id'],cid,fid,n,total) for fid,n in counts.items()])
            db.executemany('INSERT INTO document_selection VALUES (?,?,?)',[(g['group_id'],fid,basis) for fid,basis in selected.items()])
            if not g['candidates']:state='no_metadata_match'
            elif not ready:state='source_unavailable'
            elif selected:state='source_ready_with_mapping_caveats'
            else:state='source_candidates_ambiguous'
            db.execute('UPDATE title_groups SET status=? WHERE group_id=?',(state,g['group_id']))
        results.append(dict(group_id=g['group_id'],ticker=g['ticker'],title_date=g['title_date'],form_label=g['form_label'],status=state,
                            candidates=len(g['candidates']),downloaded_candidates=len(ready),selected_documents=list(selected),
                            chunks=len(g['chunk_ids']),strong_matched_chunks=matched_chunks,ambiguous_chunks=ambiguous_chunks))
        if gi%100==0:
            atomic_json(RESULT/'mapping_progress.json',dict(groups_processed=gi+1,total=len(plan['groups']),updated_at=dt.datetime.now(dt.timezone.utc).isoformat()))
            print('mapped',gi+1,'/',len(plan['groups']),flush=True)
    with db:
        db.execute("UPDATE documents SET selection_status=CASE WHEN acquisition_status!='ready' THEN 'source_unavailable' WHEN document_id IN (SELECT document_id FROM document_selection) THEN 'selected' ELSE 'candidate_not_selected' END")
        db.execute("CREATE VIEW IF NOT EXISTS corpus_documents AS SELECT * FROM documents WHERE acquisition_status='ready' AND selection_status='selected'")
        db.execute("INSERT OR REPLACE INTO store_metadata VALUES ('selection_policy','unique exact-date/form metadata or >=75% deterministic content probes with one source-byte variant; heuristic, not completeness proof')")
    atomic_json(RESULT/'group_coverage.json',results)
    audit=[];errors=[]
    for r in db.execute("SELECT * FROM documents WHERE acquisition_status='ready'"):
        guard()
        record=dict(document_id=r['document_id'],selected=r['selection_status']=='selected')
        for path_key,hash_key in [('source_path','source_sha256'),('text_path','text_sha256')]:
            path=STORE/r[path_key]
            ok=hashlib.sha256(path.read_bytes()).hexdigest()==r[hash_key]
            record[path_key+'_hash_ok']=ok
            if not ok:errors.append(r['document_id']+':'+path_key)
        audit.append(record)
    integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
    foreign=[tuple(r) for r in db.execute('PRAGMA foreign_key_check')]
    if integrity!='ok' or foreign:errors.append('database_integrity')
    company=[]
    for ticker in sorted({g['ticker'] for g in results}):
        gs=[g for g in results if g['ticker']==ticker]
        company.append(dict(ticker=ticker,total_groups=len(gs),statuses=dict(collections.Counter(g['status'] for g in gs)),
                            selected_documents=len({d for g in gs for d in g['selected_documents']})))
    acquisition=dict(db.execute('SELECT acquisition_status,count(*) FROM documents GROUP BY 1'))
    year_coverage={}
    for year in sorted({g['title_date'][:4] for g in results}):
        gs=[g for g in results if g['title_date'].startswith(year)]
        year_coverage[year]=dict(total_groups=len(gs),statuses=dict(collections.Counter(g['status'] for g in gs)))
    form_coverage={}
    for form in sorted({g['form_label'] for g in results}):
        gs=[g for g in results if g['form_label']==form]
        form_coverage[form]=dict(total_groups=len(gs),statuses=dict(collections.Counter(g['status'] for g in gs)))
    selected=db.execute("SELECT count(*),sum(source_bytes),sum(text_characters) FROM corpus_documents").fetchone()
    report=dict(completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),phase='all_candidates_processed_with_source_gaps',
                title_groups=len(results),companies=len(company),metadata_candidate_documents=len(plan['documents']),
                acquisition=acquisition,group_statuses=dict(collections.Counter(g['status'] for g in results)),
                year_coverage=year_coverage,form_coverage=form_coverage,
                http_attempt_statuses=dict(db.execute('SELECT status,count(*) FROM acquisition_attempts GROUP BY 1')),
                chunk_mapping_statuses=dict(db.execute('SELECT mapping_status,count(*) FROM finrate_mapping GROUP BY 1')),
                selected_documents=selected[0],selected_source_bytes=selected[1],selected_text_characters=selected[2],
                source_hashes_checked=len(audit),integrity=integrity,foreign_key_violations=foreign,validation_errors=errors,
                actual_store_bytes=sum(p.stat().st_size for p in STORE.rglob('*') if p.is_file()),
                free_disk_gib=shutil.disk_usage(STORE).free/1024**3,
                limitations=['2,472 title groups are not 2,472 unique SEC filings.',
                    'Primary documents only; separately filed exhibits and external images are outside this archive.',
                    'Source origin is mixed: SEC direct or public Yahoo mirror; per-document origin/verification flags distinguish them.',
                    'Content probes support identity but do not prove original chunk/item completeness.',
                    'Downloaded alternatives remain explicitly labeled candidates, not automatically FIN-RATE corpus.'])
    report['direct_sec_ready_documents']=db.execute("SELECT count(*) FROM documents WHERE acquisition_status='ready' AND sec_bytes_verified=1").fetchone()[0]
    report['previous_acquisition_status']=json.loads((RESULT/'status.json').read_text())
    report['phase']='recovery_mapped_with_remaining_identity_caveats'
    report['full_chunk_refinement']='whole normalized chunk >=1000 characters, unique available source-byte variant; exact raw offsets not inferred'
    report['original_first_pass_reports_preserved']=True
    atomic_json(RESULT/'company_coverage.json',company)
    atomic_json(RESULT/'hash_audit.json',audit)
    atomic_json(RESULT/'report.json',report)
    atomic_json(RESULT/'status.json',report)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    print(json.dumps(report,indent=2),flush=True)
    if errors:raise RuntimeError('Validation failed; staged mapping not published')
    guard()
    os.replace(staging,STORE/'manifest.sqlite')
    from finrate_raw_store_v2 import RawStore
    store=RawStore();tool_checks=[]
    try:
        for c in company:
            rows=store.list_filings(ticker=c['ticker'],limit=1)
            assert rows, c['ticker']
            fid=rows[0]['document_id']
            full=[];start=0
            while True:
                part=store.read(fid,start=start,length=24000)
                full.append(part['text'])
                if part['next_start'] is None:break
                start=part['next_start']
            row=store.db.execute('SELECT text_path FROM documents WHERE document_id=?',(fid,)).fetchone()
            expected=(STORE/row['text_path']).read_text()
            assert ''.join(full)==expected
            hits=store.grep(expected[:min(12,len(expected))],document_id=fid)
            assert hits['hits']
            tool_checks.append(dict(ticker=c['ticker'],document_id=fid,full_read_roundtrip=True,grep=True))
    finally:store.db.close()
    atomic_json(RESULT/'tool_validation.json',dict(companies=len(tool_checks),checks=tool_checks))
    report['tool_validation_companies']=len(tool_checks)
    report['mapping_published']=True
    atomic_json(RESULT/'report.json',report);atomic_json(RESULT/'status.json',report)
    update=(f"{report['completed_at']} — SEC declared-identity recovery finished: "
            f"direct SEC documents {report['direct_sec_ready_documents']}; "
            f"title-group statuses {report['group_statuses']}; "
            f"selected files {report['selected_documents']}. "
            "Hashes, SQLite and 43-company reader checks passed. Original first-pass reports and "
            "pre-recovery DB preserved. Source-only reconstruction; no QA/LLM/GPU work. "
            "Details result/finrate_sec_recovery_v1/report.json.\n\n")
    for name in ('context.md','process.md'):
        path=ROOT/name
        text=path.read_text()
        head,sep,body=text.partition('\n')
        path.write_text(head+sep+'\n'+update+body)


if __name__=='__main__':run()
