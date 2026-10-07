"""Apply only verified unique whole-normalized-chunk matches from source audit."""
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

from build_finrate_raw_corpus_v1 import ROOT,STORE,RESULT,guard,atomic_json
from finalize_finrate_raw_corpus_v1 import normalize
from finrate_raw_store_v2 import RawStore


def run():
    guard()
    audit=json.loads((RESULT/'ambiguity_full_chunk_audit.json').read_text())
    fixes=[r for r in audit if r['normalized_characters']>=1000 and len(r['full_chunk_matches'])==1]
    ids={r['chunk_id'] for r in fixes};chunks={}
    with zipfile.ZipFile(ROOT/'data/finrate_public_v1/corpus/corpus.zip') as z:
        for line in z.open(next(n for n in z.namelist() if n.endswith('corpus.jsonl'))):
            r=json.loads(line)
            if r['_id'] in ids:chunks[r['_id']]=normalize(r['text'])
    db=sqlite3.connect(STORE/'manifest.sqlite');db.row_factory=sqlite3.Row
    archive=RESULT/'before_full_chunk_refinement';archive.mkdir(exist_ok=True)
    if not (archive/'manifest.sqlite').exists():
        backup=sqlite3.connect(archive/'manifest.sqlite');db.backup(backup);backup.close()
        for name in ['report.json','status.json','company_coverage.json','group_coverage.json']:
            shutil.copyfile(RESULT/name,archive/name)
    for fix in fixes:
        group=fix['group_id'];chunk=chunks[fix['chunk_id']];matches=[]
        for r in db.execute("SELECT d.document_id,d.text_path FROM documents d JOIN group_candidates g USING(document_id) WHERE g.group_id=? AND d.acquisition_status='ready'",(group,)):
            if chunk in normalize((STORE/r['text_path']).read_text()):matches.append(r['document_id'])
        assert matches==fix['full_chunk_matches'] and len(chunk)==fix['normalized_characters']
        fid=matches[0]
        with db:
            db.execute('INSERT OR REPLACE INTO document_selection VALUES (?,?,?)',(group,fid,'unique_full_normalized_chunk_containment'))
            db.execute("UPDATE documents SET selection_status='selected' WHERE document_id=?",(fid,))
            db.execute("UPDATE title_groups SET status='source_ready_with_mapping_caveats' WHERE group_id=?",(group,))
            db.execute("UPDATE finrate_mapping SET document_id=?,mapping_status='unique_full_normalized_chunk_match_not_exact_offset' WHERE chunk_id=?",(fid,fix['chunk_id']))
    store=RawStore();trace=[]
    for fix in fixes:
        fid=fix['full_chunk_matches'][0];row=db.execute('SELECT text_path FROM documents WHERE document_id=?',(fid,)).fetchone();text=(STORE/row['text_path']).read_text();parts=[];start=0
        while True:
            result=store.read(fid,start=start,length=24000);trace.append(dict(tool='read',args=dict(document_id=fid,start=start,length=24000),result=result));parts.append(result['text'])
            if result['next_start'] is None:break
            start=result['next_start']
        assert ''.join(parts)==text
    with (RESULT/'refinement_read_trace.jsonl').open('w') as out:
        for t in trace:out.write(json.dumps(t,ensure_ascii=False)+'\n')
    groups=json.loads((RESULT/'group_coverage.json').read_text())
    for g in groups:
        g['status']=db.execute('SELECT status FROM title_groups WHERE group_id=?',(g['group_id'],)).fetchone()[0]
        g['selected_documents']=[r[0] for r in db.execute('SELECT document_id FROM document_selection WHERE group_id=? ORDER BY document_id',(g['group_id'],))]
        g['full_chunk_matches_added']=sum(f['group_id']==g['group_id'] for f in fixes)
    atomic_json(RESULT/'group_coverage.json',groups)
    import collections
    companies=[]
    for ticker in sorted({g['ticker'] for g in groups}):
        gs=[g for g in groups if g['ticker']==ticker];companies.append(dict(ticker=ticker,total_groups=len(gs),statuses=dict(collections.Counter(g['status'] for g in gs)),selected_documents=len({d for g in gs for d in g['selected_documents']})))
    atomic_json(RESULT/'company_coverage.json',companies)
    report=json.loads((RESULT/'report.json').read_text());report['group_statuses']=dict(db.execute('SELECT status,count(*) FROM title_groups GROUP BY 1'));report['chunk_mapping_statuses']=dict(db.execute('SELECT mapping_status,count(*) FROM finrate_mapping GROUP BY 1'))
    selected=db.execute('SELECT count(*),sum(source_bytes),sum(text_characters) FROM corpus_documents').fetchone();report.update(selected_documents=selected[0],selected_source_bytes=selected[1],selected_text_characters=selected[2])
    for key,field in [('year_coverage','title_date'),('form_coverage','form_label')]:
        for value,summary in report[key].items():
            gs=[g for g in groups if (g[field][:4] if key=='year_coverage' else g[field])==value];summary['statuses']=dict(collections.Counter(g['status'] for g in gs))
    report['full_chunk_refinement']=dict(resolved_groups=len(fixes),whole_read_roundtrips=len(fixes),audit='ambiguity_full_chunk_audit.json',source_bytes_changed=False)
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert not list(db.execute('PRAGMA foreign_key_check'))
    atomic_json(RESULT/'report.json',report);atomic_json(RESULT/'status.json',report)
    with db:db.execute("INSERT OR REPLACE INTO store_metadata VALUES ('full_chunk_refinement','unique full normalized chunk containment >=1000 characters; audit retained; exact raw offsets not inferred')")
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();store.db.close()
    print(json.dumps(report['group_statuses']),report['selected_documents'],flush=True)


if __name__=='__main__':run()
