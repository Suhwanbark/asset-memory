"""Freeze reader notebooks, reconstruct past snapshots, and make a text-matched flat control."""
import json
from pathlib import Path

from finrate_poc.runtime import digest,file_digest,write_json
from .config import BASE
from .workspace import Notebook,ANCHOR


def reader_records(book):
    """Expose committed period states, not merely the latest overwritten page.

    Intermediate edits are excluded; identical revisions shared by several
    checkpoints appear once. Original statements, anchors and dates survive.
    """
    selected=set()
    for row in book.db.execute('SELECT state_json FROM checkpoints ORDER BY period'):
        state=json.loads(row['state_json'])
        selected.update((path,entry['revision']) for path,entry in state.items())
    latest={r['path']:r['revision'] for r in book.db.execute('SELECT path,revision FROM pages')}
    for path,revision in sorted(selected):
        row=book.db.execute('SELECT body,asof FROM versions WHERE path=? AND revision=?',(path,revision)).fetchone()
        ident=path if latest[path]==revision else path+'@revision_'+str(revision)
        yield {'path':ident,'topic_path':path,'asset':path.split('/')[0],
               'body':row['body'],'revision':revision,'asof':row['asof'],
               'historical':latest[path]!=revision}


def snapshot(bank_path,period,destination):
    src=Notebook(bank_path,readonly=True);dst=Notebook(destination)
    try:
        if dst.catalog():raise ValueError('Snapshot destination must be empty')
        row=src.db.execute('SELECT state_json,asof FROM checkpoints WHERE period=?',(period,)).fetchone()
        if not row:raise ValueError('Checkpoint not found')
        state=json.loads(row['state_json'])
        with dst.db:
            for path,entry in sorted(state.items()):
                version=src.db.execute('SELECT body,asof FROM versions WHERE path=? AND revision=?',(path,entry['revision'])).fetchone()
                if digest(version['body'])!=entry['body_sha256']:raise ValueError('Past version changed')
                if version['asof']>row['asof']:raise ValueError('A historical checkpoint contains a future revision')
                dst.db.execute('INSERT INTO pages VALUES(?,?,?,?,?)',(path,path.split('/')[0],version['body'],entry['revision'],version['asof']))
                dst.db.execute('INSERT INTO page_fts VALUES(?,?,?)',(path,path,version['body']))
        return {'period':period,'asof':row['asof'],'pages':len(state),'future_versions_excluded':True}
    finally:src.close();dst.close()


def export_all():
    manifest=json.loads((BASE/'corpus_manifest.json').read_text());dest=BASE/'reader_banks'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():return json.loads((dest/'manifest.json').read_text())
    if (dest/'bank.sqlite').exists() or (dest/'flat.sqlite').exists():raise ValueError('Inspect incomplete export before retrying')
    books={a:BASE/'banks'/a for a in manifest['assets']}
    for asset,root in books.items():
        done=json.loads((root/'completed.json').read_text())
        if done['state']!='completed' or not done['full_asset_history'] or done['bank_sha256']!=file_digest(root/'bank.sqlite'):
            raise ValueError('Asset bank incomplete or changed: '+asset)
    bank=Notebook(dest/'bank.sqlite');flat=Notebook(dest/'flat.sqlite');mapping=[];builds={}
    try:
        for asset,root in books.items():
            builds[asset]=json.loads((root/'completed.json').read_text());src=Notebook(root/'bank.sqlite',readonly=True)
            for r in reader_records(src):
                with bank.db:
                    bank.db.execute('INSERT INTO pages VALUES(?,?,?,?,?)',(r['path'],asset,r['body'],r['revision'],r['asof']))
                    bank.db.execute('INSERT INTO page_fts VALUES(?,?,?)',(r['path'],r['topic_path']+' '+r['asof'],r['body']))
                # Identical authored body split into contiguous nonoverlapping
                # blocks. A common page label preserves the information in the
                # original title; only grouping/navigation/granularity changes.
                text=r['body'];start=0;parts=[]
                while start<len(text):
                    end=min(len(text),start+1800)
                    if end<len(text):
                        newline=text.rfind('\n',start+900,end)
                        if newline>start:end=newline+1
                        for anchor in ANCHOR.finditer(text):
                            if anchor.start()<end<anchor.end():end=anchor.end();break
                    segment=text[start:end];fid=asset+'/fragment_'+digest(r['path']+':'+str(start))[:16]
                    content='Original page: '+r['path']+'\n'+segment
                    with flat.db:
                        flat.db.execute('INSERT INTO pages VALUES(?,?,?,?,?)',(fid,asset,content,r['revision'],r['asof']))
                        flat.db.execute('INSERT INTO page_fts VALUES(?,?,?)',(fid,r['topic_path']+' '+r['asof'],content))
                    mapping.append({'fragment':fid,'original_page':r['path'],'start':start,'end':end,'body_sha256':digest(segment)})
                    parts.append(segment);start=end
                if ''.join(parts)!=text:raise ValueError('Flat control lost authored text')
            src.close()
    finally:bank.close();flat.close()
    result={'assets':manifest['assets'],'source_count':manifest['sources'],'builds':builds,
      'files':{name:file_digest(dest/name) for name in ['bank.sqlite','flat.sqlite']},
      'flat_intervention':'Identical authored text and source anchors; contiguous fragments with original page labels, no topic-page navigation. This tests grouping/granularity, not incremental update by itself.',
      'no_new_flat_model_calls':True,'reader_includes_committed_historical_revisions':True}
    write_json(dest/'flat_mapping.json',mapping,immutable=True)
    write_json(dest/'manifest.json',result,immutable=True);return result
