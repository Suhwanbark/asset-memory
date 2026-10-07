"""Source-only corpus/index preparation and an outcome-blind dataset custodian."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sqlite3
import time

from finrate_poc.data import documents, metadata, HF_REVISION
from finrate_poc.retrieval import CpuEmbedding, EMBEDDING_REVISION
from finrate_poc.runtime import ROOT, check_disk, cpu_only, digest, file_digest, watchdog, write_json
from .config import BASE, CORPUS, DATA, POLICY, SEED


def prepare_sources():
    check_disk(); BASE.mkdir(parents=True, exist_ok=True)
    write_json(BASE/'protocol.json', POLICY, immutable=True)
    manifest = BASE/'corpus_manifest.json'
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        if recorded['sqlite_sha256'] != file_digest(CORPUS): raise ValueError('Source database changed')
        return recorded
    work = BASE/'corpus.building.sqlite'
    if work.exists(): raise RuntimeError('Interrupted corpus build retained; inspect before retrying')
    con = sqlite3.connect(work)
    con.executescript('''
        CREATE TABLE sources(id TEXT PRIMARY KEY,title TEXT NOT NULL,asset TEXT NOT NULL,
          published TEXT NOT NULL,quarter TEXT NOT NULL,text TEXT NOT NULL,sha256 TEXT NOT NULL);
        CREATE INDEX source_asset_date ON sources(asset,published,id);
        CREATE VIRTUAL TABLE source_fts USING fts5(id UNINDEXED,title,text,tokenize='unicode61');
    ''')
    counts=Counter(); quarters=defaultdict(set); chars=Counter()
    for i,doc in enumerate(documents(DATA)):
        if i%100==0: check_disk()
        m=metadata(doc); day=m['filing_date']
        if not day: raise ValueError('Cannot silently omit a source with unknown date')
        quarter=day[:4]+'Q'+str((int(day[5:7])-1)//3+1)
        con.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',
          (doc['_id'],doc['title'],m['ticker'],day,quarter,doc['text'],digest(doc['text'])))
        con.execute('INSERT INTO source_fts VALUES(?,?,?)',(doc['_id'],doc['title'],doc['text']))
        counts[m['ticker']]+=1; quarters[m['ticker']].add(quarter); chars[m['ticker']]+=len(doc['text'])
    if len(counts)!=43 or sum(counts.values())!=15311: raise ValueError('Unexpected pinned corpus coverage')
    con.commit(); con.close(); work.replace(CORPUS)
    result={'dataset_revision':HF_REVISION,'corpus_zip_sha256':file_digest(DATA/'corpus/corpus.zip'),
      'sqlite_sha256':file_digest(CORPUS),'assets':sorted(counts),'sources':sum(counts.values()),
      'per_asset':{a:{'sources':counts[a],'quarters':sorted(quarters[a]),'characters':chars[a]} for a in sorted(counts)},
      'qa_access':False,'created':time.time()}
    write_json(manifest,result,immutable=True); print('CORPUS_READY',result['sources'],len(counts),flush=True)
    return result


def prepare_cohort():
    """Question/gold strings are copied, never printed or passed to bank/index builders."""
    dest=BASE/'cohort'
    if (dest/'manifest.json').exists(): return json.loads((dest/'manifest.json').read_text())
    con=sqlite3.connect(f'file:{CORPUS}?mode=ro',uri=True)
    sources={r[0]:{'asset':r[1],'published':r[2]} for r in con.execute('SELECT id,asset,published FROM sources')};con.close()
    old=json.loads((ROOT/'result/finrate_luna_v1/cohort/cohort_manifest.json').read_text())
    exposed=set(old['selected_ids']['dev'])|{'qa_3_000028','qa_3_000333','qa_3_000449'}
    excluded=set(sum(old['selected_ids'].values(),[]))|exposed
    rows=[]
    for task in POLICY['tasks']:
        for q in json.loads((DATA/f'qa/{task}.json').read_text()):
            if not q['doc_ids'] or any(s not in sources for s in q['doc_ids']):
                raise ValueError('Unresolved gold source; do not silently change the eligible population')
            rows.append({**q,'task':task,'source_ids':q['doc_ids'],
              'assets':sorted({sources[s]['asset'] for s in q['doc_ids']})})
    parent={q['q_id']:q['q_id'] for q in rows}; owners={}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    for q in rows:
        for s in q['doc_ids']:
            if s in owners: parent[find(q['q_id'])]=find(owners[s])
            owners[s]=q['q_id']
    groups=defaultdict(list)
    for q in rows: groups[find(q['q_id'])].append(q)
    pools={'dev':[],'evaluation':[]}; group_records=[]
    for g in groups.values():
        gid=min(q['q_id'] for q in g); seen=any(q['q_id'] in exposed for q in g)
        split='evaluation' if not seen and int(digest(SEED+'group'+gid)[:8],16)%3==0 else 'dev'
        pools[split].extend(q for q in g if q['q_id'] not in excluded)
        group_records.append({'id':gid,'size':len(g),'exposed':seen,'split':split})
    # Keep the seed; repair only infeasible task capacity using metadata, never
    # question content, desired difficulty, model scores or repeated seed trials.
    repairs=[]
    for task in POLICY['tasks']:
        while sum(q['task']==task for q in pools['evaluation'])<POLICY['evaluation_per_task']:
            candidates=[g for g in groups.values() if not any(q['q_id'] in exposed for q in g)
              and any(q['q_id'] in {r['q_id'] for r in pools['dev']} and q['task']==task for q in g)]
            if not candidates:raise ValueError('No source-disjoint capacity repair is possible')
            selected=min(candidates,key=lambda g:digest(SEED+'group'+min(q['q_id'] for q in g)))
            moved={q['q_id'] for q in selected};gid=min(moved)
            pools['evaluation'].extend(q for q in pools['dev'] if q['q_id'] in moved)
            pools['dev']=[q for q in pools['dev'] if q['q_id'] not in moved]
            for record in group_records:
                if record['id']==gid:record['split']='evaluation';record['capacity_repair']=task
            repairs.append({'group':gid,'required_task':task,'reason':'insufficient predeclared evaluation count'})
    chosen={}
    for split,pool in pools.items():
        n=POLICY['dev_per_task'] if split=='dev' else POLICY['evaluation_per_task']
        selected=[]
        for task in POLICY['tasks']:
            ordered=sorted((q for q in pool if q['task']==task),key=lambda q:digest(SEED+q['q_id']))
            if len(ordered)<n: raise ValueError(f'Insufficient predetermined pool: {split}/{task}')
            selected+=ordered[:n]
        chosen[split]=sorted(selected,key=lambda q:digest(SEED+'order'+q['q_id']))
    if {s for q in chosen['dev'] for s in q['doc_ids']} & {s for q in chosen['evaluation'] for s in q['doc_ids']}:
        raise ValueError('Gold-source overlap across splits')
    for split,qs in chosen.items():
        write_json(dest/f'{split}_inputs.json',[{k:q[k] for k in ['q_id','question','task']} for q in qs],immutable=True)
        write_json(dest/'gold'/f'{split}.json',qs,immutable=True)
    result={'seed':SEED,'all_sources_available':15311,'all_assets_available':43,
      'selected_ids':{s:[q['q_id'] for q in qs] for s,qs in chosen.items()},
      'groups':sorted(group_records,key=lambda r:r['id']),'excluded_prior_ids':sorted(excluded),
      'metadata_only_capacity_repairs':repairs,
      'counts':{s:dict(Counter(q['task'] for q in qs)) for s,qs in chosen.items()},
      'query_assets':{s:sorted({a for q in qs for a in q['assets']}) for s,qs in chosen.items()},
      'pool_counts':{s:dict(Counter(q['task'] for q in qs)) for s,qs in pools.items()},
      'files':{str(p.relative_to(dest)):file_digest(p) for p in dest.rglob('*.json')},
      'score_or_text_based_selection':False,'heldout_semantic_preview':False}
    write_json(dest/'manifest.json',result,immutable=True)
    print('COHORT_FIXED',result['counts'],'query_asset_counts',{s:len(a) for s,a in result['query_assets'].items()},flush=True)
    return result


def prepare_index():
    """Stream exact small source spans; checkpoint vector batches on CPU."""
    import numpy as np
    dest=BASE/'index';dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists(): return json.loads((dest/'manifest.json').read_text())
    embed=CpuEmbedding(); path=dest/'chunks.sqlite'
    chunks=sqlite3.connect(path)
    chunks.executescript('''
      CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,source_id TEXT,title TEXT,
      asset TEXT,published TEXT,start INTEGER,end INTEGER,text TEXT);
      CREATE INDEX IF NOT EXISTS chunk_source ON chunks(source_id);
      CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(id UNINDEXED,title,text,tokenize='unicode61');
    ''')
    mark=dest/'chunks_complete.json'
    if not mark.exists():
        source=sqlite3.connect(f'file:{CORPUS}?mode=ro',uri=True)
        existing={r[0] for r in chunks.execute('SELECT DISTINCT source_id FROM chunks')}
        for i,(sid,title,asset,day,text) in enumerate(source.execute('SELECT id,title,asset,published,text FROM sources ORDER BY id')):
            if sid in existing:continue
            check_disk()
            passages=embed.passages([{'_id':sid,'title':title,'text':text}])
            with chunks:
                for p in passages:
                    chunks.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)',(p['id'],sid,title,asset,day,p['start'],p['end'],p['text']))
                    chunks.execute('INSERT INTO chunk_fts VALUES(?,?,?)',(p['id'],title,p['text']))
            if i%200==0:print('CHUNKED_SOURCES',i,flush=True)
        source.close()
        write_json(mark,{'chunks':chunks.execute('SELECT count(*) FROM chunks').fetchone()[0],
                         'source_count':chunks.execute('SELECT count(DISTINCT source_id) FROM chunks').fetchone()[0]},immutable=True)
    n=chunks.execute('SELECT count(*) FROM chunks').fetchone()[0]
    ids=[r[0] for r in chunks.execute('SELECT id FROM chunks ORDER BY rowid')]
    write_json(dest/'ids.json',ids,immutable=True)
    vp=dest/'vectors.npy'; progress=dest/'vector_progress.json'
    if progress.exists():
        begin=json.loads(progress.read_text())['completed']; vectors=np.load(vp,mmap_mode='r+')
        if vectors.shape!=(n,384):raise ValueError('Vector checkpoint shape mismatch')
    else:
        begin=0;vectors=np.lib.format.open_memmap(vp,mode='w+',dtype='float32',shape=(n,384))
    started=time.monotonic()
    for start in range(begin,n,512):
        rows=chunks.execute('SELECT title,text FROM chunks ORDER BY rowid LIMIT 512 OFFSET ?',(start,)).fetchall()
        values=embed.encode([title[:150]+'\n'+body for title,body in rows])
        vectors[start:start+len(values)]=values;vectors.flush()
        write_json(progress,{'completed':start+len(values),'total':n})
        print('CPU_EMBEDDED',start+len(values),n,flush=True)
    chunks.close();del vectors
    result={'chunks':n,'sources':json.loads(mark.read_text())['source_count'],'assets':43,
      'embedding_revision':EMBEDDING_REVISION,'chunk_tokens':192,'stride':160,'device':'cpu',
      'hybrid':'equal-weight RRF k=60 over top200 dense and top200 SQLite FTS5 BM25 candidates',
      'elapsed_this_process_s':time.monotonic()-started,
      'files':{p.name:file_digest(p) for p in [path,vp,dest/'ids.json']}}
    write_json(dest/'manifest.json',result,immutable=True);print('INDEX_READY',n,flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['sources','cohort','index']);a=p.parse_args()
    cpu_only();check_disk();os.nice(10);BASE.mkdir(parents=True,exist_ok=True)
    status=BASE/'workers'/a.stage/'status.json'; stop=watchdog(status)
    write_json(status,{'state':'running','pid':os.getpid(),'stage':a.stage,'started':time.time()})
    try:
        {'sources':prepare_sources,'cohort':prepare_cohort,'index':prepare_index}[a.stage]()
        write_json(status,{'state':'completed','pid':os.getpid(),'finished':time.time()})
    except Exception as e:
        write_json(status,{'state':'failed','error':str(e),'pid':os.getpid()});raise
    finally:stop.set()

if __name__=='__main__':main()
