"""Common read/search tools and a source-anchored, versioned editable notebook.

The tools constrain access and serialization, not the agent's reasoning order.
Original sources are immutable. A source anchor proves provenance, not entailment.
"""
import json
from pathlib import Path
import re
import sqlite3
import time

from finrate_poc.model import token_count
from finrate_poc.runtime import check_disk, digest

ANCHOR = re.compile(r'\b(doc_\d+):(\d+):(\d+)\b')


def connect(path, readonly=False):
    c=sqlite3.connect(f'file:{Path(path)}?mode=ro' if readonly else str(path),uri=readonly)
    c.row_factory=sqlite3.Row
    return c


def keyword_query(query, mode='any'):
    words=re.findall(r'[A-Za-z0-9]+',query)
    stop={'a','an','the','of','to','and','or','in','on','for','from','with','is','are','was','were','did','does','do','how','what','which','between','its','their','it','as','by','at','has','have'}
    words=list(dict.fromkeys(w for w in words if w.lower() not in stop))[:40]
    if not words: return None
    return (' AND ' if mode=='all' else ' OR ').join('"'+w+'"' for w in words)


def excerpt_start(text, query):
    words=re.findall(r'[A-Za-z][A-Za-z0-9]{3,}',query)
    words=[w for w in words if w.lower() not in {'company','companies','annual','report','reporting','between','compare','their','filing'}]
    for word in sorted(words,key=lambda w:-len(w)):
        m=re.search(r'\b'+re.escape(word)+r'\b',text,re.I)
        if m: return max(0,m.start()-250)
    return 0


class Notebook:
    def __init__(self,path,*,readonly=False):
        self.path=Path(path);self.readonly=readonly
        if not readonly:self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=connect(path,readonly)
        if not readonly:
            self.db.executescript('''
              CREATE TABLE IF NOT EXISTS pages(path TEXT PRIMARY KEY,asset TEXT,body TEXT,revision INTEGER,asof TEXT);
              CREATE VIRTUAL TABLE IF NOT EXISTS page_fts USING fts5(path UNINDEXED,title,body,tokenize='unicode61');
              CREATE TABLE IF NOT EXISTS versions(path TEXT,revision INTEGER,body TEXT,asof TEXT,action_id TEXT,PRIMARY KEY(path,revision));
              CREATE TABLE IF NOT EXISTS actions(id TEXT PRIMARY KEY,result TEXT,previous_hash TEXT,event_hash TEXT);
              CREATE TABLE IF NOT EXISTS checkpoints(period TEXT PRIMARY KEY,asof TEXT,state_json TEXT,state_hash TEXT);
            ''')

    def close(self):self.db.close()

    def catalog(self,asset=None):
        sql='SELECT path,asset,revision,asof,length(body) AS characters FROM pages'
        args=[]
        if asset:sql+=' WHERE asset=?';args.append(asset)
        return [dict(r) for r in self.db.execute(sql+' ORDER BY path',args)]

    def edit(self,operations,*,asset,asof,source_db,action_id):
        if self.readonly:raise ValueError('Evaluation notebook is read-only')
        cached=self.db.execute('SELECT result FROM actions WHERE id=?',(action_id,)).fetchone()
        if cached:return json.loads(cached[0])
        if not isinstance(operations,list) or not 1<=len(operations)<=12:raise ValueError('Provide 1..12 edits')
        checked=[];seen=set()
        for op in operations:
            path=op['path'];body=op['content'];expected=op['expected_revision']
            if not isinstance(path,str) or not re.fullmatch(re.escape(asset)+r'/[a-z0-9_\-/]{1,120}',path) or '..' in path:
                raise ValueError('Path must be under the current asset with a lowercase descriptive slug')
            if path in seen:raise ValueError('Duplicate edit path');
            seen.add(path)
            if type(expected)!=int or expected<0:raise ValueError('expected_revision must be a nonnegative integer')
            old=self.db.execute('SELECT revision FROM pages WHERE path=?',(path,)).fetchone()
            revision=old[0] if old else 0
            if expected!=revision:raise ValueError(f'Revision conflict for {path}: actual {revision}')
            if not isinstance(body,str) or not body.strip() or token_count(body)>6000:
                raise ValueError('Page must be nonempty and at most6000tokens; consolidate or use multiple topics')
            refs=list(ANCHOR.finditer(body))
            if not refs:raise ValueError('Each factual page must contain exact original source anchors')
            for ref in refs:
                sid,start,end=ref.group(1),int(ref.group(2)),int(ref.group(3))
                row=source_db.execute('SELECT published,length(text) AS n FROM sources WHERE id=?',(sid,)).fetchone()
                if not row or row['published']>asof or not 0<=start<end<=row['n'] or end-start>18000:
                    raise ValueError(f'Invalid or unavailable source anchor {ref.group(0)}')
            checked.append((path,body,revision+1))
        result={'updated':[{'path':p,'revision':r,'characters':len(b)} for p,b,r in checked],
                'asof':asof,'semantic_support_verified':False}
        serial=json.dumps(result,sort_keys=True,ensure_ascii=False)
        prior=self.db.execute('SELECT event_hash FROM actions ORDER BY rowid DESC LIMIT 1').fetchone()
        previous=prior[0] if prior else '0'*64
        payload=json.dumps({'id':action_id,'edits':checked,'asof':asof,'previous':previous},sort_keys=True,ensure_ascii=False)
        check_disk()
        with self.db:
            for p,b,r in checked:
                self.db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?,?,?)',(p,asset,b,r,asof))
                self.db.execute('DELETE FROM page_fts WHERE path=?',(p,))
                self.db.execute('INSERT INTO page_fts VALUES(?,?,?)',(p,p,b))
                self.db.execute('INSERT INTO versions VALUES(?,?,?,?,?)',(p,r,b,asof,action_id))
            self.db.execute('INSERT INTO actions VALUES(?,?,?,?)',(action_id,serial,previous,digest(payload)))
        return result

    def checkpoint(self,period,asof):
        state={r['path']:{'revision':r['revision'],'body_sha256':digest(r['body'])} for r in self.db.execute('SELECT * FROM pages ORDER BY path')}
        serial=json.dumps(state,sort_keys=True)
        existing=self.db.execute('SELECT state_json,asof FROM checkpoints WHERE period=?',(period,)).fetchone()
        if existing:
            if existing[0]!=serial or existing[1]!=asof:raise ValueError('Checkpoint cannot be rewritten')
        else:
            with self.db:self.db.execute('INSERT INTO checkpoints VALUES(?,?,?,?)',(period,asof,serial,digest(serial)))
        return {'period':period,'asof':asof,'pages':len(state),'state_sha256':digest(serial)}


class Workspace:
    """All QA representations expose these same tools, filters, and source fallback."""
    def __init__(self,corpus,*,representation='raw',chunks=None,notebook=None,asof='9999-12-31',writer_asset=None):
        self.sources=connect(corpus,True);self.representation=representation;self.asof=asof
        self.chunks=connect(chunks,True) if chunks else None;self.notebook=notebook;self.writer_asset=writer_asset
        self.visible_anchors=set();self.direct_anchors=set();self.visited_sources=set();self.scratch=''

    def close(self):
        self.sources.close()
        if self.chunks:self.chunks.close()

    def observe(self,value):
        """Replay observational access, including indirect citations embedded in notes."""
        text=json.dumps(value,ensure_ascii=False)
        self.visible_anchors.update(m.group(0) for m in ANCHOR.finditer(text))
        def visit(v):
            if isinstance(v,dict):
                if v.get('source_id') and v.get('text') is not None:
                    self.visited_sources.add(v['source_id'])
                    if 'start' in v and 'end' in v:self.direct_anchors.add(f"{v['source_id']}:{v['start']}:{v['end']}")
                for x in v.values():visit(x)
            elif isinstance(v,list):
                for x in v:visit(x)
        visit(value)

    def _target(self,scope):
        if scope not in {'workspace','sources'}:raise ValueError('scope must be workspace or sources')
        if scope=='sources' or self.representation=='raw':return self.sources,'sources','source_fts','id','text','published'
        if self.representation=='chunks':return self.chunks,'chunks','chunk_fts','id','text','published'
        if self.notebook:return self.notebook.db,'pages','page_fts','path','body','asof'
        raise ValueError('No workspace representation')

    def call(self,name,args,*,action_id=''):
        check_disk()
        if name=='scratch':
            text=args.get('text','')
            if not isinstance(text,str) or token_count(text)>2500:raise ValueError('Scratch limit2500tokens')
            self.scratch=text;return {'saved':True,'text':text}
        if name=='edit':
            if not self.writer_asset or not self.notebook:raise ValueError('Shared QA workspace cannot be edited')
            for operation in args['operations']:
                for match in ANCHOR.finditer(operation['content']):
                    anchor=match.group(0)
                    if anchor not in self.visible_anchors:
                        raise ValueError('Use an EXACT anchor returned by a read/search or present in a read memory page. Do not guess subrange offsets; request a read to obtain a new anchor: '+anchor)
            return self.notebook.edit(args['operations'],asset=self.writer_asset,asof=self.asof,
                                      source_db=self.sources,action_id=action_id)
        scope=args.get('scope','workspace'); db,table,fts,idcol,bodycol,datecol=self._target(scope)
        if name=='read':
            ident=args['id'];requested_start=int(args.get('start',0));length=int(args.get('length',6000))
            if requested_start<0 or not 1<=length<=12000:raise ValueError('Read length1..12000characters, nonnegative start')
            row=db.execute(f'SELECT * FROM {table} WHERE {idcol}=? AND {datecol}<=?',(ident,self.asof)).fetchone()
            if not row:raise ValueError('Item absent or unavailable at this snapshot')
            r=dict(row);body=r[bodycol];end=min(len(body),requested_start+length)
            if requested_start>=len(body):raise ValueError('Read starts beyond the item')
            if table=='sources':
                result={'id':f'{ident}:{requested_start}:{end}','source_id':ident,'title':r['title'],
                  'asset':r['asset'],'published':r['published'],'start':requested_start,'end':end,
                  'text':body[requested_start:end],'total_characters':len(body)}
            elif table=='chunks':
                start=r['start']+requested_start;last=r['start']+end
                result={'id':f"{r['source_id']}:{start}:{last}",'source_id':r['source_id'],'title':r['title'],
                  'asset':r['asset'],'published':r['published'],'start':start,'end':last,'text':body[requested_start:end]}
            else:
                result={'id':ident,'asset':r['asset'],'revision':r['revision'],'asof':r['asof'],
                  'content':body[requested_start:end],'start':requested_start,'end':end,'total_characters':len(body),
                  'note':'Derived memory; source anchors permit verification but do not certify the summary.'}
            return result
        if name not in {'list','search'}:raise ValueError('Unknown tool')
        limit=min(max(int(args.get('limit',12)),1),40);offset=max(int(args.get('offset',0)),0)
        where=[f'd.{datecol}<=?'];params=[min(str(args.get('before',self.asof)),self.asof)]
        if args.get('asset'):where.append('d.asset=?');params.append(args['asset'])
        if args.get('after'):where.append(f'd.{datecol}>=?');params.append(args['after'])
        if name=='list':
            rows=db.execute(f'SELECT d.* FROM {table} d WHERE '+ ' AND '.join(where)+f' ORDER BY d.{datecol},d.{idcol} LIMIT ? OFFSET ?',[*params,limit,offset]).fetchall()
        else:
            query=args.get('query','');matched=keyword_query(query,args.get('match','any'))
            if not matched:return {'items':[],'message':'No searchable terms'}
            rows=db.execute(f'SELECT d.*,bm25({fts}) AS score FROM {fts} JOIN {table} d ON d.{idcol}={fts}.{idcol} '
              f'WHERE {fts} MATCH ? AND '+' AND '.join(where)+f' ORDER BY score,d.{idcol} LIMIT ? OFFSET ?',
              [matched,*params,limit,offset]).fetchall()
        items=[]
        for row in rows:
            r=dict(row);item={k:v for k,v in r.items() if k not in {bodycol,'sha256'}}
            item['id']=r[idcol];item['characters']=len(r[bodycol])
            if name=='search':
                start=excerpt_start(r[bodycol],args['query']);end=min(len(r[bodycol]),start+900)
                item['preview']=r[bodycol][start:end];item['read_start']=start
                if table in {'sources','chunks'}:
                    sid=r['id'] if table=='sources' else r['source_id'];base=0 if table=='sources' else r['start']
                    item.update(source_id=sid,start=base+start,end=base+end,text=item.pop('preview'),
                                evidence_id=f'{sid}:{base+start}:{base+end}')
            items.append(item)
        return {'items':items,'offset':offset,'next_offset':offset+len(items),'scope':scope}

    def resolve(self,anchor):
        match=ANCHOR.fullmatch(anchor)
        if not match or anchor not in self.visible_anchors:raise ValueError('Citation was not present in observed material')
        sid,start,end=match.group(1),int(match.group(2)),int(match.group(3))
        row=self.sources.execute('SELECT title,asset,published,text FROM sources WHERE id=?',(sid,)).fetchone()
        if not row or row['published']>self.asof or not 0<=start<end<=len(row['text']):raise ValueError('Invalid source citation')
        return {'id':anchor,'source_id':sid,'title':row['title'],'asset':row['asset'],'published':row['published'],
          'start':start,'end':end,'text':row['text'][start:end],
          'directly_read':anchor in self.direct_anchors,'indirect_memory_anchor':anchor not in self.direct_anchors}
