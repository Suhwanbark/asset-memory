"""Lean full-document reader; no chunking, embeddings, DOM-span DB, or QA data."""
from __future__ import annotations
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'data/finrate_raw_v1'
BLOCKS = {'div','p','li','ul','ol','section','article','header','footer','h1','h2','h3','h4','h5','h6','blockquote','pre','table','tr'}
VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
OMIT = {'head','script','style','ix:header','ix:hidden'}


class TextProjection(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.last = ''

    def hidden(self):
        return bool(self.stack and self.stack[-1][1])

    def emit(self, text):
        if text:
            self.parts.append(text)
            self.last = text[-1]

    def newline(self):
        if self.last and self.last != '\n':
            self.emit('\n')

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = self.hidden() or tag in OMIT or 'hidden' in attrs or bool(re.search(
            r'display\s*:\s*none|visibility\s*:\s*hidden',attrs.get('style') or '',re.I))
        if tag not in VOID:
            self.stack.append((tag,hidden))
        if hidden:
            return
        if tag in BLOCKS or tag in {'br','hr'}:
            self.newline()
        if tag == 'img' and attrs.get('alt'):
            self.emit('[image alt: '+attrs['alt']+']')

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag,attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        hidden = self.hidden()
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if not hidden:
            if tag in {'td','th'}:
                self.emit('\t')
            elif tag in BLOCKS:
                self.newline()

    def handle_data(self, value):
        if not self.hidden():
            value = re.sub(r'\s+',' ',value)
            if value.strip() or self.last not in {'',' ','\n','\t'}:
                self.emit(value)


def project(raw, suffix='.html'):
    # SEC HTML usually declares UTF-8 or Windows-1252; preserve original bytes.
    header = raw[:8192]
    match = re.search(br'(?:charset\s*=\s*["\x27]?|encoding=["\x27])([\w-]+)',header,re.I)
    encoding = match.group(1).decode('ascii') if match else 'utf-8'
    try:
        html = raw.decode(encoding)
    except (UnicodeError,LookupError):
        html = raw.decode('windows-1252',errors='replace')
    if suffix == '.txt' and not re.search(r'<(?:html|body|div|p)[\s>]',html,re.I):
        text = html
    else:
        parser = TextProjection()
        parser.feed(html)
        parser.close()
        text = ''.join(parser.parts)
    sections = [dict(title='Full document',start=0,end=len(text),basis='entire_readable_projection')]
    # Navigation hints, never completeness assertions or filtering boundaries.
    for match in re.finditer(r'(?im)^\s*(Item\s+\d+(?:\.\d+)?[A-Z]?\b[^\n]{0,200})',text):
        sections.append(dict(title=match.group(1).strip(),start=match.start(),end=match.end(),basis='heuristic_item_line_only'))
    return text,sections


class RawStore:
    def __init__(self, root=STORE, include_candidates=False):
        self.root = Path(root)
        self.db = sqlite3.connect(f'file:{self.root / "manifest.sqlite"}?mode=ro',uri=True)
        self.db.row_factory = sqlite3.Row
        self.include_candidates = include_candidates

    def scope(self, alias='d'):
        if self.include_candidates:
            return f"{alias}.acquisition_status='ready'"
        if 'selection_status' not in {r[1] for r in self.db.execute('PRAGMA table_info(documents)')}:
            raise RuntimeError('Source mapping is still in progress; corpus is not released for QA.')
        return f"{alias}.acquisition_status='ready' AND {alias}.selection_status='selected'"

    def list_filings(self,ticker=None,form=None,limit=50,offset=0):
        where = [self.scope()]
        params = []
        if ticker:
            where.append('EXISTS(SELECT 1 FROM json_each(f.ticker_aliases) WHERE value=?)');params.append(ticker)
        if form:
            where.append('f.form=?');params.append(form)
        return [dict(r) for r in self.db.execute('SELECT f.*,d.document_id,d.text_characters,d.source_url FROM filings f JOIN documents d USING(filing_id) WHERE '+' AND '.join(where)+' ORDER BY f.filed_at,f.filing_id LIMIT ? OFFSET ?',params+[min(limit,200),offset])]

    def read(self,document_id,start=0,length=12000):
        r = self.db.execute('SELECT text_path FROM documents d WHERE document_id=? AND '+self.scope(),(document_id,)).fetchone()
        if not r:
            raise KeyError(document_id)
        text = (self.root/r['text_path']).read_text()
        start = max(0,min(start,len(text)));end=min(len(text),start+max(1,min(length,24000)))
        return dict(document_id=document_id,start=start,end=end,total_characters=len(text),next_start=end if end<len(text) else None,text=text[start:end])

    def grep(self,query,ticker=None,document_id=None,limit=20,offset=0):
        if not query:
            raise ValueError('query must not be empty')
        sql = 'SELECT d.document_id,d.text_path FROM documents d JOIN filings f USING(filing_id) WHERE '+self.scope()
        params=[]
        if ticker:
            sql+=' AND EXISTS(SELECT 1 FROM json_each(f.ticker_aliases) WHERE value=?)';params.append(ticker)
        if document_id:
            sql+=' AND d.document_id=?';params.append(document_id)
        hits=[];seen=0
        pattern=re.compile(re.escape(query),re.I)
        for r in self.db.execute(sql+' ORDER BY d.document_id',params):
            text=(self.root/r['text_path']).read_text()
            for match in pattern.finditer(text):
                if seen>=offset:
                    start=max(0,match.start()-160);end=min(len(text),match.end()+160)
                    hits.append(dict(document_id=r['document_id'],match_start=match.start(),match_end=match.end(),start=start,end=end,text=text[start:end]))
                    if len(hits)>=min(limit,100):
                        return dict(hits=hits,next_offset=seen+1)
                seen+=1
        return dict(hits=hits,next_offset=None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tool',choices=['list_filings','read','grep']);p.add_argument('--args',default='{}');p.add_argument('--store',type=Path,default=STORE)
    a=p.parse_args();print(json.dumps(getattr(RawStore(a.store),a.tool)(**json.loads(a.args)),ensure_ascii=False,indent=2))
