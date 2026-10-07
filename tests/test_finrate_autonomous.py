import json
import sqlite3
from pathlib import Path
import sys

import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'method'))
from finrate_autonomous.workspace import Notebook,Workspace
from finrate_autonomous.agent import run,bounded_result


def make_corpus(tmp_path):
    path=tmp_path/'sources.sqlite';db=sqlite3.connect(path)
    db.executescript('CREATE TABLE sources(id TEXT PRIMARY KEY,title TEXT,asset TEXT,published TEXT,quarter TEXT,text TEXT,sha256 TEXT); CREATE VIRTUAL TABLE source_fts USING fts5(id UNINDEXED,title,text);')
    rows=[('doc_000001','AAA 2020 policy','AAA','2020-01-01','2020Q1','The company does not hedge its euro royalties.','x'),
          ('doc_000002','AAA 2021 policy','AAA','2021-01-01','2021Q1','The company began hedging euro royalties in 2021.','y')]
    db.executemany('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',rows)
    db.executemany('INSERT INTO source_fts VALUES(?,?,?)',[(r[0],r[1],r[5]) for r in rows]);db.commit();db.close()
    return path




class ScriptedModel:
    def __init__(self,decisions):self.decisions=iter(decisions);self.prompts=[]
    def invoke(self,prompt,folder,*,schema):
        self.prompts.append(prompt)
        return json.dumps(next(self.decisions))


class AutonomousTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.tmp_path=Path(self.temp.name);self.corpus=make_corpus(self.tmp_path)

    def test_asof_and_original_spans(self):
        corpus,tmp_path=self.corpus,self.tmp_path
        w=Workspace(corpus,asof='2020-12-31')
        results=w.call('search',{'query':'euro royalties'})
        assert {r['source_id'] for r in results['items']}=={'doc_000001'}
        with self.assertRaises(ValueError):w.call('read',{'id':'doc_000002'})
        read=w.call('read',{'id':'doc_000001','start':4,'length':7});w.observe(read)
        assert w.resolve(read['id'])['text']=='company'
        with self.assertRaises(ValueError):w.resolve('doc_000001:0:3')
        w.close()


    def test_notebook_revision_atomicity_provenance_and_idempotence(self):
        corpus,tmp_path=self.corpus,self.tmp_path
        book=Notebook(tmp_path/'book.sqlite');w=Workspace(corpus,notebook=book,writer_asset='AAA',asof='2020-12-31')
        good={'path':'AAA/fx','expected_revision':0,'content':'No hedge [doc_000001:0:45].'}
        bad={'path':'AAA/future','expected_revision':0,'content':'Hedges [doc_000002:0:10].'}
        with self.assertRaises(ValueError):w.call('edit',{'operations':[good,bad]},action_id='bad')
        assert book.catalog()==[]
        w.observe(w.call('read',{'id':'doc_000001','length':45}))
        a=w.call('edit',{'operations':[good]},action_id='same')
        assert w.call('edit',{'operations':[good]},action_id='same')==a
        with self.assertRaises(ValueError):w.call('edit',{'operations':[good]},action_id='conflict')
        first=book.checkpoint('2020Q1','2020-01-01')
        changed={**good,'expected_revision':1,'content':'2020: no hedge; scope euro royalties [doc_000001:0:45].'}
        w.call('edit',{'operations':[changed]},action_id='next')
        assert book.db.execute('SELECT body FROM versions WHERE revision=1').fetchone()[0]==good['content']
        assert first['pages']==1
        with self.assertRaises(ValueError):book.checkpoint('2020Q1','2020-01-01')
        w.close();book.close()


    def test_bounded_read_cannot_claim_unseen_suffix(self):
        corpus,tmp_path=self.corpus,self.tmp_path
        text='A company exposure description. '*1000
        bounded=bounded_result({'id':'doc_000001:0:31000','source_id':'doc_000001','start':0,'end':31000,'text':text},100)
        assert bounded['end']==len(bounded['text'])
        assert bounded['id']==f"doc_000001:0:{bounded['end']}"
        assert bounded['truncated']


    def test_agent_can_finish_without_mandatory_plan_or_retrieval(self):
        corpus,tmp_path=self.corpus,self.tmp_path
        model=ScriptedModel([{'finish':True,'actions':[],'answer':'Evidence is unavailable.','claims':[]}])
        w=Workspace(corpus)
        result=run({'question':'Unknown?'},w,model,tmp_path/'run')
        assert result['status']=='completed' and result['model_calls']==1 and result['tool_calls']==0
        w.close()


    def test_agent_reads_then_finishes_without_review(self):
        corpus,tmp_path=self.corpus,self.tmp_path
        model=ScriptedModel([
          {'finish':False,'actions':[{'tool':'read','arguments_json':json.dumps({'id':'doc_000001','length':46})}],'answer':'','claims':[]},
          {'finish':True,'actions':[],'answer':'No hedge.','claims':[{'text':'No hedge.','evidence_ids':['doc_000001:0:46']}]}])
        w=Workspace(corpus)
        result=run({'question':'What hedge policy?'},w,model,tmp_path/'run')
        assert result['model_calls']==2 and result['actions']==['read'] and result['invalid_citation_ids']==[]
        assert result['evidence'][0]['text']=='The company does not hedge its euro royalties.'
        w.close()


    def test_shared_qa_notebook_cannot_be_written(self):
        corpus,tmp_path=self.corpus,self.tmp_path
        book=Notebook(tmp_path/'book.sqlite');w=Workspace(corpus,representation='bank',notebook=book)
        with self.assertRaises(ValueError):w.call('edit',{'operations':[]})
        w.close();book.close()

    def test_unobserved_citation_cannot_be_written(self):
        book=Notebook(self.tmp_path/'book.sqlite')
        w=Workspace(self.corpus,notebook=book,writer_asset='AAA',asof='2020-12-31')
        with self.assertRaises(ValueError):
            w.call('edit',{'operations':[{'path':'AAA/fx','expected_revision':0,'content':'No hedge [doc_000001:0:45].'}]},action_id='unseen')
        assert book.catalog()==[]
        w.close();book.close()

    def test_historical_snapshot_uses_previous_revision(self):
        from finrate_autonomous.export import snapshot,reader_records
        book=Notebook(self.tmp_path/'book.sqlite')
        w=Workspace(self.corpus,notebook=book,writer_asset='AAA',asof='2020-01-01')
        w.observe(w.call('read',{'id':'doc_000001','length':45}))
        old='2020 no hedge [doc_000001:0:45].'
        w.call('edit',{'operations':[{'path':'AAA/fx','expected_revision':0,'content':old}]},action_id='old')
        book.checkpoint('2020Q1','2020-01-01')
        w.asof='2021-01-01';w.observe(w.call('read',{'id':'doc_000002','length':40}))
        # Use a real observed subset of the newer original.
        new=old+' 2021 hedging began [doc_000002:0:40].'
        w.call('edit',{'operations':[{'path':'AAA/fx','expected_revision':1,'content':new}]},action_id='new')
        book.checkpoint('2021Q1','2021-01-01')
        readable=list(reader_records(book))
        assert len(readable)==2 and {r['body'] for r in readable}=={old,new}
        assert sum(r['historical'] for r in readable)==1
        w.close();book.close()
        result=snapshot(self.tmp_path/'book.sqlite','2020Q1',self.tmp_path/'past.sqlite')
        past=Notebook(self.tmp_path/'past.sqlite',readonly=True)
        assert result['future_versions_excluded'] and past.db.execute('SELECT body FROM pages').fetchone()[0]==old
        past.close()

    def test_guessed_subrange_is_rejected_even_inside_observed_text(self):
        book=Notebook(self.tmp_path/'book.sqlite');w=Workspace(self.corpus,notebook=book,writer_asset='AAA')
        w.observe(w.call('read',{'id':'doc_000001'}))
        op={'path':'AAA/fx','expected_revision':0,'content':'No hedge [doc_000001:4:20].'}
        with self.assertRaises(ValueError):w.call('edit',{'operations':[op]},action_id='guessed')
        w.observe(w.call('read',{'id':'doc_000001','start':4,'length':16}))
        assert w.call('edit',{'operations':[op]},action_id='observed')['updated']
        w.close();book.close()

if __name__=='__main__':unittest.main()
