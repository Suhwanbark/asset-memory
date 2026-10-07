import sys
from pathlib import Path
import unittest
import sqlite3
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'method'))
from finrate_raw_store_v2 import project
from finrate_raw_store_v2 import RawStore
from build_finrate_raw_corpus_v1 import acceptance_date, canonical_form
from build_finrate_raw_corpus_v1 import schema


class ProjectionTest(unittest.TestCase):
    def test_hidden_and_table_periods(self):
        raw = b'''<html><head><title>IGNORE</title></head><body>
        <ix:hidden><b>HIDDEN_FACT</b></ix:hidden>
        <p style="display:none">HIDDEN_CSS</p><p>Item 7. Results</p>
        <table><tr><th>Metric</th><th>2023</th><th>2022</th></tr>
        <tr><td>Revenue</td><td>$120</td><td>$90</td></tr></table>
        <p>A &amp; B <b>remain</b> visible.</p></body></html>'''
        text,sections=project(raw)
        self.assertNotIn('IGNORE',text)
        self.assertNotIn('HIDDEN',text)
        self.assertIn('Metric\t2023\t2022\t',text)
        self.assertIn('Revenue\t$120\t$90\t',text)
        self.assertIn('A & B remain visible.',text)
        self.assertEqual(sections[0]['end'],len(text))

    def test_visible_after_hidden_void_element(self):
        text,_=project(b'<div>Before<img hidden src="x">After</div><div hidden><br>NO</div><p>End</p>')
        self.assertIn('BeforeAfter',text)
        self.assertNotIn('NO',text)
        self.assertIn('End',text)

    def test_sec_acceptance_et_and_amendment(self):
        self.assertEqual(acceptance_date('2023-05-25T00:36:47.000Z'),'2023-05-24')
        self.assertEqual(canonical_form('10-K/A'),'10-K')
        self.assertEqual(canonical_form('SC 13G/A'),'SC')

    def test_unselected_candidate_is_not_agent_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);db=sqlite3.connect(root/'manifest.sqlite');schema(db)
            db.execute('ALTER TABLE documents ADD COLUMN selection_status TEXT')
            for fid,selected in [('one','selected'),('two','candidate_not_selected')]:
                (root/(fid+'.txt')).write_text('Revenue £120 in 2023\nRevenue £90 in 2022')
                db.execute('INSERT INTO filings VALUES (?,?,?,?,?,?,?,?,?,?,?)',(fid,1,fid,'["DEMO"]','Demo','10-K','2024-01-01','2024-01-01','2023-12-31',0,'meta'))
                db.execute('INSERT INTO documents(document_id,filing_id,acquisition_status,text_path,selection_status) VALUES (?,?,?,?,?)',(fid,fid,'ready',fid+'.txt',selected))
            db.commit();db.close()
            store=RawStore(root)
            self.assertEqual([r['document_id'] for r in store.list_filings(ticker='DEMO')],['one'])
            self.assertEqual({r['document_id'] for r in store.grep('revenue')['hits']},{'one'})
            with self.assertRaises(KeyError):store.read('two')
            self.assertEqual(store.read('one',start=8,length=4)['text'],'£120')
            store.db.close()


if __name__=='__main__':unittest.main()
