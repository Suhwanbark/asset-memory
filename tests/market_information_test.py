import json
import sys
import unittest
from datetime import timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'method'))
from market_information_sources import DATA,sha,dt
from market_information_dataset import phase_label,usable_candles,ENC
from market_information_eval import baseline_ids,forecast_prompt,selector_prompt

class MarketInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=json.loads((DATA/'cases.json').read_text())['cases']
        cls.labels={x['case_id']:x for x in json.loads((DATA/'labels.json').read_text())['labels']}
        cls.cards={x['id']:x for x in json.loads((DATA/'evidence_cards.json').read_text())['cards']}

    def test_frozen_files(self):
        f=json.loads((DATA/'freeze.json').read_text())
        for field,name in [('cases_sha256','cases.json'),('labels_sha256','labels.json'),('evidence_sha256','evidence_cards.json')]:
            self.assertEqual(f[field],sha((DATA/name).read_bytes()))

    def test_temporal_boundaries(self):
        for c in self.cases:
            self.assertLess(dt(c['cutoff']),dt(self.labels[c['id']]['close_time']))
            for e in c['evidence']:
                self.assertLessEqual(dt(e['available_at']),dt(c['cutoff']))
                self.assertGreaterEqual(dt(e['available_at']),dt(c['cutoff'])-timedelta(days=90))

    def test_budget_and_unique_evidence(self):
        for c in self.cases:
            self.assertEqual(len({e['id'] for e in c['evidence']}),len(c['evidence']))
            for e in c['evidence']:self.assertEqual(len(ENC.encode(e['text'])),256)
            for method in ['recency','relevance','random']:
                ids=baseline_ids(c,method)
                self.assertEqual(len(set(ids)),2)
                self.assertEqual(sum(e['tokens'] for e in c['evidence'] if e['id'] in ids),512)

    def test_source_hashes(self):
        for e in self.cards.values():
            self.assertEqual(sha(Path(e['source_text_path']).read_bytes()),e['source_text_sha256'])
            self.assertEqual(sha(e['text']),e['excerpt_sha256'])

    def test_inference_whitelist(self):
        for c in self.cases:
            self.assertFalse({'outcome','phase','target','future','attributions','result','market_probability'}&set(c))
            for e in c['evidence']:
                self.assertEqual(set(e),{'id','published_date','available_at','title','source_type','text','tokens'})

    def test_market_history_past_only(self):
        for c in self.cases:
            hist=self.labels[c['id']]['phase']['history']
            self.assertTrue(all(x['t']<=dt(c['cutoff']).timestamp() for x in hist))

    def test_no_future_candle_influence(self):
        now=dt('2025-01-10T00:00:00Z')
        candles=[{'end_period_ts':int((now-timedelta(days=i)).timestamp()),'price':{'close':'.9'},'volume':'100'} for i in range(8)]
        a=phase_label(usable_candles(candles,now),now)
        candles.append({'end_period_ts':int((now+timedelta(days=1)).timestamp()),'price':{'close':'.1'},'volume':'100'})
        self.assertEqual(a,phase_label(usable_candles(candles,now),now))
        self.assertEqual(a['phase'],'consensus')

    def test_live_and_historical_price_schemas(self):
        now=dt('2025-01-10T00:00:00Z');ts=int(now.timestamp())
        old=[dict(end_period_ts=ts,price={'close':'0.6'},volume='100')]
        new=[dict(end_period_ts=ts,price={'close_dollars':'0.6'},volume_fp='100')]
        self.assertEqual(usable_candles(old,now),usable_candles(new,now))

    def test_no_volume_not_consensus(self):
        now=dt('2025-01-10T00:00:00Z')
        rows=[dict(t=(now-timedelta(days=i)).timestamp(),p=.95,volume=0) for i in reversed(range(7))]
        self.assertNotEqual(phase_label(rows,now)['phase'],'consensus')

    def test_no_split_event_overlap(self):
        sets={s:{c['event_id'] for c in self.cases if c['split']==s} for s in ['smoke','main']}
        self.assertFalse(sets['smoke']&sets['main'])

    def test_baseline_determinism(self):
        c=self.cases[0]
        self.assertEqual(baseline_ids(c,'random'),baseline_ids(c,'random'))
        latest=sorted(c['evidence'],key=lambda e:(e['available_at'],e['id']),reverse=True)[:2]
        self.assertEqual(baseline_ids(c,'recency'),[e['id'] for e in latest])

    def test_forecaster_ignores_selector_reason(self):
        c=self.cases[0];cards=c['evidence'][:2]
        prompt=forecast_prompt(c,cards)
        self.assertEqual(prompt,forecast_prompt(c,list(reversed(cards))))
        self.assertNotIn('selected_mean_age_days',prompt)

    def test_complete_label_mapping(self):
        self.assertEqual({c['id'] for c in self.cases},set(self.labels))
        self.assertTrue(all(x['outcome'] in [0,1] for x in self.labels.values()))

    def test_anchor_fixed_and_available(self):
        anchors=json.loads((DATA/'anchor_manifest.json').read_text())['anchors']
        for c in self.cases:
            anchor=anchors[c['event_id']]['anchor_id']
            self.assertIn(anchor,{e['id'] for e in c['evidence']})
            without=baseline_ids(c,'anchor_without');with_=baseline_ids(c,'anchor_with')
            self.assertNotIn(anchor,without)
            self.assertEqual(set(with_)-set(without),{anchor})
            self.assertEqual(len(with_),3)

    def test_news_replication_boundaries(self):
        directory=ROOT/'data/market_information_news_poc_v1'
        cases=json.loads((directory/'cases.json').read_text())['cases']
        labels={x['case_id']:x for x in json.loads((directory/'labels.json').read_text())['labels']}
        freeze=json.loads((directory/'freeze.json').read_text())
        self.assertEqual(sha((directory/'cases.json').read_bytes()),freeze['cases_sha256'])
        self.assertEqual(sha((directory/'labels.json').read_bytes()),freeze['labels_sha256'])
        for c in cases:
            self.assertLess(dt(c['cutoff']),dt(labels[c['id']]['close_time']))
            for e in c['evidence']:
                self.assertLessEqual(dt(e['available_at']),dt(c['cutoff']))
                self.assertLessEqual(e['tokens'],256)
                self.assertEqual(e['tokens'],len(ENC.encode(e['text'])))
            self.assertFalse({'outcome','phase','target','future','attributions'}&set(c))

    def test_event_weighted_scoring(self):
        from market_information_report import paired_comparison,brier
        self.assertAlmostEqual(brier(.7,1),.09)
        rows=[]
        for cid,event,delta,date in [('a1','a',.2,'2025-01-01'),('a2','a',.2,'2025-01-01'),('b1','b',-.2,'2025-02-01')]:
            for condition,loss in [('new',.3-delta),('old',.3)]:
                rows.append(dict(case_id=cid,event_id=event,split='main',condition=condition,brier=loss,
                    phase='consensus',probability=.5,outcome=1,meeting_date=date))
        c=paired_comparison(rows,'new','old')
        self.assertAlmostEqual(c['mean_brier_improvement'],0)
        self.assertEqual(c['n_events'],2)
        self.assertEqual(c['n_snapshots'],3)

    def test_literal_probability_survives_truncated_reason(self):
        from market_information_eval import parse_output
        obj=parse_output('{"probability":0.93,"evidence_ids":["a","b"],"reason":"unfinished','forecast')
        self.assertEqual(obj['probability'],.93)
        self.assertEqual(obj['evidence_ids'],['a','b'])
        self.assertEqual(obj['_parse_status'],'truncated_reason_literal_prefix')
        with self.assertRaises(Exception):parse_output('{"probability":maybe,"evidence_ids":[],"reason":"unfinished','forecast')
        with self.assertRaises(Exception):parse_output('{"probability":0.93,"evidence_ids":["unfinished','forecast')

class CompletedInferenceAudit(unittest.TestCase):
    """Opt in only once both runs and diagnostics finish."""
    def test_evidence_creation_context_uses_past_only(self):
        for name in ['market_information_poc_v1','market_information_news_poc_v1']:
            d=ROOT/'data'/name
            cs={c['id']:c for c in json.loads((d/'cases.json').read_text())['cases']}
            obj=json.loads((d/'evidence_market_context.json').read_text())
            for r in obj['rows']:
                card=next(e for e in cs[r['case_id']]['evidence'] if e['id']==r['evidence_id'])
                self.assertEqual(card['available_at'],r['available_at'])
                s=r['at_availability']
                if 'last_market_time' in s:
                    self.assertLessEqual(s['last_market_time'],dt(r['available_at']).timestamp())

    @unittest.skipUnless(__import__('os').environ.get('AUDIT_COMPLETED_MI')=='1','diagnostic audit requested on completion')
    def test_fixed_packet_repeats_and_full_source_hashes(self):
        from collections import Counter
        d=ROOT/'result/market_information_diagnostics_v1'
        manifest=json.loads((d/'manifest.json').read_text())
        self.assertEqual(len(manifest['tasks']),120)
        keys=Counter((t['group'],t['case_id'],t['condition']) for t in manifest['tasks'])
        self.assertTrue(all(n==5 for n in keys.values()))
        hashes={}
        for t in manifest['tasks']:
            key=(t['group'],t['case_id'],t['condition'])
            self.assertEqual(hashes.setdefault(key,t['prompt_sha256']),t['prompt_sha256'])
            folder=d/'calls'/t['id'];r=json.loads((folder/'result.json').read_text());response=json.loads((folder/'response.json').read_text())
            self.assertEqual(r['prompt_sha256'],sha(t['prompt']))
            self.assertEqual(response['prompt_sha256'],sha(t['prompt']))
            self.assertEqual(response['tool_calls'],[])
            for source in t['source_provenance']:
                self.assertEqual(sha(Path(source['path']).read_bytes()),source['sha256'])
        self.assertTrue(json.loads((d/'report.json').read_text())['complete'])

    @unittest.skipUnless(__import__('os').environ.get('AUDIT_COMPLETED_MI')=='1','full-run audit requested only on completion')
    def test_every_expected_result_and_exact_input(self):
        import market_information_eval as ev
        for name,conditions in [('market_information_poc_v1',ev.CONDITIONS+ev.DIAGNOSTICS),('market_information_news_poc_v1',ev.CONDITIONS)]:
            directory=ROOT/'data'/name;out=ROOT/'result'/name
            cases=json.loads((directory/'cases.json').read_text())['cases']
            frozen=json.loads((directory/'freeze.json').read_text())
            for c in cases:
                for condition in conditions:
                    folder=out/c['id']/condition
                    if not (folder/'result.json').exists():
                        failure=json.loads((folder/'failure.json').read_text())
                        self.assertFalse(failure['regenerated'])
                        self.assertTrue(failure['unknown_ids'])
                        response=json.loads(Path(failure['response_path']).read_text())
                        self.assertEqual(response['prompt_sha256'],sha(ev.selector_prompt(c,condition)))
                        self.assertEqual(response['tool_calls'],[])
                        self.assertTrue(set(failure['unknown_ids'])-set(e['id'] for e in c['evidence']))
                        continue
                    result=json.loads((folder/'result.json').read_text())
                    self.assertEqual(result['dataset_sha256'],frozen['cases_sha256'])
                    selected=[e for e in c['evidence'] if e['id'] in result['selected_ids']]
                    prompt=ev.forecast_prompt(c,selected)
                    self.assertEqual(sha(prompt),result['prompt_sha256'])
                    response=json.loads(Path(result['forecast_response_path']).read_text())
                    self.assertEqual(response['prompt_sha256'],sha(prompt))
                    self.assertEqual(response['tool_calls'],[])
                    self.assertTrue(0<=result['probability']<=1)
                    if condition not in ['no_evidence','full_pool','anchor_with']:
                        self.assertLessEqual(result['body_tokens'],512)

if __name__=='__main__':unittest.main()
