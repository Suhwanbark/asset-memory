"""Chronological source availability; an autonomous writer decides every edit."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import time

from .transport import IsolatedLuna
from finrate_poc.runtime import ROOT,check_disk,cpu_only,file_digest,watchdog,write_json
from .agent import run
from .config import BASE,CORPUS,POLICY
from .workspace import Notebook,Workspace,connect


def implementation_manifest(root):
    paths=[Path(__file__).with_name(n) for n in ['bank.py','agent.py','workspace.py','config.py','transport.py']]
    paths += [ROOT/'method/finrate_luna_v7.py',ROOT/'method/finrate_poc/runtime.py',ROOT/'method/finrate_poc/model.py']
    manifest={'policy':POLICY,'code_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in paths},
      'corpus_manifest_sha256':file_digest(BASE/'corpus_manifest.json'),
      'question_or_gold_access':False,'workflow':'Only arrival chronology and budgets are fixed; no mandated extraction, linking, planning or review stages.'}
    write_json(root/'method_manifest.json',manifest,immutable=True)
    return manifest


def build(asset,root,*,periods=None,status=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);implementation_manifest(root)
    source=connect(CORPUS,True)
    catalog=[dict(r) for r in source.execute('SELECT id,title,asset,published,quarter,length(text) AS characters FROM sources WHERE asset=? ORDER BY published,id',(asset,))]
    all_periods=sorted({r['quarter'] for r in catalog});selected=periods or all_periods
    if any(p not in all_periods for p in selected):raise ValueError('Requested period has no supplied source')
    if selected!=sorted(set(selected)):raise ValueError('Periods must be unique and chronological')
    write_json(root/'build_inputs.json',{'asset':asset,'periods':selected,'source_catalog':catalog,
      'full_asset_history':selected==all_periods,'qa_available':False},immutable=True)
    book=Notebook(root/'bank.sqlite');model=IsolatedLuna(max_calls=len(selected)*POLICY['writer_max_turns'])
    reports=[]
    try:
        for period in selected:
            folder=root/'updates'/period;report_path=folder/'report.json'
            if report_path.exists():
                report=json.loads(report_path.read_text());reports.append(report)
                if report['status']!='completed':raise RuntimeError('Prior failed update retained; inspect or version before continuing')
                continue
            incoming=[r for r in catalog if r['quarter']==period];asof=max(r['published'] for r in incoming)
            if status:write_json(status,{'state':'building_bank','asset':asset,'period':period,'pid':os.getpid(),'asof':asof})
            initial_path=folder/'initial.json'
            if initial_path.exists():initial=json.loads(initial_path.read_text())
            else:
                initial={'company':asset,'available_through':asof,'incoming_sources':incoming,
                  'existing_pages':book.catalog(asset),'task':'Update the persistent company notebook from these newly available public sources.',
                  'publication_quarter':period,'fiscal_period_note':'Publication quarter is an ingestion boundary, not the fiscal period of each assertion.'}
                write_json(initial_path,initial,immutable=True)
            before={r['path']:r['revision'] for r in initial['existing_pages']}
            w=Workspace(CORPUS,representation='bank',notebook=book,asof=asof,writer_asset=asset)
            try:result=run(initial,w,model,folder/'agent',writer=True)
            finally:w.close()
            pages=book.catalog(asset);after={r['path']:r['revision'] for r in pages}
            state=result['status']
            if not pages and state=='completed':state='failed_no_memory_created'
            checkpoint=book.checkpoint(period,asof) if state=='completed' else None
            report={'asset':asset,'period':period,'asof':asof,'status':state,
              'incoming_source_ids':[r['id'] for r in incoming],
              'visited_incoming_source_ids':sorted(set(result['visited_source_ids'])&{r['id'] for r in incoming}),
              'visited_source_ids':result['visited_source_ids'],'before_revisions':before,'after_revisions':after,
              'changed_pages':[p for p in after if before.get(p)!=after[p]],'checkpoint':checkpoint,
              'model_calls':result['model_calls'],'text_tokens':result['online_text_tokens'],
              'tool_actions':result['actions'],'writer_report':result['answer']['answer'],
              'source_availability_is_not_exhaustive_reading':True}
            write_json(report_path,report,immutable=True);reports.append(report)
            print('BANK_UPDATE',asset,period,state,len(report['changed_pages']),'calls',result['model_calls'],flush=True)
            if state!='completed':raise RuntimeError('Writer update incomplete: '+str(result['error']))
        pages=book.catalog(asset)
        active=[dict(r) for r in book.db.execute('SELECT * FROM pages ORDER BY path')]
        write_json(root/'pages.json',active,immutable=True)
        summary={'asset':asset,'state':'completed','periods':selected,'full_asset_history':selected==all_periods,
          'source_count':len({sid for r in reports for sid in r['incoming_source_ids']}),
          'visited_source_count':len({sid for r in reports for sid in r['visited_source_ids']}),
          'pages':len(pages),'model_calls':sum(r['model_calls'] for r in reports),
          'text_tokens':sum(r['text_tokens'] for r in reports),'updates':len(reports),
          'autonomous_actions':dict(Counter(a for r in reports for a in r['tool_actions']))}
    finally:source.close();book.close()
    summary['bank_sha256']=file_digest(root/'bank.sqlite')
    write_json(root/'completed.json',summary,immutable=True);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','full']);p.add_argument('--asset',default='NRT')
    p.add_argument('--periods',default='2022Q1,2022Q2');p.add_argument('--output-group');a=p.parse_args()
    cpu_only();check_disk();os.nice(10)
    root=BASE/(a.output_group or ('bank_smoke' if a.mode=='smoke' else 'banks'))/a.asset
    root.mkdir(parents=True,exist_ok=True);status=root/'status.json';stop=watchdog(status)
    try:
        result=build(a.asset,root,periods=a.periods.split(',') if a.mode=='smoke' else None,status=status)
        write_json(status,{**result,'pid':os.getpid(),'finished':time.time()})
    except Exception as e:
        write_json(status,{'state':'failed','asset':a.asset,'error':type(e).__name__+': '+str(e),'pid':os.getpid()});raise
    finally:stop.set()

if __name__=='__main__':main()
