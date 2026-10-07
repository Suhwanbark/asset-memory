"""Source-backed A/B evidence priority; author labels precede all model outputs.

Real source facts, researcher-authored analytical scenarios. These are not
naturally collected user questions or independent causal-value labels.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib, io, json, re
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from time_priority_resources import ROOT, disk_check

DATA=ROOT/'data/time_priority_v1'
def sha(x): return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def save(p,x):
    disk_check(); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def norm(x): return re.sub(r'\s+',' ',x.replace('\xa0',' ')).strip()

SOURCES=[
 ('discount','2020-03-15','https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315b.htm'),
 ('btfp_terms','2023-03-12','https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312a.htm'),
 ('btfp_close','2024-01-24','https://www.federalreserve.gov/newsevents/pressreleases/monetary20240124a.htm'),
 ('main_terms','2020-06-08','https://www.federalreserve.gov/newsevents/pressreleases/monetary20200608a.htm'),
 ('main_close','2020-12-29','https://www.federalreserve.gov/newsevents/pressreleases/monetary20201229a.htm'),
 ('cpff_terms','2020-11-30','https://www.federalreserve.gov/newsevents/pressreleases/files/monetary20201130a1.pdf'),
 ('mmlf_terms','2020-11-30','https://www.federalreserve.gov/newsevents/pressreleases/files/monetary20201130a2.pdf'),
 ('ppplf_terms','2021-06-25','https://www.federalreserve.gov/newsevents/pressreleases/files/monetary20210625a1.pdf'),
 ('ppplf_close','2021-06-25','https://www.federalreserve.gov/newsevents/pressreleases/monetary20210625a.htm'),
 ('facility_review','2021-11-19','https://www.federalreserve.gov/monetarypolicy/November-2021-Federal-Reserve-Balance-Sheet-Developments.htm'),
 ('apple_split','2020-07-30','https://www.apple.com/newsroom/2020/07/apple-reports-third-quarter-results/'),
 ('apple_dividend','2024-10-31','https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/'),
 ('nvda_split','2024-05-22','https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-first-quarter-fiscal-2025'),
 ('nvda_dividend','2024-11-20','https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2025'),
 ('avgo_split','2024-06-12','https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announces-second-quarter-fiscal-year-2024-financial'),
 ('avgo_dividend','2024-12-12','https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announces-fourth-quarter-and-fiscal-year-2024'),
]

def fetch(spec):
    sid,date,url=spec; folder=DATA/'sources'/sid; meta=folder/'meta.json'
    if meta.exists():
        m=json.loads(meta.read_text()); assert sha(Path(m['text_path']).read_bytes())==m['text_sha256']; return sid,m
    disk_check(); r=requests.get(url,timeout=60); r.raise_for_status()
    pdfmeta={}
    if r.content.startswith(b'%PDF'):
        reader=PdfReader(io.BytesIO(r.content)); text='\n'.join(p.extract_text() or '' for p in reader.pages)
        pdfmeta={str(k):str(v) for k,v in (reader.metadata or {}).items()}; ext='pdf'
    else:
        soup=BeautifulSoup(r.content,'html.parser')
        for node in soup.select('script,style,noscript,svg'): node.decompose()
        body=soup.select_one('.article-body') or soup.select_one('#article') or soup.select_one('main') or soup.select_one('article') or soup
        text=body.get_text(' ',strip=True); ext='html'
    text=norm(text); assert len(text)>300,(sid,len(text))
    for bad in ['Access Denied','verify you are human','Request Rejected']:
        assert bad not in text[:500],(sid,bad)
    folder.mkdir(parents=True,exist_ok=True); raw=folder/('source.'+ext); txt=folder/'source.txt'
    raw.write_bytes(r.content); txt.write_text(text)
    m=dict(id=sid,publication_date=date,url=url,response_url=r.url,text_path=str(txt),raw_path=str(raw),
           text_sha256=sha(text),raw_sha256=sha(r.content),pdf_metadata=pdfmeta,
           fetched_at=datetime.now(timezone.utc).isoformat(),
           date_basis='Dated official release/term-sheet; dates manually checked, not archive fetch dates.')
    save(meta,m); print('SOURCE',sid,len(text),flush=True); return sid,m

def excerpt(m,start,end=None):
    text=Path(m['text_path']).read_text(); at=text.find(start)
    if at<0: raise ValueError(f"missing anchor {m['id']}: {start}")
    if end:
        stop=text.find(end,at)
        if stop<0: raise ValueError(f"missing end {m['id']}: {end}")
        stop+=len(end)
    else:
        match=re.search(r'\.(?=\s+[A-Z]|$)',text[at:]); stop=at+match.end() if match else at+len(start)
    quote=text[at:stop]
    return dict(source_id=m['id'],published=m['publication_date'],url=m['url'],quote=quote,span=[at,stop],source_text_sha256=m['text_sha256'])

def item(name,text,*anchors):
    return dict(id=name,text=text,evidence=list(anchors),published=max(a['published'] for a in anchors))

def construct(s):
    cases=[]; gold=[]
    def add(cid,cluster,stratum,asof,q,x,y,winner,why,pair=None,change='task'):
        assert winner in [x['id'],y['id']]
        assert all(z['published']<=asof for z in [x,y]),cid
        cases.append(dict(id=cid,cluster=cluster,stratum=stratum,as_of=asof,question=q,items=[x,y],pair_id=pair,change=change))
        gold.append(dict(id=cid,winner=winner,warrant=why,review='source-grounded researcher label; no independent human agreement claimed'))
    dw=item('discount_window','The discount window offers depository institutions advances of up to 90 days; borrowers can prepay and renew daily.',excerpt(s['discount'],'To further enhance','discount window loans.'))
    # Standing terms and a new-origination shutdown govern different decisions.
    configs=[
      ('btfp','BTFP','2024-03-12','2024-04-12',
       item('btfp_loan_terms','BTFP advances to eligible depository institutions may run for up to one year, against qualifying securities valued at par.',excerpt(s['btfp_terms'],'The additional funding','valued at par.')),
       item('btfp_shutdown','BTFP will stop making new loans on March 11, 2024. The discount window remains available afterwards; other BTFP terms are unchanged.',excerpt(s['btfp_close'],'The Federal Reserve Board on Wednesday','eligible institutions.'),excerpt(s['btfp_close'],'After March 11','liquidity needs.'),excerpt(s['btfp_close'],'All other terms','unchanged.')),
       'a one-year advance drawn on March 1, 2024','Existing loans retain their contractual term; stopping new lending does not itself accelerate their maturity.'),
      ('mainstreet','Main Street Lending Program','2021-01-11','2021-02-11',
       item('main_loan_terms','Main Street loan terms were extended to five years, with principal payments deferred for two years.',excerpt(s['main_terms'],'The Board lowered','repaying the loans.')),
       item('main_shutdown','The termination date for Main Street facilities was extended to January 8, 2021, to process and fund loans submitted by December 14, 2020.',excerpt(s['main_close'],'In order to allow','January 8, 2021.')),
       'a five-year loan funded in October 2020','The programme intake deadline is different from a funded loan maturity and principal repayment schedule.'),
      ('cpff','CPFF','2021-04-02','2021-05-03',
       item('cpff_term','CPFF purchases three-month US-dollar commercial paper from eligible issuers. The SPV continues to be funded until its assets mature.',excerpt(s['cpff_terms'],'Assets of the SPV:','CPFF Dealers.'),excerpt(s['cpff_terms'],'The New York Fed will continue','assets mature.')),
       item('cpff_shutdown','CPFF will stop purchasing new commercial paper on March 31, 2021, unless the Board and Treasury extend the facility.',excerpt(s['cpff_terms'],'Termination date:','extend the facility.')),
       'three-month paper sold to the facility on March 15, 2021','Stopping purchases is not early maturity of previously purchased three-month paper.'),
      ('mmlf','MMLF','2021-04-02','2021-05-03',
       item('mmlf_term','An MMLF advance matures when its pledged eligible collateral matures, with a maximum term of 12 months.',excerpt(s['mmlf_terms'],'Advance Maturity:','12 months.')),
       item('mmlf_shutdown','MMLF will make no new credit extensions after March 31, 2021, unless the Board and Treasury extend it.',excerpt(s['mmlf_terms'],'Program Termination:','extend the Facility.')),
       'an advance made on March 15, 2021 against eligible collateral maturing March 15, 2022','The pledged collateral maturity governs existing funding; the new-extension deadline does not erase it.'),
      ('ppplf','PPPLF','2021-08-02','2021-09-02',
       item('ppplf_term','PPPLF advances mature with the pledged PPP loan. SBA forgiveness reimbursement accelerates repayment to that extent; specified default circumstances can also accelerate it.',excerpt(s['ppplf_terms'],'Maturity and Acceleration','from the SBA.')),
       item('ppplf_shutdown','The final extension of PPPLF lasts through July 30, 2021, allowing lenders more time to pledge PPP loans approved by June 30.',excerpt(s['ppplf_close'],'The Federal Reserve Board on Friday','July 30, 2021.'),excerpt(s['ppplf_close'],'The extension is being made','PPP program.')),
       'a PPPLF advance against an outstanding PPP loan maturing in 2025, with no forgiveness reimbursement or default','New lending ends separately from the repayment conditions of an existing advance.')
    ]
    for cid,name,d1,d2,terms,shutdown,position,why in configs:
        for suffix,when in [('a',d1),('b',d2)]:
            add(cid+'_legacy_'+suffix,cid,'persistent_effect',when,
                f'A treasury analyst is scheduling cash outflows for {position} under {name}. Which item deserves more weight for determining the outstanding obligation\'s repayment horizon? Assume no later contractual amendment, prepayment or default.',
                terms,shutdown,terms['id'],why,cid+'_persistence','clock_only_stable')
        add(cid+'_new',cid,'ended_origination',d1,
            f'A potential participant wants to obtain a completely new draw or sell newly issued paper to {name} today. Which item should receive priority in deciding whether this can be included as available new funding? Assume no extension beyond the dates shown.',
            terms,shutdown,shutdown['id'],'The intake/purchase cutoff controls new access; maturity terms of old positions cannot establish current origination availability.',cid+'_task_switch','task')
        add(cid+'_winddown',cid,'persistent_effect',d1,
            f'An analyst wants to decide whether closing {name} to new business is enough to remove all previously funded positions from a future cash-flow schedule. Which item is more important for that decision?',
            terms,shutdown,terms['id'],why,cid+'_task_switch','task')
    # Pure clock-change lower-bound tests: eligibility and collateral are fixed.
    clocks=[
      ('btfp','2024-03-01','2024-03-20',configs[0][4],configs[0][5],'one year'),
      ('mmlf','2021-03-15','2021-04-15',configs[3][4],configs[3][5],'12 months'),
      ('ppplf','2021-07-01','2021-08-02',configs[4][4],configs[4][5],'the 2025 maturity of its eligible collateral')]
    for cid,before,after,terms,shutdown,term in clocks:
        # Single card contains its authentic terms AND its already announced deadline.
        temporary=item(cid+'_temporary',terms['text']+' '+shutdown['text'],*(terms['evidence']+shutdown['evidence']))
        question=(f'An eligible depository institution has collateral qualifying for both described facilities, including collateral lasting beyond 90 days. It wants the longest INITIAL maturity on a new advance drawn today, without counting future renewals. Which facility information should it prioritize? Assume the standing window remains available and no extensions or term changes beyond the supplied announcements.')
        for suffix,when,win in [('before',before,temporary['id']),('after',after,dw['id'])]:
            add(cid+'_clock_'+suffix,cid,'clock_switch',when,question,dw,temporary,win,
                f'Before the announced cutoff, the temporary facility can offer {term}, longer than 90 days; after cutoff it cannot originate a new advance, while the standing window remains available.',cid+'_clock','clock_only_switch')
    # A completed corporate action can retain present analytical value.
    stock_specs=[
      ('apple','Apple','2024-11-01','2024-11-05','2020-08-20','four-for-one','2020-08-31','$0.25','2024-11-14','2024-11-11',
       excerpt(s['apple_split'],'The Board of Directors has also approved','August 31, 2020.'),
       excerpt(s['apple_dividend'],'Apple’s board of directors has declared','November 11, 2024.')),
      ('nvda','NVIDIA','2024-11-21','2024-11-25','2024-06-03','ten-for-one','2024-06-10','$0.01','2024-12-27','2024-12-05',
       excerpt(s['nvda_split'],'NVIDIA also announced','June 10, 2024.'),
       excerpt(s['nvda_dividend'],'NVIDIA will pay its next','December 5, 2024.')),
      ('avgo','Broadcom','2024-12-13','2024-12-16','2024-07-01','ten-for-one','2024-07-15','$0.59','2024-12-31','2024-12-23',
       excerpt(s['avgo_split'],'ten-for-one forward stock split','split-adjusted basis.'),
       excerpt(s['avgo_dividend'],"The Company's Board of Directors has approved",'December 23, 2024 .'))]
    for cid,name,d1,d2,presplit,ratio,splitday,amount,pay,record,e1,e2 in stock_specs:
        a=item(cid+'_split',f'{name} announced a {ratio} stock split, with split-adjusted trading beginning {splitday}.',e1)
        b=item(cid+'_dividend',f'{name} declared a cash dividend of {amount} per common share, payable {pay} to holders of record {record}.',e2)
        for suffix,when in [('a',d1),('b',d2)]:
            add(cid+'_share_basis_'+suffix,cid,'persistent_effect',when,
                f'A portfolio analyst is reconciling a {name} share count recorded on {presplit} with the share count now, with no trades between. Which item deserves priority to avoid mistaking a change in the number of units for a new purchase?',
                a,b,a['id'],'The completed split changes the unit count and still matters to current reconciliation; the cash dividend does not multiply shares.',cid+'_persistence','clock_only_stable')
        add(cid+'_cash',cid,'task_conditioned',d1,
            f'A holder is preparing the next scheduled cash-income entry from {name}, assuming the holder meets the relevant record-date conditions. Which information should receive priority?',
            a,b,b['id'],'The declared cash distribution controls scheduled income; an old split is not a new cash payment.',cid+'_task_switch','task')
        add(cid+'_price_basis',cid,'persistent_effect',d1,
            f'An analyst is comparing raw per-share quotations for {name} on {presplit} and today. Which item is more important for putting the old quote on the current share-unit basis before calculating price appreciation?',
            a,b,a['id'],'A split rescales the per-share unit across the interval. Cash income matters to total return but not to this unit conversion.',cid+'_task_switch','task')
    # Separate easy status/history controls: not counted as broad utility evidence.
    oldbase=ROOT/'data/temporal_evidence_poc_v1/sources'
    merger_specs=[
     ('irobot','Amazon–iRobot','2024-01-30','corp_irbt_20230725','corp_irbt_20240129_clean_v3',
      '$51.75 per share','mutual agreement to terminate',
      'In July 2023 Amazon and iRobot amended the cash acquisition price to $51.75 per share.',
      'On January 29, 2024 Amazon and iRobot agreed to terminate the proposed acquisition.'),
     ('adobe','Adobe–Figma','2023-12-19','corp_adbe_20231128','corp_adbe_20231218',
      'provisional','mutual agreement to terminate',
      'In November 2023 the CMA issued provisional competition findings about the Adobe–Figma deal.',
      'On December 18, 2023 Adobe and Figma agreed to terminate their merger agreement.'),
     ('jetblue','JetBlue–Spirit','2024-03-05','corp_jblu_20230307','corp_jblu_20240304',
      'continue to advance','terminate their July 2022 merger agreement',
      'In March 2023 JetBlue and Spirit said they would continue pursuing their merger despite the DOJ challenge.',
      'On March 4, 2024 JetBlue announced agreement to terminate its July 2022 merger agreement with Spirit.'),
     ('kroger','Kroger–Albertsons','2024-12-12','corp_kr_20240226','corp_aci_20241211',
      'proposed merger','exercised its right to terminate',
      'In February 2024 Kroger publicly defended its planned Albertsons merger against the FTC challenge.',
      'On December 11, 2024 Albertsons announced that it exercised its right to terminate the merger agreement.')]
    for cid,name,when,oldid,newid,anchor1,anchor2,text1,text2 in merger_specs:
        m1=json.loads((oldbase/oldid/'meta.json').read_text()); m2=json.loads((oldbase/newid/'meta.json').read_text())
        a=item(cid+'_earlier',text1,excerpt(m1,anchor1)); b=item(cid+'_later',text2,excerpt(m2,anchor2))
        add(cid+'_completion',cid,'status_control',when,
            f'An event analyst is deciding whether to keep completion of the existing {name} agreement as a live pending catalyst. Which item should receive priority?',a,b,b['id'],
            'The issuer termination supersedes a past intention for the pending-deal task.',cid+'_task_switch')
        add(cid+'_audit',cid,'historical_control',when,
            f'An analyst is reconstructing the public information available before the termination of {name}, to explain what investors had been told at the earlier announcement. Which item is more directly useful for that historical reconstruction?',a,b,a['id'],
            'The old announcement directly records the earlier information set; later termination must not be projected backward.',cid+'_task_switch')
    # Four explicit reading controls, independent FOMC releases.
    for stamp in ['20230201','20230726','20240918','20251210']:
        m=json.loads((oldbase/('fed_'+stamp+'_statement')/'meta.json').read_text())
        rate=excerpt(m,'Committee decided to')
        objective=excerpt(m,'The Committee seeks to')
        a=item('fomc_'+stamp+'_rate',rate['quote'],rate); b=item('fomc_'+stamp+'_goal',objective['quote'],objective)
        add('reading_'+stamp,'fomc_'+stamp,'reading_control',m['publication_date'],
            'Which item directly states the numerical target range selected for the federal funds rate at this meeting?',a,b,a['id'],
            'Direct reading control, not temporal utility: the explicit rate decision contains the requested range.')
    assert len(cases)==50,len(cases)
    return cases,gold

def main():
    with ThreadPoolExecutor(max_workers=4) as pool: sources=dict(pool.map(fetch,SOURCES))
    save(DATA/'sources.json',sources)
    cases,gold=construct(sources)
    public={'version':'time_priority_v1','cases':cases,'scope':'source-grounded task-specific temporal evidence priority, not causal financial value','authoring':'researcher-authored scenarios over real official facts'}
    save(DATA/'cases.json',public); save(DATA/'gold.json',{'labels':gold})
    save(DATA/'freeze.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'cases_sha256':sha((DATA/'cases.json').read_bytes()),'gold_sha256':sha((DATA/'gold.json').read_bytes()),'n_cases':len(cases),'clusters':sorted({c['cluster'] for c in cases}),'selection':'fixed before any model inference; source/rubric errors corrected before freezing'})
    print('FROZEN',len(cases),'questions',len({c['cluster'] for c in cases}),'clusters',flush=True)
if __name__=='__main__': main()
