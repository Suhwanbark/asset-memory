"""Standalone research plots from frozen data and offline results."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from market_information_sources import ROOT,DATA,RAW,dt
from market_information_dataset import usable_candles,phase_label

OUT=ROOT/'result/market_information_poc_v1'
ASSETS=OUT/'figures'
COLORS={'consensus':'#2b8c72','fluctuation':'#df9330','reversal':'#b54f59','intermediate':'#8994a4'}

def timelines():
    ASSETS.mkdir(parents=True,exist_ok=True)
    cases=json.loads((DATA/'cases.json').read_text())['cases']
    selected=['FEDDECISION-24SEP-H0','KXFEDDECISION-25SEP-H0','KXFEDDECISION-26SEP-H0']
    fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
    for ax,event in zip(axes,selected):
        raw=json.loads((RAW/('candles_'+event+'.json')).read_text())['candlesticks']
        cs=[c for c in cases if c['event_id']==event]
        stop=dt(max(c['meeting_date'] for c in cs)+'T00:00:00Z')
        rows=usable_candles(raw,stop)
        ts=[datetime.fromtimestamp(r['t'],timezone.utc) for r in rows]
        ax.plot(ts,[r['p'] for r in rows],color='#23364d',lw=1.6)
        for i,r in enumerate(rows):
            phase=phase_label(rows[:i+1],ts[i])['phase']
            ax.scatter(ts[i],r['p'],color=COLORS.get(phase,'#c8c8c8'),s=15,zorder=3)
        for c in cs:
            cut=dt(c['cutoff']);ax.axvline(cut,color='#64748b',ls='--',lw=1)
            ax.text(cut,.97,f"-{c['lead_days']}d",va='top',ha='right',fontsize=9)
        ax.set(title=event,ylabel='P(no rate change)',ylim=(-.03,1.03))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=col,label=key) for key,col in COLORS.items()]
    fig.legend(handles=handles,loc='outside lower center' if hasattr(matplotlib,'__version__') and int(matplotlib.__version__.split('.')[1])>=7 else 'lower center',ncol=4,frameon=False)
    fig.suptitle('Actual Kalshi market paths; phase labels use past data only',fontsize=14)
    fig.savefig(ASSETS/'market_timeline_examples.png',dpi=180,bbox_inches='tight');plt.close(fig)

def results():
    report=json.loads((OUT/'report.json').read_text())
    rs=[r for r in report['summary'] if r['split']=='main']
    names={'recency':'Latest','relevance':'Relevance','random':'Random','llm_priority':'LLM priority (71/72)',
           'llm_temporal':'LLM temporal','no_evidence':'No evidence','full_pool':'Full pool','market_price':'Market price',
           'anchor_without':'Anchor absent','anchor_with':'Anchor present'}
    main=[r for r in rs if r['condition'] not in ['anchor_without','anchor_with']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    axes[0].barh([names[r['condition']] for r in main],[r['event_mean_brier'] for r in main],color=['#3c8d83' if r['condition']=='llm_temporal' else '#697b96' for r in main])
    axes[0].invert_yaxis();axes[0].set(xlabel='Brier score (lower is better)',title='Main retrospective PoC; equal weight per meeting')
    cmp=next(c for c in report['comparisons'] if c['treatment']=='llm_temporal' and c['reference']=='recency')
    if cmp['n_events']:
        values=cmp['event_differences'];ev=list(values);v=np.array(list(values.values()))
        axes[1].bar(range(len(v)),v,color=np.where(v>=0,'#2b8c72','#b54f59'))
        axes[1].axhline(0,color='black',lw=.8)
        axes[1].set(xlabel='Meeting in chronological order',ylabel='Brier improvement over latest',title='Both helpful and harmful cases retained')
    fig.savefig(ASSETS/'selection_results.png',dpi=180,bbox_inches='tight');plt.close(fig)

def boj_repeats():
    d=ROOT/'result/market_information_diagnostics_v1'
    if not (d/'report.json').exists():return
    r=json.loads((d/'report.json').read_text());cs=json.loads((ROOT/'data/market_information_news_poc_v1/cases.json').read_text())['cases']
    dates={c['id']:c['cutoff'][:10] for c in cs};rows=[s for s in r['summary'] if s['group']=='news_repeats']
    ids=list(dict.fromkeys(s['case_id'] for s in rows));x=np.arange(len(ids))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for shift,k,label,color in [(-.1,'recency','Latest two','#64748b'),(.1,'llm_temporal','Time-aware selected pair','#258270')]:
        ss=[next(s for s in rows if s['case_id']==cid and s['condition']==k) for cid in ids]
        means=np.array([s['mean_probability'] for s in ss]);lo=means-np.array([s['min_p'] for s in ss]);hi=np.array([s['max_p'] for s in ss])-means
        axes[0].errorbar(x+shift,means,yerr=[lo,hi],fmt='o',capsize=4,color=color,label=label)
        axes[1].bar(x+shift*1.6,[s['mean_brier'] for s in ss],width=.28,color=color,label=label)
    for ax in axes:
        ax.set_xticks(x,[dates[cid] for cid in ids]);ax.grid(axis='y',alpha=.15)
    axes[0].set(ylabel='P(21–40 bp hike)',ylim=(0,1.03),title='Five fresh forecasts per fixed packet')
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].set(ylabel='Mean Brier loss (lower is better)',title='Actual outcome: YES')
    fig.suptitle('BOJ December 2025 — exploratory case; whiskers show min–max, not confidence intervals',fontsize=11)
    fig.savefig(ASSETS/'boj_repeated_case.png',dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    timelines()
    if (OUT/'report.json').exists():results()
    boj_repeats()
