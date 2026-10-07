"""Write a readable, complete DEV result when the guarded pipeline finishes."""
from collections import Counter
import json
import os
import time

from finrate_poc.runtime import check_disk,cpu_only,file_digest,watchdog,write_json
from finrate_autonomous_v10.config import BASE,POLICY
from finrate_autonomous_v10.evaluate import verify_online


def render():
    verify_online()
    report=json.loads((BASE/'dev/report.json').read_text())
    diagnostic=json.loads((BASE/'dev/diagnostics.json').read_text())
    grounding=json.loads((BASE/'dev/source_grounding.json').read_text())
    lines=['Fin-RATE autonomous v10 — fixed development results',
      'Model: gpt-5.6-luna / low. Corpus: all43 companies. DEV: LT10 + EC10.',
      'This is a development result; heldout has not been released by this program.',
      'All original questions and failures remain in the denominator. No cutoff exclusion.', '']
    for task,conditions in report['summary'].items():
        lines.extend([task,'condition | strict correct/N | unassessed | key-point lower..upper'])
        for name in POLICY['conditions']:
            r=conditions[name]
            lines.append(f"{name} | {r['correct']}/{r['n']} | {r['unassessed']} | {r['key_point_lower']:.2%}..{r['key_point_upper']:.2%}")
        lines.append('')
    lines.append('Paired bank-minus-control key-point differences (descriptive bootstrap intervals):')
    for name,r in diagnostic['paired'].items():
        lo,hi=r['paired_bootstrap95']
        lines.append(f"{name}: {100*r['mean_key_point_difference']:+.2f} percentage points, interval [{100*lo:+.2f},{100*hi:+.2f}], assessed pairs {r['n_assessed_pairs']}, unassessed {r['unassessed_pairs']}")
    lines.extend(['','Generation / source-grounding / memory use:'])
    for condition in POLICY['conditions']:
        rows=[r for r in diagnostic['rows'] if r['condition']==condition]
        gs=[r for r in grounding if r['condition']==condition]
        states=dict(Counter(r['status'] for r in rows))
        grounded=dict(Counter(r.get('overall_grounding',r['state']) for r in gs))
        used=sum(r['memory_items_observed']>0 for r in rows)
        lines.append(f'{condition}: generation={states}; source labels={grounded}; memory-observed questions={used}/{len(rows)}')
    costs=diagnostic.get('bank_build_costs',{});usage=Counter()
    for cost in costs.values():usage.update(cost['provider_usage'])
    limits=diagnostic.get('writer_budget_boundaries',{})
    lines.extend(['',f'Offline bank accepted response calls: {sum(c["logged_response_calls"] for c in costs.values())}',
      f'Offline provider token usage (cached input is a subset of input): {dict(usage)}',
      f'Budget-exhausted quarter checkpoints: {sum(len(v) for v in limits.values())}',
      'These checkpoints preserve validated edits, not a claim of self-selected completion or exhaustive reading.',
      'Per-company observed-source coverage and all online costs are in diagnostics.json/report.json.',
      '', 'Interpretation limits:',
      'Small fixed development sample; same-family model judges; uncontrolled training exposure.',
      'Reference matching and original-source faithfulness are distinct. A supported abstention is not a complete answer.',
      'Bank construction costs are additional to online QA. Identical online ceilings do not equal total compute.',
      'Retrospective LT/EC scores alone do not demonstrate incremental-versus-rebuild efficiency.',
      'Inspect all paired examples, operational failures and reference-quality diagnostics before a one-time heldout release.',
      'No automatic rerun or question selection is authorized by this report.'])
    output=BASE/'dev/RESULTS.txt';output.write_text('\n'.join(lines)+'\n')
    write_json(BASE/'dev/readable_report_manifest.json',{'file':str(output),'sha256':file_digest(output),
      'generator_sha256':file_digest(__file__),'heldout_access':False,'scores_changed':False},immutable=True)


def main():
    cpu_only();check_disk();os.nice(10);root=BASE/'report_worker';status=root/'status.json';stop=watchdog(status)
    try:
        while True:
            check_disk();pipeline=json.loads((BASE/'supervisors/pipeline/status.json').read_text())
            if pipeline['state']=='development_ready_for_review':
                render();write_json(status,{'state':'completed','pid':os.getpid(),'finished':time.time()});return
            if pipeline['state'] in {'needs_review','stopped_low_disk'}:
                write_json(root/'pipeline_issue.json',pipeline,immutable=True)
                write_json(status,{'state':'pipeline_requires_review','pid':os.getpid()});return
            write_json(status,{'state':'waiting_for_development_results','pid':os.getpid()});time.sleep(20)
    except Exception as exc:
        write_json(status,{'state':'failed','error':str(exc),'pid':os.getpid()});raise
    finally:stop.set()


if __name__=='__main__':main()
