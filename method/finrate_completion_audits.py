"""Finish zero-inference development audits as the existing workers complete.

Never starts/restarts an upstream worker, edits a source bank, freezes a method,
or accesses held-out contents. Quarter replays disallow new model calls.
"""
import json
import os
import time

from finrate_memory_build_report import report
from finrate_paired_development import analyze
from finrate_quarter_replay import replay
from finrate_poc.runtime import ROOT, check_disk, cpu_only, file_digest, watchdog, write_json


def main():
    cpu_only(); check_disk(); os.nice(10)
    base = ROOT/'result/finrate_luna_v1'; out = base/'v6/completion_audits'
    out.mkdir(parents=True, exist_ok=True); status = out/'status.json'; started = time.time()
    names = ['finrate_completion_audits.py', 'finrate_memory_build_report.py',
             'finrate_paired_development.py', 'finrate_quarter_replay.py']
    code = {name: file_digest(ROOT/'method'/name) for name in names}
    write_json(out/'manifest.json', {'code_sha256': code, 'assets': ['FIX','LW','NRT'],
        'new_model_calls_allowed': 0, 'evaluation_allowed': False,
        'purpose': 'Full-build accounting, cached-response quarter replay, paired complete-grid development analysis.'}, immutable=True)
    common = {'pid': os.getpid(), 'started': started, 'cpu_only': True,
              'disk_stop_bytes': 10_000_000_000, 'evaluation_touched': False, 'new_model_calls': 0}
    done = {}; stop = watchdog(status)
    try:
        while True:
            check_disk()
            if any(file_digest(ROOT/'method'/name) != h for name,h in code.items()):
                raise ValueError('Running audit code changed')
            for asset in ['NRT','LW','FIX']:
                bank = base/'asset_memory/v5'/asset
                if asset in done or not (bank/'structural_audit.json').exists(): continue
                write_json(status, {**common, 'state': 'auditing_completed_bank', 'asset': asset, 'completed_assets': done})
                cost = report(bank)
                cost_path = base/'v5/build_accounting'/f'{asset}.json'; write_json(cost_path, cost, immutable=True)
                checked = replay(asset, base/'quarter_replay_v5'/asset)
                done[asset] = {'bank_sha256': cost['bank_sha256'], 'cost_report_sha256': file_digest(cost_path),
                               'replay_report_sha256': file_digest(base/'quarter_replay_v5'/asset/'report.json'),
                               'new_model_calls': checked['new_model_calls']}
                print('ASSET_AUDITS_READY', asset, flush=True)
            main_ready = (base/'v6/source_grounding.json').exists()
            ablation_ready = (base/'v6/history_ablation/source_grounding.json').exists()
            if len(done) == 3 and main_ready and ablation_ready:
                write_json(status, {**common, 'state': 'analyzing_completed_development', 'completed_assets': done})
                paired = analyze(); path = base/'v6/paired_development_analysis.json'
                write_json(path, paired, immutable=True)
                write_json(status, {**common, 'state': 'audits_ready_for_review', 'finished': time.time(),
                    'completed_assets': done, 'paired_analysis_sha256': file_digest(path),
                    'conditions': paired['conditions'], 'n_questions': paired['n_questions']})
                print('DEVELOPMENT_AUDITS_READY_FOR_REVIEW', flush=True)
                return
            upstream = [base/f'v5/workers/{a}/status.json' for a in ['FIX','LW','NRT']]
            upstream += [base/f'v6/workers/{c}/status.json' for c in ['agentic','history_disabled']]
            for path in upstream:
                state = json.loads(path.read_text())
                if state['state'] in {'worker_failed','stopped_low_disk'}:
                    raise RuntimeError(f'Upstream worker stopped: {path} ({state["state"]})')
            if time.time()-started > 24*3600: raise TimeoutError('Audit wait exceeded 24h; no completion claimed')
            write_json(status, {**common, 'state': 'waiting_for_upstream_completion', 'completed_assets': done,
                'pending_assets': [a for a in ['FIX','LW','NRT'] if a not in done],
                'main_grounding_ready': main_ready, 'ablation_grounding_ready': ablation_ready})
            time.sleep(15)
    except Exception as exc:
        write_json(status, {**common, 'state': 'audit_worker_failed', 'finished': time.time(),
                            'error': type(exc).__name__+': '+str(exc), 'completed_assets': done})
        raise
    finally: stop.set()


if __name__ == '__main__': main()
