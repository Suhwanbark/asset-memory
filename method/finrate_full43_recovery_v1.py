"""Resume the authorized full43 DEV pipeline with a recorded transport fix.

Frozen v10 code, completed banks, old failures and existing predictions remain
unchanged. Only previously unaccepted successful fallback responses are restored;
new terminal failures still halt bank construction. No heldout is released.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from finrate_poc.runtime import ROOT, check_disk, cpu_only, digest, file_digest, watchdog, write_json
from finrate_autonomous_v10.config import BASE, POLICY, SEED

RUN = BASE / 'recovery_v1'
MODULE = 'finrate_full43_recovery_v1'


def freeze():
    from finrate_autonomous_v10.evaluate import verify_online
    verify_online();check_disk()
    if (RUN / 'manifest.json').exists():
        return verify()
    old = json.loads((BASE / 'supervisors/pipeline/status.json').read_text())
    if old['state'] != 'needs_review' or old['active']:
        raise ValueError('Require the stopped original pipeline')
    if (Path('/proc') / str(old['pid']) / 'cmdline').exists():
        raise ValueError('Original supervisor PID is still present')
    if {r['label'] for r in old['failed']} != {'bank_NWN'}:
        raise ValueError('Unexpected original failed worker set')
    files = [Path(__file__), ROOT / 'method/finrate_transport_recovery_v1.py',
             ROOT / 'tests/test_transport_recovery_v1.py']
    preserved = {}
    completed = []
    for p in sorted((BASE / 'banks').glob('*/completed.json')):
        j = json.loads(p.read_text())
        if j['bank_sha256'] != file_digest(p.parent / 'bank.sqlite'):
            raise ValueError('Completed bank changed')
        completed.append(p.parent.name)
        for f in [p, p.parent / 'bank.sqlite']:
            preserved[str(f.relative_to(ROOT))] = file_digest(f)
    if len(completed) != 5:
        raise ValueError('Unexpected completed-bank set')
    for p in (BASE / 'dev').glob('*/*/prediction.json'):
        preserved[str(p.relative_to(ROOT))] = file_digest(p)
    snapshot = RUN / 'pre_resume/NWN'
    shutil.copytree(BASE / 'banks/NWN', snapshot)
    snapshot_hashes = {str(p.relative_to(snapshot)): file_digest(p)
                       for p in snapshot.rglob('*') if p.is_file()}
    write_json(RUN / 'pre_resume/original_pipeline_status.json', old, immutable=True)
    for p in files:
        target = RUN / 'code_snapshot' / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    manifest = {
        'code_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in files},
        'original_online_manifest_sha256': file_digest(BASE / 'online_manifest.json'),
        'corpus_manifest_sha256': file_digest(BASE / 'corpus_manifest.json'),
        'cohort_manifest_sha256': file_digest(BASE / 'cohort/manifest.json'),
        'completed_banks_reused': completed, 'preserved_files_sha256': preserved,
        'nwn_snapshot_files_sha256': snapshot_hashes,
        'transport_policy': 'Restore the exact completed original response only for a verified informational HTTPS fallback notice; no resampling, terminal retry or native tool acceptance.',
        'unchanged': ['model', 'prompts', 'tools', 'budgets', 'cohort', 'source corpus', 'existing predictions'],
        'model': POLICY['model'], 'cpu_only': True,
        'min_free_disk_bytes': POLICY['min_free_disk_bytes'],
        'concurrency': 2, 'heldout_release': False,
        'authorization': 'User requested continued work to completion; later status requests do not cancel it.',
    }
    write_json(RUN / 'manifest.json', manifest, immutable=True)
    return manifest


def verify():
    from finrate_autonomous_v10.evaluate import verify_online
    verify_online();check_disk()
    m = json.loads((RUN / 'manifest.json').read_text())
    for group in ['code_sha256', 'preserved_files_sha256']:
        for name, sha in m[group].items():
            if file_digest(ROOT / name) != sha:
                raise ValueError('Recovery input or code changed: ' + name)
    for name, key in [('online_manifest.json', 'original_online_manifest_sha256'),
                      ('corpus_manifest.json', 'corpus_manifest_sha256'),
                      ('cohort/manifest.json', 'cohort_manifest_sha256')]:
        if file_digest(BASE / name) != m[key]:
            raise ValueError('Original manifest changed: ' + name)
    return m


def bank_worker(asset, status):
    from finrate_autonomous_v10 import bank
    from finrate_transport_recovery_v1 import RecoveredLuna
    verify()
    root = BASE / 'banks' / asset
    if (root / 'completed.json').exists():
        raise ValueError('Do not rerun a completed bank')
    write_json(root / 'recovery_execution.json', {
        'recovery_manifest_sha256': file_digest(RUN / 'manifest.json'),
        'recovery_module': MODULE, 'original_frozen_bank_implementation_unchanged': True,
    }, immutable=True)
    bank.IsolatedLuna = RecoveredLuna
    result = bank.build(asset, root, status=status)
    write_json(root / 'status.json', {**result, 'pid': os.getpid(), 'finished': time.time(),
                                     'recovery_run': str(RUN)})


def qa_worker(stage, condition, status):
    from finrate_autonomous_v10 import evaluate
    from finrate_transport_recovery_v1 import RecoveredLuna
    verify();evaluate.IsolatedLuna = RecoveredLuna
    if stage == 'infer':evaluate.infer('dev', condition, status)
    elif stage == 'grade':evaluate.grade('dev', status)
    else:evaluate.ground('dev', status)


def pipeline(status):
    from finrate_autonomous_v10 import orchestrate, audit, export, diagnostics
    from finrate_final_evaluation import single_executor
    m = verify()
    with single_executor(RUN / 'pipeline.lock'):
        # Only the supervisor's worker-record destination changes. Bank and QA
        # workers keep canonical v10 paths, preserving action IDs and cached data.
        orchestrate.BASE = RUN
        assets = sorted(json.loads((BASE / 'corpus_manifest.json').read_text())['assets'],
                        key=lambda a: digest(SEED + 'build' + a))
        jobs = [('bank_' + a, MODULE, ['bank', '--asset', a])
                for a in assets if a not in m['completed_banks_reused']]
        orchestrate.supervise(jobs, status, concurrency=2)
        verify()
        audits = {a: audit.structural(BASE / 'banks' / a) for a in assets}
        write_json(BASE / 'bank_structural_audits.json', audits, immutable=True)
        export.export_all()
        if not (BASE / 'index/manifest.json').exists():
            raise ValueError('Full CPU index must be complete before DEV inference')
        jobs = [('dev_' + c, MODULE, ['infer', '--condition', c]) for c in POLICY['conditions']]
        orchestrate.supervise(jobs, status, concurrency=2)
        for stage in ['grade', 'ground']:
            orchestrate.supervise([('dev_' + stage, MODULE, [stage])], status, concurrency=1)
        diagnostics.collect('dev');verify()
        report = json.loads((BASE / 'dev/report.json').read_text())
        lines = ['Full43 DEV comparison, fixed LT10+EC10; heldout not released.']
        for c, row in report['summary']['all'].items():
            lines.append(f"{c}: correct {row['correct']}/{row['n']}; keypoints {row['key_point_lower']:.2%}..{row['key_point_upper']:.2%}; unassessed {row['unassessed']}")
        (BASE / 'dev/RESULTS.txt').write_text('\n'.join(lines) + '\n')
        write_json(status, {'state': 'development_ready_for_review', 'pid': os.getpid(),
                            'finished': time.time(), 'heldout_release': False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'pipeline', 'bank', 'infer', 'grade', 'ground'])
    p.add_argument('--asset');p.add_argument('--condition', choices=POLICY['conditions'])
    a = p.parse_args();cpu_only();check_disk();os.nice(10)
    if a.stage == 'prepare':freeze();return
    label = ('bank_' + a.asset if a.stage == 'bank' else
             'dev_' + a.condition if a.stage == 'infer' else 'dev_' + a.stage)
    status = RUN / ('status.json' if a.stage == 'pipeline' else 'workers/' + label + '/status.json')
    stop = watchdog(status)
    try:
        if a.stage == 'pipeline':pipeline(status)
        elif a.stage == 'bank':bank_worker(a.asset, status)
        else:qa_worker(a.stage, a.condition, status)
        if a.stage != 'pipeline':
            write_json(status, {'state': 'completed', 'pid': os.getpid(), 'finished': time.time()})
    except Exception as exc:
        prior = json.loads(status.read_text()) if status.exists() else {}
        write_json(status, {**prior, 'state': 'needs_review', 'error': str(exc), 'pid': os.getpid()})
        raise
    finally:stop.set()


if __name__ == '__main__':main()
