"""Recover a closed-quarter snapshot and replay only its later source delta.

The completed full bank and parent snapshot are immutable. The last-quarter
replay uses exact logged responses from the original source-only build, with
zero new model calls allowed. This is a state/update equivalence check, not a
new unseen-data experiment or a measurement of fresh update latency/cost.
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3

from finrate_batched_memory import batches
from finrate_memory_audit import audit
from finrate_spanned_memory import build_asset
from finrate_v5_development import copy_frozen_tree
from finrate_poc.bank import Bank
from finrate_poc.data import documents, metadata
from finrate_poc.memory import quarter
from finrate_poc.model import Luna
from finrate_poc.runtime import ROOT, check_disk, digest, file_digest, write_json


def table_hashes(path):
    bank = Bank(path, readonly=True)
    try:
        return {table: digest(json.dumps([list(r) for r in bank.db.execute(f'SELECT * FROM {table} ORDER BY 1')],
                                          ensure_ascii=False, separators=(',', ':')))
                for table in ['sources', 'facts', 'links', 'events', 'checkpoints']}
    finally: bank.close()


def snapshot(source_folder, destination, label):
    check_disk(); source_folder = Path(source_folder); destination = Path(destination)
    completion = json.loads((source_folder / 'completed.json').read_text())
    original_hash = file_digest(source_folder / 'bank.sqlite')
    if original_hash != completion['bank_sha256']: raise ValueError('Original bank changed')
    if destination.exists(): raise ValueError('Snapshot destination must be new')
    source = Bank(source_folder / 'bank.sqlite', readonly=True)
    try:
        row = source.db.execute('SELECT body FROM checkpoints WHERE label=?', (label,)).fetchone()
        if row is None: raise ValueError('Unknown closed-quarter checkpoint')
        checkpoint = json.loads(row[0])
        cutoff = source.db.execute('SELECT seq FROM events WHERE sha256=?', (checkpoint['event_hash'],)).fetchone()
        if cutoff is None: raise ValueError('Checkpoint event hash is absent')
        cutoff = cutoff[0]
        prefix = [json.loads(r[0]) for r in source.db.execute('SELECT body FROM events WHERE seq<=? ORDER BY seq', (cutoff,))]
        allowed = {'sources': [e['id'] for e in prefix if e['op'] == 'register_source'],
                   'facts': [e['fact_id'] for e in prefix if e['op'] == 'add_fact'],
                   'links': [e['link_id'] for e in prefix if e['op'] == 'link']}
        kept_sources = set(allowed['sources'])
        source_rows = [dict(r) for r in source.db.execute('SELECT * FROM sources ORDER BY id') if r['id'] in kept_sources]
        destination.mkdir(parents=True)
        with sqlite3.connect(destination / 'bank.sqlite') as target:
            source.db.backup(target)
            target.execute('PRAGMA foreign_keys=ON')
            for table in ['links', 'facts', 'sources']:
                target.execute('CREATE TEMP TABLE kept_ids (id TEXT PRIMARY KEY)')
                target.executemany('INSERT INTO kept_ids VALUES(?)', [(key,) for key in allowed[table]])
                target.execute(f'DELETE FROM {table} WHERE id NOT IN (SELECT id FROM kept_ids)')
                target.execute('DROP TABLE kept_ids')
            target.execute('DELETE FROM events WHERE seq>?', (cutoff,))
            target.execute('UPDATE sqlite_sequence SET seq=? WHERE name=?', (cutoff, 'events'))
            target.execute('DELETE FROM checkpoints WHERE label>?', (label,))
    finally: source.close()
    checked = audit(destination / 'bank.sqlite')
    if checked['event_hash'] != checkpoint['event_hash']: raise ValueError('Recovered history hash differs')
    if any(checked[k] != checkpoint[k] for k in ['sources', 'facts', 'links']):
        raise ValueError('Recovered state differs from the checkpoint')
    if file_digest(source_folder / 'bank.sqlite') != original_hash: raise ValueError('Original bank was mutated')
    manifest = json.loads((source_folder / 'build_manifest.json').read_text())
    # A completed incremental child's manifest lists only its delta. Recover
    # the full historical source inventory from audited bank rows instead.
    manifest['sources'] = [{'id': r['id'], 'sha256': r['sha256'], 'title': r['title'], 'published': r['published']}
                           for r in source_rows]
    manifest['batch_count'] = len(list(batches([{'_id': r['id'], 'title': r['title'], 'text': r['text']} for r in source_rows])))
    manifest['parent'] = None
    write_json(destination / 'build_manifest.json', manifest, immutable=True)
    shutil.copyfile(source_folder / 'span_method_manifest.json', destination / 'span_method_manifest.json')
    result = {'asset': completion['asset'], 'final_quarter': label, 'event_hash': checked['event_hash'],
              'new_sources': checked['sources'], 'inherited_sources': 0, 'total_sources': checked['sources'],
              'facts': checked['facts'], 'links': checked['links'], 'batches': manifest['batch_count'],
              'bank_sha256': file_digest(destination / 'bank.sqlite'), 'derived_checkpoint_snapshot': True}
    write_json(destination / 'completed.json', result, immutable=True)
    write_json(destination / 'derivation.json', {'original_bank_sha256': original_hash,
        'original_path': str(source_folder.resolve()), 'checkpoint': checkpoint, 'event_sequence_cutoff': cutoff,
        'original_bank_unchanged': True, 'snapshot_audit': checked,
        'cost_note': 'No source ingestion or model invocation occurred; original build call logs remain in the full bank folder.'}, immutable=True)
    return result


def replay(asset, out):
    check_disk(); out = Path(out); base = ROOT / 'result/finrate_luna_v1'; source_folder = base / 'asset_memory/v5' / asset
    complete = json.loads((source_folder / 'completed.json').read_text())
    if (out / 'report.json').exists():
        result = json.loads((out / 'report.json').read_text())
        for path, expected in result['immutable_bank_hashes'].items():
            if file_digest(path) != expected: raise ValueError('Previously checked bank changed')
        return result
    checked = audit(source_folder / 'bank.sqlite')
    labels = [r['quarter'] for r in checked['checkpoints']]
    if len(labels) < 2: raise ValueError('Need at least two completed quarters')
    label = labels[-2]; parent = out / 'parent'; child = out / 'updated'
    parent_completion = (json.loads((parent / 'completed.json').read_text()) if (parent / 'completed.json').exists()
                         else snapshot(source_folder, parent, label))
    if parent_completion['final_quarter'] != label: raise ValueError('Parent checkpoint changed')
    parent_hash = file_digest(parent / 'bank.sqlite'); full_hash = file_digest(source_folder / 'bank.sqlite')
    if parent_hash != parent_completion['bank_sha256'] or full_hash != complete['bank_sha256']:
        raise ValueError('Frozen bank bytes changed')
    docs = [d for d in documents(ROOT / 'data/finrate_public_v1') if metadata(d)['ticker'] == asset]
    packets = list(batches(docs)); selected = [(i, batch) for i, batch in enumerate(packets) if quarter(batch['published']) > label]
    delta_ids = {w['source_id'] for _, batch in selected for w in batch['windows']}
    delta = [d for d in docs if d['_id'] in delta_ids]
    copied = []
    for new_index, (old_index, batch) in enumerate(selected):
        old = source_folder / 'calls' / f'{old_index:04d}_{batch["published"]}'
        new = child / 'calls' / f'{new_index:04d}_{batch["published"]}'
        copied.append({'original_batch': old.name, 'delta_batch': new.name,
                       'file_sha256': copy_frozen_tree(old, new)})
    write_json(out / 'replay_manifest.json', {'asset': asset, 'parent_quarter': label,
        'full_bank_sha256': full_hash, 'parent_bank_sha256': parent_hash, 'delta_sources': sorted(delta_ids),
        'cache_copies': copied, 'new_model_calls_allowed': 0, 'questions_or_gold_available': False,
        'code_sha256': file_digest(__file__)}, immutable=True)
    model = Luna(max_calls=0)
    updated = build_asset(delta, child, model, parent=parent)
    after = audit(child / 'bank.sqlite')
    original_tables = table_hashes(source_folder / 'bank.sqlite'); updated_tables = table_hashes(child / 'bank.sqlite')
    if original_tables != updated_tables: raise ValueError('Incremental replay did not recover the original logical bank')
    if file_digest(parent / 'bank.sqlite') != parent_hash or file_digest(source_folder / 'bank.sqlite') != full_hash:
        raise ValueError('Replay modified immutable source/parent banks')
    result = {'asset': asset, 'parent_quarter': label, 'final_quarter': complete['final_quarter'],
        'parent_sources': parent_completion['total_sources'], 'delta_sources': len(delta),
        'parent_facts': parent_completion['facts'], 'final_facts': updated['facts'],
        'parent_links': parent_completion['links'], 'final_links': updated['links'],
        'new_model_calls': model.calls, 'identical_logical_tables': True,
        'table_sha256': original_tables, 'event_hash': after['event_hash'],
        'immutable_bank_hashes': {str((parent / 'bank.sqlite').resolve()): parent_hash,
                                  str((source_folder / 'bank.sqlite').resolve()): full_hash,
                                  str((child / 'bank.sqlite').resolve()): file_digest(child / 'bank.sqlite')},
        'structural_audit_passed': after['passed'], 'questions_or_gold_used': False,
        'limitation': 'Exact cached-response replay validates state continuity and delta-only execution; not a fresh unseen-quarter test or measured new-inference cost.'}
    write_json(out / 'report.json', result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--asset', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = replay(args.asset, args.output)
    print(json.dumps({k: result[k] for k in ['asset', 'parent_quarter', 'final_quarter', 'parent_sources', 'delta_sources',
                                          'final_facts', 'final_links', 'new_model_calls', 'identical_logical_tables']}))


if __name__ == '__main__': main()
