"""Append later-quarter disclosures to an immutable asset-memory snapshot.

This companion does not modify the running v2 package. A completed parent is
copied with SQLite backup; only new sources invoke extraction/linking. Parent
facts, historical links and checkpoint/event hashes retain their identities.
Late-arriving sources within an already closed quarter require a new branch
and are deliberately rejected here, rather than silently rewriting history.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3

from finrate_poc import bank as bank_module, memory as memory_module, model as model_module
from finrate_poc.bank import Bank, source_span
from finrate_poc.data import metadata
from finrate_poc.memory import extract_prompt, quarter, source_windows
from finrate_poc.model import fit_evidence, parse_json
from finrate_poc.runtime import check_disk, digest, file_digest, write_json
from finrate_poc.schemas import EXTRACTION, LINKS


def _append_source(bank, asset, source, folder, model, policy):
    bank.register_source(source)
    for start, text in source_windows(source['text']):
        call_dir = folder / 'calls' / source['id'] / str(start)
        proposal = parse_json(model.invoke(extract_prompt(asset, source, text, policy),
                                          call_dir / 'extract', schema=EXTRACTION))
        accepted, rejected = [], []
        for fact in proposal['facts']:
            try:
                if fact.get('subject') != asset or fact.get('source_id') != source['id']:
                    raise ValueError('Extraction changed asset or source')
                source_span(text, fact.get('quote', ''))
                accepted.append(bank.add_fact(fact))
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append({'proposal': fact, 'reason': str(exc)})
        write_json(call_dir / 'extraction_validation.json', {
            'accepted_ids': accepted, 'rejected': rejected,
            'limitation': 'Structural quote validation does not establish semantic accuracy.'}, immutable=True)
        new_facts = [bank.read_fact(fact_id) for fact_id in accepted]
        past = {}
        for fact in new_facts:
            for old in bank.grep(fact['topic'][:200], subject=asset, before=source['published'], limit=8):
                if old['source_id'] != source['id']:
                    past[old['id']] = old
        candidates = fit_evidence(list(past.values()), 4500)
        if not candidates or not new_facts:
            continue
        prompt = ('No tools or external knowledge. Compare the new source-grounded observation with the '
                  'candidate history. Link only genuinely comparable subjects, scope, metrics and periods. '
                  'A later observation does not invalidate a historically true earlier one. Silence does not '
                  'mean termination. Use updates, continues, contradicts, retracts, or same_topic. '
                  'Return JSON {"links":[{"old_id":"...", "new_id":"...", "relation":"...", '
                  '"rationale":"source-supported reason", "evidence_quote":"exact quote from new source"}]}; '
                  'return an empty list if no justified link.\n' +
                  json.dumps({'new': new_facts, 'past': candidates}, ensure_ascii=False))
        proposed = parse_json(model.invoke(prompt, call_dir / 'links_batch', schema=LINKS))
        link_ids, invalid = [], []
        for link in proposed['links']:
            try:
                if link.get('new_id') not in accepted or link.get('old_id') not in {f['id'] for f in candidates}:
                    raise ValueError('Link endpoint was not supplied')
                link_ids.append(bank.link(**link))
            except (ValueError, TypeError, KeyError) as exc:
                invalid.append({'proposal': link, 'reason': str(exc)})
        write_json(call_dir / 'links_batch/validation.json', {
            'accepted_ids': link_ids, 'rejected': invalid}, immutable=True)


def append_snapshot(parent_dir, target_dir, docs, model):
    """Create a later immutable snapshot; never mutate or re-extract the parent."""
    check_disk()
    parent_dir, target_dir = Path(parent_dir).resolve(), Path(target_dir).resolve()
    if parent_dir == target_dir or parent_dir in target_dir.parents:
        raise ValueError('Target must be a separate snapshot directory')
    parent_completion = json.loads((parent_dir / 'completed.json').read_text())
    parent_build = json.loads((parent_dir / 'build_manifest.json').read_text())
    asset = parent_completion['asset']
    if parent_build['asset'] != asset:
        raise ValueError('Parent asset identity mismatch')
    policy = parent_build['extraction_policy']
    parent_hash = file_digest(parent_dir / 'bank.sqlite')
    if parent_completion.get('bank_sha256', parent_hash) != parent_hash:
        raise ValueError('Parent snapshot bytes changed')
    parent = Bank(parent_dir / 'bank.sqlite', readonly=True)
    try:
        event_hash = parent.verify_event_chain()
        if event_hash != parent_completion['event_hash']:
            raise ValueError('Parent event history changed')
        source_ids = {r[0] for r in parent.db.execute('SELECT id FROM sources')}
        rows = sorted([(metadata(d), d) for d in docs],
                      key=lambda pair: (pair[0]['filing_date'] or '', pair[0]['doc_id']))
        if not rows or len({d['_id'] for _, d in rows}) != len(rows):
            raise ValueError('Delta must contain distinct new sources')
        for meta, doc in rows:
            if meta['ticker'] != asset or not meta['filing_date']:
                raise ValueError('Delta source asset/date mismatch')
            if doc['_id'] in source_ids:
                raise ValueError('Existing sources must not be reprocessed')
            if quarter(meta['filing_date']) <= parent_completion['final_quarter']:
                raise ValueError('Delta is within an already closed quarter; branch/rebuild required')
        code_paths = [Path(__file__), Path(bank_module.__file__), Path(memory_module.__file__),
                      Path(model_module.__file__)]
        manifest = {
            'asset': asset, 'extraction_policy': policy,
            'kind': 'incremental_later_quarters', 'parent': str(parent_dir),
            'parent_bank_sha256': parent_hash, 'parent_event_hash': event_hash,
            'parent_completion_sha256': file_digest(parent_dir / 'completed.json'),
            'parent_build_sha256': file_digest(parent_dir / 'build_manifest.json'),
            'parent_final_quarter': parent_completion['final_quarter'],
            'code_sha256': {str(p): file_digest(p) for p in code_paths},
            'sources': [{'id': d['_id'], 'title': d['title'], 'sha256': digest(d['text']),
                         'published': m['filing_date']} for m, d in rows],
            'questions_and_answers_available': False,
        }
        write_json(target_dir / 'build_manifest.json', manifest, immutable=True)
        if (target_dir / 'completed.json').exists():
            completed = json.loads((target_dir / 'completed.json').read_text())
            if file_digest(target_dir / 'bank.sqlite') != completed['bank_sha256']:
                raise ValueError('Completed snapshot changed')
            return completed
        pending = target_dir / 'bank.pending.sqlite'
        pending.unlink(missing_ok=True)
        # SQLite backup is consistent, and no write handle is opened on parent.
        with sqlite3.connect(pending) as cloned:
            parent.db.backup(cloned)
    finally:
        parent.close()
    bank = Bank(pending)
    try:
        current_quarter = None
        for meta, doc in rows:
            check_disk()
            q = quarter(meta['filing_date'])
            if current_quarter and q != current_quarter:
                bank.checkpoint(current_quarter)
            source = {'id': doc['_id'], 'title': doc['title'], 'text': doc['text'],
                      'published': meta['filing_date']}
            _append_source(bank, asset, source, target_dir, model, policy)
            current_quarter = q
        bank.checkpoint(current_quarter)
        completed = {'asset': asset, 'final_quarter': current_quarter,
                     'parent_event_hash': event_hash, 'event_hash': bank.verify_event_chain(),
                     'new_sources': len(rows), 'inherited_sources': len(source_ids),
                     'total_sources': bank.db.execute('SELECT COUNT(*) FROM sources').fetchone()[0]}
    finally:
        bank.close()
    if file_digest(parent_dir / 'bank.sqlite') != parent_hash:
        raise ValueError('Parent was modified during incremental build')
    check_disk()
    os.replace(pending, target_dir / 'bank.sqlite')
    completed['bank_sha256'] = file_digest(target_dir / 'bank.sqlite')
    write_json(target_dir / 'completed.json', completed, immutable=True)
    return completed
