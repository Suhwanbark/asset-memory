"""Validate materialized memory against its event history and source spans.

A valid hash chain alone does not prove the SQL fact tables were not changed.
This audit reconstructs the expected identities from events and checks the
actual tables, quotation offsets, content hashes and checkpoint prefix counts.
It is structural, not an LLM fact-entailment judge.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from finrate_poc.bank import Bank, canonical
from finrate_poc.memory import quarter
from finrate_poc.runtime import check_disk, digest, file_digest, write_json


def audit(path):
    bank = Bank(path, readonly=True)
    try:
        bank.db.execute('BEGIN')
        final_hash = bank.verify_event_chain()
        expected = {'sources': {}, 'facts': {}, 'links': {}}
        zero = '0' * 64
        prefixes = {zero: {'sources': 0, 'facts': 0, 'links': 0, 'last_quarter': None}}
        last_date = None
        for row in bank.db.execute('SELECT body,sha256 FROM events ORDER BY seq'):
            event = json.loads(row['body'])
            op = event['op']
            if op == 'register_source':
                key = event['id']
                if key in expected['sources'] or (last_date and event['published'] < last_date):
                    raise ValueError('Source events duplicate IDs or run backwards')
                expected['sources'][key] = event
                last_date = event['published']
            elif op == 'add_fact':
                key = event['fact_id']
                if key in expected['facts'] or event['body']['source_id'] not in expected['sources']:
                    raise ValueError('Fact event has duplicate/unknown source identity')
                if key != 'fact_' + digest(canonical(event['body']))[:24]:
                    raise ValueError('Fact content identity differs from event')
                expected['facts'][key] = event['body']
            elif op == 'link':
                key = event['link_id']
                body = event['body']
                if key in expected['links'] or any(body[k] not in expected['facts'] for k in ('old_id', 'new_id')):
                    raise ValueError('Link event has duplicate/unknown endpoints')
                if key != 'link_' + digest(canonical(body))[:24]:
                    raise ValueError('Link content identity differs from event')
                expected['links'][key] = body
            else:
                raise ValueError('Unknown bank event operation')
            prefixes[row['sha256']] = {**{t: len(v) for t, v in expected.items()},
                                       'last_quarter': quarter(last_date) if last_date else None}
        sources = {r['id']: dict(r) for r in bank.db.execute('SELECT * FROM sources')}
        if set(sources) != set(expected['sources']):
            raise ValueError('Source table IDs differ from event history')
        for source_id, source in sources.items():
            event = expected['sources'][source_id]
            if source['published'] != event['published'] or source['sha256'] != event['text_sha256']:
                raise ValueError('Source metadata differs from event history')
            if digest(source['text']) != source['sha256']:
                raise ValueError('Original source text changed')
        facts = {r['id']: dict(r) for r in bank.db.execute('SELECT * FROM facts')}
        if set(facts) != set(expected['facts']):
            raise ValueError('Fact table IDs differ from event history')
        for fact_id, row in facts.items():
            body = json.loads(row['body'])
            if body != expected['facts'][fact_id]:
                raise ValueError('Fact table content differs from event history')
            source = sources[body['source_id']]
            if row['source_id'] != source['id'] or row['observed_at'] != source['published']:
                raise ValueError('Fact table metadata differs from source')
            if body['source_sha256'] != source['sha256'] or body['observed_at'] != source['published']:
                raise ValueError('Fact provenance differs from source')
            if source['text'][body['quote_start']:body['quote_end']] != body['quote']:
                raise ValueError('Fact quote offsets do not match original source')
        links = {r['id']: dict(r) for r in bank.db.execute('SELECT * FROM links')}
        if set(links) != set(expected['links']):
            raise ValueError('Link table IDs differ from event history')
        for link_id, row in links.items():
            body = json.loads(row['body'])
            if body != expected['links'][link_id] or any(row[k] != body[k] for k in ('old_id', 'new_id')):
                raise ValueError('Link table content differs from event history')
            old, new = (expected['facts'][body[k]] for k in ('old_id', 'new_id'))
            if old['subject'] != new['subject'] or old['observed_at'] > new['observed_at']:
                raise ValueError('Link crosses entity or reverses time')
            if body['source_id'] != new['source_id'] or body['evidence_quote'] not in sources[new['source_id']]['text']:
                raise ValueError('Link evidence is not grounded in the newer source')
        checkpoints = []
        for row in bank.db.execute('SELECT * FROM checkpoints ORDER BY label'):
            body = json.loads(row['body']); prefix = prefixes.get(row['event_hash'])
            if not prefix or row['event_hash'] != body['event_hash'] or row['label'] != body['quarter']:
                raise ValueError('Checkpoint has no valid event prefix')
            if any(body[t] != prefix[t] for t in ('sources', 'facts', 'links')):
                raise ValueError('Checkpoint counts differ from its event prefix')
            if prefix['last_quarter'] != row['label']:
                raise ValueError('Checkpoint quarter differs from source chronology')
            checkpoints.append(body)
        return {'event_hash': final_hash, 'sources': len(sources), 'facts': len(facts), 'links': len(links),
                'checkpoints': checkpoints,
                'quote_alignment': dict(Counter(f.get('quote_alignment', 'exact') for f in expected['facts'].values())),
                'modalities': dict(Counter(f['modality'] for f in expected['facts'].values())),
                'unknown_period_facts': sum(f['period'] is None for f in expected['facts'].values()),
                'passed': True, 'scope': 'structural provenance and history consistency; semantic truth not assessed'}
    finally:
        bank.close()


def snapshot_audit(source, destination):
    """Save a consistent audit snapshot of a live worker's staging database."""
    check_disk()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Audit destination must be new')
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
    try:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)
    finally:
        src.close()
    report = {'original_path': str(source), 'snapshot_path': str(destination),
              'snapshot_sha256': file_digest(destination),
              'captured_at': datetime.now(timezone.utc).isoformat(), **audit(destination)}
    write_json(destination.with_suffix('.audit.json'), report, immutable=True)
    return report
