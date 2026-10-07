"""Read-only construction accounting for a completed source-only asset bank.

No benchmark questions or gold files are read. Per-quarter costs are attributed
to the original chronological build; they are not measured fresh incremental
update costs. Unknown/missing transport usage is never counted as zero.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from finrate_memory_audit import audit
from finrate_poc.bank import Bank
from finrate_poc.memory import quarter
from finrate_poc.model import token_count
from finrate_poc.runtime import check_disk, file_digest, write_json


def merged_length(intervals):
    end = -1; covered = 0
    for start, stop in sorted(intervals):
        covered += max(0, stop-max(end, start)); end = max(end, stop)
    return covered


def report(folder):
    check_disk(); folder = Path(folder)
    completed = json.loads((folder / 'completed.json').read_text())
    if completed.get('derived_checkpoint_snapshot'):
        raise ValueError('Derived snapshots omit original call logs; use the original full-build bank for costs')
    if completed.get('inherited_sources', 0):
        raise ValueError('Use the full-build bank for total construction accounting; a delta omits parent calls')
    if file_digest(folder / 'bank.sqlite') != completed['bank_sha256']:
        raise ValueError('Completed bank bytes changed')
    checked = audit(folder / 'bank.sqlite')
    quarters = defaultdict(Counter); relations = Counter(); source_tokens = fact_tokens = quote_tokens = 0
    span_lengths = []; quote_intervals = defaultdict(list); publication_by_source = {}
    semantic_links = Counter(); periods_unknown = 0; facts_with_edges = set(); cross_date_facts = set()
    bank = Bank(folder / 'bank.sqlite', readonly=True)
    try:
        for row in bank.db.execute('SELECT id,published,text FROM sources'):
            size = token_count(row['text']); source_tokens += size
            publication_by_source[row['id']] = row['published']
            quarters[quarter(row['published'])].update(sources=1, original_source_tokens=size)
        facts = {}
        for row in bank.db.execute('SELECT id,body FROM facts'):
            fact = json.loads(row['body']); facts[row['id']] = fact
            size = token_count(row['body']); fact_tokens += size; quote_tokens += token_count(fact['quote'])
            quarters[quarter(fact['observed_at'])].update(facts=1, serialized_fact_tokens=size)
            periods_unknown += fact['period'] is None
            span_lengths.append(fact['quote_end']-fact['quote_start'])
            quote_intervals[fact['source_id']].append((fact['quote_start'], fact['quote_end']))
        for row in bank.db.execute('SELECT old_id,new_id,body FROM links'):
            link = json.loads(row['body']); old = facts[row['old_id']]; new = facts[row['new_id']]
            quarters[quarter(link['observed_at'])].update(links=1)
            relations.update([link['relation']]); facts_with_edges.update([row['old_id'], row['new_id']])
            same = old['observed_at'] == new['observed_at']
            semantic_links.update(['same_publication_date' if same else 'different_publication_dates'])
            if not same: cross_date_facts.update([row['old_id'], row['new_id']])
        checkpoints = [json.loads(row[0]) for row in bank.db.execute('SELECT body FROM checkpoints ORDER BY label')]
    finally: bank.close()
    calls = {}; unknown_requests = []; response_times = []; rejects = Counter(); phases = defaultdict(Counter)
    for batch in sorted((folder / 'calls').iterdir()):
        if not batch.is_dir(): continue
        day = batch.name.split('_', 1)[1]; q = quarter(day)
        quarters[q].update(extraction_batches=1)
        for path in batch.rglob('request.json'):
            response_path = path.parent / 'response.json'; event_path = path.parent / 'events.jsonl'
            if not response_path.exists() or not event_path.exists():
                unknown_requests.append(str(path.relative_to(folder))); continue
            response = json.loads(response_path.read_text()); events = [json.loads(s) for s in event_path.read_text().splitlines() if s.strip()]
            identities = {e['thread_id'] for e in events if e.get('type') == 'thread.started'}
            if len(identities) != 1: raise ValueError('Ambiguous logged transport identity')
            identity = next(iter(identities)); usage = Counter()
            for e in events:
                if e.get('type') == 'turn.completed': usage.update({k: v for k, v in e.get('usage', {}).items() if type(v) is int})
            if not usage:
                unknown_requests.append(str(path.relative_to(folder))); continue
            phase = 'extraction' if path.parent.name == 'extract' else 'linking'
            value = {'quarter': q, 'phase': phase, 'usage': dict(usage), 'request_sha256': file_digest(path)}
            if identity in calls:
                if calls[identity] != value: raise ValueError('Duplicate trace with inconsistent attribution')
                continue
            calls[identity] = value
            phases[phase].update(usage); phases[phase].update(logged_calls=1)
            quarters[q].update({'model_calls': 1, **usage})
            if isinstance(response.get('elapsed_s'), (int, float)): response_times.append(response['elapsed_s'])
        for path in batch.rglob('validation.json'):
            for rejection in json.loads(path.read_text())['rejected']: rejects.update([rejection['reason']])
        for path in batch.rglob('span_conversion.json'):
            for rejection in json.loads(path.read_text())['rejected']: rejects.update([rejection['reason']])
    unique_quote_chars = sum(merged_length(spans) for spans in quote_intervals.values())
    total_usage = Counter()
    for row in calls.values(): total_usage.update(row['usage'])
    return {'asset': completed['asset'], 'bank_sha256': completed['bank_sha256'],
        'completed': completed, 'structural_audit_passed': checked['passed'],
        'source_only_analysis': True, 'questions_or_gold_read': False,
        'quarterly_original_build': {q: dict(v) for q, v in sorted(quarters.items())},
        'checkpoints': checkpoints, 'relations': dict(relations), 'publication_date_link_types': dict(semantic_links),
        'facts_with_any_edge': len(facts_with_edges), 'facts_with_cross_date_edge': len(cross_date_facts),
        'unknown_reporting_period_facts': periods_unknown,
        'original_source_tokens': source_tokens, 'serialized_fact_tokens': fact_tokens, 'stored_quote_tokens': quote_tokens,
        'fact_to_source_token_ratio': fact_tokens/source_tokens if source_tokens else None,
        'median_quote_chars': statistics.median(span_lengths) if span_lengths else None,
        'quote_repetition_ratio': sum(span_lengths)/unique_quote_chars if unique_quote_chars else None,
        'unique_logged_model_calls': len(calls), 'provider_usage': dict(total_usage),
        'by_phase': {p: dict(v) for p, v in phases.items()},
        'unknown_usage_requests': unknown_requests, 'structural_rejections': dict(rejects),
        'median_call_elapsed_seconds': statistics.median(response_times) if response_times else None,
        'limitations': ['Structural provenance does not establish semantic correctness.',
            'Different publication dates need not represent different reporting periods or real-world changes.',
            'Quarterly attribution describes the original build, not a fresh incremental-update experiment.',
            'Serialized fact size includes repeated quotations; compression is not assumed.',
            'Input includes cached input. Missing usage is unknown. No monetary billing estimate is inferred.']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('folder'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = report(args.folder); write_json(args.output, result, immutable=True)
    print(json.dumps({k: result[k] for k in ['asset', 'unique_logged_model_calls', 'provider_usage',
                     'fact_to_source_token_ratio', 'quote_repetition_ratio']}, ensure_ascii=False))


if __name__ == '__main__': main()
