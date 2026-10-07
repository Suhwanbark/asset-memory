"""Count real logged model invocations once despite copied response caches.

Ephemeral Codex thread IDs identify transport invocations. Provider usage is
read from completed-turn events, not inferred from local token counts or the
number of prediction artifacts. Pending/interrupted requests remain explicit.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from finrate_poc.runtime import check_disk, file_digest, write_json


def phase(path):
    parts = path.parts
    if 'development_gold_audit_v1' in parts: return 'reference_quality_audit'
    if 'blinded_judge' in parts: return 'original_gold_grading'
    if 'source_grounding_v1' in parts or 'source_grounding' in parts: return 'source_grounding_grading'
    if 'asset_memory' in parts or 'asset_prefetch_v2' in parts: return 'asset_memory_build'
    if 'dev' in parts:
        return 'online_qa_or_investigation'
    return 'implementation_smokes'


def collect(base):
    base = Path(base)
    invocations = {}
    missing_usage = []
    request_only = []
    parse_errors = []
    for path in sorted(base.rglob('events.jsonl')):
        if 'evaluation' in path.relative_to(base).parts: continue
        events = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip(): continue
            try: events.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors.append({'path': str(path.relative_to(base)), 'line': number})
        thread_ids = {e['thread_id'] for e in events if e.get('type') == 'thread.started' and e.get('thread_id')}
        if len(thread_ids) != 1:
            missing_usage.append({'path': str(path.relative_to(base)), 'reason': 'missing_or_ambiguous_transport_thread_id'})
            continue
        thread_id = next(iter(thread_ids))
        usage = Counter(); completed_turns = 0
        for e in events:
            if e.get('type') != 'turn.completed': continue
            completed_turns += 1
            usage.update({k: v for k, v in e.get('usage', {}).items() if type(v) is int})
        folder = path.parent
        request = json.loads((folder / 'request.json').read_text()) if (folder / 'request.json').exists() else {}
        response = json.loads((folder / 'response.json').read_text()) if (folder / 'response.json').exists() else {}
        record = {'thread_id': thread_id, 'paths': [str(path.relative_to(base))],
                  'phases': [phase(path.relative_to(base))], 'provider_usage': dict(usage),
                  'completed_turns': completed_turns,
                  'requested_model': request.get('model', response.get('model')),
                  'prompt_sha256': request.get('prompt_sha256', response.get('prompt_sha256')),
                  'elapsed_s': response.get('elapsed_s'), 'event_sha256': file_digest(path)}
        if thread_id in invocations:
            old = invocations[thread_id]
            if any(old[k] != record[k] for k in ['provider_usage', 'completed_turns', 'prompt_sha256', 'event_sha256']):
                raise ValueError('Copied transport identity has inconsistent trace or inputs: ' + thread_id)
            old['paths'].extend(record['paths'])
            old['phases'] = sorted(set(old['phases']) | set(record['phases']))
        else: invocations[thread_id] = record
        if not completed_turns:
            missing_usage.append({'path': str(path.relative_to(base)), 'thread_id': thread_id,
                                  'reason': 'no_completed_turn_usage; not assumed zero cost'})
    for path in sorted(base.rglob('request.json')):
        if 'evaluation' in path.relative_to(base).parts: continue
        if not (path.parent / 'events.jsonl').exists():
            request_only.append(str(path.relative_to(base)))
    total = Counter(); grouped = defaultdict(lambda: {'unique_invocations': 0, 'provider_usage': Counter()})
    for record in invocations.values():
        total.update(record['provider_usage'])
        # A smoke reused in an incremental smoke is still one invocation. An
        # unexpected cross-purpose reuse is exposed rather than double-counted.
        group = record['phases'][0] if len(record['phases']) == 1 else 'cross_phase_cache_reuse'
        grouped[group]['unique_invocations'] += 1
        grouped[group]['provider_usage'].update(record['provider_usage'])
    return {'captured_at': datetime.now(timezone.utc).isoformat(),
            'unique_logged_transport_invocations': len(invocations),
            'response_trace_copies_ignored': sum(len(r['paths'])-1 for r in invocations.values()),
            'total_provider_usage': dict(total), 'by_phase': dict(grouped),
            'missing_completed_usage': missing_usage, 'request_only_pending_or_interrupted': request_only,
            'event_parse_warnings': parse_errors,
            'requested_models': dict(Counter(r['requested_model'] for r in invocations.values())),
            'records': list(invocations.values()), 'evaluation_responses_read': False,
            'interpretation': 'Live development snapshot. Cached input is already part of provider input; reasoning fields are retained separately without adding them again to output. Missing/in-flight usage is unknown, not zero. No dollar cost inferred from Codex transport.'}


def save(base, output):
    check_disk()
    report = collect(base)
    write_json(output, report, immutable=True)
    return report
