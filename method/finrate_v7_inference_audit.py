"""Independent full-grid provenance/identity audit after DEV generation."""
from collections import Counter
import json

from finrate_final_evaluation import prediction_grid, online_usage
from finrate_v5_development import verify_frozen
from finrate_poc.data import documents
from finrate_poc.model import token_count
from finrate_poc.runtime import ROOT, check_disk, file_digest, write_json


def main():
    check_disk(); base = ROOT / 'result/finrate_luna_v1'; out = base / 'v7'
    manifest = json.loads((out / 'method_manifest.json').read_text()); verify_frozen(ROOT, manifest)
    cohort = json.loads((base / 'cohort/cohort_manifest.json').read_text())
    ids = cohort['selected_ids']['dev']; conditions = ['naive_llm', 'embedding', 'hybrid', 'raw_agent', 'agentic']
    prediction_grid(out / 'dev', ids, conditions)
    prediction_grid(out / 'history_ablation/dev', ids, ['history_disabled'])
    corpus = {d['_id']: d for d in documents(ROOT / 'data/finrate_public_v1')}
    banks = {}
    for asset, row in json.loads((out / 'memory_manifest.json').read_text()).items():
        actual = file_digest(ROOT / row['path']); expected = row['completion']['bank_sha256']
        if actual != expected: raise ValueError('Bank bytes changed')
        banks[asset] = actual
    rows = []; calls = Counter(); events = Counter(); issues = []; input_hashes = {}
    for condition in [*conditions, 'history_disabled']:
        root = out / 'history_ablation/dev' if condition == 'history_disabled' else out / 'dev'
        for q in ids:
            path = root / q / condition / 'prediction.json'; record = json.loads(path.read_text())
            input_hashes[str(path.relative_to(out))] = file_digest(path)
            spans = 0; evidence = record.get('evidence', [])
            for item in evidence:
                source = corpus[item['source_id']]; a = item.get('quote_start', item.get('start'))
                b = item.get('quote_end', item.get('end')); text = item.get('quote', item.get('text'))
                if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(source['text']) or source['text'][a:b] != text:
                    issues.append({'q_id': q, 'condition': condition, 'kind': 'source_span_mismatch', 'id': item['id']})
                spans += 1
            reserialized_tokens = token_count(json.dumps(evidence, ensure_ascii=False))
            # Artifact JSON is sorted for determinism. Key order changes token
            # counts; the budget applies to evidence actually sent to the model.
            answer_request = path.parent/'answer/request.json'
            if record['status'] == 'completed':
                payload = json.loads(json.loads(answer_request.read_text())['prompt'].rsplit('\n', 1)[1])
                if payload['evidence'] != evidence:
                    issues.append({'q_id': q, 'condition': condition, 'kind': 'final_request_evidence_mismatch'})
                evidence_tokens = token_count(json.dumps(payload['evidence'], ensure_ascii=False))
                if evidence_tokens != record['evidence_tokens']:
                    issues.append({'q_id': q, 'condition': condition, 'kind': 'recorded_evidence_count_mismatch'})
            else: evidence_tokens = reserialized_tokens
            if evidence_tokens > manifest['policy']['evidence_tokens'] or record.get('online_budget_exceeded'):
                issues.append({'q_id': q, 'condition': condition, 'kind': 'budget_exceeded'})
            allowed = {e['id'] for e in evidence}
            bad = [i for c in record.get('answer', {}).get('claims', []) for i in c['evidence_ids'] if i not in allowed]
            if bad: issues.append({'q_id': q, 'condition': condition, 'kind': 'invalid_citation', 'ids': bad})
            actual_text_tokens = 0
            for p in path.parent.rglob('request.json'):
                if 'blinded_judge' in p.parts: continue
                req = json.loads(p.read_text()); calls[condition] += 1
                actual_text_tokens += token_count(req['prompt'])
                if (req.get('model'), req.get('reasoning'), req.get('transport')) != (
                    'gpt-5.6-luna', 'low', 'codex_exec_ephemeral_v7_shell_sleep_disabled'):
                    issues.append({'kind': 'request_identity', 'path': str(p.relative_to(out))})
                input_hashes[str(p.relative_to(out))] = file_digest(p)
                response = p.parent/'response.json'
                if response.exists():
                    saved = json.loads(response.read_text())
                    actual_text_tokens += token_count(saved['text'])
                    if saved.get('model') != req['model'] or saved.get('prompt_sha256') != req['prompt_sha256']:
                        issues.append({'kind': 'response_identity', 'path': str(response.relative_to(out))})
                    input_hashes[str(response.relative_to(out))] = file_digest(response)
                event_path = p.parent/'events.jsonl'
                if event_path.exists():
                    input_hashes[str(event_path.relative_to(out))] = file_digest(event_path)
                    for line in event_path.read_text().splitlines():
                        event = json.loads(line)
                        if event.get('type') != 'item.completed': continue
                        item = event.get('item', {}); kind = item.get('type'); events[kind] += 1
                        if kind in {'command_execution', 'mcp_tool_call', 'web_search', 'file_change'}:
                            issues.append({'kind': 'native_tool_use', 'path': str(event_path.relative_to(out))})
            if record['status'] == 'completed' and actual_text_tokens != record['online_text_tokens']:
                issues.append({'q_id': q, 'condition': condition, 'kind': 'online_text_accounting_mismatch'})
            if actual_text_tokens > manifest['policy']['online_token_ceiling']:
                issues.append({'q_id': q, 'condition': condition, 'kind': 'actual_online_text_budget_exceeded'})
            rows.append({'q_id': q, 'condition': condition, 'status': record['status'], 'source_spans': spans,
                         'evidence_tokens': evidence_tokens, 'reserialized_artifact_evidence_tokens': reserialized_tokens,
                         'actual_online_text_tokens': actual_text_tokens, 'online_usage': online_usage(path.parent),
                         'has_memory_evidence': any(e['id'].startswith('fact_') for e in evidence),
                         'visible_history_edges': len(record.get('visible_history_links', []))})
    result = {'version': 'v7', 'audit_revision': 2, 'split': 'dev', 'all120_predictions_present': len(rows) == 120,
              'status_counts': dict(Counter(r['status'] for r in rows)), 'checked_source_spans': sum(r['source_spans'] for r in rows),
              'online_requests_by_condition': dict(calls), 'online_completed_item_types': dict(events),
              'issues': issues, 'structural_checks_passed': not issues, 'bank_sha256': banks,
              'rows': rows, 'input_sha256': input_hashes, 'audit_code_sha256': file_digest(__file__),
              'new_model_calls': 0, 'gold_or_heldout_read': False,
              'supersedes_budget_measurement_of': 'inference_integrity_audit.json',
              'measurement_correction': 'Measure exact request evidence order rather than sorted-key artifact reserialization; compare semantic evidence equality and independently recompute all online prompt/response text tokens.',
              'limitations': ['Exact source spans and citation IDs do not establish semantic correctness or completeness.',
                              'Provider model alias is verified; immutable weights are not independently verified.',
                              'Completed error items may be transport fallback notices; raw events retain the full messages.']}
    write_json(out/'inference_integrity_audit_v2.json', result, immutable=True)
    print(json.dumps({k: v for k, v in result.items() if k not in {'rows', 'input_sha256'}}), flush=True)
    if issues: raise SystemExit(1)


if __name__ == '__main__': main()
