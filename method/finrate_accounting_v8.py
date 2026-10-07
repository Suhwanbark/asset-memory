"""Separate unique shared planning from standalone attributed method costs."""
from collections import Counter
from pathlib import Path

from finrate_final_evaluation import online_usage, summarize


def combined_usage(*parts):
    usage = Counter(); result = {'logged_response_calls': 0, 'requests_with_unknown_usage': 0,
                                'known_usage_requests_without_saved_response': 0}
    for part in parts:
        usage.update(part['provider_usage'])
        for key in result: result[key] += part.get(key, 0)
    return {**result, 'provider_usage': dict(usage)}


def components(folder, shared_folder=None):
    local = online_usage(folder)
    shared = online_usage(shared_folder) if shared_folder is not None else combined_usage()
    return {'online_usage': local, 'shared_planning_usage': shared,
            'standalone_online_usage': combined_usage(local, shared),
            'shared_planning_path': str(Path(shared_folder).resolve()) if shared_folder is not None else None}


def summary(rows):
    local = summarize(rows)
    standalone = summarize([{**r, 'online_usage': r.get('standalone_online_usage', r['online_usage'])} for r in rows])
    for condition, values in local.items():
        other = standalone[condition]
        values.update(standalone_attributed_provider_usage=other['online_provider_usage'],
                      standalone_component_response_calls=other['logged_response_calls'],
                      standalone_unknown_usage_requests=other['requests_with_unknown_usage'])
    return local


def unique_inference_accounting(rows):
    """Previously run static calls are labeled, never charged as new v8 calls."""
    unique = {}; new_local = []; reused = []
    for row in rows:
        if row['shared_planning_path'] is not None:
            path = row['shared_planning_path']; value = row['shared_planning_usage']
            if path in unique and unique[path] != value: raise ValueError('Shared usage differs across conditions')
            unique[path] = value
        (reused if row.get('reused_baseline', False) else new_local).append(row['online_usage'])
    return {'shared_planning_unique_questions': len(unique),
            'shared_planning_usage_counted_once': combined_usage(*unique.values()),
            'new_condition_local_usage': combined_usage(*new_local),
            'new_physical_inference_usage': combined_usage(*unique.values(), *new_local),
            'previously_executed_reused_baseline_usage': combined_usage(*reused),
            'scope': 'Inference only; exclude judges, offline memory/index construction and earlier development. Standalone per-condition attribution repeats shared components for comparison, not for the actual total.'}
