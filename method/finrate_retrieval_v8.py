"""Prototype: preserve a common source core, use memory for additional spans.

All conditions receive the same plan and exact common source evidence. Derived
facts/edge labels only route retrieval; final evidence consists of exact source
spans with canonical source IDs. No questions/gold/results are loaded here.
"""
import json

from finrate_agent_v7 import diverse, interleave
from finrate_poc.model import fit_evidence, token_count

CORE_TOKENS = 4800
FINAL_TOKENS = 6000


def canonical_evidence(item):
    # Immutable artifacts sort JSON keys. Use that same order before budget
    # measurement so saving/loading a common packet cannot change its size.
    return {key: item[key] for key in sorted(item)}


class CoreAugmentRetriever:
    def __init__(self, temporal):
        self.temporal = temporal
        self.cached_plan = None

    def prepare(self, plan):
        key = json.dumps(plan, sort_keys=True)
        if key == self.cached_plan: return
        self.plan = plan; t = self.temporal
        self.ranked = t.ranking(t.index, plan['queries'])
        groups = t.scoped(self.ranked, plan)
        self.scoped_source_ids = {p['id'] for values in groups.values() for p in values}
        self.source_candidates = diverse([canonical_evidence(t.evidence(p)) for p in interleave(groups, limit=80)], per_source=3)[:32]
        self.core = fit_evidence(self.source_candidates, CORE_TOKENS)
        self.fact_ranked = None; self.cached_plan = key

    def _memory_candidates(self, *, history):
        t = self.temporal
        if t.fact_index is None: raise ValueError('Frozen fact index required')
        if self.fact_ranked is None: self.fact_ranked = t.ranking(t.fact_index, self.plan['queries'])
        groups = t.scoped(self.fact_ranked, self.plan)
        seeds = interleave(groups, limit=8)
        allowed = {p['id'] for values in groups.values() for p in values}
        rank = {p['id']: i for i, p in enumerate(self.fact_ranked)}
        seed_ids = {p['id'] for p in seeds}; neighbors = {}; traversals = []
        if history:
            for seed in seeds[:4]:
                asset = t.meta[seed['source_id']]['ticker']; result = t.banks[asset].history(seed['id'], limit=32)
                traversals.append({'seed': seed['id'], 'links': result['links']})
                for f in result['facts']:
                    if f['id'] not in seed_ids and f['id'] in allowed and rank.get(f['id'], 10**9) < 500:
                        neighbors[f['id']] = f
        neighbor_list = sorted(neighbors.values(), key=lambda f: (rank[f['id']], f['id']))[:8]
        routed = interleave({0: seeds, 1: neighbor_list}, limit=16)
        candidates = []; routes = []
        for record in routed:
            f = t.facts[record['id']]; source = t.docs[f['source_id']]
            if source['text'][f['quote_start']:f['quote_end']] != f['quote']:
                raise ValueError('Fact routing span no longer matches its original source')
            # Use the shared passage index to locate the best query-relevant
            # source span inside the fact's original quotation. Broad extraction
            # windows and model-written fact fields do not occupy final context.
            matches = [p for p in self.ranked if p['id'] in self.scoped_source_ids
                       and p['source_id'] == f['source_id']
                       and p['start'] < f['quote_end'] and p['end'] > f['quote_start']]
            for p in matches[:2]:
                e = canonical_evidence(t.evidence(p))
                candidates.append(e)
                routes.append({'evidence_id': e['id'], 'fact_id': f['id'],
                               'via_history': f['id'] in neighbors})
        return candidates, {'seed_ids': [p['id'] for p in seeds], 'traversals': traversals, 'routes': routes}

    def collect(self, plan, *, memory, history=False):
        self.prepare(plan)
        routed, trace = self._memory_candidates(history=history) if memory else (
            [], {'seed_ids': [], 'traversals': [], 'routes': []})
        candidates = diverse([*self.core, *routed, *self.source_candidates], per_source=4)
        evidence = fit_evidence(candidates, FINAL_TOKENS)
        by_id = {e['id']: e for e in evidence}
        if any(by_id.get(e['id']) != e for e in self.core):
            raise ValueError('Augmentation dropped or altered common source evidence')
        core_ids = {e['id'] for e in self.core}; added_ids = set(by_id)-core_ids
        selected_routes = [r for r in trace['routes'] if r['evidence_id'] in added_ids]
        return evidence, {**trace, 'shared_core_ids': sorted(core_ids), 'extra_ids': sorted(added_ids),
            'selected_memory_routes': selected_routes,
            'selected_graph_extra_ids': sorted({r['evidence_id'] for r in selected_routes if r['via_history']}),
            'all_core_evidence_preserved': True, 'core_token_budget': CORE_TOKENS,
            'core_tokens': token_count(json.dumps(self.core, ensure_ascii=False)),
            'final_token_budget': FINAL_TOKENS, 'final_tokens': token_count(json.dumps(evidence, ensure_ascii=False)),
            'derived_memory_assertions_presented_as_truth': False}
