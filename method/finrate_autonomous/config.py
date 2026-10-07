from finrate_poc.runtime import ROOT

BASE = ROOT / 'result/finrate_autonomous_v9'
DATA = ROOT / 'data/finrate_public_v1'
CORPUS = BASE / 'corpus.sqlite'
SEED = 'finrate-full43-autonomous-20261005-v1'
POLICY = {
    'version': 'v9', 'model': 'gpt-5.6-luna', 'reasoning': 'low',
    'cutoff_policy': 'exploratory_exposure_uncontrolled',
    'cutoff_authorization': 'User explicitly waived cutoff filtering; resumption authorized 2026-10-05.',
    'seed': SEED, 'assets': 'all43', 'tasks': ['LT-QA', 'EC-QA'],
    'dev_per_task': 10, 'evaluation_per_task': 20,
    'conditions': ['naive_llm', 'embedding', 'hybrid', 'raw_agent', 'chunk_agent', 'bank_agent', 'flat_bank_agent'],
    'agent_max_turns': 12, 'agent_token_ceiling': 160000,
    'writer_max_turns': 12, 'writer_token_ceiling': 160000,
    'tool_result_tokens': 3000, 'max_actions_per_turn': 4,
    'static_top_k': 24, 'static_evidence_tokens': 8000,
    'chunk_tokens': 192, 'chunk_stride': 160,
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'device': 'cpu', 'cpu_threads': 2, 'min_free_disk_bytes': 10000000000,
    'ingestion': 'Chronological publication-quarter batches; agent chooses which originals to inspect and how to edit. Availability is not a claim of exhaustive reading.',
    'agent_policy': 'Identical QA prompt, tool interface, budgets, original-source fallback and scratch. No prescribed plan/retrieval/review sequence.',
    'flat_control': 'Same authored page contents, dates and source anchors, flattened to separate source-anchored blocks without company/topic navigation; no new LLM preprocessing.',
    'selection': 'Fixed hash sampling within source-disjoint groups and task strata; previously exposed source groups excluded from heldout; no difficulty/gold-count/performance filtering.',
    'cost': 'Report actual provider calls/tokens, CPU indexing, bank construction and incremental update separately. Equal online ceilings do not mean equal total compute.',
    'heldout': 'One frozen release after development review. No heldout outcomes used in method choice.',
}
