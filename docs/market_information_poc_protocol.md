# Market information selection PoC — 2026-09-29

## Authorization and scope

The user authorized setting a goal, constructing the dataset, and running the PoC. This supersedes the earlier experiment's no-ablation/status-only scope. Research question: can an LLM reassess information value as a financial prediction market evolves, and does its selection improve a fixed forecaster relative to recency and relevance?

Do not recreate an expiry/contract-status benchmark. Use genuine traded economic/financial event questions, dated pre-outcome evidence, market histories, and externally resolved outcomes. No trading or remote publication. Preserve all earlier experiments. All new outputs go under `market_information_poc_v1`.

## Design fixed before model outcomes

- Unit: event × historical cutoff; report event and snapshot counts separately.
- Outcome: actual binary market resolution; no researcher-assigned item utility labels.
- Compare latest-first, lexical relevance, utility-aware model selection, seeded random; fixed forecaster, identical candidate pool and selected evidence budget.
- Include no-evidence and supplied-market-probability baselines. Market histories used for phase assignment are not news utility gold.
- Multiple cutoffs per event, selected by time/availability rather than model failure. Phase labels use only observations through cutoff. Distinguish active consensus from no trading where activity data permit.
- Forecast Brier score paired within snapshot and averaged within event before aggregate comparisons. Uncertainty clustered by event, with shared economic release/calendar dependencies disclosed.
- Information addition/removal is a downstream, model-conditional contribution estimate, not intrinsic/causal market value. Reserve full-pool and ablation for diagnostic analyses.
- Separate labels/market future from whitelisted inference inputs. Enforce evidence availability at cutoff. No current mutable news body treated as a verified historical snapshot without qualification.
- Keep every attempted condition, raw response, parse/transport failure and exclusion reason. Retry transport failures only. Freeze dataset and prompts before outcome inspection. Prompt/data revisions get new versions and are evaluated across the affected sample, not selected failures.
- Success means an auditable experiment and a measured result, not a forced positive effect. A target pattern can be exploratory; claims of an advantage require held-out confirmation. Null/opposite results remain in the report.

## Execution plan

1. Audit actual data availability and freeze feasible source/domain/sample criteria.
2. Build first source-grounded snapshots, validate publication time/outcome boundaries.
3. Run a small end-to-end smoke to validate parsing/runtime, then freeze the confirmatory protocol on a separate event set.
4. Expand toward 30 economic events and 2–3 cutoffs each if archive coverage permits.
5. Compare selectors, audit representative positive/negative/null cases, run relevant temporal ablations and publish local reproducible results.

## Resource and provenance constraints

- Preserve the existing one-idle-GPU / 20 GiB free-disk constraints when applicable. Current GPUs contain other work; use existing CPU/CLI inference transport unless an idle GPU becomes available.
- Previous direct Polymarket API access returned HTTP451. No bypass. Independently public research releases or another accessible market's public API are acceptable and must be disclosed.
- Kalshi public read-only markets endpoint responded HTTP200 during this session; audit historical prices and outcomes next.
- Retrospective model knowledge contamination cannot be excluded by a prompt. Include no-evidence control and disclose model release/cutoff limits. Prospective evaluation is a later confirmation step, not falsely reported as completed.

This document will receive availability amendments before inference; execution results belong in separate result files.

## Availability amendment / frozen primary PoC (before forecasting)

- Actual population: all finalized Kalshi H0 (no-rate-change) FOMC contracts returned by public live + historical series endpoints, one per meeting date. 29 contract records; **28 meetings and 78 snapshots** pass source, market-open and price-history eligibility. The empty Jan-26-2024 contract is excluded; all exclusions recorded.
- 212 official dated Federal Reserve source documents: monetary/economic speeches, preceding FOMC statements/minutes. This is a primary-source financial information setting, not an expiry/status task or a comprehensive news benchmark. The publicly released SWM-Bench news archive was inspected but is **not** used in this primary dataset: its upstream volatility sampling and posterior labels would complicate interpretation.
- Cutoffs: 21, 7, 2 days before market close, at 00:00 UTC. Source availability conservatively next UTC day. Lookback 90 days; up to 24 chronologically spaced documents, minimum six. Fixed topical TF-IDF passage extraction; exactly 256 body tokens per item and exactly two items (512 body tokens) per budgeted condition. Metadata length can vary and is reported.
- Phase: past seven-day price range >=0.15 = fluctuation; past seven-day prices all >=0.85 or <=0.15 and range <=0.10 = consensus, both with at least three traded days. Reversal requires a prior observed consensus and >=0.20 departure. Other cases intermediate. These are operational strata, not utility ground truth.
- Phase counts: 57 consensus, 10 fluctuation, 1 reversal, 10 intermediate. Imbalanced coverage is preserved, not relabeled or selectively removed.
- First two chronological meetings (six snapshots) are runtime smoke; remaining **26 meetings (72 snapshots)** main retrospective PoC. No model selection results inspected before dataset freeze.
- Conditions: recency, TF-IDF relevance, seeded random, neutral LLM priority selection, explicitly time-aware LLM selection. Also no-evidence and full-pool diagnostics, plus observed market probability scored offline. Same stateless `gpt-5.6-luna` low forecaster for all conditions; selection rationale is not passed to forecaster. Identical forecast prompts share cached responses.
- Exact inference/response trace logging uses the existing text-only isolated CLI transport. No GPU needed. Outcome labels and phase histories are separate from whitelisted inference cases.
- An explicitly time-aware selection prompt is an intervention, not proof that the model has the ability without instruction. Compare with neutral selection.
- Before forecasting, source-QC removed bibliography paragraphs from candidate passages and normalized historical `price.close`/`volume` versus live `price.close_dollars`/`volume_fp`. This restored three September-2026 snapshots. The initial pre-inference freeze is retained under `data/market_information_poc_v1/preinference_revisions/`; no performance-based data changes.
- Descriptive target-pattern threshold before main scoring: temporal selector retains an older item than both recency items, with a paired Brier improvement of at least 0.02. Count adverse differences <= -0.02 and small/tied cases too. This identifies examples, not a statistical superiority claim.
- Aggregate uncertainty: paired event bootstrap plus a circular three-meeting block sensitivity interval, since monetary-policy meetings are serially dependent. Neither establishes cross-domain generality with 26 main events.
- Before main inference: fix one anchor per event, the most recent prior statement appearing at every cutoff (else most recent common item). Compare two latest non-anchor records with the same two records plus anchor at each cutoff. These 512 vs 768 token conditions are **addition diagnostics, not equal-budget selector comparisons**. The anchor is not selected by outcome or observed model error. Manifest `anchor_manifest.json`.
- Smoke found one valid probability with an unsupported citation ID in the full-pool condition. Preserve the output and its Brier score, flag citation invalidity separately, and do not regenerate. Apply this handling to every condition before main inference.
- Runtime format recovery: one response completed its numeric probability and citation list but did not close the final reason string. Read only the exact completed JSON-prefix fields, keep the original response and a `truncated_reason_literal_prefix` flag. Do not infer a value from prose or regenerate. Apply the same syntactic recovery to all conditions; incomplete probabilities/ID arrays remain errors.

- Runtime selector failures: if returned IDs are outside the frozen candidate pool, retain the response as a task failure with no forecast score; do not repair IDs or regenerate. Continue the remaining independent conditions. Report attempt coverage separately from successful scores and give aggregate score bounds for the unscored failure.
