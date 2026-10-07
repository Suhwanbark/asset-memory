# Temporal Evidence POC — v1 protocol (2026-09-28)

## Objective and boundary

Test whether a reader distinguishes a historically accurate record from a claim that can still be used as a current premise. This is a necessary, limited component of time-conditioned information utility, not a complete measure of predictive information value or a demonstrated financial-memory research gap.

The user selected a real prediction-market/event setting and requested more than ten events. Target: 30 independently dated financial/economic events; count multiple questions or evaluation times within one event separately from event count. Actual available counts, exclusions and source restrictions must be disclosed. No trading or remote writes. Polymarket API returned HTTP 451; do not bypass it. Publicly released research datasets and independently public official announcements may be used, with source changes disclosed.

## Selection, before inference

- Select by official source availability, timestamp, unambiguous statement scope and event category, never by model accuracy.
- Preserve original public documents, their URLs, publication times/dates, retrieval timestamps and hashes.
- Label only statements whose status is supported by documents available at the evaluation cutoff. An eventual market resolution is not a label for the value of earlier evidence.
- A closed market flag does not establish final settlement. No made-up market questions are to be reported as actual traded markets.
- Include changed and unchanged official states. Do not manufacture a balanced failure rate by excluding successful cases.
- Closely related monetary-policy events and overlapping source windows are correlated; report event/family counts and do not treat all probes as independent.

## Task

The reader receives an as-of date, dated source history or its memory representation, and neutrally phrased factual premises that may be used in a current event briefing. Each probe has an explicit `current` or `historical` scope. Classify `supported`, `contradicted`, or `insufficient` IN THAT SCOPE, with short source-grounded evidence. A historically true statement can coexist with a changed current state.

The task does not assign a causal importance score to a news item, infer its effect from market-price movement, or reward reproducing an original agent's action. No add/remove-information intervention (explicitly excluded by the user). Forecast scoring is not run retrospectively as leakage-free forecasting: model parametric knowledge of resolved events cannot be excluded merely by hiding later documents.

## Conditions

1. Full dated source history: basic evidence-reading upper comparison, not a predetermined high-score gate.
2. Generic concise research memory formed from the same history without evaluation probes or labels. This is a simple baseline, not a branded memory framework.
3. Naive retrieval of dated source chunks: a transparent information-selection baseline, with selected chunks and actual token count retained.

Start with these interpretable conditions; native memory methods may follow only if the evidence-validity task itself is sound. The latest update belongs to the stored history in each condition, not a full current document handed only to the final reader. No need to synthetically lengthen documents for the first POC. Store size, selected length and final prompt length must be reported.

## Execution and analysis

- Reuse the authenticated Codex CLI transport, GPT-5.6 Luna, reasoning low, isolated stateless invocations, no browsing/tools in final inference. Record actual model/usage/prompts/responses and nonzero exits. Check official CLI guidance for execution flags.
- Prompt and source hashes prevent reuse across changed inputs. Retries repair runtime failures only; do not re-answer wrong judgments selectively.
- Score structured labels deterministically against separate evidence-backed gold. Malformed output is an execution/format issue, not silent content failure. Independently review mismatches and an appropriate sample of correct judgments.
- Main error types: obsolete premise reused; still-valid premise discarded; past truth confused with current applicability; insufficient evidence overclaimed. Separately inspect whether supporting source survived storage/retrieval.
- Report null results, ambiguity exclusions, complete and partial denominators. Do not claim a memory-specific effect from errors also made with full source history, nor a causal financial decision effect from this diagnostic alone.

## Iteration rule

Fix data/implementation errors in versioned artifacts, rerun affected full conditions and keep previous results. Do not alter sources, gold or selected events to force a failure. If the task is near-perfect, conclude that this simple form of the proposed problem has not been reproduced, then inspect naturally occurring, independently selected longer histories rather than inventing contradictions.

## Frozen first diagnostic stratum

- 24 FOMC decisions: every regularly scheduled 2023-2025 decision, with the previous three statements, current statement and complete current-meeting minutes. Cutoff is end of the official minutes-publication day, NOT the meeting day. Inputs are naturally long documents, not generated noise.
- 6 corporate events: JetBlue/Spirit, Adobe/Figma, Amazon/iRobot, Microsoft/Activision, Kroger/Albertsons and VanEck's US spot-bitcoin ETF launch. Full publisher sources are archived. All are explicitly official-event diagnostics, not 30 verified prediction markets.
- 96 FOMC probes plus 18 corporate probes = 114 probes over 30 events. Template probes (history, 2-percent objective, uncertain future rate) are controls, not 114 independent demonstrations. Main errors are old current premises accepted and still-valid current premises rejected.
- ForecastBench archive connects 6 FOMC cutoffs to 9 later binary market contracts. These are linkage candidates; no future resolution enters inference and no forecast is scored in this diagnostic. Remaining official events are not represented as linked markets.

## v2 scope clarification, preserving v1 (before v2 inference)

v1 outputs revealed an open-world ambiguity: a summary can explicitly retain a January 29 policy decision while the reader declines to say it still holds at the February 19 minutes-publication cutoff. This is NOT evidence of an obsolete premise being reused and may be reasonable abstention under the original wording.

v2 clarifies uniformly for every condition: CURRENT means the latest applicable state established by the supplied records, not guaranteed exhaustive knowledge of every world event through the cutoff. No same-day reannouncement is required for an explicitly continuing status. Future outcomes still require actual evidence. Exact sources, questions, gold, RAG selections and v1-generated summaries are held fixed; only the general scope instruction changes. Run all cases/conditions, not just previous errors. Both versions are retained and reported; this is protocol clarification, not an improved memory method or new independent validation set.

## Post-run source QC and v3 reviewed cohort

Source inspection found a separate dynamic recommended-news module inside the extracted Amazon 2024 article container. Remove only that DOM module, preserve and hash-check all four article text blocks, retain the original archive, then form a new summary and rerun all three conditions for the corrected case with the unchanged v2 task. Give the corrected source a new ID and update gold citation IDs, not claim meanings or labels.

The VanEck launch PDF has creation/modification metadata later than its historical cutoff. This does not prove its article contains later facts, but a contemporaneous artifact cannot be established. Exclude the entire case from the reviewed main cohort regardless of response correctness (all original conditions answered it correctly). Keep the original case and results for audit.

The reviewed cohort comprises 29 events and 111 probes: 28 unchanged v2 cases plus the corrected Amazon case. Save exact response references and hashes. Published dates and body inspection are not a guarantee of historical web-snapshot identity; disclose this remaining limitation. This source-QC stage is not another independent holdout evaluation or a completed prediction-market information-utility experiment.
