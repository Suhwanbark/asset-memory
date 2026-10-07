# Time-aware evidence priority — feasibility protocol

2026-09-28. This replaces the previous memory/status diagnostic as the active question.

## Construct

Does an LLM select the more decision-relevant of two truthful, dated evidence items for a specified task and evaluation time? No memory formation, compression, retrieval, persona, trading or source-agent action matching. The initial operationalization is **source-grounded temporal applicability and decision relevance**, not proven causal economic value of information.

## Justification and exclusions, before inference

- Every evidence item must be traceable to an official public source, with a publication date, excerpt, hash and a distinction between report date and affected period.
- Both items must be available before the question's cutoff. Neither item is fabricated or a knowingly false statement.
- The question specifies an analytical decision and horizon. Gold priority is warranted by an explicit scope/period, a binding ongoing condition, a resolved uncertainty, or the information explicitly needed for that decision. Write this warrant before model inference.
- Do not infer informational causality from a price move. Do not call all ended events irrelevant; their effects may persist.
- Do not force a ranking where both inputs are needed and neither defensibly has priority. Log such exclusions. Source-grounded researcher labels remain provisional until independent human review; no fabricated human agreement claims.
- Use harder older-wins examples as well as newer-wins examples. Balance displayed A/B positions by evaluating both orders. Do not sample based on model mistakes.
- Where a decision/horizon actually changes the relevant evidence, create paired queries over the exact same evidence. Separate task/horizon-change pairs from genuine clock-only pairs. Do not pretend all pairs establish a causal effect of time alone.
- Include simple reading/format controls to separate a small model's general task failure from temporal prioritization failure.
- Real documents do not make the queries naturally occurring: researcher-authored questions/excerpts and purposive selection must be disclosed.

## Models and resources

Qwen3-1.7B in non-thinking mode and SmolLM2-1.7B-Instruct, frozen official model revisions. English prompts to avoid conflating Korean capability. Direct A/B choice plus short public explanation. Same prompts, complete source snippets, no hidden gold in input. Greedy decoding for reproducibility; log unsupported/invalid output separately and retain label likelihoods when possible.

Only ONE verified-idle GPU, UUID pinned with CUDA_VISIBLE_DEVICES. Models run sequentially on that GPU, never one GPU each. Never stop unrelated jobs. A subprocess monitor polls free disk every 2 seconds; below 20 GiB (stricter than user 20 GB) terminates only this task's child process group. Check projected space before model downloads.

## Reporting

Report events/source families separately from authored questions, A/B order variants and model responses. Metrics: gold-choice accuracy, older-wins accuracy, paired-query correctness, order invariance, invalid responses, comparison to always-latest and simple lexical relevance. Errors require inspecting actual response and gold defensibility. A null result is valid. This small-model POC cannot establish that all LLMs lack temporal awareness.

## Frozen sample and diagnostic update

- Frozen before inference: 50 authored questions / 16 source clusters. Main set: 38 questions from 8 clusters (5 funding programmes and 3 stock-action histories). Controls: 12 questions from 8 further clusters (4 merger histories and 4 FOMC releases).
- Pure date-change switching comparisons: only 3 pairs, using identical facts/questions at two dates. Additional date-change stable comparisons: 8 pairs. Task-change comparisons are reported separately and are not evidence of a clock-only effect.
- First run (v1): answer-first with a 110-token ceiling. Qwen selected A in 98/100 presentations, SmolLM2 in 100/100. Both scored only 4/8 on elementary reading controls; some explanations contradicted the initial label. These scores are not clean evidence of time-priority failure.
- Before running v2: retain the same frozen cases, facts, gold, dates, both orders, models, greedy decoding and GPU. Uniformly ask for one short explanatory sentence followed by a final `Answer: A/B` line, with a 160-token ceiling so the final label has room. Re-evaluate all 50 questions for both models, not just mistakes. Save v2 independently; v1 remains accessible. The two changed factors (answer position and output ceiling) make this a format diagnostic, not an isolated causal test of answer order.
- v2 is a development diagnostic motivated by v1, not a fresh holdout. Labels are parsed from the requested final answer line; no LLM judge or semantic reinterpretation of wrong labels is used.
- Scoring audit after first v2 model: some outputs end with an unambiguous `Answer: A/B` on the same line as the explanation. Normalize whitespace/newline only and accept that terminal label; keep raw strict-newline parsing in each response and report strict vs normalized scores. This is not a model re-run and does not infer a label from explanatory prose.
- Before v3: v2 still showed large order effects and imperfect elementary reading (Qwen 6/8, SmolLM2 4/8). Run one final uniform interface diagnostic with a minimal generic multiple-choice system instruction and letter-only output (8-token ceiling). No demonstrations, extra factual hints, date reasoning instructions, or changes to evidence/question/gold. All 100 presentations/model are rerun and retained separately. Do not treat low task accuracy as temporal-specific unless simple controls and order behavior support that interpretation. Stop this bounded feasibility batch after v3, including a null/inconclusive interpretation if required.
