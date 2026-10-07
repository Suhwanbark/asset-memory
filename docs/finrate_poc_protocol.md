# Fin-RATE Luna PoC protocol

## Current state: stopped by user; v8 incomplete

The latest user instruction is to stop experiments and explain the task/method. All five v8 workers and their child process groups were stopped and verified absent. `active_run.json` and `v8/user_stop.json` disable automatic restart. Do not launch experiments or model calls until the user resumes. Prior continuous-running authorization is superseded.

Exact Luna low, cutoff filter waived (`exploratory_exposure_uncontrolled`), CPU only, 10GB disk stop. Fixed FIX/LW/NRT corpus1101sources, DEV20/evaluation10 unchanged. Heldout unopened/unreleased. Latest implemented method: v5 memory + v8 common-source-plan/core augmentation. Shared plan/review costs two calls per question, each condition answer one; actual shared cost and standalone attribution are separate. Nine new tests and five-call synthetic Luna smoke passed before launch. Sixty unchanged v7 static outputs/grades/source checks were reused verbatim; v8 comparison stopped incomplete.

| v7 condition | Strict correct/20 | Mean key-point fraction |
| --- | ---: | ---: |
| Naive | 0 known;1 unassessed | 8.5–13.5% |
| Embedding | 2 | 21% |
| Hybrid | 1 | 21% |
| Raw agent | 5 | 46.75% |
| Ours | 3 | 38.75% |
| History disabled | 4 | 44.5% |

All120 predictions completed without generation failure. One naive judge key-point-length mismatch remains UNASSESSED. All source assessments finished. Ours used memory20/20 and exposed history edges10/20, but no memory superiority is established. Original-gold and source-grounding remain separate; reference quality flags do not change questions/scores. See v7/paired_development_analysis.json and development_review.json.

Full inference audit checked1591 exact source spans,240 exact-Luna online requests, valid citations, unchanged banks and actual request evidence/text budgets. The v2 audit corrects an artifact reserialization token-count issue without altering predictions or experimental code. Native tool executions0;45 logged transport fallback notices retained. Disk group-kill behavior also passed a disposable subprocess test without touching real experiment processes or filling disk.

The source-only immutable v5 banks remain:7583facts/3398links;1054successful build invocations, input19,087,234/output1,213,708 plus one unknown-usage capacity failure. Cached quarter replays prove continuity, not unseen-quarter QA or compression. Semantic relation audits remain applicable.

The v8 procedure shares a fresh source-only plan/review across three agent controls, preserves up to4800tokens of identical original-source evidence, and adds passages through raw retrieval, direct facts, or fact-history traversal within6000total. Derived fact/edge claims are routing hints only; final answers receive exact original passages. The complete implementation and separate v8 evaluation harness were frozen before launch. `v8/development_plan.json` is an earlier prototype-stage plan, superseded for runtime status by the user stop record. No general KG baseline or streaming new-quarter QA evaluation has been performed.

The material below is historical: pending/running statements there describe earlier checkpoints. Current runtime truth is `context.md`, `active_run.json`, and each worker status. Full prior protocol archived in `docs/context_archive/protocol_before_v6_completion_review.md`.

### Historical v3/v2 setup

**Active online version: v3.** The memory builder remains frozen v2. A verified tool-contract bug in v2/old raw-agent planning sent regex patterns to literal substring search; the old outputs are retained as diagnostics. New `finrate_agent_v3.py` applies the same corrected source tools, regex/whitespace rules, structured actions, date/asset validation and bounded feedback to memory-on and memory-off conditions. Its 5 tests passed. Replay of all 46 old grep calls recovered results for 31 formerly empty searches; this is tool validation, not a new answer-accuracy score. All 60 static predictions and their grades are reused verbatim.

The v3 raw-agent worker PID864792 runs the same 20 development questions. The v3 memory-agent worker PID864793 waits for all three completed v2 banks, audits their materialized histories, then runs development inference, original-gold grading and source-only grounding assessment. No evaluation inputs are loaded. Each run has a separate manifest/code snapshot/status and the shared active modules must not be edited in place. The original v2 worker may also finish its old online path; those outputs do not substitute for the corrected v3 comparison.

The user authorized implementation and background execution with `gpt-5.6-luna`, CPU-only embeddings, no outcome-based question selection, and stopping our worker if free disk is below 10,000,000,000 bytes. The durable method requirement is **memory per asset tracked across quarters and years**. Atomic facts and the exact consolidation pipeline are changeable development choices.

The official [Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna) gives a 2026-02-16 knowledge cutoff. The pinned public Fin-RATE release has 15,311 chunks, dated 2020-04-24 through 2025-06-09. All 7,500 questions reference only pre-cutoff sources. There are **zero date-eligible candidates** under the originally requested restriction; the complete audit is retained.

The user subsequently explicitly authorized **original Fin-RATE with Luna and no cutoff filter**: “우리도 일단 luna로 기존처럼 해줘”. The active policy is `exploratory_exposure_uncontrolled`. This is a retrospective method comparison and does not rule out pretraining exposure. The question-selection rule, CPU restriction and disk-stop threshold remain in force.

Frozen toy assets: FIX, LW, NRT, selected by a predeclared hash seed. All 1,101 released source chunks for those assets are shared by the methods. Twenty development and ten evaluation LT questions are frozen; source-overlap components stay in the same split. Background development worker **PID757469 runs v2**, builds the asset memories and completes and grades the development grid, leaving evaluation untouched. Its log is `result/finrate_luna_v1/development_v2.log`.

All 60 v1 static baseline responses are reused verbatim in v2, including one embedding JSON-format failure; 59 parsed successfully. Provenance and file hashes are in `v2/baseline_reuse.json`. The v1 worker was stopped after that grid and its code and outputs preserved. v2 fixes whitespace-only quote alignment, adds extraction/link/judge output schemas, and clarifies the difference between an observed action and potential risk. Baseline prompts and responses are unchanged. A source-only smoke on doc_006125 recovered three formerly rejected quotations without regeneration; a fresh structured extraction accepted 12 observations with no structural rejection. This does not establish their semantic correctness or comparative benchmark performance.

## Evidence and artifacts

- Data snapshot: `GGLabYale/Fin-RATE`, revision `284b1b4f6cd1e58574f1f1d17c29b35c6aa72cce`.
- `data/finrate_public_v1/source_manifest.json`: source URLs, byte sizes and SHA-256 checksums.
- `result/finrate_luna_v1/cutoff_audit.json`: complete audit summary.
- `result/finrate_luna_v1/question_audit.json`: every question and exclusion reason.
- `result/finrate_luna_v1/document_metadata.json`: all parsed source dates; zero unknown dates.
- `result/finrate_luna_v1/status.json`: worker status; `launch.json` and `worker.log` retain the completed launch.
- `result/finrate_luna_v1/embedding_smoke.json`: real CPU-only MiniLM invocation on two implementation-test strings, **not benchmark performance**.
- `data/finrate_luna_protocol_v1.json`: explicit model, data, cohort, compute and cutoff policy.

## Method and comparisons

Four conditions: naive LLM (no evidence), embedding, hybrid, agentic asset memory. Same requested Luna model and reasoning setting, same original sources, same question set, same 6,000-token final evidence cap. Embedding and hybrid use identical exact source passages. CPU MiniLM is pinned to an already-cached revision; hybrid uses equal-weight reciprocal-rank fusion. This is a documented baseline inspired by Fin-RATE, **not a claim to exactly reproduce its inconsistent hybrid descriptions**.

Agentic memory v1: ingest sources in publication order; extract atomic observations with subject, comparable topic, predicate/value, reporting period, modality, original quote and source; locate earlier observations; add explicit continuing/updating/conflicting links; write immutable quarterly checkpoints. Old facts are not deleted merely because newer facts arrive. Filing time and reporting period are separate. Missing mentions do not establish termination. A quotation match is a structural provenance check, not a proof of semantic correctness.

The agent has typed `grep`, history and source-reading operations, plus controlled add/link operations during memory construction. It never gets arbitrary shell writes. Evaluation opens shared asset memories read-only and records query-specific investigations separately, avoiding cross-question contamination. All proposed/accepted/rejected edits and LLM requests are retained. Facts and links are content-addressed; the event log is hash-chained.

The extraction granularity, lookup and consolidation strategy may change in development versions. Per-asset identity, time history and source provenance remain required. Additional agent calls, reasoning usage, index construction and memory construction costs must be reported separately; equal final context does not mean equal actual compute.

The independent companion `method/finrate_memory_incremental.py` implements later-quarter updates from a completed asset snapshot. It copies the parent read-only, processes only new sources, keeps existing fact IDs/history/checkpoints, and publishes a new immutable snapshot with parent hashes. Same-quarter late arrivals require a separate rebuild branch and are rejected by this append path. Four continuity/failure-isolation tests passed. A real FIX April-to-July source-only smoke reused the April extraction and made two new Luna calls for July extraction/linking, producing 20 observations across the two sources with the parent unchanged. This validates incremental execution, not its benchmark benefit; it is not injected into the already-running v2 comparison.

The static execution audit includes all 60 calls and their provider usage, including the failed answer. Average provider input tokens are 8,796.7 / 14,709 / 14,720.65 for naive / embedding / hybrid, while measured prompt text averages 162 / 6,084.2 / 6,086.35. CLI system overhead is substantial. Report actual usage separately from the local 24,000-text-token ceiling; do not claim equal total compute. Cached input is already included in provider input totals.

The development-only raw-agent control in `method/finrate_dev_controls.py` removes memory tools while retaining the initial hybrid pool, source tools, final answer prompt, adaptive 4-step/24,000-text-token ceiling and 6,000-token final evidence cap. Planner wording is adjusted to describe only available tools. This controls for much of the extra online investigation; it is not an exact token-matched intervention. The runner first grades all static baseline answers with a prompt tested byte-identical to v2's blind grader, then runs and grades the raw agent on all 20 fixed development questions. Cache reuse preserves each judgement for the eventual main report. Separate status, manifest, code snapshot and reports are under `controls/raw_agent_v1/`; held-out inputs are not loaded.

The full three-asset build contains 1,520 source windows, with at most one extraction and one link-batch call each (3,040 calls before QA). Early call latency averaged about 20 seconds; total construction may take many hours. This is a cost of the current method and must accompany any quality gain, rather than being hidden as free preprocessing.

The static development grading is complete: strict CORRECT is **0/20 naive, 1/20 embedding, 1/20 hybrid**; mean key-point coverage is 8.75%, 18.25%, and 24.75%. All fixed questions and the one format failure remain included. The source-ID retrieval diagnostic found all linked gold sources for 6/20 questions in each retrieval condition; average source-ID recall was 0.38 and 0.405. This is not answer-evidence sufficiency and is affected by the documented gold/source-year issues. Agentic and raw-agent comparison results are pending.

Judge reasoning itself needs caution: for q001837, both retrieval answers said the relevant Item 3 evidence was absent, which matches the inspected supplied excerpts, but the judge's explanation treated reference-answer information as if it had been supplied to the candidate. The answers still fail full-answer completeness, but the error-type/citation rationale is unreliable in this case. Preserve these original-gold scores and use an independent source-only grounding assessment before substantive claims. No gold rewrite or question exclusion has been made.

The materialized-bank audit independently checks sources/facts/links against events, content identities, quote spans and checkpoint prefix counts. A saved live-build snapshot of 26 sources, 289 facts and 37 links passed, as did the two-source incremental smoke. A fixed-hash sample of eight early FIX facts was inspected by Codex against source context: all eight values were supported in that context, but three table quotations omitted the unit/period headers needed to interpret the isolated quote. Neither result is independent human annotation or a whole-bank semantic accuracy estimate.

The old raw-agent run completed with 2/20 CORRECT and 26.75% mean key-point coverage, using 82 response calls. Those numbers are qualified by the search-contract defect: 33 of 46 grep calls returned empty results, mostly because regex alternatives were interpreted literally. The new comparison reruns all 20 questions in both online conditions under the same corrected contract. Original outputs, failures and diagnostics remain available.

Independent v2 asset prefetch workers (LW PID831384; NRT PID831385) call the unchanged builder in separate staging trees and atomically publish completed asset directories for the serial consumer. If the consumer starts that asset first, prefetch stops rather than overwriting its work. Four tests cover publication, incomplete snapshots, existing work and a concurrent consumer start. This changes scheduling only, not data, prompts, model settings or chronological update order within an asset.

`finrate_source_grading.py` separately checks source grounding without reference answers/key points. It verifies original quote offsets, omits derived memory values as source truth and uses the same bounded additional source context for all conditions. Additional verification context is distinguished from evidence the candidate actually received. Four input-integrity tests passed. No-evidence responses are marked not assessable rather than declared false. The judge's abstention sublabels remain diagnostic; original-gold correctness scores are not replaced.

## Selection and evaluation

The active policy is `exploratory_exposure_uncontrolled`: the user explicitly authorized original Fin-RATE with `gpt-5.6-luna` and no cutoff filter. The earlier strict post-cutoff audit found no eligible questions; that earlier execution gate was superseded by the user's decision. This run cannot establish freedom from pretraining exposure. The same Luna model is used across conditions.

The cohort is already frozen: FIX/LW/NRT, 20 development and 10 evaluation LT questions, selected using the fixed hash seed `finrate-toy-20261004`. Overlapping gold-source groups stay in the same split. All 1,101 public source chunks for the selected assets are available to the methods; the corpus is not restricted to gold evidence. Answers, key points and gold source IDs remain in separate scoring artifacts. Source/year and reference-quality concerns discovered during development are recorded as limitations without dropping questions, changing gold or changing denominators. Held-out question and answer contents remain unopened until the reviewed release is frozen.

Improve only on development. Freeze code/config before evaluation and lock the evaluation set to one method release. Keep failures in the denominator and retain all negative results. Never rerun or replace held-out questions until a desired ranking appears. Single-Luna blinded rubric grading is a diagnostic and does not reproduce Fin-RATE's official three-judge evaluation. Source-grounded human spot checks are still needed before substantive claims.

## Implementation status and commands

The active run uses v5 offline memory construction and v6 online comparison. The primary conditions are naive LLM, embedding, hybrid and agentic memory; a raw-source agent and a history-edge-removal ablation provide additional controls. Static and raw-agent development outputs are complete. The memory conditions wait for all three asset banks; NRT and LW are complete and FIX is still building. Main comparative performance remains **unverified until those conditions finish**. Previously recorded integrity, CPU embedding and transport tests cover their stated components, not comparative QA quality.

From `/home/hob/workspace/alignment/replicate`:

```bash
/home/hob/anaconda3/envs/su/bin/python -u method/finrate_observe.py --seconds 0
/home/hob/anaconda3/envs/su/bin/python -u method/finrate_observe.py --seconds 600
```

These are read-only observation commands. They verify the actual registered process argv, distinguish completed workers from missing active workers, record status and progress, and stop observation when the memory comparison begins or a terminal problem appears. They do not launch, restart or edit any experiment, and do not read question/gold contents. Their snapshots are saved under `result/finrate_luna_v1/v6/monitoring/`. Current worker identities and their authoritative status paths are in `result/finrate_luna_v1/active_run.json`; older launch commands and PID records are historical and must not be used to start duplicate runs.

The current launch modules are `finrate_v5_development` for asset construction and `finrate_v6_development` for the two memory conditions. `finrate_completion_audits` performs completed-bank cost accounting, cached quarter replay and the full paired development analysis without new model calls. The prepared `finrate_final_evaluation` harness separately freezes an actually reviewed method and runs a single locked held-out release. It has not been frozen or run. Use versioned artifacts and preserve every failed prediction and negative result; never edit running frozen code in place.

Background workers use their own process groups with CPU settings applied. The watchdog checks free disk every two seconds and terminates only our process group below 10,000,000,000 free bytes. It never removes other work or stops another user's GPU process. A failed no-answer capacity request was preserved and retried once with the same model and frozen inputs; successful responses and scored failures are not regenerated. Observation timeouts are not evidence that an experiment stopped.


## 2026-10-05 — LW 전체 구축 완료 검토

LW 전체 은행은 307 sources / 2314 facts / 766 links이며, 2024-Q3 부모에 Q4 출처 16개만 캐시 재생해 전체 논리 상태와 event hash가 일치했다. 이는 저장·증분 상태 연속성 검증이고, 신규 미래 자료의 성능 검증은 아니다. 원본 구축은 성공 320호출, provider input 5,649,280 / output 342,974 tokens이며 capacity 실패 1건의 사용량은 unknown이다. 직렬화된 fact 크기는 원문 토큰의 2.452배다.

LW의 `contradicts` 1개는 2020년 발행 기대와 2022년 신규 옵션 승인 사이의 조건부 계획 변화로 해석할 문맥은 있다. 그러나 선택 new endpoint는 vesting schedule이고 원래 Stock Plan과의 범위 동일성은 미확인이다. `v5/audits/LW_contradiction_review.json`에 승인일·미래 효력일과 endpoint의 차이를 보존했다. 이는 NRT의 지원되지 않는 기간 모순과 동일한 확정 오류로 합산하지 않는다. 관계 원본과 모든 질문·점수는 유지했다.
