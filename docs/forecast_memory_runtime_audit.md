# Forecast Memory smoke — independent runtime / source audit

Audit date: 2026-09-28. Read-only inspection of the runner, official SDK, recorded source tool calls, stored memories, and Intel follow-up outputs. No extra model calls, source changes, or reruns were performed by this auditor.

## Scope and overall finding

The three source sessions actually read the supplied releases. The runner keeps future releases and the evaluator-only gold file out of memory formation. The Intel case does **not** demonstrate the target forecast-to-realized-fact failure. The current experiment is a short, one-session ingestion smoke, not evidence about long-term incremental memory accumulation.

## 1. Source acquisition / reading QC

The complete archived release is supplied as `source.txt` in an isolated temporary working directory. I compared each recorded `aggregated_output` against the corresponding exact line ranges of the dated source packet.

| Case | Source packet lines | Verified tool-output coverage | Source tool calls |
|---|---:|---:|---:|
| NVIDIA | 376 | 376 / 376 | 2 |
| AMD | 361 | 361 / 361 | 2 |
| Intel | 674 | 674 / 674 | 5 |

All recorded commands are read-only `wc` / `sed` operations on `source.txt`. No network access, reads of the current release, evaluator gold, or other project files appear in these source call traces. No truncation marker occurs in the recorded outputs. Intel reread overlapping bounded sections; the union covers the full file. This verifies output availability in the recorded trace, not the model's attention to every line.

The three public source notes preserve the target revenue guidance as guidance:

- NVIDIA: Q2 FY2023 revenue $8.10 billion, plus or minus 2%.
- AMD: Q3 2022 revenue approximately $6.7 billion, plus or minus $200 million.
- Intel: Q2 2024 revenue $12.5–13.5 billion.

Files: `result/forecast_memory_smoke_v1/<case>/source/response.json` and `events.jsonl`; source text paths and hashes are in the dataset manifest.

## 2. Input / inference boundaries

- `source()` reads only the past release and makes a research note without the future question or gold.
- Memory formation receives the past request, a dated full release normalized as an assistant observation, and the actual source note. It is not literal replay of every shell wrapper or repeated tool-output chunk; the observation content is preserved once.
- The future date and neutral update question enter retrieval / final inference only after formation.
- Each CLI call uses a fresh `codex exec --ephemeral` session without resume/fork, an isolated temporary directory, ignored user configuration, and zero project-document allowance. The runner disallows tool calls in final evaluation and SDK extraction calls.
- The full-history and memory evaluations use the same current release and question. No Memory can legitimately state that it lacks the previous review.
- The source and current material are actual archived company releases, not rewritten failure-inducing evidence.

## 3. Official SDK path and implementation qualifications

The installed official package is `mem0ai 2.2.1`. The inspected `Memory.add`, `search`, and `get_all` signatures match the runner's use of `agent_id`, scoped `filters`, and `top_k`.

The runner uses native extraction, vector persistence, entity linking, and search. It replaces only the LLM and dense-embedding objects after initialization. The LLM is GPT-5.6-luna / low via Codex CLI; the embedding is pinned all-MiniLM-L6-v2, CPU, 256-token window, attention-mask mean pooling and L2 normalization.

Important qualifications:

1. **The current SDK is an ADD-only additive extraction pipeline.** It should not be described as reproducing an older Mem0 UPDATE/DELETE implementation or the earlier paper protocol.
2. **Transport is an adaptation.** SDK system/user prompt contents are embedded with role tags in a CLI user prompt. This preserves their text but is not identical to native Chat Completions message-role semantics. The CLI also has its own agent harness instructions.
3. **One completed research session is added once.** This smoke does not test repeated cross-session consolidation.
4. **Dates in the SDK prompt conflict with historical replay.** All three recorded extraction requests contain `Observation Date: 2026-09-28` and `Current Date: 2026-09-28`, while the supplied conversation/release dates are 2022 or 2024. The SDK's native builder defaults to execution day; the OSS `timestamp` argument is unsupported. The historical dates remain in the content and metadata. No incorrect 2026/2027 date normalization was found in these three saved stores, but any such error must be separated from the proposed phenomenon and attributed to this replay limitation.
5. **Text-only delivery drops existing metadata.** Publication date and source URL are correctly retained in every saved memory's metadata, but `format_memories()` sends only the memory text. Missing metadata in the final prompt is a delivery choice, not evidence of formation loss. A metadata-delivery diagnostic would be needed if this causes a date/source error.
6. **Top-10 equals the complete store here.** Each case has nine stored memories, and all nine are retrieved. The Intel ID sets match exactly; the global run also reports nine retrieved for each case. Top-10 vs all therefore cannot establish retrieval omission. Differences can reflect presentation order or generation variability.
7. **Response caching is not content-addressed.** Existing `response.json` files are reused without checking a prompt hash. This first fixed run is unaffected, but changed protocols must use a new run directory or explicit cache invalidation; do not edit inputs and silently reuse old responses.

## 4. Intel target adjudication

This is narrow target adjudication, not a claim that every sentence of each generated financial report is error-free.

Target evidence: the April 25 release forecasts Q2 2024 revenue of $12.5–13.5 billion. The August 1 release reports Q2 revenue of $12.8 billion. Thus actual revenue is within the earlier range; this does not mean all profitability guidance was met. The new Q3 revenue range happens also to be $12.5–13.5 billion and must not be mistaken for an explicit comparison with the previous Q2 forecast.

| Condition | Forecast treated as realized fact / wrong target period? | Explicit previous-Q2-revenue comparison coverage |
|---|---|---|
| No Memory | Not observed | No; appropriately says the previous review was not supplied |
| Full History | Not observed | No explicit old Q2 revenue-range versus actual comparison; does compare Q1/Q2 performance and profitability |
| Mem0 Top-10 | Not observed | No explicit old Q2 revenue-range versus actual comparison |
| Mem0 all | Not observed | Qualitative success: says actual $12.8 billion is broadly within prior Q2 outlook; does not restate the old numerical range |

Exact answer excerpts:

- No Memory: `No previous review was supplied, so changes versus the prior note cannot be determined.`
- Full History: `Intel’s Q2 results were weaker than the Q1 trajectory implied.` This is not a statement that the earlier Q2 revenue range was missed. Its subsequent profitability comparisons support the broad deterioration description.
- Mem0 Top-10: `Intel’s Q2 results were weaker than the prior outlook implied and led to a major shift toward cost and liquidity management.` This general statement should not be scored as an explicit revenue-range miss: profits did miss the prior outlook.
- Mem0 all: `Revenue was $12.8 billion, down 1% year over year, broadly within the prior Q2 outlook, but profitability was weaker`.

The stored and retrieved memory correctly says:

> Agent reported Intel's Q2 2024 outlook of $12.5–13.5 billion in revenue, 40.2% GAAP gross margin, 43.5% non-GAAP gross margin, negative $0.05 GAAP EPS, and $0.10 non-GAAP EPS. Management expected full-year 2024 revenue and non-GAAP EPS growth plus roughly 200 basis points of gross-margin improvement.

Therefore, the missing explicit comparison in Top-10 is an answer-coverage issue despite available information, not proof of storage or retrieval loss. Under a strict rubric requiring old numerical range + target period + actual result in the answer, none of the four restates the complete tuple; the qualitative success of Mem0 all should still be reported separately.

## 5. Sierra Forest wording audit

Past source (`intel_2024_q2/past.txt`, line 76):

> the next-generation E-core Intel® Xeon®, code-named Sierra Forest, achieved product release this week, and Intel expects Granite Rapids to be released in the third quarter.

Source note:

> Sierra Forest Xeons launched, with Granite Rapids expected in Q3.

Stored memory:

> Agent reported that Intel introduced the Gaudi 3 AI accelerator, citing projected advantages versus Nvidia's H100; launched Sierra Forest Xeons; expected Granite Rapids in Q3 2024; and expanded AI and edge partnerships and deployments.

The original release describes Sierra Forest's product-release milestone as achieved, not merely forecast. Consequently this is **not a clear forecast-to-fact error**. `Launched` may be less precise than the specific product-release milestone, and the first paraphrase occurs in the Source note before Mem0. Do not count it as a newly introduced Mem0 formation error or claim that a forecast was converted to a realized launch.

## 6. Interpretation boundaries

- No target failure was established in the independently audited Intel case.
- Correct storage plus omitted comparison is different from wrong factualization.
- Metadata omission, the SDK observation clock, Source-note paraphrases, and output ordering must be traced separately before attributing downstream errors to Mem0.
- These three hand-selected companies are a mechanism smoke, not an unbiased estimate of financial-memory error prevalence.
