# Temporal evidence source audit — clean v3 handoff

Date: 2026-09-28. Scope: the **12 corporate-source archives** and the structural correction of the Amazon termination article. This audit does not independently certify the separate 24-event Federal Reserve corpus. Detailed original findings: [corporate source audit](temporal_evidence_corporate_source_audit.md).

## Main-set decision

- Original proposal: 24 Federal Reserve events + 6 corporate events = 30 cases.
- Clean main set: **24 Federal Reserve events + 5 corporate events = 29 cases** (111 probes if original four-Fed/three-corporate probe schemas are unchanged).
- Keep Amazon–iRobot with the newly archived, complete article-body copy. Rebuild summaries/retrieval inputs and rerun every condition for this case; never reuse its old derived artifacts as if already clean.
- Exclude the complete VanEck spot-bitcoin case because the launch PDF's exact pre-cutoff availability is unverified. Do not seek an answer-preserving replacement or move its cutoff merely to retain 30 cases.
- Preserve original 30-case inputs/results for provenance. The source audit occurred after those model outputs existed; report the clean rerun as a source-QC correction, not a prospectively uncontaminated original run.

## Amazon correction completed

| Item | Recorded value |
| --- | --- |
| Original document | `corp_irbt_20240129` |
| New document | `corp_irbt_20240129_clean_v3` |
| Removal selector | `.articles-links` within the original article container |
| Removed material | One separately structured `Trending news and stories` module with four unrelated headlines, including 2026 retail headlines |
| Text length | 2,605 → 2,226 characters; 379 characters removed |
| Preserved content | All four original article-body blocks, including both companies' statements and the termination-fee sentence |
| Raw HTML | Original archived bytes copied unchanged; no network refetch |
| Clean-text SHA256 | `3d581af8e19229e63d2c6e5cc0cf10a28f069db1af6c61090124558d630c055b` |

Reproduction: `python method/temporal_evidence_corporate_clean_copy.py`. The script uses exclusive writes and refuses to overwrite existing correction files. It verifies the one-module/four-article-block DOM structure, complete preservation of article characters, old/new hashes, and unchanged original files. It does not modify cases, gold, or results and does not invoke a model.

Patch manifest: `data/temporal_evidence_poc_v1/corporate_clean_case_patch.json`. The `new_doc_metadata` field can replace the original document metadata in a new dataset copy; any gold evidence source-ID references should likewise be mapped to the new document ID without changing claims, labels, or explanations. The manifest records exact removed text and its hash, removed links, original raw/text hashes, DOM evidence, and the four preserved block hashes.

This correction removes a website recommendation component by its publisher-defined structural role. It does **not** select only paragraphs that support the intended answer. The copied raw HTML intentionally still contains the module as evidence; model input must use the corrected `text_path`, not reparse that HTML with the old unfiltered selector.

## Remaining eleven original documents

| Document | Finding and handling |
| --- | --- |
| `corp_jblu_20230307` | No confirmed later factual addition; 2040 appears as a future corporate commitment. Retain complete release. |
| `corp_jblu_20240304` | No confirmed later factual addition; later 2024 dates are announced plans/deadlines, not subsequent outcomes. Retain. |
| `corp_adbe_20231128` | Extracted article has no confirmed later factual addition. Raw GOV.UK internal update timestamp is 2026, while public update/structured modification dates remain November 2023. Retain with historical-snapshot limitation. |
| `corp_adbe_20231218` | No confirmed later factual addition. A trailing content-card block duplicates the same 2023 article's information. Retain. |
| `corp_irbt_20230725` | No confirmed post-January-2024-cutoff fact. Metadata indicates July 28, 2023 modification after July 25 publication. Retain for this case, but do not claim exact July-25 snapshot verification. |
| `corp_atvi_20230426` | No confirmed later factual addition. A 2026 market-size number is explicitly a forecast. GOV.UK internal/public date distinction applies. Retain. |
| `corp_atvi_20231013` | No confirmed later factual addition. GOV.UK internal/public date distinction applies. Retain; keep original-blocked versus narrower-approved scope distinction. |
| `corp_kr_20240226` | No confirmed later factual addition. 2025/2030 references are prospective corporate targets. Retain issuer claims as issuer claims, not regulator findings. |
| `corp_aci_20241211` | No confirmed later factual addition. January 2025 earnings call is described as forthcoming. Retain entire release, including the non-merger capital-return/outlook sections. |
| `corp_vaneck_20231207` | Full 15-prediction article has no confirmed realized post-cutoff addition. Hypothetical future-past election narration is not itself evidence of a later update. Archive retained, but its paired case is excluded because of the launch PDF below. |
| `corp_vaneck_20240111` | **Excluded from clean main set:** January-11-dated launch release exists in a PDF created April 22 and modified April 23, 2024, after the January 12 cutoff. Five-page text contains no identified explicit post-cutoff event, but exact artifact availability is not established. |

## Interpretation limits

1. A publication date earlier than cutoff is necessary but not sufficient: dynamic article modules and later file regeneration can defeat that check.
2. A future year is not automatically contamination. Distinguish a prospective forecast/goal from a statement asserting that a later event has occurred.
3. Current-page acquisition is not a contemporaneous historical snapshot. Apart from the identified issues, absence of detected later facts does not prove that every retained article was unchanged since publication.
4. Correct answers on the supplied text do not establish source integrity, a memory failure, or prediction-market contract alignment. The corporate cases remain source-backed event-status diagnostics with unverified market-contract linkage.
