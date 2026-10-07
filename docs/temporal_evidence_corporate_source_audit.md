# Corporate source temporal-integrity audit

Audit date: 2026-09-28. Scope: the 12 archived documents used in `corporate_cases.json` (six events, three probes each). This is a post-run source audit, not another model evaluation. Existing sources, cases, gold labels, and results were not modified.

## Conclusion

- **One confirmed post-cutoff contamination:** the Amazon–iRobot termination article includes a dynamic trending-news module mentioning 2026. The article's publication-date metadata did not reveal this problem.
- **One unresolved artifact-provenance issue:** the VanEck launch PDF is dated January 11, 2024 in its text, but its PDF creation/modification metadata is April 22–23, 2024, after the January 12 case cutoff. No clearly post-cutoff factual addition was identified in its five-page text; nevertheless, its availability in this exact form at the cutoff is unverified.
- No confirmed post-cutoff factual addition was found in the other ten extracted documents. This is **not** a certification that their present-day bodies are identical to historical snapshots: contemporaneous captures have not been checked.
- For a strict point-in-time result, quarantine **the two affected cases**, not only individual probes. All probes within a case share the same document input. Keep original evaluation artifacts unchanged and clearly identify them as using the unaudited source version.

## Document-by-document findings

| Source ID | Publication date | Audit finding |
| --- | --- | --- |
| `corp_jblu_20230307` | 2023-03-07 | No confirmed future factual contamination. A 2040 net-zero commitment is explicitly a prospective target, not later information. Contacts and generic navigation remain. |
| `corp_jblu_20240304` | 2024-03-04 | No confirmed future factual contamination. References to May 2024 Investor Day and July 2024 outside date are prospective statements in the release. |
| `corp_adbe_20231128` | 2023-11-28 | No confirmed future factual contamination in extracted text. GOV.UK internal update timestamp is 2026; public update and structured `dateModified` remain 2023. See below. |
| `corp_adbe_20231218` | 2023-12-18 | No confirmed future factual contamination. A duplicated content-card metadata block follows the release; its date and description refer to the same 2023 announcement. |
| `corp_irbt_20230725` | 2023-07-25 | No confirmed post-case-cutoff factual contamination. HTML `article:modified_time` is July 28, 2023, three days after publication but before the January 2024 case cutoff. Exact publication-day body is not established. |
| `corp_irbt_20240129` | 2024-01-29 | **Confirmed contamination:** four dynamic trending headlines follow the complete announcement, including 2026 sale headlines. |
| `corp_atvi_20230426` | 2023-04-26 | No confirmed future factual contamination. Market size in 2026 is explicitly a forecast. GOV.UK internal/public timestamp distinction applies. |
| `corp_atvi_20231013` | 2023-10-13 | No confirmed future factual contamination in text. GOV.UK internal/public timestamp distinction applies. |
| `corp_kr_20240226` | 2024-02-26 | No confirmed future factual contamination. References to 2025/2030 are previously stated corporate goals, not claims about realized outcomes. |
| `corp_aci_20241211` | 2024-12-11 | No confirmed future factual contamination. January 2025 earnings call is described as upcoming. Capital-return and fiscal-outlook sections belong to the same release; these were not removed merely because irrelevant to the gold. |
| `corp_vaneck_20231207` | 2023-12-07 | Full 15-prediction article, without the broader page's recommendation content. Future dates and future-past narration occur within explicitly speculative scenarios. No confirmed actual post-cutoff result was identified. |
| `corp_vaneck_20240111` | 2024-01-11 | **Provenance unresolved:** PDF was generated/modified in April 2024. Textual announcement date alone does not establish availability of this artifact by January 12. |

## 1. Confirmed Amazon contamination and a structurally valid correction

Archived source: `data/temporal_evidence_poc_v1/sources/corp_irbt_20240129/source.txt`.

The selected `article.ArticlePage-mainContent` contains a nested `.ArticlePage-articleBody > article`. Its children are four `div.contentContainer.block` nodes containing the complete announcement, followed by a `div.contentContainer` containing `.articles-links`.

The final module starts with `Trending news and stories`. Its links include:

- `/news/retail/amazon-prime-big-deals-day-2026-when-october-6-7?utm_medium=trending_module`
- `/news/retail/prime-big-deal-days-2026-best-deals-amazon?utm_medium=trending_module`

Thus 2026 material is present **inside model-visible extracted text**, despite the source metadata retaining the January 29, 2024 publication and modification timestamps. The headings concern unrelated retail news rather than a different merger outcome; this does not make them acceptable in an as-of-2024 corpus.

A principled new source version can remove the explicitly identified `.articles-links` navigation/recommendation module while retaining all four article-body blocks. This is document-boundary correction, not selection of gold-supporting sentences. Keep the original raw HTML and provenance, archive the corrected full-article text as a new version, rebuild derived summaries/retrieval inputs, and rerun affected conditions. None of those corrections was made during this read-only audit.

## 2. VanEck PDF: body date is not artifact date

Archived PDF: `sources/corp_vaneck_20240111/source.pdf`.

Its five-page text contains the January 11 launch announcement, November 30, 2023 asset figures, a January 8, 2024 S-1 footnote, and risk disclosures. No clearly realized post-January-12 event was identified in this text. However, the PDF's own metadata says:

```text
/CreationDate: D:20240422162146-04'00'
/ModDate: D:20240423152012+05'30'
/SourceModified: D:20240422202117
/Creator: Acrobat PDFMaker 24 for Word
```

A later rebuild may leave the announcement unchanged, but that is not established here. A strict historical replay should use a verifiable contemporaneous complete release or snapshot. Do not simply relabel the current PDF as January-available based on its dateline. Do not change the cutoff merely to rescue the existing experiment without declaring a new task version.

## 3. GOV.UK update timestamps

For all three GOV.UK sources, raw HTML has `govuk:updated-at` on September 24, 2026, while `govuk:first-published-at`, `govuk:public-updated-at`, and JSON-LD `datePublished`/`dateModified` remain the original 2023 dates.

The 2026 internal timestamps are not in the extracted article text. They may reflect publishing-system maintenance, but the audit cannot establish what changed. They are **not sufficient evidence of a 2026 factual article update**, and should not be described as confirmed model-visible leakage. Generic share/topic links are included, with no later event headlines found.

## 4. VanEck forecast article: future references are not automatically leakage

The source explicitly introduces a speculative 2024 outlook, includes all 15 numbered predictions, and dates its supporting charts to 2023. Its Bitcoin scenario narrates a hypothetical 2024 election in past tense inside a future-looking passage. This wording is awkward, but it is not by itself proof of a post-election update. Likewise, references to a future accounting-rule effective date are not realized outcomes.

The current selector `div.blog-post.content` excludes the broader page's additional modules. Retain the complete article; do not remove an inconvenient forecast paragraph to improve temporal-label behavior. Historical snapshot verification remains a separate unresolved provenance step.

## Reporting boundary

The already observed all-correct corporate model outputs do not establish source integrity. The appropriate conclusion is: **the status task was solved on the provided archived texts, but two cases require temporal-source qualification before a strict historical benchmark claim**. This audit neither establishes a memory failure nor validates actual prediction-market contract linkage.
