# Exploratory news replication — fixed before inference

This separate sample tests whether the primary official-source FOMC setup transfers to other financial/economic questions and genuine archived news snippets. It is not pooled with the primary study and is not a clean independent prospective confirmation.

## Sources and eligibility

- SWM-Bench frozen revision `6358edee035750dd635de4e6c3fea1668cd0d8fe`: use only article title, description, URL, publication time and market/event identity. Do not use posterior attributions, target/future prices, or z-scores to select or label items.
- Join exact tickers to official Kalshi historical market metadata and daily candles. Use actual final contract results for scoring.
- Exclude FOMC questions to broaden domain; use fixed-time macro releases, company quantities, stock indices, treasury yield and Bank of Japan decisions. Avoid anytime-threshold/early-resolution questions and already-expired status rules.
- Select first eligible lexicographic ticker per event, >=8 dated snippets across>=3 dates, and >=2 of the standard 21/7/2-day cutoffs. Minimum6 candidate snippets at each cutoff. Preserve all screening exclusions. Neither model performance nor winner label used for inclusion.
- Frozen sample: 8 event groups, 23 snapshots. Source archive contains239 cards including some inspected but ultimately unused records; count actually used cards separately during final audit.

## Evaluation

Same selection and forecasting prompts, same model and seven conditions as the primary experiment. Select exactly2 original snippets, each capped256 tokens, total<=512 body tokens. Unlike primary equal-length cards, snippets may be short; report delivered token lengths and do not combine scores with the exact512-token primary study.

Each input is restricted to documents published by cutoff. Outcomes, market phases and future candles are separate. Article bodies are not fetched after the fact to extend the supplied snippets.

## Limitations fixed before results

- Upstream news coverage was selected around market moves; this is a convenience archive, not an unbiased contemporaneous news crawl.
- Publication timestamps and title/description versions come from the archive and are not independently verified historical full-page versions.
- Retrospective contamination, small event count, correlated macro/index outcomes, and varying snippet lengths limit interpretation.
- Matching to Kalshi resolution does not create a per-news utility ground truth.
- No tuning based on primary or replication model scores; all eligible frozen cases reported.

- Runtime selector failures: if returned IDs are outside the frozen candidate pool, retain the response as a task failure with no forecast score; do not repair IDs or regenerate. Continue the remaining independent conditions. Report attempt coverage separately from successful scores and give aggregate score bounds for the unscored failure.
