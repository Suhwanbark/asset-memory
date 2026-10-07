# Exploratory case diagnostics, frozen before repeat calls

These follow inspection of initial outputs. They are not independent confirmation or a replacement for the complete primary and news samples.

Run five fresh stateless forecast calls for each fixed evidence packet at all three cutoffs. Original selection decisions, source dates, prompts and responses remain unchanged. Repeats measure forecast sampling sensitivity conditional on selection, not selector stability. Compare mean Brier across calls, not Brier of the mean probability.

- BOJ December 2025: recency versus temporal packets. The initial case illustrates duplicate latest reports versus a previous day's policy preparation report; later cutoffs provide contrasting results. Selected after seeing that pattern.
- FOMC September 2025: repeat both original excerpt packets, then replace each selected excerpt with its complete archived source text and run five calls again. Source inspection found that the latest Powell excerpt omitted current policy guidance and the minutes excerpt selected procedural opening text. This is an important compression confound. Full-source conditions have unequal and larger token lengths and **cannot** establish equal-budget superiority. Keep every cutoff and both conditions.
- FOMC November 2024: repeat the fixed common-anchor addition comparison at every cutoff; initial runs showed a change of contribution sign. The anchor was fixed before initial scores, but this illustrative event was chosen after results. It is a 512 vs 768 token addition diagnostic.

Total: four groups × three cutoffs × two packets × five repeats = 120 fresh forecasts. No label, price history or original model reason is passed to the forecaster. No stopping after a favorable replicate and no regenerated invalid choices. All response traces preserved. Main sample results are reported regardless of these diagnostics.
