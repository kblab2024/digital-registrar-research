# Evaluation documentation

ELI5-style explanations of every metric the `scripts/eval/` pipeline produces, plus paper-ready citations.

## Decision tree — which doc do I read?

| Question | File |
|---|---|
| "What does this CSV column mean?" | [reading_outputs.md](reading_outputs.md) |
| "What's accuracy / κ / MCC / balanced accuracy?" | [non_nested_metrics.md](non_nested_metrics.md) |
| "What does MAE / RMSE / Bland-Altman LoA / CCC mean?" | [continuous_metrics.md](continuous_metrics.md) |
| "How are lymph nodes / margins / biomarkers scored?" | [nested_metrics.md](nested_metrics.md) |
| "Why is missing field different from wrong field?" | [completeness.md](completeness.md) |
| "What's Fleiss κ / vote calibration / ensemble Δ?" | [multirun.md](multirun.md) |
| "Cohen κ vs Krippendorff α — when do I use which?" | [iaa_basics.md](iaa_basics.md) |
| "What's the anchoring index? Δκ? Convergence to preann?" | [preann_effect.md](preann_effect.md) |
| "Wilson vs BCa vs two-source bootstrap CI?" | [ci_methods.md](ci_methods.md) |
| "Source-of-error decomposition / difficulty tiers / worst cases?" | [diagnostics.md](diagnostics.md) |
| "Cross-dataset Δ / KL / JS / Wasserstein?" | [cross_dataset.md](cross_dataset.md) |
| "Holm-Bonferroni vs BH? Effect sizes?" | [multiple_comparisons.md](multiple_comparisons.md) |
| "Anatomic vs pathologic stage / curated semantic neighbors?" | [confusion_pairs.md](confusion_pairs.md) |
| "What does field type / section / scope mean?" | [glossary.md](glossary.md) |
| "Show me the canonical CLI invocations." | [recipes.md](recipes.md) |
| "How do I cite this in the paper Methods section?" | [methods_citations.md](methods_citations.md) |

## CSV → metric crosswalk

| Output file | Headline metric | Doc |
|---|---|---|
| `non_nested/per_field_overall.csv` | attempted_accuracy, effective_accuracy, completeness penalty | [non_nested_metrics.md](non_nested_metrics.md), [completeness.md](completeness.md) |
| `non_nested/headline_classification.csv` | Cohen's κ (unweighted, quadratic), MCC, balanced accuracy | [non_nested_metrics.md](non_nested_metrics.md) |
| `non_nested/missingness_summary.csv` | parse_error_rate, field_missing_rate, attempted_rate | [completeness.md](completeness.md) |
| `non_nested/schema_conformance.csv` | out_of_vocab_rate (modularity-advantage signal) | [completeness.md](completeness.md) |
| `non_nested/run_consistency.csv` | Fleiss κ, flip rate, missing-flip rate, stability accuracy | [multirun.md](multirun.md) |
| `non_nested/confusion_pairs.csv` + `accuracy_collapsing_neighbors.csv` | top confusion pairs + curated neighbor analysis | [confusion_pairs.md](confusion_pairs.md) |
| `nested/per_field_per_organ.csv` | bipartite F1, hallucination/miss rate, count MAE | [nested_metrics.md](nested_metrics.md) |
| `nested/nested_missingness.csv` | parse_error / field_key_absent / empty_list / attempted | [completeness.md](completeness.md) |
| `iaa/pair_*.csv` | Cohen's κ (un/weighted), CCC, ICC, BA LoA, F1, Krippendorff α | [iaa_basics.md](iaa_basics.md) |
| `iaa/preann/delta_kappa_per_field__*.csv` | Δκ with vs without preann + paired bootstrap CI | [preann_effect.md](preann_effect.md) |
| `iaa/preann/anchoring_index__*.csv` | AI = P(human=preann \| with) − P(human=preann \| without) | [preann_effect.md](preann_effect.md) |
| `completeness/modularity_advantage.csv` | sorted method-pair Δ on attempted_rate (ablation headline) | [completeness.md](completeness.md) |
| `diagnostics/error_source_decomposition.csv` | model_error / report_ambiguity / report_silent buckets | [diagnostics.md](diagnostics.md) |
| `diagnostics/accuracy_by_difficulty_tier.csv` | accuracy stratified by IAA-derived difficulty | [diagnostics.md](diagnostics.md) |
| `cross_dataset/per_field_delta.csv` | CMUH vs TCGA Δ accuracy with bootstrap CI | [cross_dataset.md](cross_dataset.md) |
| `headline/headline_forest.csv` | unified long-form for forest-plot rendering | [reading_outputs.md](reading_outputs.md) |

## Conventions

- **Units of analysis** are typically `(case, field, run)`; a run is one full re-execution of a model on the same case set.
- Three-way outcome model: every model-vs-gold scoring distinguishes **correct / wrong / missing**. Missing further splits into `parse_error` (whole case failed) vs `field_missing` (case loaded, this field absent). See [completeness.md](completeness.md).
- All accuracy is reported in two flavors: **attempted_accuracy** (correct / attempted) and **effective_accuracy** (correct / eligible). The gap is the *completeness penalty* — primary signal for the modularity ablation.
- All proportions are reported with Wilson 95% CI by default; bootstrap CI for non-binary statistics (BCa, n_boot=2000 unless overridden).
- Multi-run statistics use case-stratified bootstrap and / or GLMM with random intercepts for `(case_id, run_id)` (see [ci_methods.md](ci_methods.md)).
- Endpoint pre-registration in `configs/eval_endpoints.yaml` separates **primary** (Holm-Bonferroni) from **secondary** (Benjamini-Hochberg) p-value adjustment families.

Every numerical claim in the paper traces back to a column in one of these CSVs, with provenance captured in the corresponding `manifest.json` (git SHA + UTC timestamp + full args).
