# Glossary

## Field types (`classify_field`)

| Type | Examples | Scoring approach |
|---|---|---|
| `binary` | `lymphovascular_invasion`, `cancer_excision_report` | Cohen's κ, MCC, McNemar |
| `nominal` | `cancer_category`, `procedure`, `histology` | Unweighted κ, balanced accuracy |
| `ordinal` | `grade`, `pt_category`, `nuclear_grade` | Quadratic-weighted κ, Kendall's τ-b, top-k, rank distance |
| `continuous` | `tumor_size`, `dcis_size`, `maximal_ln_size` | MAE, RMSE, CCC, ICC, Bland-Altman |
| `nested_list` | `regional_lymph_node`, `margins`, `biomarkers` | Bipartite F1, hallucination/miss rate, count MAE |

Single source of truth: `digital_registrar_research.benchmarks.eval.iaa.classify_field(field, organ)`. Per-organ taxonomy in `scope_organs.py`.

## Sections (`classify_section`)

| Section | Examples |
|---|---|
| `top_level` | `cancer_category`, `cancer_excision_report` |
| `staging` | `pt_category`, `pn_category`, `pm_category`, `tnm_descriptor`, stage groups |
| `grading` | `grade`, `nuclear_grade`, `tubule_formation`, `mitotic_rate`, `total_score`, `dcis_grade` |
| `invasion` | `lymphovascular_invasion`, `perineural_invasion` |
| `size` | `tumor_size`, `dcis_size`, `maximal_ln_size` |
| `biomarker` | `biomarker_er`, `biomarker_pr`, `biomarker_her2` |
| `other` | everything else |

Used in `non_nested/section_rollup.csv` to aggregate field-level metrics into clinically-meaningful sections.

## Outcome flags (three-way model)

See [completeness.md](completeness.md). Quick reference:

| Flag | Definition |
|---|---|
| `gold_present` | Gold has the field non-null. |
| `attempted` | Model produced a value. |
| `correct` | `attempted AND value matches gold`. |
| `wrong` | `attempted AND not correct`. |
| `field_missing` | Case loaded but this field absent. |
| `parse_error` | Whole-case load failed. |

## Scope terms

- **`FAIR_SCOPE`** — fields all four method families (DSPy, GPT-4, ClinicalBERT, rules) can produce. The head-to-head comparison set.
- **`NESTED_LIST_FIELDS`** — list-of-dict fields (margins, biomarkers, regional_lymph_node, plus organ-specific extras).
- **`BREAST_BIOMARKERS`** — `er`, `pr`, `her2`. Scored conditionally when `cancer_category == "breast"`.
- **`ORDINAL_FIELDS`** — fields with natural ranking (grade, T/N/M categories, stage groups).
- **`SPAN_FIELDS`** — integer span fields (the ClinicalBERT-QA head's domain).

Defined in `src/digital_registrar_research/benchmarks/eval/scope.py` and `scope_organs.py`.

## Annotators

Disk subdirectories under `<root>/data/<dataset>/annotations/`:

- `gold` — adjudicated reference. No `_meta` block.
- `nhc_with_preann`, `nhc_without_preann` — annotator NHC, with or without LLM pre-annotation visible.
- `kpc_with_preann`, `kpc_without_preann` — annotator KPC, same modes.

Human annotations carry a `_meta` block: `{annotator, mode, annotated_at}`.

## Run IDs

String format: `runNN[-slug]` where `NN` is zero-padded slot number and the optional slug supports multi-machine parallelism. Examples:

- `run01` — single-machine canonical.
- `run02-alpha` — slot 2 from machine "alpha".
- `run01-beta` — slot 1 from machine "beta".

Regex: `^run(\d+)(?:-([a-z0-9][a-z0-9-]*))?$`. See `scripts/_run_id.py` for parsing.

## Multi-primary

A case is `multi_primary` if it describes more than one distinct tumor (bilateral breast, double primary lung, multifocal disease). Detection heuristics in `digital_registrar_research.benchmarks.eval.multi_primary`:

- `cancer_laterality == "bilateral"` → `multi_primary`.
- `tumor_focality` ∈ {multifocal, multicentric, ...} → `multi_primary`.
- Multi-clock / multi-quadrant strings (e.g. `"12 and 3"`) → `multi_primary`.

Subgroup column: `single_primary` / `multi_primary` / `unknown`. Used to stratify all metrics so the writeup can say "in the multi-primary subgroup, accuracy was X" — direct response to **Reviewer 2.2**.

## Endpoint tiers

Defined in `configs/eval_endpoints.yaml`:

- **Primary** (12 fields) — Holm-Bonferroni p-value adjustment. Headline claims.
- **Secondary** (the rest) — Benjamini-Hochberg FDR adjustment. Exploratory.

See [multiple_comparisons.md](multiple_comparisons.md).

## Time conventions

All timestamps in manifests are UTC ISO 8601 (e.g. `2026-04-26T16:24:49.010290+00:00`). `_meta.annotated_at` from human annotations is also UTC.

Heuristic per-case duration estimation (in `iaa/preann/edit_distance__*.csv`): order an annotator's saves chronologically, take diffs, drop diffs > 30 min as session breaks. Approximate — documented as such in the CSV.

## File extensions

| Extension | When |
|---|---|
| `.parquet` | Atomic tables (`correctness_table.parquet`, `nested_atomic.parquet`, `missingness_atomic.parquet`). Compact, typed. |
| `.csv` | Aggregated reduce outputs. Human-readable, spreadsheet-friendly. |
| `.json` | Manifests, ensemble predictions. |
| `.yaml` | Endpoint pre-registration. |

The parquet → CSV split: every CSV is a deterministic *reduce* from a parquet. Re-running with new parameters (different bootstrap n, etc.) doesn't require re-scoring — just re-aggregate.
