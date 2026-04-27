# Per-method evaluation

The eval pipeline lives at `scripts/eval/cli.py` with subcommand dispatch. For benchmark-style accuracy / coverage on scalar fields, use the `non_nested` subcommand. Other subcommands (`nested`, `iaa`, `completeness`, `diagnostics`, `cross_dataset`, `headline`) are documented separately under [`docs/eval/`](../eval/).

The whole eval surface is **method-agnostic** — pass `--method {rule_based|clinicalbert|llm}` and it consumes the canonical predictions tree. So the rule, BERT, and LLM baselines all produce comparable metrics CSVs out of the box.

## Per-method `non_nested`

```bash
python -m scripts.eval.cli non_nested \
    --root workspace --dataset tcga \
    --method clinicalbert --model merged \
    --annotator gold \
    --out workspace/results/eval/non_nested_bert_merged_tcga
```

| Flag | Required | Notes |
|---|---|---|
| `--root` | yes | Experiment root — same `dummy` / `workspace` / abs-path shorthand as the runners. |
| `--dataset` | yes | `cmuh` or `tcga`. |
| `--method` | yes | `rule_based`, `clinicalbert`, or `llm`. |
| `--model` | required for `clinicalbert` and `llm` | For `clinicalbert`: `cls`, `qa`, or `merged`. For `llm`: the model slug, e.g. `gpt_oss_20b`. Not used for `rule_based`. |
| `--run-ids` | optional (LLM only) | Specific run IDs to score. Default: auto-discover every `run*` subdir under the model. |
| `--annotator` | default `gold` | Annotator subdir to score against. |
| `--organs` | default all | Restrict to organ indices (1..10) or names. |
| `--cases` | optional | Allowlist of case IDs (inline or `@path/to/list.txt`). |
| `--out` | default `workspace/results/eval/non_nested` | Output directory. |
| `--n-boot` | 2000 | Bootstrap replicates for CIs. |
| `--alpha` | 0.05 | CI coverage. |

## Outputs

```
{--out}/
├── manifest.json                     CLI args + git sha + UTC timestamp
├── correctness_table.parquet         atomic per-(case, organ, field, run) outcome
├── per_field_overall.csv             accuracy + Wilson CI per field, across all organs / runs
├── per_field_by_organ.csv            per-(field, organ) cell
├── per_field_by_subgroup.csv         stratified by multi-primary subgroup label
├── per_organ_overall.csv             across-field aggregate per organ
├── headline_classification.csv       precision / recall / F1 per (field, organ)
├── per_class_prf1.csv                per-class P/R/F1
├── confusion/<field>__<organ>.csv    confusion matrices
├── confusion_pairs.csv               most-confused class pairs
├── accuracy_collapsing_neighbors.csv accuracy after collapsing semantic-neighbor classes
├── rank_distance.csv                 ordinal rank distance
├── top_k_ordinal.csv                 top-k accuracy for ordinal fields
├── schema_conformance.csv            does the prediction respect the field's enum?
├── refusal_calibration.csv           is the method's "I don't know" rate calibrated?
├── run_consistency.csv               cross-run variance (LLM only)
├── section_rollup.csv                accuracy rolled up by report section
└── missingness_summary.csv           where each method drops out
```

The atomic `correctness_table.parquet` is what the side-by-side comparison consumes — see [`05_compare.md`](05_compare.md).

## Per-field schema

`correctness_table.parquet` columns:

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | Empty `""` for rule_based and clinicalbert; populated for llm. |
| `method` | str | `rule_based` / `clinicalbert` / `llm`. |
| `model` | str | Model name (head for BERT, slug for LLM, empty for rule_based). |
| `annotator` | str | `gold` etc. |
| `case_id` | str | e.g. `tcga1_17`. |
| `organ_idx` | int | 1..10. |
| `organ` | str | `breast`, `lung`, ... — derived from `organ_idx` and the gold's `cancer_category` (gold takes precedence when not `others`). |
| `subgroup` | str | Multi-primary subgroup label. |
| `field` | str | Schema field, e.g. `pt_category`, `tumor_size`, `lymphovascular_invasion`. |
| `field_kind` | str | `binary` / `nominal` / `ordinal` / `continuous` / `list_of_literals`. |
| `gold_present` | bool | Did the gold record have a non-null value? |
| `attempted` | bool | Did the method emit a non-null value? |
| `correct` | bool | Was the attempted value correct? (None when `attempted=False`.) |
| `wrong` | bool | Inverse of `correct` for attempted cells. |
| `field_missing` | bool | Method's prediction had no key for this field (vs `attempted` with explicit `null`). |
| `parse_error` | bool | Prediction file was malformed / sentinel. |
| `error_mode` | str | One of `json_parse`, `schema_invalid`, `timeout`, `refusal`, `file_missing`, `other`, or null. |
| `gold_value`, `pred_value` | Any | Raw values for human inspection. |

## Per-organ scope

`non_nested` pulls the field list **per organ** from `digital_registrar_research.benchmarks.eval.scope.get_organ_scoreable_fields(organ)` plus `cancer_category` and `cancer_excision_report`. That gives ~25–30 fields per organ rather than the 12 in the legacy `FAIR_SCOPE`. Methods are scored against the right scope automatically — there's no need to pass an explicit `--scope`.

## Filtering tricks

- **Just the test set** (BERT-style coverage): `--cases @workspace/data/{dataset}/splits.json` ... actually `splits.json` is JSON, not a flat list. Convert it first: `python -c "import json; print('\n'.join(json.load(open('dummy/data/tcga/splits.json'))['test']))" > /tmp/test_ids.txt`, then `--cases @/tmp/test_ids.txt`.
- **One organ**: `--organs breast` or `--organs 1`.
- **A specific case set**: `--cases tcga1_17 tcga1_22 cmuh3_5`.

## What to read first

Headlines for human eyes: `per_field_overall.csv` and `per_field_by_organ.csv`. Everything else is downstream of the atomic table — useful for deeper questions like "where does the model refuse?", "what's the top confusion pair?", "is the schema conformance high?".
