# Side-by-side comparison

## Defaults (cross-corpus baseline)

The convenience wrappers (`eval_rule_vs_llm`, `eval_bert_vs_llm`, `eval_rule_bert_llm`) default to:

- `--folder workspace`
- `--datasets tcga` (the LLM-comparable held-out corpus)

Every gold case under `<folder>/data/<dataset>/annotations/gold/` is scored — there is no train/test split inside a corpus. Disjointness is guaranteed by the dataset boundary, and the predict-step leakage guard in `clinicalbert_*.predict` already refuses to score BERT on a dataset that was in its training set.

So a one-liner:

```bash
python scripts/baselines/eval_bert_vs_llm.py --llm-model gpt_oss_20b \
    --out workspace/results/eval/bert_vs_llm
```

evaluates BERT (trained on CMUH) vs LLM (TCGA-only) on the full TCGA corpus.

## Multi-dataset output layout

When `--datasets cmuh tcga`, per-method directories carry both per-dataset outputs AND a combined parquet with a `dataset` column:

```
{--out}/
├── non_nested_<label1>/
│   ├── cmuh/        # full per-dataset non_nested output (per_field_overall.csv etc.)
│   ├── tcga/        # ditto
│   └── correctness_table.parquet   # combined (with `dataset` column)
├── non_nested_<label2>/...
└── compare/
    ├── manifest.json
    ├── wide.csv         # rows keyed on (dataset, organ, field, case_id)
    ├── per_field.csv    # stratified by (label, dataset, organ, field)
    ├── pairwise.csv     # paired-bootstrap deltas per (dataset, organ, field) plus per-dataset ALL/ALL plus cross-dataset ALL/ALL
    └── headline.csv     # one row per (label, dataset) plus (label, ALL)
```

If you pass `--datasets tcga` (single dataset, the default), the `dataset` column is still present (with one value); compare's per_field, pairwise, and headline tables degenerate to the single-dataset case.

## Generic compare (manual)

Once each method has its own `non_nested` output (with `correctness_table.parquet`), the comparison step joins them on (case_id, organ, field) and produces:

- **`wide.csv`** — one row per cell, columns per method (e.g. `rule_based_correct`, `bert_merged_correct`, `llm_correct`, plus `_attempted` and `_pred` columns). For spreadsheet inspection.
- **`per_field.csv`** — long-form, one row per (label, organ, field) with accuracy + Wilson CI.
- **`pairwise.csv`** — every pair of methods × (organ, field), with paired-bootstrap delta + 95% CI + McNemar p-value. Plus an `ALL` rollup row per pair.
- **`headline.csv`** — single row per method: total cells, attempted, correct, coverage, accuracy.
- **`manifest.json`** — input parquets, RNG seed, n_boot, etc.

## Generic compare

```bash
python -m scripts.eval.compare.run_compare \
    --inputs rule_based:workspace/results/eval/non_nested_rule_tcga \
             bert_merged:workspace/results/eval/non_nested_bert_merged_tcga \
             llm_gptoss:workspace/results/eval/non_nested_llm_gptoss_tcga \
    --out workspace/results/eval/compare/all_three_tcga \
    --n-boot 2000 --seed 0
```

Each `--inputs` entry is `LABEL:DIR` where `DIR` contains a `correctness_table.parquet`. Labels are arbitrary; they appear in the column names of `wide.csv` and the `a_label`/`b_label` columns of `pairwise.csv`. You can compare two methods or many — the script auto-iterates every pair for `pairwise.csv`.

## Convenience wrappers

For the three most common comparisons, single-shot scripts exist that run `non_nested` for each method and then call `run_compare`:

### Rule vs LLM

```bash
python scripts/baselines/eval_rule_vs_llm.py \
    --folder workspace --dataset tcga \
    --llm-model gpt_oss_20b --llm-runs run01 run02 run03 \
    --out workspace/results/eval/rule_vs_llm_tcga \
    [--organs breast colorectal] [--n-boot 2000] [-v]
```

### BERT vs LLM

```bash
python scripts/baselines/eval_bert_vs_llm.py \
    --folder workspace --dataset tcga \
    --bert-head merged \
    --llm-model gpt_oss_20b --llm-runs run01 run02 run03 \
    --out workspace/results/eval/bert_vs_llm_tcga \
    [--organs breast colorectal] [--n-boot 2000] [-v]
```

`--bert-head` accepts `cls`, `qa`, or `merged`. `merged` is the default — it's what the eval contract assumes.

### Rule + BERT + LLM (three-way)

```bash
python scripts/baselines/eval_rule_bert_llm.py \
    --folder workspace --dataset tcga \
    --bert-head merged \
    --llm-model gpt_oss_20b --llm-runs run01 run02 run03 \
    --out workspace/results/eval/rule_bert_llm_tcga
```

Same flags as the two-way wrappers.

## Output layout (any of the above)

```
{--out}/
├── non_nested_<label1>/        # per-method non_nested output
│   ├── manifest.json
│   ├── correctness_table.parquet
│   └── ... (full per-method outputs)
├── non_nested_<label2>/        # ditto
├── non_nested_<label3>/        # ditto (three-way)
└── compare/
    ├── manifest.json
    ├── wide.csv
    ├── per_field.csv
    ├── pairwise.csv
    └── headline.csv
```

## Statistical conventions

- **Accuracy CI**: Wilson 95% on the (correct, attempted) ratio per cell.
- **Pairwise delta CI**: Paired bootstrap on the per-cell binary correctness vector, 2000 replicates by default. Delta is `mean(a_correct) - mean(b_correct)` over the **paired** subset (rows where both methods attempted the field).
- **McNemar p-value**: Two-sided, exact binomial when `n < 25` discordant cells, normal approximation otherwise. Tests whether the discordant-cell asymmetry between the two methods is significant.
- **Coverage**: `attempted / n_cases`, separate from accuracy. A method can have low coverage and high accuracy-on-attempted (rule_based on narrative fields) or high coverage and lower accuracy-on-attempted (BERT, which always emits a class).

## Reading the headline table

```
     label  n_cells_total  n_cells_attempted  n_cells_correct  coverage  accuracy_attempted    ci_lo    ci_hi
       llm            932                932              931  1.000000            0.998927 0.993947 0.999811
rule_based            604                321              291  0.531457            0.906542 0.869718 0.933750
```

- `n_cells_total`: how many (case, field) cells the method **could** have emitted (after the per-organ scope).
- `n_cells_attempted`: how many cells the method actually wrote a value for. `coverage = attempted / total`.
- `n_cells_correct`: of the attempted, how many matched gold. `accuracy_attempted = correct / attempted`.
- `ci_lo`/`ci_hi`: Wilson 95% CI on `accuracy_attempted`.

Numbers above are from a smoke run on synthetic dummy data — production numbers look different.

## Reading the pairwise table

```
a_label    b_label   organ    field             n_paired  acc_a  acc_b  delta  delta_lo  delta_hi  mcnemar_p
rule_based llm       breast   pt_category       17        0.65   0.94   -0.29  -0.47     -0.12     0.0007
rule_based llm       lung     histology         12        0.83   1.00   -0.17  -0.42      0.00     0.083
...
rule_based llm       ALL      ALL               342       0.71   0.96   -0.25  -0.30     -0.20     <1e-12
```

Each row is one (a_label, b_label) pair, organ, field cell. The `ALL/ALL` row at the bottom of each pair is the global rollup. `n_paired` is the size of the paired subset (where both methods attempted that cell). `delta_lo`/`delta_hi` is the bootstrap 95% CI on `delta = acc_a - acc_b`.

## Filtering / customizing

Pass `--organs` to either the convenience wrapper or the underlying `non_nested` calls to scope the comparison to a subset of organs. Pass `--n-boot 5000` for tighter delta CIs (slower).
