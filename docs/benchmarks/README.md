# Benchmarks

A reproducible comparison of three baseline methods against gold annotations on pathology reports:

| Method | What it is | Training? | Outputs at |
|---|---|---|---|
| **Rule-based** | Per-organ regex + lexicon. Deterministic floor LLMs have to beat. | None | `{root}/results/predictions/{dataset}/rule_based/{organ_n}/{case_id}.json` |
| **ClinicalBERT** | Bio_ClinicalBERT fine-tuned as multi-head CLS (categorical/boolean) + extractive QA (numeric spans). Outputs are merged per case. | Pooled CMUH+TCGA train | `{root}/results/predictions/{dataset}/clinicalbert/{cls,qa,merged}/{organ_n}/{case_id}.json` |
| **LLM** | DSPy + Ollama-style cancer extraction pipeline (gptoss, gemma3, qwen3.5, medgemma, ...). Multi-run by design. | None (zero / few-shot) | `{root}/results/predictions/{dataset}/llm/{model}/{run_id}/{organ_n}/{case_id}.json` |

All three methods write into the **canonical predictions tree**, so a single eval pipeline produces directly-comparable metrics CSVs.

## Defaults

- **`--folder workspace`** — the live data tree. Override to `dummy` for synthetic fixtures or absolute path for elsewhere.
- **`--datasets cmuh tcga`** — multi-dataset by default. Pass a single name to focus.
- **`--split test`** — the eval wrappers restrict every method's `non_nested` call to the test fold of `splits.json`. Mandatory when BERT is involved; keeps coverage comparable for rule-vs-LLM. Override with `--split all`.
- **`--test-fraction 0.34`** — the default split ratio used by `registrar-split` when generating `splits.json`.

## Quickstart (5-minute end-to-end on dummy data)

```bash
# 1. Generate the dummy data tree (cmuh + tcga, splits.json included).
python scripts/data/gen_dummy_skeleton.py --out dummy --clean --cases-per-organ 5 --llm-runs 1

# 2. Run rule_based predictions on both datasets in one call.
python scripts/baselines/run_rule.py --folder dummy --overwrite

# 3. Score rule vs LLM, both datasets, test split only.
python scripts/baselines/eval_rule_vs_llm.py \
    --folder dummy --llm-model gpt_oss_20b \
    --out dummy/results/eval/rule_vs_llm

# Or do the same thing with three methods on workspace defaults:
python scripts/baselines/eval_rule_bert_llm.py \
    --llm-model gpt_oss_20b \
    --out workspace/results/eval/rule_bert_llm
```

The convenience wrappers loop over datasets internally and concatenate per-dataset parquets with a `dataset` column, so the comparison tables (`headline.csv`, `per_field.csv`, `pairwise.csv`) come out stratified by (method, dataset, organ, field) automatically.

## Documentation map

1. [01_data_layout.md](01_data_layout.md) — canonical input + output paths
2. [02_train_bert.md](02_train_bert.md) — training the ClinicalBERT heads
3. [03_run_baselines.md](03_run_baselines.md) — running prediction for each method
4. [04_evaluate.md](04_evaluate.md) — running per-method `non_nested` and aggregating
5. [05_compare.md](05_compare.md) — side-by-side comparison with `run_compare` and the convenience wrappers
6. [06_methods.md](06_methods.md) — descriptions, scope, and limitations of each method

## Conventions used throughout

- **`{folder}`** in commands means `dummy` (synthetic data), `workspace` (live data on this box), or any absolute path. Resolved via `scripts/_config_loader.py:resolve_folder`.
- **`{dataset}`** is `cmuh` or `tcga`.
- **`{organ_n}`** is the 1-based numeric organ index (1=breast, 2=colorectal, 3=esophagus, 4=liver, 5=stomach, 6=lung, 7=prostate, 8=pancreas, 9=stomach, 10=cervix, ...). Defined in `scripts/eval/_common/stratify.py`.
- **`{case_id}`** is the corpus-prefixed report id, e.g. `cmuh1_17` (cmuh dataset, organ index 1, case 17).
- **`{annotator}`** is `gold`, `nhc_with_preann`, `nhc_without_preann`, `kpc_with_preann`, or `kpc_without_preann`. Eval against `gold` is the default.

## Migration notes

The legacy benchmark workflow (`registrar-benchmark` / `benchmarks.eval.run_all` / `benchmarks.eval.pairwise_compare`) has been retired. Predictions used to land at `workspace/results/benchmarks/<method>/<dataset>/<case_id>.json` (flat, no organ subdir); they now land at `{folder}/results/predictions/{dataset}/<method>/<model>/<run_id>/<organ_n>/<case_id>.json` (canonical, hierarchical).

The retired entry points still resolve but print a migration message pointing at the new tools — no silent breakage.

## Literature-comparison precedents

Scoped comparisons of clinical extraction pipelines: Alsentzer et al. 2019 (ClinicalBERT), Agrawal et al. 2022 (LLM zero-shot extraction), Goel et al. 2023 (LLM with prompting), Sushil et al. 2024 (LLM vs encoder).
