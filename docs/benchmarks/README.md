# Benchmarks

A reproducible comparison of three baseline methods against gold annotations on pathology reports:

| Method | What it is | Training? | Outputs at |
|---|---|---|---|
| **Rule-based** | Per-organ regex + lexicon. Deterministic floor LLMs have to beat. | None | `{root}/results/predictions/{dataset}/rule_based/{organ_n}/{case_id}.json` |
| **ClinicalBERT** | Bio_ClinicalBERT fine-tuned as multi-head CLS (categorical/boolean) + extractive QA (numeric spans). Outputs are merged per case. | Pooled CMUH+TCGA train | `{root}/results/predictions/{dataset}/clinicalbert/{cls,qa,merged}/{organ_n}/{case_id}.json` |
| **LLM** | DSPy + Ollama-style cancer extraction pipeline (gptoss, gemma3, qwen3.5, medgemma, ...). Multi-run by design. | None (zero / few-shot) | `{root}/results/predictions/{dataset}/llm/{model}/{run_id}/{organ_n}/{case_id}.json` |

All three methods write into the **canonical predictions tree**, so a single eval pipeline produces directly-comparable metrics CSVs.

## Quickstart (5-minute end-to-end on dummy data)

```bash
# 1. Generate a synthetic dataset under dummy/.
python scripts/data/gen_dummy_skeleton.py --out dummy --clean --cases-per-organ 5 --llm-runs 1

# 2. Run rule_based predictions on every report.
python scripts/baselines/run_rule.py --folder dummy --dataset cmuh --overwrite

# 3. (Skip BERT training on dummy — checkpoints aren't shipped.) Run LLM via the
#    pipeline runner already in the repo, or rely on the LLM predictions
#    populated by gen_dummy_skeleton.

# 4. Score each method with the canonical eval pipeline.
python -m scripts.eval.cli non_nested --root dummy --dataset cmuh \
    --method rule_based --annotator gold \
    --out dummy/results/eval/non_nested_rule

python -m scripts.eval.cli non_nested --root dummy --dataset cmuh \
    --method llm --model gpt_oss_20b --annotator gold \
    --out dummy/results/eval/non_nested_llm

# 5. Side-by-side compare (paired bootstrap + McNemar + Wilson CIs).
python -m scripts.eval.compare.run_compare \
    --inputs rule_based:dummy/results/eval/non_nested_rule \
             llm:dummy/results/eval/non_nested_llm \
    --out dummy/results/eval/compare/rule_vs_llm

# Or the convenience wrapper that does steps 4 + 5 in one call:
python scripts/baselines/eval_rule_vs_llm.py \
    --folder dummy --dataset cmuh --llm-model gpt_oss_20b \
    --out dummy/results/eval/rule_vs_llm
```

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
