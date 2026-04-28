# Benchmarks

A reproducible cross-corpus comparison of three baseline methods against gold annotations on pathology reports:

| Method | What it is | Trained on (default) | Evaluated on (default) |
|---|---|---|---|
| **Rule-based** | Per-organ regex + lexicon. Deterministic floor LLMs have to beat. | None | TCGA |
| **ClinicalBERT** | Bio_ClinicalBERT fine-tuned as multi-head CLS (categorical/boolean) + extractive QA (numeric spans). Outputs are merged per case. | **CMUH** | TCGA (held out) |
| **LLM** | DSPy + Ollama / OpenAI cancer extraction pipeline (gptoss, gemma3, qwen3.5, medgemma, gpt-4o, ...). Multi-run by design. | None (zero / few-shot) | TCGA |

## Why TCGA-only evaluation is the canonical default

The LLM comparator (especially the OpenAI API) **cannot see CMUH** for privacy reasons. So **TCGA is the only corpus all three methods can be evaluated on as a fair like-for-like baseline**.

For BERT and rule-based to be a fair comparison against the LLM:

- **Rule-based** has no training, so it just predicts on TCGA.
- **ClinicalBERT** is trained on **CMUH only**, with TCGA fully held out. Pooled training (CMUH + TCGA) would let BERT see the test corpus at training time, and the cross-method comparison would no longer be cross-domain.
- **LLM** has no training; predicted on TCGA via its API.

This is reflected in the script defaults:

| Script | Default datasets | Notes |
|---|---|---|
| `registrar-split` | `cmuh tcga` | Both get a `splits.json` for internal use; the cross-corpus default eval doesn't need them but ablations do. |
| `train_bert.py` | `cmuh` | TCGA stays held out. Pass `--datasets cmuh tcga` for pooled-training ablation. |
| `run_rule.py` | `tcga` | Rule has no training; predict on the LLM-comparable corpus. |
| `run_bert.py` | `tcga` | BERT predicts on TCGA; `--split all` (TCGA fully held out). |
| `eval_*_vs_*.py` | `tcga` | Eval on TCGA, `--split all` (no leakage filter needed since BERT didn't train on TCGA). |

## Quickstart (cross-corpus, the default)

```bash
# 1. Generate the train/test split for both corpora.
registrar-split

# 2. Train BERT on CMUH (default). TCGA is untouched.
python scripts/baselines/train_bert.py \
    --heads cls qa --epochs-cls 5 --epochs-qa 3

# 3. Run BERT and rule on TCGA (default). Rule and BERT now have
#    predictions for the full TCGA corpus.
python scripts/baselines/run_rule.py
python scripts/baselines/run_bert.py

# 4. Run the LLM (e.g. via run_dspy_ollama_single.py or your OpenAI pipeline)
#    on TCGA. Same canonical layout under
#    {workspace}/results/predictions/tcga/llm/{model}/{run}/.

# 5. Side-by-side compare. Default --datasets tcga --split all.
python scripts/baselines/eval_rule_bert_llm.py \
    --llm-model gpt_oss_20b \
    --out workspace/results/eval/rule_bert_llm
```

## Quickstart (synthetic dummy data — same defaults)

```bash
python scripts/data/gen_dummy_skeleton.py --out dummy --clean
python scripts/baselines/run_rule.py --folder dummy
python scripts/baselines/eval_rule_vs_llm.py \
    --folder dummy --llm-model gpt_oss_20b \
    --out dummy/results/eval/rule_vs_llm
```

## Ablations: pooled training and intra-corpus eval

If you want to deviate from the cross-corpus default for an ablation:

```bash
# Pooled training — destroys TCGA's held-out status, but useful as an upper-bound ablation.
python scripts/baselines/train_bert.py --datasets cmuh tcga

# Intra-corpus benchmark (CMUH-only). The leakage guard requires --split test
# so BERT is not scored on its own training cases.
python scripts/baselines/run_bert.py --datasets cmuh --split test
python scripts/baselines/eval_bert_vs_llm.py \
    --datasets cmuh --split test \
    --llm-model gpt_oss_20b \
    --out workspace/results/eval/bert_vs_llm_intra_cmuh

# Both corpora at once with proper test-fold filtering everywhere.
python scripts/baselines/eval_rule_bert_llm.py \
    --datasets cmuh tcga --split test \
    --llm-model gpt_oss_20b \
    --out workspace/results/eval/rule_bert_llm_both
```

## Documentation map

1. [01_data_layout.md](01_data_layout.md) — canonical input + output paths
2. [02_train_bert.md](02_train_bert.md) — training the ClinicalBERT heads (CMUH-only by default)
3. [03_run_baselines.md](03_run_baselines.md) — predicting with rule, BERT, LLM
4. [04_evaluate.md](04_evaluate.md) — per-method `non_nested` evaluation
5. [05_compare.md](05_compare.md) — side-by-side comparison via `run_compare` and convenience wrappers
6. [06_methods.md](06_methods.md) — descriptions, scope, and limitations of each method

## Conventions used throughout

- **`{folder}`** in commands means `dummy` (synthetic data), `workspace` (live data on this box), or any absolute path. Resolved via `scripts/_config_loader.py:resolve_folder`. Default is `workspace`.
- **`{dataset}`** is `cmuh` or `tcga`.
- **`{organ_n}`** is the 1-based numeric organ index, **dataset-specific** per [`configs/organ_code.yaml`](../../configs/organ_code.yaml). TCGA covers 5 organs (1=breast, 2=colorectal, 3=thyroid, 4=stomach, 5=liver); CMUH covers 10 (1=pancreas, 2=breast, 3=cervix, 4=colorectal, 5=esophagus, 6=liver, 7=lung, 8=prostate, 9=stomach, 10=thyroid). The cross-corpus baseline restricts both folds to the 5 shared organs. See [01_data_layout.md](01_data_layout.md#input-layout-folderdatadataset).
- **`{case_id}`** is the corpus-prefixed report id, e.g. `cmuh1_17` (cmuh dataset, organ index 1, case 17).
- **`{annotator}`** is `gold`, `nhc_with_preann`, `nhc_without_preann`, `kpc_with_preann`, or `kpc_without_preann`. Eval against `gold` is the default.

## Migration notes

The default training contract has shifted. Earlier sessions ran pooled CMUH+TCGA training because per-dataset training would undertrain a 110M-parameter encoder. That reasoning still applies — BERT trained on ~100 CMUH cases will be undertrained — but the **fair-baseline-against-OpenAI** constraint outweighs it. The comparison the paper wants to support is "LLMs (TCGA-only) vs BERT (CMUH-only)", which requires honest cross-domain held-out evaluation.

The legacy benchmark workflow (`registrar-benchmark` / `benchmarks.eval.run_all` / `benchmarks.eval.pairwise_compare`) has been retired. Predictions used to land at `workspace/results/benchmarks/<method>/<dataset>/<case_id>.json` (flat); they now land at `{folder}/results/predictions/{dataset}/<method>/<model>/<run_id>/<organ_n>/<case_id>.json` (canonical, hierarchical).

The retired entry points still resolve but print a migration message — no silent breakage.

## Literature-comparison precedents

Scoped comparisons of clinical extraction pipelines: Alsentzer et al. 2019 (ClinicalBERT), Agrawal et al. 2022 (LLM zero-shot extraction), Goel et al. 2023 (LLM with prompting), Sushil et al. 2024 (LLM vs encoder).
