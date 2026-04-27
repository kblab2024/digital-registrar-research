# Running predictions (rule, BERT, LLM)

Each baseline has a dedicated runner under `scripts/baselines/` (and `scripts/pipeline/` for LLM). All three write to the canonical predictions tree, share the same `--folder` / `--dataset` arg shape, and emit comparable side files (`_summary.json`, `_log.jsonl`, `_run.log`, `_run_meta.json`).

## Rule-based

```bash
python scripts/baselines/run_rule.py \
    [--folder workspace] [--datasets cmuh tcga] \
    [--organs 1 2 3] [--limit 50] [--overwrite] [-v]
```

| Flag | Default | Effect |
|---|---|---|
| `--folder` | `workspace` | Experiment root (`dummy` / `workspace` / abs path). |
| `--datasets` | `cmuh tcga` | One or more datasets. Default loops over both. |
| `--organs` | Numeric organ subdirs to keep, e.g. `1 2 3`. Default: every organ subdir under `reports/` that has at least one `.txt`. |
| `--limit N` | Cap cases per organ (debugging). |
| `--overwrite` | Reprocess cases even if a valid output already exists. |
| `--tolerate-errors` | Always exit 0 if the script completes (even with per-case failures). |
| `-v` | DEBUG console logging. |

**Notes:**

- The rule baseline classifies the organ from the report itself (lexicon vote, `rules.classify_organ`). It does **not** read the gold annotation. This keeps the floor honest — when the lexicon classifier is wrong, the per-organ extraction emits an empty `cancer_data`.
- Runtime: <1 second per case on a laptop CPU. Pure Python, no GPU, no model.

Output: `{folder}/results/predictions/{dataset}/rule_based/{organ_n}/{case_id}.json` plus side files.

## ClinicalBERT (CLS + QA + merged)

```bash
python scripts/baselines/run_bert.py \
    [--folder workspace] [--datasets cmuh tcga] \
    [--heads cls qa merged] \
    [--ckpt-cls ckpts/clinicalbert_cls.pt] \
    [--ckpt-qa  ckpts/clinicalbert_qa] \
    [--organs breast colorectal esophagus liver stomach] [--overwrite] [-v]
```

| Flag | Default | Effect |
|---|---|---|
| `--folder` | `workspace` | Experiment root. |
| `--datasets` | `cmuh tcga` | One or more datasets. Default loops over both. |
| `--heads` | `cls qa merged` | Heads to run. `merged` requires both `cls` and `qa` outputs (run them in the same call or beforehand). |
| `--ckpt-cls` | `ckpts/clinicalbert_cls.pt` | Path to CLS checkpoint. |
| `--ckpt-qa` | `ckpts/clinicalbert_qa` | Path to QA checkpoint dir. |
| `--organs` | `breast colorectal esophagus liver stomach` | Cancer-category names to keep. |
| `--overwrite` | off | Reprocess cases even if valid outputs exist. |

**Notes:**

- Predicts on the **test split** (per `splits.json`) — BERT was trained on the train split, so scoring it on training cases is memorization, not generalization. By contrast, rule and LLM predict on every report (they have no training phase).
- Device auto-detected: MPS / CUDA / CPU.
- `merged` does a per-case key-merge: CLS provides the base (carries `cancer_category` + `cancer_excision_report`); QA's `cancer_data` scalars overlay onto CLS's, with CLS winning on collisions.

Output: `{folder}/results/predictions/{dataset}/clinicalbert/{cls|qa|merged}/{organ_n}/{case_id}.json`.

## LLM (DSPy + Ollama)

The LLM pipeline is unchanged from the existing canonical workflow — see `scripts/pipeline/run_dspy_ollama_single.py` for the single-run runner and `run_dspy_ollama_multirun.py` / `run_gpt_oss_multirun.py` for multi-run sweeps.

```bash
python scripts/pipeline/run_dspy_ollama_single.py \
    --model gptoss --folder workspace --dataset tcga \
    [--run run01] [--organs 1 2] [--limit N] [--overwrite] [-v]
```

`--model` accepts: `gptoss`, `gemma3`, `gemma4`, `qwen3_5`, `medgemmalarge`, `medgemmasmall` (each auto-loads `configs/dspy_ollama_{alias}.yaml` for decoding overrides).

Output: `{folder}/results/predictions/{dataset}/llm/{model_slug}/{run_id}/{organ_n}/{case_id}.json`. The model slug is derived from the model id (`ollama_chat/gpt-oss:20b` → `gpt_oss_20b`).

**Multi-run pattern:** LLMs are stochastic, so the canon is to run 3-10 seeded runs per model. The aggregated `_manifest.yaml` at `{folder}/results/predictions/{dataset}/llm/{model_slug}/_manifest.yaml` lists every run and its parse-error rate.

## Caching and resume

All three runners check for valid existing outputs before re-running a case. To force a fresh run, pass `--overwrite`. Partial / errored outputs (those with `_pipeline_error: true` in the JSON) are NOT considered valid and will be retried automatically.

## Side files (uniform across all three)

| File | Content |
|---|---|
| `_summary.json` | Run-level totals: `n_cases`, `n_ok`, `n_pipeline_error`, `n_cached`, per-organ counts, wall time, parse error rate. |
| `_log.jsonl` | One JSONL row per case: `{case_id, organ, status, latency_s, parse_success, is_cancer, cancer_category, error, started_at}`. |
| `_run.log` | Full-verbosity log. |
| `_run_meta.json` | Provenance: git sha, host, argv, started_at / finished_at, full path resolution. |

For LLM runs only, the model-level `_manifest.yaml` is appended/updated idempotently with one entry per run (so re-running a particular run slot updates its row in place).
