# Training the ClinicalBERT heads

ClinicalBERT is the only baseline that needs training. Rule-based is deterministic; LLMs are zero / few-shot.

## What the two heads do

| Head | What it predicts | Loss | Output shape |
|---|---|---|---|
| **CLS** | Categorical / boolean fields per organ (TNM, histology, procedure, LVI, perineural, ...) and the top-level `cancer_category` | Multi-head softmax cross-entropy over `[CLS]` pooled embedding | One classification per field |
| **QA** | Numeric span fields (tumor_size, AJCC version, maximal LN size, Gleason percentages, ...) | Span-prediction (start/end logits) on per-organ question bank | Integer per field |

The two heads are trained **independently** on the **pooled** train split across all configured datasets, then merged at predict time. See [`docs/benchmarks/06_methods.md`](06_methods.md) for the architectural-scope rationale.

## Training command

```bash
python scripts/baselines/train_bert.py \
    --folder workspace --datasets cmuh tcga \
    --heads cls qa \
    --ckpt-cls ckpts/clinicalbert_cls.pt \
    --ckpt-qa  ckpts/clinicalbert_qa \
    --epochs-cls 5 --epochs-qa 3
```

Arguments:

| Flag | Default | What it controls |
|---|---|---|
| `--folder` | required | Experiment root: `dummy`, `workspace`, or absolute path. Reads inputs from `{folder}/data/{dataset}/`. |
| `--datasets` | `cmuh tcga` | Datasets to **pool** for training (concat `cmuh.train ∪ tcga.train`). Pooling is load-bearing — the annotated pool is too small to train per-dataset heads. |
| `--heads` | `cls qa` | Which heads to train. Both run sequentially in a single invocation. |
| `--organs` | breast colorectal esophagus liver stomach | Cancer-category names to keep. Other organs' rows are dropped from the train set. |
| `--ckpt-cls` | `ckpts/clinicalbert_cls.pt` | Output checkpoint path for CLS. |
| `--ckpt-qa` | `ckpts/clinicalbert_qa` | Output checkpoint dir for QA (HuggingFace `save_pretrained` format). |
| `--epochs-cls` | 5 | CLS epochs. |
| `--epochs-qa` | 3 | QA epochs. |
| `--included-only` | off | CLS-only flag: drop train cases where `cancer_excision_report=False` (no organ-specific labels to learn from). |

## Device selection

The trainer auto-detects:

1. Apple-silicon **MPS** (Mac M1/M2/M3) — preferred when available.
2. **CUDA** GPUs.
3. **CPU** fallback.

No flag is needed; the device is picked once per head. `PYTORCH_ENABLE_MPS_FALLBACK=1` and `TOKENIZERS_PARALLELISM=false` are set automatically so the trainer is friendly to laptop runs.

## Why pool train splits

Documented at length in the project memory:

- Per-dataset training would starve each classification head — at the current scale (~100 train cases across 3 TCGA organs, similar in CMUH), a 110M-parameter encoder needs the pooled train set to converge meaningfully.
- One shared encoder per baseline (no per-organ models). The CLS multi-head architecture *requires* a shared encoder so the top-level `cancer_category` head can exist; per-organ encoders would also waste capacity re-learning shared pathology vocabulary across organs.
- Predict-time eval stays per-dataset (and per-organ) so per-corpus accuracy stays visible.

Reconsider per-dataset training only when each dataset alone exceeds ~500 annotated cases.

## Training output

```
ckpts/
├── clinicalbert_cls.pt      torch.save dict with state + field_to_idx + card metadata
└── clinicalbert_qa/
    ├── config.json
    ├── pytorch_model.bin     or model.safetensors (HF defaults)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

The CLS checkpoint stores both the model weights and the per-field index→value vocabularies, so prediction time doesn't need to re-derive them from the schema.

## Smoke training on dummy data

The dummy fixtures are too small for meaningful learning, but the smoke run validates the training loop end-to-end:

```bash
python scripts/data/gen_dummy_skeleton.py --out dummy --clean --cases-per-organ 5 --llm-runs 1
python scripts/baselines/train_bert.py --folder dummy --datasets cmuh tcga --heads cls qa \
    --epochs-cls 1 --epochs-qa 1
```

Expect both heads to write checkpoints in <1 minute on CPU; accuracy will be near-random because the dummy data is synthetic noise.
