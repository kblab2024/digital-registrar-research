# Benchmarks

A reproducible comparison of The Digital Registrar against three standard approaches to structured extraction from pathology reports:

| Method | What it is |
|---|---|
| **Digital Registrar** (this work) | Modular DSPy pipeline with per-organ subsection signatures |
| **GPT-4 (DSPy swap)** | Same pipeline, but `dspy.configure(lm=openai/gpt-4-turbo)` — isolates model capacity from pipeline design |
| **ClinicalBERT (CLS / QA)** | Bio_ClinicalBERT fine-tuned as a classifier (categorical fields) and as a QA model (span fields) |
| **Rule-based** | Pure-regex + lexicon — establishes the floor LLM methods have to beat |

## Layout

```
src/digital_registrar_research/benchmarks/
├── baselines/
│   ├── gpt4.py                 # GPT-4 via DSPy model swap
│   ├── clinicalbert_cls.py     # ClinicalBERT classification head
│   ├── clinicalbert_qa.py      # ClinicalBERT QA head
│   └── rules.py                # regex + lexicon
├── data/
│   ├── split.py                # deterministic 100/51 stratified split
│   └── splits.json             # generated; checked-in for reproducibility
└── eval/
    ├── scope.py                # FAIR_SCOPE whitelist + field-type definitions (canonical)
    ├── metrics.py              # field-level accuracy + nested bipartite matching (canonical)
    └── run_all.py              # aggregator → workspace/results/benchmarks/tables/
```

## Scoring

`scope.py` and `metrics.py` are the canonical scoring harness — both the benchmark `run_all` aggregator and the [ablation grid](ablations.md) score against this single source so numbers are directly comparable across studies.

`FAIR_SCOPE` is the whitelist of fields used in the head-to-head comparison. `NESTED_LIST_FIELDS` (margins / biomarkers / regional_lymph_node) are intentionally N/A for the encoder/rule baselines, by design.

## Running

```bash
# 1. generate the train/test split (run once)
registrar-split

# 2. run each baseline (each writes to workspace/results/benchmarks/<method>/)
python -m digital_registrar_research.benchmarks.baselines.rules data/tcga_dataset_20251117/tcga1/<some>.txt
OPENAI_API_KEY=... python -m digital_registrar_research.benchmarks.baselines.gpt4 --split test --out workspace/results/benchmarks/gpt4
python -m digital_registrar_research.benchmarks.baselines.clinicalbert_cls --phase train --ckpt workspace/results/benchmarks/clinicalbert_cls/cls.pt
python -m digital_registrar_research.benchmarks.baselines.clinicalbert_cls --phase predict --ckpt workspace/results/benchmarks/clinicalbert_cls/cls.pt

# 3. aggregate
registrar-benchmark
```

## Literature-comparison precedents

Scoped comparisons of clinical extraction pipelines: Alsentzer et al. 2019 (ClinicalBERT), Agrawal et al. 2022 (LLM zero-shot extraction), Goel et al. 2023 (LLM with prompting), Sushil et al. 2024 (LLM vs encoder).
